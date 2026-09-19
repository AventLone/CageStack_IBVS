#include "perception/nodes/Localization_LIO.h"
#include "perception/tools/cloud_conversion.hpp"
#include <algorithm>
#include <cmath>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl_conversions/pcl_conversions.h>
#include <tf2_eigen/tf2_eigen.hpp>

Localization_LIO::Localization_LIO(const std::string& node_name) : Node(node_name)
{
    const auto extrinsic = [this](const std::string& name)
    {
        const auto t = declare_parameter<std::vector<double>>(name + ".translation", {0.0, 0.0, 0.0});
        const auto q = declare_parameter<std::vector<double>>(name + ".quaternion_xyzw", {0.0, 0.0, 0.0, 1.0});
        return Sophus::SE3d(Eigen::Quaterniond(q[3], q[0], q[1], q[2]).normalized(),
                            Eigen::Vector3d(t[0], t[1], t[2]));
    };
    mT_il = extrinsic("lidar_to_imu");
    mT_ib = extrinsic("base_to_imu");
    mMapFrame = declare_parameter<std::string>("map_frame", "map");
    mInitDuration = declare_parameter<double>("imu_init_duration", 0.5);
    mMapResolution = declare_parameter<double>("map_resolution", 0.1);
    mKeyframeTranslation = declare_parameter<double>("keyframe_translation", 0.1);
    mKeyframeRotation = declare_parameter<double>("keyframe_rotation", 0.035);
    mIntensityThreshold = declare_parameter<double>("intensity_threshold", -1.0);
    mIntensityKeepRatio = std::clamp(declare_parameter<double>("intensity_keep_ratio", 0.6), 0.01, 1.0);
    mSelfFilterHalfWidth = declare_parameter<double>("self_filter_half_width", 0.8);
    mPathMaxPoses = declare_parameter<int>("path_max_poses", 2000);
    mFilterConfig.accel_noise_density = declare_parameter<double>("eskf.accel_noise_density", 0.03);
    mFilterConfig.gyro_noise_density = declare_parameter<double>("eskf.gyro_noise_density", 0.00034906585);
    mFilterConfig.innovation_gate = declare_parameter<double>("eskf.innovation_gate", 22.458);
    const double position_std = declare_parameter<double>("eskf.lidar_position_std", 0.03);
    const double rotation_std = declare_parameter<double>("eskf.lidar_rotation_std", 0.00523598776);
    mLidarCov.topLeftCorner<3, 3>().diagonal().setConstant(position_std * position_std);
    mLidarCov.bottomRightCorner<3, 3>().diagonal().setConstant(rotation_std * rotation_std);

    GICP::Config config;
    config.voxel_size = declare_parameter<double>("gicp.voxel_size", 0.2);
    config.max_correspondence_distance = declare_parameter<double>("gicp.max_correspondence_distance", 0.5);
    config.max_fitness_score = declare_parameter<double>("gicp.max_fitness_score", 0.01); // MSE, 10 cm RMSE
    config.min_correspondences = declare_parameter<int>("gicp.min_correspondences", 30);
    config.max_iterations = declare_parameter<int>("gicp.max_iterations", 30);
    config.convergence_translation = 1e-4;
    config.convergence_rotation = 1e-4;
    mGicp.setConfig(config);

    mProcessedScanVisPub = create_publisher<sensor_msgs::msg::PointCloud2>("/scan_vis", rclcpp::SensorDataQoS());
    mVoxelMapPub = create_publisher<sensor_msgs::msg::PointCloud2>("/voxel_map", rclcpp::SensorDataQoS());
    mBasePosePub = create_publisher<geometry_msgs::msg::PoseStamped>("/base_pose", rclcpp::SensorDataQoS());
    mBasePosePathPub = create_publisher<nav_msgs::msg::Path>("/base_pose_path", rclcpp::SensorDataQoS());
    mLidarScanSub = create_subscription<sensor_msgs::msg::PointCloud2>(
        declare_parameter<std::string>("lidar_topic", "/hesai/pandar"), rclcpp::SensorDataQoS(),
        [this](sensor_msgs::msg::PointCloud2::ConstSharedPtr msg)
        {
            {
                std::lock_guard lock(mMutex);
                mPendingScan = std::move(msg); // Bound latency; IMU history is retained across dropped scans.
            }
            mTrigger.notify_one();
        });
    mImuSub = create_subscription<sensor_msgs::msg::Imu>(
        declare_parameter<std::string>("imu_topic", "/imu"), rclcpp::SensorDataQoS().keep_last(1000),
        [this](sensor_msgs::msg::Imu::ConstSharedPtr msg) { onImu(*msg); });
    mLidarWorker = std::thread(&Localization_LIO::lidarWorkerLoop, this);
    RCLCPP_INFO(get_logger(), "Keep the IMU stationary for the first %.2f s.", mInitDuration);
}

Localization_LIO::~Localization_LIO()
{
    {
        std::lock_guard lock(mMutex);
        mShutdown = true;
    }
    mTrigger.notify_one();
    mLidarWorker.join();
}

void Localization_LIO::onImu(const sensor_msgs::msg::Imu& msg)
{
    const lio::ImuData sample{
        rclcpp::Time(msg.header.stamp).seconds(),
        {msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z},
        {msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z}};
    {
        std::lock_guard lock(mMutex);
        mImuBuffer.push_back(sample);
        if (!mLiveFilter)
        {
            if (mImuBuffer.size() >= 20 && sample.timestamp - mImuBuffer.front().timestamp >= mInitDuration)
            {
                mLiveFilter.emplace(mFilterConfig);
                mLiveFilter->initialize({mImuBuffer.begin(), mImuBuffer.end()});
                mInitialFilter = mLiveFilter;
                RCLCPP_INFO(get_logger(), "IMU initialized. Waiting for a complete LiDAR scan.");
            }
        }
        else
        {
            mLiveFilter->predict(sample);
            mBasePosePub->publish(basePose(*mLiveFilter));
        }
    }
    mTrigger.notify_one();
}

bool Localization_LIO::alignICP(const pcl::PointCloud<pcl::PointXYZ>& scan, lio::ESKF& filter)
{
    const Sophus::SE3d prediction = filter.pose() * mT_il;
    const auto result = mGicp.align(scan, Eigen::Isometry3d(prediction.matrix()));
    if (!result.converged || result.num_correspondences < mGicp.config().min_correspondences ||
        result.fitness_score > mGicp.config().max_fitness_score)
    {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000,
                            "GICP rejected: matches=%zu, MSE=%.4f, converged=%d.",
                            result.num_correspondences, result.fitness_score, result.converged);
        return false;
    }

    const Sophus::SE3d T_wl(result.transform.rotation(), result.transform.translation());
    // Convert LiDAR pose noise [dp_world, dtheta_lidar] to IMU pose noise,
    // including the position/rotation correlation introduced by the lever arm.
    const Sophus::SE3d T_li = mT_il.inverse();
    lio::ESKF::MeasurementCov J = lio::ESKF::MeasurementCov::Identity();
    J.topRightCorner<3, 3>() = -T_wl.rotationMatrix() * Sophus::SO3d::hat(T_li.translation());
    J.bottomRightCorner<3, 3>() = mT_il.rotationMatrix();
    const bool accepted = filter.observe(T_wl * T_li, J * mLidarCov * J.transpose());
    if (!accepted)
    {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000, "LiDAR pose rejected by ESKF innovation gate.");
    }
    return accepted;
}

void Localization_LIO::updateVoxelMap(const pcl::PointCloud<pcl::PointXYZ>& scan, const Sophus::SE3d& T_wl)
{
    pcl::PointCloud<pcl::PointXYZ> world_scan;
    pcl::transformPointCloud(scan, world_scan, T_wl.matrix().cast<float>().eval());
    mGicp.insertTargetPoints(world_scan);
    *mMap += world_scan;
    pcl::VoxelGrid<pcl::PointXYZ> voxel;
    voxel.setLeafSize(mMapResolution, mMapResolution, mMapResolution);
    voxel.setInputCloud(mMap);
    auto downsampled = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    voxel.filter(*downsampled);
    mMap = std::move(downsampled);
    mLastKeyframe = T_wl;
}

geometry_msgs::msg::PoseStamped Localization_LIO::basePose(const lio::ESKF& filter) const
{
    geometry_msgs::msg::PoseStamped msg;
    msg.header.frame_id = mMapFrame;
    msg.header.stamp = rclcpp::Time(static_cast<int64_t>(std::llround(filter.state().timestamp * 1e9)));
    msg.pose = tf2::toMsg(Eigen::Isometry3d((filter.pose() * mT_ib).matrix()));
    return msg;
}

void Localization_LIO::publishScan(const pcl::PointCloud<pcl::PointXYZ>& scan, const lio::ESKF& filter)
{
    const auto pose = basePose(filter);
    if (mProcessedScanVisPub->get_subscription_count() > 0)
    {
        pcl::PointCloud<pcl::PointXYZ> transformed;
        pcl::transformPointCloud(scan, transformed, (filter.pose() * mT_il).matrix().cast<float>().eval());
        sensor_msgs::msg::PointCloud2 msg;
        pcl::toROSMsg(transformed, msg);
        msg.header = pose.header;
        mProcessedScanVisPub->publish(msg);
    }
    if (mVoxelMapPub->get_subscription_count() > 0)
    {
        sensor_msgs::msg::PointCloud2 msg;
        pcl::toROSMsg(*mMap, msg);
        msg.header = pose.header;
        mVoxelMapPub->publish(msg);
    }
    if (mPathMaxPoses > 0)
    {
        mBasePosePath.header = pose.header;
        mBasePosePath.poses.push_back(pose);
        if (mBasePosePath.poses.size() > static_cast<std::size_t>(mPathMaxPoses))
        {
            mBasePosePath.poses.erase(mBasePosePath.poses.begin());
        }
        mBasePosePathPub->publish(mBasePosePath);
    }
}

void Localization_LIO::lidarWorkerLoop()
{
    std::optional<lio::ESKF> filter;
    lio::ImuProcessor imu_processor(mT_il);
    while (true)
    {
        sensor_msgs::msg::PointCloud2::ConstSharedPtr msg;
        {
            std::unique_lock lock(mMutex);
            mTrigger.wait(lock, [this] { return mShutdown || (mInitialFilter && mPendingScan); });
            if (mShutdown) return;
            if (!filter) filter = mInitialFilter;
            msg = std::move(mPendingScan);
        }
        StampedCloud cloud;
        if (!parseCloudMsg(*msg, cloud))
        {
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000,
                                "Expected nonempty Hesai cloud with FLOAT64 absolute timestamp [s].");
            continue;
        }
        // Initialization may finish inside a scan. Start with the next complete scan.
        if (cloud.begin_time < filter->state().timestamp) continue;

        std::vector<lio::ImuData> imu;
        {
            std::unique_lock lock(mMutex);
            mTrigger.wait(lock, [this, &cloud] { return mShutdown || mImuBuffer.back().timestamp >= cloud.end_time; });
            if (mShutdown) return;
            const auto end = std::lower_bound(mImuBuffer.begin(), mImuBuffer.end(), cloud.end_time,
                [](const lio::ImuData& sample, double t) { return sample.timestamp < t; });
            imu.assign(mImuBuffer.begin(), std::next(end));
        }
        imu_processor.process(imu, cloud, *filter);

        if (!mIntensityAnalyzed && mIntensityThreshold < 0.0f)
        {
            std::vector<float> intensities;
            intensities.reserve(cloud.points.size());
            for (const auto& point : cloud.points) intensities.push_back(point.intensity);
            const auto index = static_cast<std::size_t>((1.0f - mIntensityKeepRatio) * (intensities.size() - 1));
            std::nth_element(intensities.begin(), intensities.begin() + index, intensities.end());
            mIntensityThreshold = intensities[index];
            mIntensityAnalyzed = true;
            RCLCPP_INFO(get_logger(), "Intensity threshold: %.2f.", mIntensityThreshold);
        }
        auto selected = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
        selected->reserve(cloud.points.size());
        const auto T_il = mT_il.cast<float>();
        for (const auto& point : cloud.points)
        {
            const Eigen::Vector3f p = T_il * point.position; // Self filter in IMU frame.
            if (point.intensity >= mIntensityThreshold &&
                (std::abs(p.x()) > mSelfFilterHalfWidth || std::abs(p.y()) > mSelfFilterHalfWidth))
            {
                selected->emplace_back(point.position.x(), point.position.y(), point.position.z());
            }
        }
        pcl::PointCloud<pcl::PointXYZ> scan;
        pcl::VoxelGrid<pcl::PointXYZ> voxel;
        voxel.setLeafSize(mMapResolution, mMapResolution, mMapResolution);
        voxel.setInputCloud(selected);
        voxel.filter(scan);

        bool accepted = false;
        if (scan.size() >= mGicp.config().min_correspondences)
        {
            accepted = !mGicp.hasTarget() || alignICP(scan, *filter);
        }
        const Sophus::SE3d T_wl = filter->pose() * mT_il;
        if (accepted && (!mGicp.hasTarget() ||
            (T_wl.translation() - mLastKeyframe.translation()).norm() >= mKeyframeTranslation ||
            (mLastKeyframe.so3().inverse() * T_wl.so3()).log().norm() >= mKeyframeRotation))
        {
            updateVoxelMap(scan, T_wl);
        }

        // Correct at scan end, then replay every later IMU on a filter copy.
        // The next callback continues from the corrected latest-time state.
        {
            std::lock_guard lock(mMutex);
            mLiveFilter = filter;
            for (const auto& sample : mImuBuffer)
            {
                if (sample.timestamp > mLiveFilter->state().timestamp) mLiveFilter->predict(sample);
            }
            // Keep one sample at/before scan end for interpolation on the next scan.
            while (mImuBuffer.size() > 1 && mImuBuffer[1].timestamp <= cloud.end_time)
            {
                mImuBuffer.pop_front();
            }
        }
        publishScan(scan, *filter);
    }
}
