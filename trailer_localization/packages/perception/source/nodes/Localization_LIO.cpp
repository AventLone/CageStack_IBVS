#include "perception/nodes/Localization_LIO.h"
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl_conversions/pcl_conversions.h>
#include <tf2_eigen/tf2_eigen.hpp>

Localization_LIO::Localization_LIO(const std::string& node_name) : Node(node_name), mTfBuffer(this->get_clock()), mTfListener(mTfBuffer)
{
    while (rclcpp::ok())
    {
        try
        {
            const auto transform = tf2::transformToEigen(mTfBuffer.lookupTransform("base_link", "imu", tf2::TimePointZero));
            mTi2b = Sophus::SE3d(transform.rotation(), transform.translation());
            break;
        }
        catch (const tf2::TransformException& ex)
        {
            RCLCPP_ERROR(this->get_logger(), "Could not transform fork to body: %s", ex.what());
        }
    }

    Sophus::SE3d T_IL;

    while (rclcpp::ok())
    {
        try
        {
            const auto transform = tf2::transformToEigen(mTfBuffer.lookupTransform("imu", "PandarXT-32", tf2::TimePointZero));
            T_IL = Sophus::SE3d(transform.rotation(), transform.translation());
            break;
        }
        catch (const tf2::TransformException& ex)
        {
            RCLCPP_ERROR(this->get_logger(), "Could not transform fork to body: %s", ex.what());
        }
    }

    // const auto extrinsic = [this](const std::string& prefix)
    // {
    //     const auto t = declare_parameter<std::vector<double>>(prefix + ".translation", {0.0, 0.0, 0.0});
    //     const auto q = declare_parameter<std::vector<double>>(prefix + ".quaternion_xyzw", {0.0, 0.0, 0.0, 1.0});
    //     if (t.size() != 3 || q.size() != 4) throw std::invalid_argument("Extrinsic requires xyz and quaternion xyzw.");
    //     const Eigen::Vector3d translation(t[0], t[1], t[2]);
    //     Eigen::Quaterniond rotation(q[3], q[0], q[1], q[2]);
    //     if (!translation.allFinite() || !rotation.coeffs().allFinite() || rotation.norm() < 1e-9)
    //         throw std::invalid_argument("Invalid extrinsic calibration.");
    //     rotation.normalize();
    //     return Sophus::SE3d(rotation, translation);
    // };

    // const auto T_IL = extrinsic("lio.lidar_to_imu");
    // mTi2b = extrinsic("lio.base_to_imu");

    lio::ESKF::Config filter;
    filter.gyro_noise = declare_parameter("lio.gyro_noise", filter.gyro_noise);
    filter.accel_noise = declare_parameter("lio.accel_noise", filter.accel_noise);
    filter.gyro_bias_noise = declare_parameter("lio.gyro_bias_noise", filter.gyro_bias_noise);
    filter.accel_bias_noise = declare_parameter("lio.accel_bias_noise", filter.accel_bias_noise);
    filter.gravity_magnitude = declare_parameter("lio.gravity_magnitude", filter.gravity_magnitude);
    filter.max_imu_gap = declare_parameter("lio.max_imu_gap", filter.max_imu_gap);
    filter.max_iterations = declare_parameter("lio.max_iterations", filter.max_iterations);
    const int min_matches = declare_parameter("lio.min_correspondences", 30);

    if (min_matches < 3)
    {
        throw std::invalid_argument("min_correspondences must be at least three.");
    }

    filter.min_correspondences = static_cast<std::size_t>(min_matches);
    filter.max_plane_rmse = declare_parameter("lio.max_plane_rmse", filter.max_plane_rmse);
    mEskf = std::make_unique<lio::ESKF>(filter, T_IL);

    lio::LidarMeasurement::Config measurement;
    measurement.lidar_noise = declare_parameter("lio.lidar_noise", measurement.lidar_noise);
    measurement.max_residual = declare_parameter("lio.max_residual", measurement.max_residual);
    measurement.max_neighbor_distance = declare_parameter("lio.max_neighbor_distance", measurement.max_neighbor_distance);
    measurement.max_plane_distance = declare_parameter("lio.max_plane_distance", measurement.max_plane_distance);
    measurement.voxel_radius = declare_parameter("lio.voxel_radius", measurement.voxel_radius);

    mLidarMeasurement = std::make_unique<lio::LidarMeasurement>(measurement, T_IL);
    mMapConfig.estimate_covariances = false;
    mMapConfig.voxel_size = static_cast<float>(declare_parameter("lio.map_voxel_size", 0.5));
    const int max_voxels = declare_parameter("lio.max_map_voxels", 200000);
    if (max_voxels <= 0) throw std::invalid_argument("max_map_voxels must be positive.");
    mMapConfig.max_voxels_num = static_cast<std::size_t>(max_voxels);
    mMap = std::make_unique<SparseVoxel>(mMapConfig);
    mScanResolution = declare_parameter("lio.scan_resolution", mScanResolution);
    mImuWaitTimeout = declare_parameter("lio.imu_wait_timeout", mImuWaitTimeout);
    mInitializationDuration = declare_parameter("lio.initialization_duration", mInitializationDuration);
    for (const double value : {mScanResolution, mImuWaitTimeout, mInitializationDuration})
        if (!std::isfinite(value) || value <= 0.0) throw std::invalid_argument("LIO resolutions and durations must be positive.");
    if (mInitializationDuration < 0.5) throw std::invalid_argument("Initialization needs at least 0.5 s.");
    mImuTimeOffset = declare_parameter("lio.imu_time_offset", 0.0);
    if (!std::isfinite(mImuTimeOffset)) throw std::invalid_argument("IMU time offset must be finite.");
    mTimingConfig.time_field = declare_parameter("lio.time_field", mTimingConfig.time_field);
    mTimingConfig.time_mode = declare_parameter("lio.time_mode", mTimingConfig.time_mode);
    mTimingConfig.time_scale = declare_parameter("lio.time_scale", mTimingConfig.time_scale);
    mTimingConfig.stamp_is_end = declare_parameter("lio.stamp_is_end", mTimingConfig.stamp_is_end);
    mTimingConfig.allow_untimed_cloud = declare_parameter("lio.allow_untimed_cloud", false);
    mTimingConfig.scan_duration = declare_parameter("lio.scan_duration", mTimingConfig.scan_duration);
    mTimingConfig.max_scan_duration = declare_parameter("lio.max_scan_duration", mTimingConfig.max_scan_duration);
    mWorldFrame = declare_parameter<std::string>("lio.world_frame", "map");
    mImuFrame = declare_parameter<std::string>("lio.imu_frame", "imu_sensor_frame");
    mLidarFrame = declare_parameter<std::string>("lio.lidar_frame", "PandarXT-32");

    mIntensityThreshold = static_cast<float>(declare_parameter("intensity_threshold", -1.0));
    mIntensityKeepRatio = static_cast<float>(declare_parameter("intensity_keep_ratio", 0.6));
    if (!std::isfinite(mIntensityThreshold) || !std::isfinite(mIntensityKeepRatio) ||
        mIntensityKeepRatio <= 0.0f || mIntensityKeepRatio > 1.0f)
    {
        throw std::invalid_argument("Invalid intensity filter parameters.");
    }

    mProcessedScanVisPub = create_publisher<sensor_msgs::msg::PointCloud2>("/scan_vis", rclcpp::SensorDataQoS());
    mVoxelMapPub = create_publisher<sensor_msgs::msg::PointCloud2>("/voxel_map", rclcpp::SensorDataQoS());
    mBasePosePub = create_publisher<geometry_msgs::msg::PoseStamped>("/base_pose", rclcpp::SensorDataQoS());
    mBasePosePathPub = create_publisher<nav_msgs::msg::Path>("/base_pose_path", rclcpp::SensorDataQoS());

    mLidarScanSub = create_subscription<sensor_msgs::msg::PointCloud2>(
        declare_parameter<std::string>("lio.lidar_topic", "/hesai/pandar"), rclcpp::SensorDataQoS(),
        [this](sensor_msgs::msg::PointCloud2::ConstSharedPtr msg)
        {
            std::lock_guard lock(mBufferMutex);
            if (mLidarFrame.empty()) mLidarFrame = msg->header.frame_id;
            if (msg->header.frame_id != mLidarFrame)
            {
                RCLCPP_WARN(get_logger(), "Dropping LiDAR with unexpected frame_id.");
                return;
            }
            if (mScanBuffer.size() >= 4) mScanBuffer.pop_front();
            mScanBuffer.push_back(std::move(msg));
            mTrigger.notify_one();
        });

    mImuSub = create_subscription<sensor_msgs::msg::Imu>(
        declare_parameter<std::string>("lio.imu_topic", "/alphasense/imu"), rclcpp::SensorDataQoS().keep_last(2000),
        [this](const sensor_msgs::msg::Imu::ConstSharedPtr& msg)
        {
            const lio::ImuData imu{rclcpp::Time(msg->header.stamp).seconds() + mImuTimeOffset,
                Eigen::Vector3d(msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z),
                Eigen::Vector3d(msg->linear_acceleration.x, msg->linear_acceleration.y, msg->linear_acceleration.z)};

            std::lock_guard lock(mBufferMutex);
            if (mImuFrame.empty())
            {
                mImuFrame = msg->header.frame_id;
            }

            if (msg->header.frame_id != mImuFrame || !lio::ImuProcessor::finite(imu) || imu.timestamp <= mLastImuTime)
            {
                RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "Dropping invalid, duplicate or out-of-order IMU.");
                return;
            }
            mLastImuTime = imu.timestamp;
            mImuBuffer.push_back(imu);
            // Preserve the full integration history, not just the latest sample.
            while (mImuBuffer.size() > 20000)
            {
                mImuBuffer.pop_front();
            }
            mTrigger.notify_one();
        });
    mLidarWorker = std::thread(&Localization_LIO::lidarWorkerLoop, this);
    RCLCPP_INFO(get_logger(), "Tightly coupled LIO started; keep the IMU stationary for initialization.");
}

Localization_LIO::~Localization_LIO()
{
    {
        std::lock_guard lock(mBufferMutex);
        mIsShutdown = true;
    }
    mTrigger.notify_all();
    if (mLidarWorker.joinable())
    {
        mLidarWorker.join();
    }
}

bool Localization_LIO::collectImu(const double scan_begin, double scan_end, std::vector<lio::ImuData>& imu)
{
    std::unique_lock lock(mBufferMutex);
    const bool covered = mTrigger.wait_for(lock, std::chrono::duration<double>(mImuWaitTimeout), [this, scan_end]
        {
            return mIsShutdown || (!mImuBuffer.empty() && mImuBuffer.back().timestamp >= scan_end);
        });

    if (mIsShutdown)
    {
        return false;
    }

    if (!covered)
    {
        RCLCPP_WARN(get_logger(), "Waiting for IMU coverage failed; check timestamp clock/units and imu_time_offset.");
        return false;
    }

    const double begin = mEskf->initialized() ? mEskf->nominalState().timestamp : scan_begin - mInitializationDuration;
    if (mEskf->initialized() && scan_begin < begin)
    {
        RCLCPP_WARN(get_logger(), "Skipping overlapping or stale scan.");
        return false;
    }

    if (mImuBuffer.front().timestamp > begin)
    {
        if (mEskf->initialized())
        {
            mEskf->clear();
            mMap->clear();
            mBasePosePath.poses.clear();
            RCLCPP_ERROR(get_logger(), "IMU history was lost: resetting local map; stationary reinitialization required.");
        }
        else RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "Insufficient IMU history before scan; waiting for initialization.");
        return false;
    }

    // Keep one sample <= state time and the first sample >= scan end.
    while (mImuBuffer.size() > 2 && mImuBuffer[1].timestamp <= begin)
    {
        mImuBuffer.pop_front();
    }

    for (const auto& sample : mImuBuffer)
    {
        imu.push_back(sample);
        if (sample.timestamp >= scan_end)
        {
            break;
        }
    }

    for (std::size_t i = 1; i < imu.size(); ++i)
    {
        if (imu[i].timestamp - imu[i - 1].timestamp > mEskf->config().max_imu_gap)
        {
            mEskf->clear();
            mMap->clear();
            mBasePosePath.poses.clear();
            RCLCPP_ERROR(get_logger(), "IMU gap: resetting local map; stationary reinitialization required.");
            return false;
        }
    }
    return true;
}

void Localization_LIO::lidarWorkerLoop()
{
    while (rclcpp::ok())
    {
        sensor_msgs::msg::PointCloud2::ConstSharedPtr msg;
        {
            std::unique_lock lock(mBufferMutex);
            mTrigger.wait(lock, [this] { return mIsShutdown || !mScanBuffer.empty(); });
            if (mIsShutdown) return;
            msg = std::move(mScanBuffer.front());
            mScanBuffer.pop_front();
        }

        try
        {
            auto scan = lio::readTimedCloud(*msg, mTimingConfig);
            if (!scan.has_point_time)
            {
                RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
                    "Untimed cloud explicitly enabled: within-scan deskew is unavailable.");
            }

            std::vector<lio::ImuData> imu;
            if (!collectImu(scan.begin, scan.end, imu))
            {
                continue;
            }

            if (!mEskf->initialized())
            {
                if (const auto initialization = lio::ImuProcessor::buildImuSequence(imu, scan.begin - mInitializationDuration, scan.begin);
                    !mEskf->initialize(initialization))
                {
                    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                        "IMU initialization requires stationary, gravity-including acceleration in m/s^2.");
                    continue;
                }
            }

            if (!mIntensityAnalyzed && !scan.intensities.empty())
            {
                std::vector<float> values;
                for (float value : scan.intensities) if (std::isfinite(value)) values.push_back(value);
                if (!values.empty())
                {
                    std::ranges::sort(values);
                    const auto index = std::min(values.size() - 1, static_cast<std::size_t>((1.0f - mIntensityKeepRatio) * values.size()));
                    if (mIntensityThreshold < 0.0f) mIntensityThreshold = values[index];
                    mIntensityAnalyzed = true;
                }
            }
            std::vector<lio::PointXYZT> points;
            points.reserve(scan.points.size());
            for (std::size_t i = 0; i < scan.points.size(); ++i)
            {
                const auto& point = scan.points[i];
                if (!std::isfinite(scan.intensities[i]) || scan.intensities[i] < mIntensityThreshold)
                {
                    continue;
                }
                if (std::abs(point.x) > 0.8f || std::abs(point.y) > 0.8f) points.push_back(point);
            }
            // Check raw (not interpolated) gaps before any propagation/deskew.
            mEskf->processScan(imu, points, scan.begin, scan.end);
            const auto cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
            cloud->reserve(points.size());
            for (const auto& point : points) cloud->emplace_back(point.x, point.y, point.z);
            pcl::PointCloud<pcl::PointXYZ> filtered;
            pcl::VoxelGrid<pcl::PointXYZ> voxel_filter;
            const auto resolution = static_cast<float>(mScanResolution);
            voxel_filter.setLeafSize(resolution, resolution, resolution);
            voxel_filter.setInputCloud(cloud);
            voxel_filter.filter(filtered);
            if (mMap->empty())
            {
                if (filtered.size() >= mEskf->config().min_correspondences)
                {
                    pcl::PointCloud<pcl::PointXYZ> world;
                    pcl::transformPointCloud(filtered, world, mEskf->lidarPose().cast<float>());
                    mMap->initialize(world);
                }
            }
            else
            {
                const auto result = mEskf->update([&](const lio::ImuState& state)
                    {
                        return mLidarMeasurement->build(state, filtered, *mMap);
                    });

                if (result.accepted)
                {
                    pcl::PointCloud<pcl::PointXYZ> world;
                    pcl::transformPointCloud(filtered, world, mEskf->lidarPose().cast<float>());
                    mMap->insert(world);
                }
                else
                {
                    RCLCPP_WARN(get_logger(), "LiDAR update rejected: %zu planes, RMSE %.4f, iterations %d; keeping IMU prior.",
                                result.num_correspondences, result.plane_rmse, result.iterations);
                }
            }
            publish(filtered, scan.end);
        }
        catch (const std::exception& exception)
        {
            RCLCPP_ERROR(get_logger(), "LIO scan skipped: %s", exception.what());
        }
    }
}

void Localization_LIO::publish(const pcl::PointCloud<pcl::PointXYZ>& scan, double timestamp)
{
    const auto stamp = rclcpp::Time(std::llround(timestamp * 1e9), get_clock()->get_clock_type());
    pcl::PointCloud<pcl::PointXYZ> transformed;
    pcl::transformPointCloud(scan, transformed, mEskf->lidarPose().cast<float>());
    sensor_msgs::msg::PointCloud2 scan_msg;
    pcl::toROSMsg(transformed, scan_msg);
    scan_msg.header.stamp = stamp;
    scan_msg.header.frame_id = mWorldFrame;
    mProcessedScanVisPub->publish(scan_msg);
    if (mVoxelMapPub->get_subscription_count() > 0)
    {
        pcl::PointCloud<pcl::PointXYZ> map_cloud;
        for (const auto* point : mMap->points())
        {
            map_cloud.emplace_back(point->position.x(), point->position.y(), point->position.z());
        }
        sensor_msgs::msg::PointCloud2 map_msg;
        pcl::toROSMsg(map_cloud, map_msg);
        map_msg.header = scan_msg.header;
        mVoxelMapPub->publish(map_msg);
    }
    geometry_msgs::msg::PoseStamped pose_msg;
    pose_msg.header = scan_msg.header;
    const Eigen::Isometry3d base_pose(mEskf->state().matrix() * mTi2b.matrix());
    pose_msg.pose = tf2::toMsg(base_pose);
    mBasePosePub->publish(pose_msg);
    mBasePosePath.header = pose_msg.header;
    mBasePosePath.poses.push_back(pose_msg);
    if (mBasePosePath.poses.size() > 10000)
    {
        mBasePosePath.poses.erase(mBasePosePath.poses.begin());
    }
    mBasePosePathPub->publish(mBasePosePath);
}
