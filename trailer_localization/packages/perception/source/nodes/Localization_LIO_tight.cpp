#include "perception/nodes/Localization_LIO_tight.h"
#include <pcl_conversions/pcl_conversions.h>
#include "perception/tools/OrthographicProjector.hpp"
#include "perception/tools/feature_detect_3d.hpp"
#include <opencv2/opencv.hpp>

#include "perception/types/stamped_cloud.hpp"

namespace
{
struct IntensityAnalysis
{
    float minimum{};
    float p25{};
    float p40{};
    float median{};
    float p75{};
    float p90{};
    float p95{};
    float p99{};
    float maximum{};
    float suggested_threshold{};
    std::size_t valid_count{};
    std::size_t retained_count{};
};

std::optional<IntensityAnalysis> analyzeIntensity(const pcl::PointCloud<pcl::PointXYZI>& cloud, const float keep_ratio)
{
    std::vector<float> intensities;
    intensities.reserve(cloud.size());
    for (const auto& point : cloud)
    {
        if (std::isfinite(point.intensity))
        {
            intensities.push_back(point.intensity);
        }
    }

    if (intensities.empty())
    {
        return std::nullopt;
    }

    std::ranges::sort(intensities);
    const auto percentile = [&intensities](const double fraction)
        {
            const auto index = static_cast<std::size_t>(
                std::round(fraction * static_cast<double>(intensities.size() - 1)));
            return intensities[index];
        };

    const float minimum = intensities.front();
    const float maximum = intensities.back();
    const std::size_t target_retained_count = std::min(intensities.size(),
        static_cast<std::size_t>(std::ceil(static_cast<double>(keep_ratio) * intensities.size())));
    const std::size_t threshold_index = intensities.size() - target_retained_count;
    const float suggested_threshold = intensities[threshold_index];
    const auto first_retained = std::ranges::lower_bound(intensities, suggested_threshold);
    const std::size_t retained_count = static_cast<std::size_t>(intensities.end() - first_retained);
    return IntensityAnalysis{minimum, percentile(0.25), percentile(0.40), percentile(0.50), percentile(0.75),
                             percentile(0.90), percentile(0.95), percentile(0.99), maximum,
                             suggested_threshold, intensities.size(), retained_count};
}
}

void Localization_LIO_T::updateCloudMap(const pcl::PointCloud<pcl::PointXYZ>& scan)
{
    // Transform scan into trailer local template frame
    pcl::PointCloud<pcl::PointXYZ> scan_in_trailer;
    pcl::transformPointCloud(scan, scan_in_trailer, mIESKF.pose().cast<float>());

    *mMap += scan_in_trailer;   // Merge into voxel map

    const auto filtered_map = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    pcl::VoxelGrid<pcl::PointXYZ> voxel_filter;
    voxel_filter.setLeafSize(MAP_RESOLUTION * 2, MAP_RESOLUTION * 2, MAP_RESOLUTION * 2);
    voxel_filter.setInputCloud(mMap);
    voxel_filter.filter(*filtered_map);

    mMap = filtered_map;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr Localization_LIO_T::denoiseAndDownsample(const pcl::PointCloud<pcl::PointXYZI>& src) const
{
    /* Filter out points below the selected intensity threshold */
    const auto denoised_scan = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    denoised_scan->reserve(src.size());
    for (const auto& point : src)
    {
        if (point.intensity < mIntensityThreshold)
        {
            continue;
        }

        if (point.x > 0.8f || point.x < -0.8f || point.y > 0.8f || point.y < -0.8f)
        {
            denoised_scan->emplace_back(point.x, point.y, point.z);
        }
    }

    /* Preprocess the cloud */
    const auto processed_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    pcl::VoxelGrid<pcl::PointXYZ> voxel_filter;
    voxel_filter.setLeafSize(MAP_RESOLUTION, MAP_RESOLUTION, MAP_RESOLUTION);
    voxel_filter.setInputCloud(denoised_scan);
    voxel_filter.filter(*processed_cloud);

    pcl::transformPointCloud(*processed_cloud, *processed_cloud, mT_il);

    return processed_cloud;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr Localization_LIO_T::denoiseAndDownsample(const StampedCloud& src) const
{
    /* Filter out points below the selected intensity threshold */
    const auto denoised_scan = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();

    denoised_scan->reserve(src.points.size());
    for (const auto& point : src.points)
    {
        if (point.intensity < mIntensityThreshold)
        {
            continue;
        }

        if (point.position.x() > 0.8f || point.position.x() < -0.8f || point.position.y() > 0.8f || point.position.y() < -0.8f)
        {
            denoised_scan->emplace_back(point.position.x(), point.position.y(), point.position.z());
        }
    }

    /* Preprocess the cloud */
    const auto processed_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    pcl::VoxelGrid<pcl::PointXYZ> voxel_filter;
    voxel_filter.setLeafSize(MAP_RESOLUTION, MAP_RESOLUTION, MAP_RESOLUTION);
    voxel_filter.setInputCloud(denoised_scan);
    voxel_filter.filter(*processed_cloud);

    pcl::transformPointCloud(*processed_cloud, *processed_cloud, mT_il);
    return processed_cloud;
}

void Localization_LIO_T::workerLoop()
{
    StampedCloud stamped_cloud;
    double last_timestamp = 0.0;
    while (rclcpp::ok())
    {
        /* Wait untill received scan data */
        sensor_msgs::msg::PointCloud2::ConstSharedPtr scan_msg;
        {
            std::unique_lock lock(mScanBufferMutex);
            mTrigger.wait(lock, [this]() -> bool { return !mScanBuffer.empty() || mIsShutdown; });
            if (mIsShutdown)
            {
                break;
            }
            scan_msg = std::move(mScanBuffer.front());
            mScanBuffer.pop_front();
        }

        const auto frame_start_time = std::chrono::high_resolution_clock::now();

        pcl::PointCloud<pcl::PointXYZI> lidar_points;
        if (!parseCloudMsg(*scan_msg, stamped_cloud))
        {
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000, "Failed to parse point cloud message!");
            continue;
        }

        /* Initialize the whole system */
        if (!mInitialized)
        {
            mInitialized = true;
            last_timestamp = stamped_cloud.end_time;

            const auto cloud = getCloudXYZI(stamped_cloud.points);

            if (const auto analysis = analyzeIntensity(cloud, mIntensityKeepRatio))
            {
                const double retained_percentage = 100.0 * static_cast<double>(analysis->retained_count) /
                                                   static_cast<double>(analysis->valid_count);
                RCLCPP_INFO(get_logger(),
                            "First-frame intensity distribution (%zu valid points): min %.2f, P25 %.2f, "
                            "P40 %.2f, P50 %.2f, P75 %.2f, P90 %.2f, P95 %.2f, P99 %.2f, max %.2f.",
                            analysis->valid_count, analysis->minimum, analysis->p25, analysis->p40, analysis->median,
                            analysis->p75, analysis->p90, analysis->p95, analysis->p99, analysis->maximum);
                RCLCPP_INFO(get_logger(),
                            "Suggested intensity threshold: %.2f (target keep ratio %.1f%%, actual %.1f%%).",
                            analysis->suggested_threshold, 100.0 * mIntensityKeepRatio, retained_percentage);
                if (mIntensityThreshold < 0.0f)
                {
                    mIntensityThreshold = analysis->suggested_threshold;
                    RCLCPP_INFO(get_logger(), "Using automatically selected intensity threshold %.2f.",
                                mIntensityThreshold);
                }
                else
                {
                    RCLCPP_INFO(get_logger(), "Using configured intensity threshold %.2f.", mIntensityThreshold);
                }
            }
            else
            {
                RCLCPP_WARN(get_logger(), "First frame contains no finite intensity values; no threshold was selected.");
            }

            const auto denoised_scan = denoiseAndDownsample(cloud);
            if (denoised_scan->size() < 200)
            {
                RCLCPP_WARN(get_logger(), "denoised_scan has too few points!");
                mInitialized = false;
                continue;
            }

            std::vector<lio::ImuData> imu_datas;
            {
                std::lock_guard<std::mutex> lock(mImuBufferMutex);
                imu_datas = std::vector(mImuBuffer.begin(), mImuBuffer.end());
            }
            if (!mIESKF.initialize(imu_datas))
            {
                RCLCPP_WARN(get_logger(), "ESKF failed to initialize!");
                mInitialized = false;
                continue;
            }
            mIESKF.setTimestamp(stamped_cloud.end_time);

            mMap = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
            pcl::transformPointCloud(*denoised_scan, *mMap, mIESKF.pose().cast<float>());
            // mGicp.initializeTarget(*mMap);
            mIESKF.initialize(*mMap);
            continue;
        }

        /* ESKF prediction */
        const auto last_state = mIESKF.state();
        auto eskf_predict_future = std::async(std::launch::async,
            [this, last_timestamp, current_timestamp = stamped_cloud.end_time]()
            {
                std::vector<lio::ImuData> imu_datas;
                {
                    std::lock_guard<std::mutex> lock(mImuBufferMutex);
                    imu_datas = std::vector(mImuBuffer.begin(), mImuBuffer.end());
                }
                const auto sequence = lio::ImuProcessor::buildImuSequence(imu_datas, last_timestamp, current_timestamp);
                mIESKF.predict(sequence);
            });
        last_timestamp = stamped_cloud.end_time;

        /* Deskew the point cloud and transform it into IMU frame */
        while (rclcpp::ok() && !mIsShutdown)
        {
            std::vector<lio::ImuData> imu_datas;
            {
                std::lock_guard<std::mutex> lock(mImuBufferMutex);
                imu_datas = std::vector(mImuBuffer.begin(), mImuBuffer.end());
            }
            try
            {
                mImuProcessor->process(imu_datas, stamped_cloud, last_state);
            }
            catch (const std::exception& e)
            {
                // RCLCPP_WARN(get_logger(), e.what());
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }
            break;
        }

        // const auto scan_in_imu = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
        const auto denoised_cloud = denoiseAndDownsample(stamped_cloud);

        /* Observe and update the cloud map */
        eskf_predict_future.get();
        mIESKF.observe(*denoised_cloud);
        updateCloudMap(*denoised_cloud);

        pcl::PointCloud<pcl::PointXYZ> transformed_scan;
        pcl::transformPointCloud(*denoised_cloud, transformed_scan, mIESKF.pose().cast<float>());

        sensor_msgs::msg::PointCloud2 scan_vis_msg;
        pcl::toROSMsg(transformed_scan, scan_vis_msg);
        scan_vis_msg.header.stamp = this->now();
        scan_vis_msg.header.frame_id = "map";
        mProcessedScanVisPub->publish(scan_vis_msg);

        sensor_msgs::msg::PointCloud2 map_msg;
        pcl::toROSMsg(*mMap, map_msg);
        map_msg.header.stamp = this->now();
        map_msg.header.frame_id = "map";
        mVoxelMapPub->publish(map_msg);

        geometry_msgs::msg::PoseStamped pose_msg;
        pose_msg.header.stamp = rclcpp::Time(static_cast<int64_t>(stamped_cloud.end_time * 1e9));
        pose_msg.header.frame_id = "map";
        pose_msg.pose = tf2::toMsg(mIESKF.pose());
        mBasePosePub->publish(pose_msg);

        mBasePosePath.header = pose_msg.header;
        mBasePosePath.poses.push_back(pose_msg);
        mBasePosePathPub->publish(mBasePosePath);

        const auto frame_end_time = std::chrono::high_resolution_clock::now();
        const double frame_duration_ms = std::chrono::duration<double, std::milli>(frame_end_time - frame_start_time).count();
        RCLCPP_INFO(get_logger(), "Whole frame processing time: %.2f ms", frame_duration_ms);
    }
}