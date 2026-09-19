#include "perception/nodes/Localization_LIO.h"
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

std::optional<IntensityAnalysis> analyzeIntensity(const pcl::PointCloud<pcl::PointXYZI>& cloud,
                                                  const float keep_ratio)
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


void Localization_LIO::updateVoxelMap(const pcl::PointCloud<pcl::PointXYZ>& scan_in_truck)
{
    // Transform scan into trailer local template frame
    pcl::PointCloud<pcl::PointXYZ> scan_in_trailer;
    pcl::transformPointCloud(scan_in_truck, scan_in_trailer, mBasePose.cast<float>());

    *mMap += scan_in_trailer;   // Merge into voxel map

    const auto filtered_map = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    pcl::VoxelGrid<pcl::PointXYZ> voxel_filter;
    voxel_filter.setLeafSize(MAP_RESOLUTION, MAP_RESOLUTION, MAP_RESOLUTION);
    voxel_filter.setInputCloud(mMap);
    voxel_filter.filter(*filtered_map);

    mMap = filtered_map;

    mGicp.insertTargetPoints(scan_in_trailer);
}

bool Localization_LIO::alignICP(const pcl::PointCloud<pcl::PointXYZ>::Ptr& current_scan)
{
    RCLCPP_INFO(get_logger(), "CUDA sparsity-aware GICP SE(3) start.");
    if (mMap == nullptr || mMap->empty() || current_scan == nullptr || current_scan->empty())
    {
        const std::size_t target_point_count = mMap == nullptr ? 0 : mMap->size();
        const std::size_t source_point_count = current_scan == nullptr ? 0 : current_scan->size();
        RCLCPP_WARN(get_logger(), "GICP skipped: target map has %zu points; source ROI has %zu points.",
                    target_point_count, source_point_count);
        return false;
    }

    GICP::Result result;
    try
    {
        result = mGicp.align(*current_scan, mBasePose);
    }
    catch (const std::exception& exception)
    {
        RCLCPP_ERROR(get_logger(), "CUDA sparse GICP failed: %s", exception.what());
        return false;
    }

    if (result.converged)
    {
        const double fitness_score = result.fitness_score;
        const auto& gicp_config = mGicp.config();

        if (result.num_correspondences < gicp_config.min_correspondences)
        {
            RCLCPP_WARN(get_logger(), "GICP converged with too few correspondences (%zu < %zu), skipping map update.",
                        result.num_correspondences, gicp_config.min_correspondences);
            return false;
        }

        mBasePose = result.transform;

        if (fitness_score > gicp_config.max_fitness_score)
        {
            RCLCPP_WARN(get_logger(), "GICP converged but fitness score (%.4f) > threshold (%.4f), skipping map update.",
                        fitness_score, gicp_config.max_fitness_score);
            return false;
        }

        return true;
    }

    RCLCPP_WARN(get_logger(), "CUDA sparse GICP did not converge: iter: %d, correspondences: %zu/%zu, fitness: %.6f.",
                result.iterations, result.num_correspondences, result.num_source_points,
                result.fitness_score);
    return false;
}

void Localization_LIO::lidarWorkerLoop()
{
    StampedCloud stamped_cloud;

    while (rclcpp::ok())
    {
        /* Wait and receive scan data */
        sensor_msgs::msg::PointCloud2 scan_msg;
        {
            std::unique_lock lock(mScanBufferMutex);
            mLidarTrigger.wait(lock, [this]() -> bool { return !mScanBuffer.empty() || mIsShutdown; });
            if (mIsShutdown)
            {
                break;
            }
            scan_msg = std::move(mScanBuffer.front());
            mScanBuffer.pop();
        }

        const auto frame_start_time = std::chrono::high_resolution_clock::now();

        pcl::PointCloud<pcl::PointXYZI> lidar_points;

        if (!parseCloudMsg(scan_msg, stamped_cloud))
        {
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000, "Failed to parse point cloud message!");
            continue;
        }


        if (!mIntensityAnalyzed)
        {
            mIntensityAnalyzed = true;
            if (const auto analysis = analyzeIntensity(lidar_points, mIntensityKeepRatio))
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
        }

        /* Filter out points below the selected intensity threshold */
        const auto denoised_scan = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
        denoised_scan->reserve(lidar_points.size());
        for (const auto& point : lidar_points)
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

        if (mMap == nullptr)
        {
            if (processed_cloud->size() > 100)
            {
                mMap = processed_cloud;
                mGicp.initializeTarget(*mMap);
            }
            else
            {
                RCLCPP_WARN(get_logger(), "processed_cloud is empty!");
            }
            continue;
        }

        if (processed_cloud->empty())
        {
            RCLCPP_WARN(get_logger(), "Skipping GICP: current scan has no points inside the trailer ROI.");
            continue;
        }

        if (alignICP(processed_cloud))
        {
            updateVoxelMap(*processed_cloud);
        }

        pcl::PointCloud<pcl::PointXYZ> transformed_scan;
        pcl::transformPointCloud(*processed_cloud, transformed_scan, mBasePose.cast<float>());

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
        pose_msg.header = scan_msg.header;
        pose_msg.header.frame_id = "map";
        pose_msg.pose = tf2::toMsg(mBasePose.cast<double>());
        mBasePosePub->publish(pose_msg);

        mBasePosePath.header = pose_msg.header;
        mBasePosePath.poses.push_back(pose_msg);
        mBasePosePathPub->publish(mBasePosePath);

        const auto frame_end_time = std::chrono::high_resolution_clock::now();
        const double frame_duration_ms = std::chrono::duration<double, std::milli>(frame_end_time - frame_start_time).count();
        RCLCPP_INFO(get_logger(), "Whole frame processing time: %.2f ms", frame_duration_ms);
    }
}

void Localization_LIO::imuWorkerLoop()
{
    while (rclcpp::ok())
    {
        /* Wait and receive scan data */
        sensor_msgs::msg::Imu img_msg;
        {
            std::unique_lock lock(mImuBufferMutex);
            mImuTrigger.wait(lock, [this]() -> bool { return !mImuBuffer.empty() || mIsShutdown; });
            if (mIsShutdown)
            {
                break;
            }
            img_msg = std::move(mImuBuffer.front());
            mImuBuffer.pop();
        }



    }
}