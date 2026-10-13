#include "localization/nodes/Localization_LO.h"
#include <pcl_conversions/pcl_conversions.h>
#include "localization/tools/OrthographicProjector.hpp"
#include "localization/tools/feature_detect_3d.hpp"
#include "localization/tools/filter_3d.h"
#include <opencv2/opencv.hpp>
#include "localization/types/common.hpp"

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

void Localization_LO::updateVoxelMap(const pcl::PointCloud<pcl::PointXYZ>& scan_in_truck)
{
    // Transform scan into trailer local template frame
    pcl::PointCloud<pcl::PointXYZ> scan_in_trailer;
    pcl::transformPointCloud(scan_in_truck, scan_in_trailer, mBasePose.cast<float>());

    *mTrailerVoxelMap += scan_in_trailer;   // Merge into voxel map

    const auto filtered_map = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    pcl::VoxelGrid<pcl::PointXYZ> voxel_filter;
    voxel_filter.setLeafSize(MAP_RESOLUTION, MAP_RESOLUTION, MAP_RESOLUTION);
    voxel_filter.setInputCloud(mTrailerVoxelMap);
    voxel_filter.filter(*filtered_map);

    mTrailerVoxelMap = filtered_map;

    mGicp.insertTargetPoints(scan_in_trailer);
}

void Localization_LO::recordGicpDuration(const double duration_ms)
{
    if (constexpr std::size_t statistics_window_size = 200;
        mGicpDurationsMs.size() == statistics_window_size)
    {
        mGicpDurationsMs.erase(mGicpDurationsMs.begin());
    }
    mGicpDurationsMs.push_back(duration_ms);
    ++mGicpDurationCount;

    if (constexpr std::size_t report_interval = 50;
        mGicpDurationsMs.size() < report_interval || mGicpDurationCount % report_interval != 0)
    {
        return;
    }

    std::vector<double> sorted_durations = mGicpDurationsMs;
    std::ranges::sort(sorted_durations);
    const auto percentile = [&sorted_durations](const double fraction)
        {
            const std::size_t index = static_cast<std::size_t>(std::ceil(fraction * static_cast<double>(sorted_durations.size()))) - 1;
            return sorted_durations[std::min(index, sorted_durations.size() - 1)];
        };
    const double mean = std::accumulate(mGicpDurationsMs.begin(), mGicpDurationsMs.end(), 0.0) /
                        static_cast<double>(mGicpDurationsMs.size());

    RCLCPP_INFO(get_logger(),
                "Sparse GICP latency (%zu-frame window): mean %.2f ms, P50 %.2f ms, P95 %.2f ms, P99 %.2f ms, max %.2f ms",
                mGicpDurationsMs.size(), mean, percentile(0.50), percentile(0.95), percentile(0.99), sorted_durations.back());
}

bool Localization_LO::alignICP(const pcl::PointCloud<pcl::PointXYZ>::Ptr& current_scan)
{
    // RCLCPP_INFO(get_logger(), "Sparsity-aware GICP start.");
    if (mTrailerVoxelMap == nullptr || mTrailerVoxelMap->empty() || current_scan == nullptr || current_scan->empty())
    {
        const std::size_t target_point_count = mTrailerVoxelMap == nullptr ? 0 : mTrailerVoxelMap->size();
        const std::size_t source_point_count = current_scan == nullptr ? 0 : current_scan->size();
        RCLCPP_WARN(get_logger(), "GICP skipped: target map has %zu points; source ROI has %zu points.",
                    target_point_count, source_point_count);
        return false;
    }

    const auto start_time = std::chrono::high_resolution_clock::now();
    GICP::Result result;
    try
    {
        result = mGicp.align(*current_scan, mBasePose);
    }
    catch (const std::exception& exception)
    {
        RCLCPP_ERROR(get_logger(), "Sparse GICP failed: %s", exception.what());
        return false;
    }

    const auto end_time = std::chrono::high_resolution_clock::now();
    const double duration_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();

    if constexpr (PRINT_BENCHMARK)
    {
        RCLCPP_INFO(get_logger(), "Sparse GICP computation time: %.2f ms", duration_ms);
        recordGicpDuration(duration_ms);
    }


    if (result.converged)
    {
        const double fitness_score = result.fitness_score;
        const auto& gicp_config = mGicp.config();
        if constexpr (PRINT_BENCHMARK)
        {
            RCLCPP_INFO(get_logger(), "Sparse GICP fitness score: %.4f, iter: %d, correspondences: %zu/%zu",
                    fitness_score, result.iterations, result.num_correspondences, result.num_source_points);
        }

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

    RCLCPP_WARN(get_logger(), "Sparse GICP did not converge: iter: %d, correspondences: %zu/%zu, fitness: %.6f.",
                result.iterations, result.num_correspondences, result.num_source_points,
                result.fitness_score);
    return false;
}

void Localization_LO::workerLoop()
{
    while (rclcpp::ok())
    {
        /* Wait and receive scan data */
        sensor_msgs::msg::PointCloud2::ConstSharedPtr scan_msg;
        {
            std::unique_lock lock(mScanBufferMutex);
            mTrigger.wait(lock, [this]() -> bool { return !mScanBuffer.empty() || mIsShutdown; });
            if (mIsShutdown)
            {
                break;
            }
            scan_msg = std::move(mScanBuffer.front());
            mScanBuffer.pop();
        }

        const auto frame_start_time = std::chrono::high_resolution_clock::now();

        pcl::PointCloud<pcl::PointXYZI> lidar_points;
        pcl::fromROSMsg(*scan_msg, lidar_points);

        pcl::transformPointCloud(lidar_points, lidar_points, T_truck2lidar);

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

            if (point.x > 1.8f || point.x < -1.8f || point.y > 0.8f || point.y < -0.8f)
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

        if (mTrailerVoxelMap == nullptr)
        {
            if (processed_cloud->size() > 100)
            {
                mTrailerVoxelMap = processed_cloud;
                mGicp.insertTargetPoints(*mTrailerVoxelMap);
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
        pcl::toROSMsg(*mTrailerVoxelMap, map_msg);
        map_msg.header.stamp = this->now();
        map_msg.header.frame_id = "map";
        mVoxelMapPub->publish(map_msg);

        geometry_msgs::msg::PoseStamped pose_msg;
        pose_msg.header = scan_msg->header;
        pose_msg.header.frame_id = "map";
        pose_msg.pose = tf2::toMsg(mBasePose.cast<double>());
        mBasePosePub->publish(pose_msg);

        geometry_msgs::msg::TransformStamped tf;
        // tf.header.stamp = scan_msg->header.stamp;
        tf.header.stamp = this->get_clock()->now();
        tf.header.frame_id = "map";
        tf.child_frame_id = "LOLA";
        tf.transform.translation.x = pose_msg.pose.position.x;
        tf.transform.translation.y = pose_msg.pose.position.y;
        tf.transform.translation.z = pose_msg.pose.position.z;
        tf.transform.rotation.x = pose_msg.pose.orientation.x;
        tf.transform.rotation.y = pose_msg.pose.orientation.y;
        tf.transform.rotation.z = pose_msg.pose.orientation.z;
        tf.transform.rotation.w = pose_msg.pose.orientation.w;
        mTfBroadcaster->sendTransform(tf);

        mBasePosePath.header = pose_msg.header;
        mBasePosePath.poses.push_back(pose_msg);
        mBasePosePathPub->publish(mBasePosePath);

        const auto frame_end_time = std::chrono::high_resolution_clock::now();
        const double frame_duration_ms = std::chrono::duration<double, std::milli>(frame_end_time - frame_start_time).count();

        if constexpr (PRINT_BENCHMARK)
        {
            RCLCPP_INFO(get_logger(), "Whole frame processing time: %.2f ms", frame_duration_ms);
        }
    }
}