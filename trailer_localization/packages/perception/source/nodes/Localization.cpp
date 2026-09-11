#include "perception/nodes/Localization.h"
#include <pcl_conversions/pcl_conversions.h>
#include "perception/tools/OrthographicProjector.hpp"
#include "perception/tools/feature_detect_3d.hpp"
#include "perception/tools/filter_3d.h"
#include <opencv2/opencv.hpp>
#include "perception/types/common.hpp"

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

void Localization::makeTemplate(const pcl::PointCloud<pcl::PointXYZ>& src_scan)
{
    /* 2. Get lidar scan within the ROI */
    auto scan_in_roi = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    scan_in_roi->reserve(src_scan.size() / 2);
    for (const auto& point : src_scan)
    {
        if (point.x < 1.0f && point.y > -10.0f && point.y < 10.0f && point.z > 0.1f)
        {
            scan_in_roi->emplace_back(point);
        }
    }

    /* Get the top-down view image */
    constexpr float resolution = 0.03f;
    // Trailer side walls must be 1-5 m apart; convert the limits to image pixels.
    constexpr float min_side_separation = 1.0f / resolution;
    constexpr float max_side_separation = 5.0f / resolution;
    OrthographicProjector<pcl::PointXYZ> projector(View::TOP, resolution);
    projector.setCloud(scan_in_roi);

    cv::Mat top_down_img = projector.projection();
    cv::Mat debug_img;
    cv::cvtColor(top_down_img, debug_img, cv::COLOR_GRAY2BGR);

    std::vector<cv::Vec4i> lines;
    cv::HoughLinesP(top_down_img, lines, 1.0, CV_PI / 180.0, 60, 40.0, 30.0);

    std::vector<int> candidate_indices;
    for (int index = 0; index < static_cast<int>(lines.size()); ++index)
    {
        const cv::Vec4i& line = lines[index];

        if (const double angle = std::atan2(static_cast<double>(line[3] - line[1]), static_cast<double>(line[2] - line[0])) * 180.0 / CV_PI;
            std::abs(angle) <= 30.0)
        {
            candidate_indices.push_back(index);
        }
    }

    int best_first = -1;
    int best_second = -1;
    double best_score = -1.0;

    for (std::size_t i = 0; i < candidate_indices.size(); ++i)
    {
        const cv::Vec4i& first_line = lines[candidate_indices[i]];
        const cv::Point2f first_start(static_cast<float>(first_line[0]), static_cast<float>(first_line[1]));
        const cv::Point2f first_end(static_cast<float>(first_line[2]), static_cast<float>(first_line[3]));
        const cv::Point2f first_direction = first_end - first_start;
        const double first_length = cv::norm(first_direction);
        const cv::Point2f first_unit = first_direction / static_cast<float>(first_length);
        const cv::Point2f first_midpoint = (first_start + first_end) * 0.5f;

        for (std::size_t j = i + 1; j < candidate_indices.size(); ++j)
        {
            const cv::Vec4i& second_line = lines[candidate_indices[j]];
            const cv::Point2f second_start(static_cast<float>(second_line[0]), static_cast<float>(second_line[1]));
            const cv::Point2f second_end(static_cast<float>(second_line[2]), static_cast<float>(second_line[3]));
            const cv::Point2f second_direction = second_end - second_start;
            const double second_length = cv::norm(second_direction);

            if (const cv::Point2f second_unit = second_direction / static_cast<float>(second_length);
                std::abs(first_unit.dot(second_unit)) < std::cos(10.0 * CV_PI / 180.0))
            {
                continue;
            }

            const cv::Point2f second_midpoint = (second_start + second_end) * 0.5f;
            const cv::Point2f midpoint_delta = second_midpoint - first_midpoint;

            if (const double separation = std::abs(first_unit.x * midpoint_delta.y - first_unit.y * midpoint_delta.x);
                separation < min_side_separation || separation > max_side_separation)
            {
                continue;
            }

            const double first_min = first_start.dot(first_unit);
            const double first_max = first_end.dot(first_unit);
            const double second_min = second_start.dot(first_unit);
            const double second_max = second_end.dot(first_unit);
            const double overlap = std::min(first_max, second_max) - std::max(first_min, second_min);

            if (overlap < 0.5 * std::min(first_length, second_length))
            {
                continue;
            }

            if (const double score = std::min(first_length, second_length) + 0.5 * overlap; score > best_score)
            {
                best_score = score;
                best_first = candidate_indices[i];
                best_second = candidate_indices[j];
            }
        }
    }

    for (const auto candidate : candidate_indices)
    {
        const auto& line = lines[candidate];
        cv::line(debug_img, cv::Point(line[0], line[1]), cv::Point(line[2], line[3]), cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
    }

    if (best_first < 0)
    {
        RCLCPP_WARN(get_logger(), "Failed to find the trailer template!");
        return;
    }

    std::vector<cv::Mat> side_masks(2);
    const auto line_indices = std::vector{best_first, best_second};
    for (std::size_t i = 0; i < 2; ++i)
    {
        side_masks[i] = cv::Mat::zeros(top_down_img.size(), CV_8UC1);
        const cv::Vec4i& line = lines[line_indices[i]];
        cv::line(debug_img, cv::Point(line[0], line[1]), cv::Point(line[2], line[3]), cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
        cv::line(side_masks[i], cv::Point(line[0], line[1]), cv::Point(line[2], line[3]), cv::Scalar(255), 5, cv::LINE_AA);
    }

    const cv::Vec4i& first_line = lines[best_first];
    cv::Point2f first_start(static_cast<float>(first_line[0]), static_cast<float>(first_line[1]));
    cv::Point2f first_end(static_cast<float>(first_line[2]), static_cast<float>(first_line[3]));

    const cv::Vec4i& second_line = lines[best_second];
    cv::Point2f second_start(static_cast<float>(second_line[0]), static_cast<float>(second_line[1]));
    cv::Point2f second_end(static_cast<float>(second_line[2]), static_cast<float>(second_line[3]));

    // Orient both side lines in the same direction before comparing their ends.
    if (const cv::Point2f first_direction = first_end - first_start;
        first_direction.dot(second_end - second_start) < 0.0f)
    {
        std::swap(second_start, second_end);
    }

    const cv::Point2f rear_start = first_end;
    const cv::Point2f rear_end = second_end;

    cv::line(debug_img, first_start, second_start, cv::Scalar(255, 0, 0), 2, cv::LINE_AA);

    /* Get 2 side wall point cloud */
    const auto side_wall_1 = projector.extractCloud(side_masks[0]);
    const auto side_wall_2 = projector.extractCloud(side_masks[1]);

    /* Get 2 refined side wall point cloud */
    RawCloud refined_side_wall_1, refined_side_wall_2;
    feature3d::findInliers(side_wall_1, refined_side_wall_1, 0.04f);
    feature3d::findInliers(side_wall_2, refined_side_wall_2, 0.04f);

    const auto side_wall_1_plane = feature3d::fitPlaneByPCA(refined_side_wall_1, 2);
    auto side_wall_2_plane = feature3d::fitPlaneByPCA(refined_side_wall_2, 2);
    if (side_wall_1_plane.head<3>().dot(side_wall_2_plane.head<3>()) < 0.0f)
    {
        side_wall_2_plane = -side_wall_2_plane;
    }
    Eigen::Vector4f mid_plane = 0.5f * (side_wall_1_plane + side_wall_2_plane);
    if (mid_plane[1] < 0.0f)
    {
        mid_plane = -mid_plane;
    }
    const float angle = std::atan2(-mid_plane[0], mid_plane[1]);

    RawCloud side_walls = refined_side_wall_1 + refined_side_wall_2;

    // 1. Rotation transformation from truck frame to rotated frame
    Eigen::Isometry3f T_rot = Eigen::Isometry3f::Identity();
    T_rot.rotate(Eigen::AngleAxisf(-angle, Eigen::Vector3f::UnitZ()));
    pcl::transformPointCloud(side_walls, side_walls, T_rot);

    Eigen::Vector4f rotated_mid_plane;
    rotated_mid_plane.head<3>() = T_rot.rotation() * mid_plane.head<3>();
    rotated_mid_plane[3] = mid_plane[3];

    /* Find out the max x of `side_walls` */
    float max_x = std::numeric_limits<float>::lowest();
    for (const auto& point : side_walls)
    {
        max_x = std::max(max_x, point.x);
    }
    const float y_offset = -rotated_mid_plane[3] / rotated_mid_plane[1];

    // 2. Translation transformation from rotated frame to template frame
    Eigen::Isometry3f T_trans = Eigen::Isometry3f::Identity();
    T_trans.translate(Eigen::Vector3f(-max_x, -y_offset, 0.0f));
    pcl::transformPointCloud(side_walls, side_walls, T_trans);

    Eigen::Isometry3f T_truck2template = T_trans * T_rot;   // Combined transformation: points_template = T_truck2template * points_truck
    mBasePose = T_truck2template.inverse();   // Trailer pose in truck frame

    mTrailerRoi = ROI::getBBox(side_walls);
    mTrailerRoi.max_x += 0.5f;
    mTrailerRoi.min_x -= 9.9f;
    mTrailerRoi.max_y += 1.0f;
    mTrailerRoi.min_y -= 1.0f;
    mTrailerRoi.min_z = -0.1f;
    mTrailerRoi.max_z += 3.1f;

    pcl::PointCloud<pcl::PointXYZ> raw_template;
    pcl::transformPointCloud(src_scan, raw_template, T_truck2template);
    auto initial_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    filter3d::getCloud(raw_template, initial_cloud, nullptr, mTrailerRoi);

    mTrailerVoxelMap = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    pcl::VoxelGrid<pcl::PointXYZ> voxel_filter;
    voxel_filter.setLeafSize(0.01f, 0.01f, 0.01f);
    voxel_filter.setInputCloud(initial_cloud);
    voxel_filter.filter(*mTrailerVoxelMap);

    mGicp.initializeTarget(*mTrailerVoxelMap);
}

void Localization::updateVoxelMap(const pcl::PointCloud<pcl::PointXYZ>& scan_in_truck)
{
    // Transform scan into trailer local template frame
    pcl::PointCloud<pcl::PointXYZ> scan_in_trailer;
    pcl::transformPointCloud(scan_in_truck, scan_in_trailer, mBasePose);

    *mTrailerVoxelMap += scan_in_trailer;   // Merge into voxel map

    const auto filtered_map = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    pcl::VoxelGrid<pcl::PointXYZ> voxel_filter;
    voxel_filter.setLeafSize(MAP_RESOLUTION, MAP_RESOLUTION, MAP_RESOLUTION);
    voxel_filter.setInputCloud(mTrailerVoxelMap);
    voxel_filter.filter(*filtered_map);

    mTrailerVoxelMap = filtered_map;

    mGicp.insertTargetPoints(scan_in_trailer);
}

void Localization::recordGicpDuration(const double duration_ms)
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
                "CUDA sparse GICP latency (%zu-frame window): mean %.2f ms, P50 %.2f ms, P95 %.2f ms, P99 %.2f ms, max %.2f ms",
                mGicpDurationsMs.size(), mean, percentile(0.50), percentile(0.95), percentile(0.99), sorted_durations.back());
}

bool Localization::alignICP(const pcl::PointCloud<pcl::PointXYZ>::Ptr& current_scan)
{
    RCLCPP_INFO(get_logger(), "CUDA sparsity-aware GICP SE(3) start.");
    if (mTrailerVoxelMap == nullptr || mTrailerVoxelMap->empty() || current_scan == nullptr || current_scan->empty())
    {
        const std::size_t target_point_count = mTrailerVoxelMap == nullptr ? 0 : mTrailerVoxelMap->size();
        const std::size_t source_point_count = current_scan == nullptr ? 0 : current_scan->size();
        RCLCPP_WARN(get_logger(), "GICP skipped: target map has %zu points; source ROI has %zu points.",
                    target_point_count, source_point_count);
        return false;
    }

    const auto start_time = std::chrono::high_resolution_clock::now();
    SparsityAwareGICP::Result result;
    try
    {
        result = mGicp.align(*current_scan, mBasePose);
    }
    catch (const std::exception& exception)
    {
        RCLCPP_ERROR(get_logger(), "CUDA sparse GICP failed: %s", exception.what());
        return false;
    }

    const auto end_time = std::chrono::high_resolution_clock::now();
    const double duration_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    RCLCPP_INFO(get_logger(), "CUDA sparse GICP computation time: %.2f ms", duration_ms);
    recordGicpDuration(duration_ms);

    if (result.converged)
    {
        const double fitness_score = result.fitness_score;
        const auto& gicp_config = mGicp.config();
        RCLCPP_INFO(get_logger(), "CUDA sparse GICP fitness score: %.4f, iter: %d, correspondences: %zu/%zu",
                    fitness_score, result.iterations, result.num_correspondences, result.num_source_points);

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

void Localization::workerLoop()
{
    while (rclcpp::ok())
    {
        /* Wait and receive scan data */
        sensor_msgs::msg::PointCloud2 scan_msg;
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

        pcl::PointCloud<pcl::PointXYZI> lidar_points;
        pcl::fromROSMsg(scan_msg, lidar_points);

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

        if (mTrailerVoxelMap == nullptr)
        {
            if (processed_cloud->size() > 100)
            {
                mTrailerVoxelMap = processed_cloud;
                mGicp.initializeTarget(*mTrailerVoxelMap);
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
        pcl::transformPointCloud(*processed_cloud, transformed_scan, mBasePose);

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
        pose_msg.header = scan_msg.header;
        pose_msg.header.frame_id = "map";
        pose_msg.pose = tf2::toMsg(mBasePose.cast<double>());
        mBasePosePub->publish(pose_msg);

        mBasePosePath.header = pose_msg.header;
        mBasePosePath.poses.push_back(pose_msg);
        mBasePosePathPub->publish(mBasePosePath);
    }
}