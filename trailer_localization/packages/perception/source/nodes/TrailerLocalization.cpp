#include "perception/nodes/TrailerLocalization.h"
#include <pcl_conversions/pcl_conversions.h>
#include "perception/tools/OrthographicProjector.hpp"
#include "perception/tools/feature_detect_3d.hpp"
#include "perception/tools/filter_3d.h"
#include <opencv2/opencv.hpp>
#include <chrono>

#include "perception/types/common.hpp"

void TrailerLocalization::makeTemplate(const pcl::PointCloud<pcl::PointXYZ>& src_scan)
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
                separation < 10.0)
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

    // cv::line(debug_img, cv::Point(cvRound(rear_start.x), cvRound(rear_start.y)),
    //     cv::Point(cvRound(rear_end.x), cvRound(rear_end.y)), cv::Scalar(255, 0, 0), 3, cv::LINE_AA);
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
    mTrailerPose = T_truck2template.inverse();   // Trailer pose in truck frame

    mTrailerRoi = ROI::getBBox(side_walls);
    mTrailerRoi.min_x -= 9.9f;
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

    mTrailerTemplate = mTrailerVoxelMap;
}

void TrailerLocalization::updateVoxelMap(const pcl::PointCloud<pcl::PointXYZ>& scan_in_truck)
{
    if (mTrailerVoxelMap == nullptr)
    {
        mTrailerVoxelMap = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    }

    // Transform scan into trailer local template frame
    const Eigen::Isometry3f T_truck2trailer = mTrailerPose.inverse();
    pcl::PointCloud<pcl::PointXYZ> scan_in_trailer;
    pcl::transformPointCloud(scan_in_truck, scan_in_trailer, T_truck2trailer);

    // Filter points inside trailer ROI
    const auto scan_in_roi = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    filter3d::getCloud(scan_in_trailer, scan_in_roi, nullptr, mTrailerRoi);

    // Merge into voxel map
    *mTrailerVoxelMap += *scan_in_roi;

    // Apply 0.01m voxel grid filter to update voxel map
    pcl::VoxelGrid<pcl::PointXYZ> voxel_filter;
    voxel_filter.setLeafSize(0.01f, 0.01f, 0.01f);
    voxel_filter.setInputCloud(mTrailerVoxelMap);

    const auto filtered_map = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    voxel_filter.filter(*filtered_map);
    mTrailerVoxelMap = filtered_map;

    // Synchronize template with the updated voxel map
    mTrailerTemplate = mTrailerVoxelMap;
}

bool TrailerLocalization::alignICP(const pcl::PointCloud<pcl::PointXYZ>::Ptr& current_scan, Eigen::Isometry3f& out_pose)
{
    RCLCPP_INFO(get_logger(), "ICP SE(2) start.");
    if (mTrailerTemplate == nullptr || mTrailerTemplate->empty() || current_scan == nullptr || current_scan->empty())
    {
        RCLCPP_WARN(get_logger(), "ICP failed!");
        return false;
    }

    const auto start_time = std::chrono::high_resolution_clock::now();

    // Downsample input clouds for fast and robust registration
    pcl::VoxelGrid<pcl::PointXYZ> voxel_filter;
    voxel_filter.setLeafSize(0.01f, 0.01f, 0.01f);

    const auto src_downsampled = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    voxel_filter.setInputCloud(mTrailerTemplate);
    voxel_filter.filter(*src_downsampled);

    const auto tgt_downsampled = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    voxel_filter.setInputCloud(current_scan);
    voxel_filter.filter(*tgt_downsampled);

    // Standard ICP with 2D Transformation Estimation (SE(2): x, y, yaw)
    pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
    using TransformEst2D = pcl::registration::TransformationEstimation2D<pcl::PointXYZ, pcl::PointXYZ>;
    icp.setTransformationEstimation(std::make_shared<TransformEst2D>());

    icp.setInputSource(src_downsampled);
    icp.setInputTarget(tgt_downsampled);
    icp.setMaxCorrespondenceDistance(0.5);
    icp.setMaximumIterations(80);
    icp.setTransformationEpsilon(1e-6);
    icp.setEuclideanFitnessEpsilon(1e-6);

    pcl::PointCloud<pcl::PointXYZ> aligned_cloud;
    icp.align(aligned_cloud, mTrailerPose.matrix());

    const auto end_time = std::chrono::high_resolution_clock::now();
    const double duration_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    RCLCPP_INFO(get_logger(), "2D ICP computation time: %.2f ms", duration_ms);

    if (icp.hasConverged())
    {
        const double fitness_score = icp.getFitnessScore();
        RCLCPP_INFO(get_logger(), "2D ICP fitness score: %.4f", fitness_score);

        // Only update pose & map if the alignment quality is high (fitness score within threshold)
        constexpr double max_acceptable_fitness = 0.05; // Mean squared distance threshold
        if (fitness_score > max_acceptable_fitness)
        {
            RCLCPP_WARN(get_logger(), "ICP converged but fitness score (%.4f) > threshold (%.4f), skipping map update.",
                        fitness_score, max_acceptable_fitness);
            return false;
        }

        out_pose = Eigen::Isometry3f(icp.getFinalTransformation());
        out_pose.translation().z() = 0.0f; // Ensure z is exactly 0 in 2D plane
        return true;
    }

    return false;
}

void TrailerLocalization::workerLoop()
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

        pcl::PointCloud<pcl::PointXYZ> lidar_points;
        pcl::fromROSMsg(scan_msg, lidar_points);

        /* 1. Get lidar scan in base truck frame */
        rclcpp::Time lidar_scan_stamp(scan_msg.header.stamp);
        pcl::PointCloud<pcl::PointXYZ> lidar_points_truck;
        transformLidarScan(lidar_points, lidar_scan_stamp, lidar_points_truck);

        /* 2. Get lidar scan within ROI */
        auto scan_in_roi = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
        scan_in_roi->reserve(lidar_points_truck.size() / 2);
        for (const auto& point : lidar_points_truck)
        {
            if (point.x < 1.0f && point.y > -10.0f && point.y < 10.0f && point.z > 0.1f)
            {
                scan_in_roi->emplace_back(point);
            }
        }

        if (mTrailerTemplate == nullptr)
        {
            makeTemplate(lidar_points_truck);
        }
        else
        {
            if (alignICP(scan_in_roi, mTrailerPose))
            {
                updateVoxelMap(lidar_points_truck);

                pcl::PointCloud<pcl::PointXYZ> aligned_map;
                pcl::transformPointCloud(*mTrailerVoxelMap, aligned_map, mTrailerPose);

                sensor_msgs::msg::PointCloud2 scan_vis_msg;
                pcl::toROSMsg(aligned_map, scan_vis_msg);
                scan_vis_msg.header = scan_msg.header;
                scan_vis_msg.header.frame_id = "LOLA";
                mProcessedScanVisPub->publish(scan_vis_msg);
            }
        }

        if (mTrailerTemplate != nullptr)
        {
            geometry_msgs::msg::PoseStamped pose_msg;
            pose_msg.header = scan_msg.header;
            pose_msg.header.frame_id = "LOLA";
            pose_msg.pose = tf2::toMsg(Eigen::Isometry3d(mTrailerPose.cast<double>()));
            mTrailerPosePub->publish(pose_msg);
        }
    }
}