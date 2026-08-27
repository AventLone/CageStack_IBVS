#include "perception/nodes/TrailerLocalization.h"
#include <pcl_conversions/pcl_conversions.h>
#include "perception/tools/OrthographicProjector.hpp"
#include <opencv2/opencv.hpp>

void TrailerLocalization::makeTemplate(const pcl::PointCloud<pcl::PointXYZ>::Ptr& src_scan)
{
    /* Get the top-down view image */
    constexpr float resolution = 0.03f;
    OrthographicProjector<pcl::PointXYZ> projector(View::TOP, resolution);
    projector.setCloud(src_scan);

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

    for (std::size_t first = 0; first < candidate_indices.size(); ++first)
    {
        const cv::Vec4i& first_line = lines[candidate_indices[first]];
        const cv::Point2f first_start(static_cast<float>(first_line[0]), static_cast<float>(first_line[1]));
        const cv::Point2f first_end(static_cast<float>(first_line[2]), static_cast<float>(first_line[3]));
        const cv::Point2f first_direction = first_end - first_start;
        const double first_length = cv::norm(first_direction);
        const cv::Point2f first_unit = first_direction / static_cast<float>(first_length);
        const cv::Point2f first_midpoint = (first_start + first_end) * 0.5f;

        for (std::size_t second = first + 1; second < candidate_indices.size(); ++second)
        {
            const cv::Vec4i& second_line = lines[candidate_indices[second]];
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
                best_first = candidate_indices[first];
                best_second = candidate_indices[second];
            }
        }
    }

    for (const cv::Vec4i& line : lines)
    {
        cv::line(debug_img, cv::Point(line[0], line[1]), cv::Point(line[2], line[3]), cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
    }

    if (best_first < 0)
    {
        return;
    }

    std::vector<cv::Mat> side_masks(2);
    const auto line_indices = std::vector{best_first, best_second};
    for (std::size_t i = 0; i < 2; ++i)
    {
        side_masks[i] = cv::Mat::zeros(top_down_img.size(), CV_8UC1);
        const cv::Vec4i& line = line_indices[i];
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


    // mTrailerTemplate = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();

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

        /* 2. Get lidar scan within the ROI */
        auto scan_in_roi = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
        scan_in_roi->reserve(lidar_points_truck.size() / 2);
        for (const auto& point : lidar_points_truck)
        {
            if (point.x < 1.0f && point.y > -10.0f && point.y < 10.0f && point.z > 0.1f)
            {
                scan_in_roi->emplace_back(point);
            }
        }

        sensor_msgs::msg::PointCloud2 scan_vis_msg;
        pcl::toROSMsg(*scan_in_roi, scan_vis_msg);
        scan_vis_msg.header = scan_msg.header;
        scan_vis_msg.header.frame_id = "LOLA";
        mProcessedScanVisPub->publish(scan_vis_msg);

        if (mTrailerTemplate == nullptr)
        {
            makeTemplate(scan_in_roi);
        }



    }
}