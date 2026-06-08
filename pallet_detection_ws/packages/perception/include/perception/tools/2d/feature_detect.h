#pragma once
#include "filter.h"
#include <optional>
#include <opencv2/opencv.hpp>

namespace feature2d
{
enum class EdgeType
{
    UPPER, LOWER, LEFT, RIGHT
};

struct Line
{
    explicit Line(const cv::Point& p_1, const cv::Point& p_2) : p1(p_1), p2(p_2)
    {}

    cv::Point p1, p2;

    [[nodiscard]] double length() const
    {
        return cv::norm(p1 - p2);
    }

    [[nodiscard]] cv::Point center() const
    {
        return (p1 + p2) / 2;
    }
};

static inline const cv::Mat RIGHT_EDGE_KERNEL = (cv::Mat_<int>(3, 3) <<
                                                 1, 0, -1,
                                                 2, 0, -2,
                                                 1, 0, -1);
static inline const cv::Mat LEFT_EDGE_KERNEL = (cv::Mat_<int>(3, 3) <<
                                                -1, 0, 1,
                                                -2, 0, 2,
                                                -1, 0, 1);
static inline const cv::Mat UPPER_EDGE_KERNEL = (cv::Mat_<int>(3, 3) <<
                                                 -1, -2, -1,
                                                 0, 0, 0,
                                                 1, 2, 1);
static inline const cv::Mat LOWER_EDGE_KERNEL = (cv::Mat_<int>(3, 3) <<
                                                 1, 2, 1,
                                                 0, 0, 0,
                                                 -1, -2, -1);

static cv::Mat getEdgeKernel(const EdgeType edge_type)
{
    switch (edge_type)
    {
        case EdgeType::RIGHT:
            return RIGHT_EDGE_KERNEL;
        case EdgeType::LEFT:
            return LEFT_EDGE_KERNEL;
        case EdgeType::UPPER:
            return UPPER_EDGE_KERNEL;
        case EdgeType::LOWER:
            return LOWER_EDGE_KERNEL;
        default:
            return RIGHT_EDGE_KERNEL;
    }
}

std::optional<std::vector<Line>> detectConvexHullEdge(const cv::Mat& src_img, EdgeType edge_type);

std::vector<cv::Point2f> detectMinRect(const cv::Mat& src_img);

Line detectRectEdge(const std::vector<cv::Point>& src_points, EdgeType edge_type, cv::Mat* debug_img = nullptr);

Line detectRectEdge(const cv::Mat& src_img, EdgeType edge_type);

inline void detectEdge(const cv::Mat& src, cv::Mat& dst, const EdgeType edge_type = EdgeType::RIGHT)
{
    const cv::Mat& kernel = getEdgeKernel(edge_type);
    // 1. Convert source to 32-bit Float
    cv::Mat temp;
    src.convertTo(temp, CV_32F);
    // 2. Perform filter using CV_32F for the destination
    // Passing -1 for ddelt (3rd param) tells OpenCV to match the src depth (CV_32F)
    cv::filter2D(temp, dst, CV_32F, kernel);
    // 3. Post-processing: Suppress negative values
    // Note: dst is still CV_32F here.
    cv::max(dst, 0, dst);
    // 4. (Optional) Convert back to 8-bit if needed for display
    dst.convertTo(dst, CV_8U);
}

bool findInliers(const cv::Mat& src_img, std::vector<cv::Point>& inliers, float dist_thresh, int iters = 300);
}
