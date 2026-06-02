#pragma once
#include <opencv2/opencv.hpp>

namespace feature2d
{
inline void open(const cv::Mat& src, cv::Mat& dst)
{
    CV_Assert(!src.empty());
    static const cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2));
    cv::morphologyEx(src, dst, cv::MORPH_OPEN, kernel);
}

inline void close(const cv::Mat& src, cv::Mat& dst)
{
    CV_Assert(!src.empty());
    static const cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2));
    cv::morphologyEx(src, dst, cv::MORPH_CLOSE, kernel, cv::Point(-1, -1), 2);
}

inline void removeIsolatedPoints(const cv::Mat& src, cv::Mat& dst)
{
    CV_Assert(src.type() == CV_8UC1);
    // Kernel without center (8-neighborhood only)
    static const cv::Mat kernel = (cv::Mat_<uchar>(3, 3) <<
                                   1, 1, 1,
                                   1, 0, 1,
                                   1, 1, 1);
    cv::Mat mask;
    cv::filter2D(src, mask, CV_8U, kernel);
    src.copyTo(dst, mask);
}

inline void denoise(const cv::Mat& src, cv::Mat& dst)
{
    CV_Assert(!src.empty() && src.type() == CV_8UC1);
    static const cv::Mat structuring_element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2));
    static const cv::Mat kernel = (cv::Mat_<uchar>(3, 3) <<
                                   1, 1, 1,
                                   1, 0, 1,
                                   1, 1, 1);

    cv::Mat temp, mask, closed_img;
    cv::filter2D(src, mask, CV_8U, kernel);
    src.copyTo(temp, mask);
    cv::morphologyEx(temp, closed_img, cv::MORPH_CLOSE, structuring_element);
    cv::morphologyEx(closed_img, dst, cv::MORPH_OPEN, structuring_element);
}

inline float IoU(const cv::Rect& a, const cv::Rect& b)
{
    const auto inter_x1 = std::max(a.x, b.x);
    const auto inter_y1 = std::max(a.y, b.y);
    const auto inter_x2 = std::min(a.x + a.width, b.x + b.width);
    const auto inter_y2 = std::min(a.y + a.height, b.y + b.height);

    const auto inter_w = std::max(0, inter_x2 - inter_x1);
    const auto inter_h = std::max(0, inter_y2 - inter_y1);
    const auto inter_area = inter_w * inter_h;

    const auto area_a = std::max(0, a.width) * std::max(0, a.height);
    const auto area_b = std::max(0, b.width) * std::max(0, b.height);
    const auto union_area = area_a + area_b - inter_area;

    if (union_area <= 0)
    {
        return 0.0f;
    }

    return static_cast<float>(inter_area) / static_cast<float>(union_area);
}
}
