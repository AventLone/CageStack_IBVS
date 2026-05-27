#pragma once
#include <opencv2/opencv.hpp>

namespace filter2d
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
}
