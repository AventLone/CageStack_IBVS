#pragma once
#include <opencv2/opencv.hpp>
// class features_detect_2d
// {
// public:
//     explicit features_detect_2d() = default;
// };
namespace features_2d
{
cv::RotatedRect detectMinRect(const cv::Mat& src_img);
}
