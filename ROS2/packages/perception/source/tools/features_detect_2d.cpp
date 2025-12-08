#include "perception/tools/features_detect_2d.h"

cv::RotatedRect features_2d::detectMinRect(const cv::Mat& src_img)
{
    std::vector<cv::Point> non_zero_points;
    cv::findNonZero(src_img, non_zero_points);
    const cv::RotatedRect rr = cv::minAreaRect(non_zero_points); // 最小外接矩形
    cv::Point2f corners[4];
    rr.points(corners); // corners 是 Point2f[4]

    cv::Mat debug_img;
    cv::cvtColor(src_img, debug_img, cv::COLOR_GRAY2BGR);
    for (int j = 0; j < 4; ++j)
        cv::line(debug_img, corners[j], corners[(j + 1) % 4], cv::Scalar(0, 255, 0), 2);

    return rr;
}
