#include "perception/tools/feature_detect_2d.h"

cv::RotatedRect feature2d::detectMinRect(const cv::Mat& src_img)
{
    std::vector<cv::Point> non_zero_points;
    cv::findNonZero(src_img, non_zero_points);
    const cv::RotatedRect rr = cv::minAreaRect(non_zero_points); // 最小外接矩形
    cv::Point2f corners[4];
    rr.points(corners); // corners 是 Point2f[4]

    cv::Mat debug_img;
    cv::cvtColor(src_img, debug_img, cv::COLOR_GRAY2BGR);
    for (int j = 0; j < 4; ++j)
    {
        cv::line(debug_img, corners[j], corners[(j + 1) % 4], cv::Scalar(0, 255, 0), 1);
        cv::circle(debug_img, corners[j], 3, cv::Scalar(0, 0, 255), 2);
    }

    return rr;
}

feature2d::Line feature2d::detectRectEdge(const cv::Mat& src_img, const EdgeType edge_type)
{
    std::vector<cv::Point> non_zero_points;
    cv::findNonZero(src_img, non_zero_points);
    const cv::RotatedRect rr = cv::minAreaRect(non_zero_points);
    std::vector<cv::Point2f> corners(4);
    rr.points(corners.data());

    if (edge_type == EdgeType::LEFT || edge_type == EdgeType::RIGHT)
    {
        std::sort(corners.begin(), corners.end(), [](const cv::Point2f& a, const cv::Point2f& b) -> bool
                      {
                          return a.x < b.x;
                      });
        if (edge_type == EdgeType::LEFT)
        {
            return feature2d::Line(corners[0], corners[1]);
        }
        else
        {
            return feature2d::Line(corners[2], corners[3]);
        }
    }
    else
    {
        std::sort(corners.begin(), corners.end(), [](const cv::Point2f& a, const cv::Point2f& b) -> bool
                      {
                          return a.y < b.y;
                      });
        if (edge_type == EdgeType::UPPER)
        {
            return feature2d::Line(corners[0], corners[1]);
        }
        else
        {
            return feature2d::Line(corners[2], corners[3]);
        }
    }
}
