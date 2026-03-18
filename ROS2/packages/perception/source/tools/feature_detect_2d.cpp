#include "perception/tools/feature_detect_2d.h"
#include <random>

namespace feature2d
{
cv::RotatedRect detectMinRect(const cv::Mat& src_img)
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

Line detectRectEdge(const cv::Mat& src_img, const EdgeType edge_type)
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

bool findInliers(const cv::Mat& src_img, std::vector<cv::Point>& inliers, const float dist_thresh, const int iters)
{
    std::vector<cv::Point> points;
    cv::findNonZero(src_img, points);

    const int N = static_cast<int>(points.size());
    if (N < 10)
    {
        return false;
    }
    std::vector<int> best_inliers_indices;

    std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<int> uni(0, N - 1);

    for (int iter = 0; iter < iters; ++iter)
    {
        const int i = uni(rng);
        const int j = uni(rng);
        if (i == j)
        {
            continue;
        }
        const auto& p1 = points[i];
        const auto& p2 = points[j];

        //line in xy-plane: a*x + b*y + d = 0
        const float a = static_cast<float>(p1.y - p2.y);
        const float b = static_cast<float>(p2.x - p1.x);
        const float d = static_cast<float>(p1.x * p2.y - p2.x * p1.y);

        const float norm = std::hypot(a, b);
        if (norm < 1e-6f)
        {
            continue;
        }

        std::vector<int> inliers_indices;
        for (int k = 0; k < N; ++k)
        {
            const auto& p = points[k];
            if (const float dist = std::abs(a * static_cast<float>(p.x) + b * static_cast<float>(p.y) + d) / norm; dist < dist_thresh)
            {
                inliers_indices.push_back(k);
            }
        }

        if (inliers_indices.size() > best_inliers_indices.size())
        {
            best_inliers_indices = std::move(inliers_indices);
        }
    }

    inliers.clear();
    inliers.reserve(N);

    for (const auto idx : best_inliers_indices)
    {
        inliers.push_back(points[idx]);
    }

    return true;
}
}
