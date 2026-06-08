#include "perception/tools/2d/feature_detect.h"
#include <random>

namespace feature2d
{
std::optional<std::vector<Line>> detectConvexHullEdge(const cv::Mat& src_img, const EdgeType edge_type)
{
    static const auto pair_lines = [](const std::vector<cv::Point>& points1,
                                      const std::vector<cv::Point>& points2) -> std::optional<std::vector<Line>>
        {
            if (points1.empty() or points2.empty())
            {
                return std::nullopt;
            }

            std::vector<Line> lines;
            lines.reserve(points1.size() * points2.size());
            for (const auto& point1 : points1)
            {
                for (const auto& point2 : points2)
                {
                    lines.emplace_back(point1, point2);
                }
            }
            return lines;
        };

    if (src_img.empty())
    {
        return std::nullopt;
    }

    std::vector<cv::Point> non_zero_points, hull_points;
    cv::findNonZero(src_img, non_zero_points);
    if (non_zero_points.size() < 4)
    {
        return std::nullopt;
    }
    cv::convexHull(non_zero_points, hull_points);
    if (hull_points.size() < 4)
    {
        return std::nullopt;
    }
    
    const auto [min_x, max_x] = std::minmax_element(hull_points.begin(), hull_points.end(),
                                                    [](const cv::Point& a, const cv::Point& b) -> bool
                                                        {
                                                            return a.x < b.x;
                                                        });
    const auto [min_y, max_y] = std::minmax_element(hull_points.begin(), hull_points.end(),
                                                    [](const cv::Point& a, const cv::Point& b) -> bool
                                                        {
                                                            return a.y < b.y;
                                                        });
    const int mid_x = (min_x->x + max_x->x) / 2;
    const int mid_y = (min_y->y + max_y->y) / 2;

    /* Group points by quadrant */
    std::vector<cv::Point> quadrant_1, quadrant_2, quadrant_3, quadrant_4;
    for (const auto& point : hull_points)
    {
        if (point.y < mid_y)
        {
            if (point.x > mid_x)
            {
                quadrant_1.push_back(point);
            }
            else
            {
                quadrant_2.push_back(point);
            }
        }
        else
        {
            if (point.x < mid_x)
            {
                quadrant_3.push_back(point);
            }
            else
            {
                quadrant_4.push_back(point);
            }
        }
    }

    switch (edge_type)
    {
        case EdgeType::LEFT:
            return pair_lines(quadrant_2, quadrant_3);

        case EdgeType::RIGHT:
            return pair_lines(quadrant_1, quadrant_4);

        case EdgeType::UPPER:
            return pair_lines(quadrant_1, quadrant_2);

        case EdgeType::LOWER:
            return pair_lines(quadrant_3, quadrant_4);
    }

    return std::nullopt;
}

std::vector<cv::Point2f> detectMinRect(const cv::Mat& src_img)
{
    std::vector<cv::Point> non_zero_points;
    cv::findNonZero(src_img, non_zero_points);
    const cv::RotatedRect rr = cv::minAreaRect(non_zero_points); // 最小外接矩形
    std::vector<cv::Point2f> corners(4);
    rr.points(corners.data()); // corners 是 Point2f[4]

    return corners;
}

Line detectRectEdge(const std::vector<cv::Point>& src_points, const EdgeType edge_type, cv::Mat* debug_img)
{
    const cv::RotatedRect rr = cv::minAreaRect(src_points);
    std::vector<cv::Point2f> corners(4);
    rr.points(corners.data());

    if (debug_img != nullptr)
    {
        for (int i = 0; i < 4; ++i)
        {
            cv::line(*debug_img, corners[i], corners[(i + 1) % 4], cv::Scalar(0, 255, 0), 1);
        }
    }

    if (edge_type == EdgeType::LEFT || edge_type == EdgeType::RIGHT)
    {
        std::sort(corners.begin(), corners.end(), [](const cv::Point2f& a, const cv::Point2f& b) -> bool
                      {
                          return a.x < b.x;
                      });
        if (edge_type == EdgeType::LEFT)
        {
            return Line(corners[0], corners[1]);
        }
        else
        {
            return Line(corners[2], corners[3]);
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
            return Line(corners[0], corners[1]);
        }
        else
        {
            return Line(corners[2], corners[3]);
        }
    }
}

Line detectRectEdge(const cv::Mat& src_img, const EdgeType edge_type)
{
    std::vector<cv::Point> non_zero_points;
    cv::findNonZero(src_img, non_zero_points);
    return detectRectEdge(non_zero_points, edge_type);
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
        const auto a = static_cast<float>(p1.y - p2.y);
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
