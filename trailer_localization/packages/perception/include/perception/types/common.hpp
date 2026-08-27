#pragma once
#include <opencv2/core/types.hpp>
#include "./point_cloud.hpp"

using RawCloud = pcl::PointCloud<pcl::PointXYZ>;
using ColoredCloud = pcl::PointCloud<pcl::PointXYZRGB>;

using SemanticCloud = pcl::PointCloud<SemanticPoint>;
using SemanticCloudPtr = std::unique_ptr<SemanticCloud>;

struct ROI
{
    float min_x, max_x;
    float min_y, max_y;
    float min_z, max_z;

    template<class PointT>
    bool enclose(const PointT& point) const noexcept
    {
        return point.x > min_x && point.x < max_x &&
               point.y > min_y && point.y < max_y &&
               point.z > min_z && point.z < max_z;
    }

    template<class PointT>
    static ROI getBBox(const pcl::PointCloud<PointT>& cloud)
    {
        ROI roi{};
        if (cloud.empty())
        {
            return roi;
        }

        roi.min_x = roi.max_x = cloud.points[0].x;
        roi.min_y = roi.max_y = cloud.points[0].y;
        roi.min_z = roi.max_z = cloud.points[0].z;

        for (const auto& point : cloud.points)
        {
            roi.min_x = std::min(roi.min_x, point.x);
            roi.max_x = std::max(roi.max_x, point.x);
            roi.min_y = std::min(roi.min_y, point.y);
            roi.max_y = std::max(roi.max_y, point.y);
            roi.min_z = std::min(roi.min_z, point.z);
            roi.max_z = std::max(roi.max_z, point.z);
        }

        return roi;
    }
};

/**
 * @brief Template type for covariance matrices
 * @tparam Type The vector type for which to generate a covariance (usually a state or measurement type)
 */
template<class Type>
using CovarianceMatrix = Eigen::Matrix<typename Type::Scalar, Type::RowsAtCompileTime, Type::RowsAtCompileTime>;

/**
 * @class CovarianceSquareRoot
 * @brief Template type for covariance square roots
 * @tparam Type The vector type for which to generate a covariance (usually a state or measurement type)
 */
template<class Type>
using CovarianceSquareRoot = Eigen::LLT<CovarianceMatrix<Type>, Eigen::Lower>;

/**
 * @class Jacobian
 * @brief Template type of jacobian of VecA to VecB
 */
template<class VecA, class VecB>
using Jacobian = Eigen::Matrix<typename VecA::Scalar, VecA::RowsAtCompileTime, VecB::RowsAtCompileTime>;
