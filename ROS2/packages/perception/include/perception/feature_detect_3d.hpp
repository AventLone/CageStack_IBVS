#pragma once
#include <pcl/common/common.h>
#include <pcl/common/eigen.h>
#include <pcl/common/centroid.h>

template<class PointT>
Eigen::Vector3f getCloudSize(const pcl::PointCloud<PointT>& cloud)
{
    Eigen::Vector4f min_point, max_point;
    pcl::getMinMax3D(cloud, min_point, max_point);
    return max_point.head<3>() - min_point.head<3>();
}

template<class PointT>
Eigen::Vector4f fitPlaneByPCA(const pcl::PointCloud<PointT>& cloud, const int parallel_to = -1)
{
    Eigen::Vector4f centroid;
    Eigen::Matrix3f covariance_matrix;
    pcl::computeMeanAndCovarianceMatrix(cloud, covariance_matrix, centroid);

    Eigen::Matrix3f eigen_vectors;
    Eigen::Vector3f eigen_values;
    pcl::eigen33(covariance_matrix, eigen_vectors, eigen_values);

    Eigen::Vector3f::Index min_row, min_col;
    eigen_values.minCoeff(&min_row, &min_col);
    Eigen::Vector3f normal = eigen_vectors.col(min_col);
    normal.normalize();

    if (parallel_to != -1)
    {
        normal[parallel_to] = 0.0f;
    }

    Eigen::Vector4f plane_coeff{};
    plane_coeff.head<3>() = normal;
    plane_coeff[3] = -normal.dot(centroid.head<3>());

    return plane_coeff;
}

template<class PointT>
float computeAngleByPCA(const pcl::PointCloud<PointT>& cloud)
{
    Eigen::Vector4f centroid;
    Eigen::Matrix3f covariance_matrix;
    pcl::computeMeanAndCovarianceMatrix(cloud, covariance_matrix, centroid);

    Eigen::Matrix3f eigen_vectors;
    Eigen::Vector3f eigen_values;
    pcl::eigen33(covariance_matrix, eigen_vectors, eigen_values);

    Eigen::Vector3f::Index min_row, min_col;
    eigen_values.minCoeff(&min_row, &min_col);
    Eigen::Vector3f normal = eigen_vectors.col(min_col);
    normal.normalize();

    return std::atan2(normal[1], normal[0]);
}
