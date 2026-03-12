#pragma once
#include <pcl/common/common.h>
#include <pcl/common/eigen.h>
#include <pcl/common/centroid.h>
#include <pcl/common/transforms.h>
#include <opencv2/opencv.hpp>
#include <random>

namespace feature3d
{
template<class PointT>
Eigen::Vector3f getCloudSize(const pcl::PointCloud<PointT>& cloud)
{
    Eigen::Vector4f min_point, max_point;
    pcl::getMinMax3D(cloud, min_point, max_point);
    return max_point.head<3>() - min_point.head<3>();
}

template<class PointT>
float calculateLineAngle(const pcl::PointCloud<PointT>& cloud)
{
    // Validate the cloud
    if (cloud.size() < 2)
    {
        throw std::invalid_argument("The number of the points can't be less than 2！");
    }

    std::vector<cv::Point2f> points;
    for (const auto& point : cloud.points)
    {
        points.emplace_back(point.y, point.x);
    }
    cv::Vec4f line; // Fitting result: [vx, vy, x0, y0]; y = k*x + b; k = vy / vx, b = y0 - k*x0
    cv::fitLine(points, line, cv::DIST_L2, 0.0, 0.01, 0.01);
    return -std::atan2(line[1], line[0]);
}

template<class PointT>
void getCloud(const pcl::PointCloud<PointT>& src, pcl::PointCloud<PointT>& dst, const std::vector<int>& indices)
{
    dst.reserve(indices.size());

    for (const int index : indices)
    {
        dst.push_back(src[index]);
    }
}

/* Assume by default that this plane is perpendicular to the x-y plane. */
template<class PointT>
bool findInliers(const pcl::PointCloud<PointT>& cloud, pcl::PointCloud<PointT>& inliers, const float dist_thresh, const int iters = 300)
{
    const int N = cloud.size();
    if (N < 10)
    {
        return false;
    }
    std::vector<int> best_inliers_indices;

    std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<int> uni(0, N - 1);

    for (int iter = 0; iter < iters; ++iter)
    {
        int i = uni(rng);
        int j = uni(rng);
        if (i == j)
        {
            continue;
        }
        const auto& p1 = cloud[i];
        const auto& p2 = cloud[j];

        //line in xy-plane: a*x + b*y + d = 0
        float a = p1.y - p2.y;
        float b = p2.x - p1.x;
        float d = p1.x * p2.y - p2.x * p1.y;

        float norm = std::hypot(a, b);
        if (norm < 1e-6f)
        {
            continue;
        }

        std::vector<int> inliers_indices;
        for (int k = 0; k < N; ++k)
        {
            const auto& p = cloud[k];
            if (const float dist = std::abs(a * p.x + b * p.y + d) / norm; dist < dist_thresh)
            {
                inliers_indices.push_back(k);
            }
        }

        if (inliers_indices.size() > best_inliers_indices.size())
        {
            best_inliers_indices = std::move(inliers_indices);
        }
    }

    getCloud(cloud, inliers, best_inliers_indices);

    return true;
}

template<class PointT>
Eigen::Vector4f fitPlaneByPCA(const pcl::PointCloud<PointT>& cloud, const int parallel_to = -1)
{
    if (cloud.size() < 6)
    {
        throw std::invalid_argument("The input cloud is empty");
    }
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

template<class PointT>
Eigen::Vector3f computeCenter(const pcl::PointCloud<PointT>& cloud)
{
    Eigen::Vector4f min_point, max_point;
    pcl::getMinMax3D(cloud, min_point, max_point);

    return 0.5f * (min_point.head<3>() + max_point.head<3>());
}

inline pcl::PointCloud<pcl::PointXYZ> createPalletCloud(const float angle)
{
    pcl::PointCloud<pcl::PointXYZ> front_face, left_side_face, right_side_face, cloud, angel_cloud;

    std::random_device rd; // Random device for seeding
    std::mt19937 gen(rd()); // Mersenne Twister engine

    static std::uniform_real_distribution<float> unifor(0.02, 0.2);
    static std::cauchy_distribution<float> dist_chaos(0.0f, 0.001f);

    for (float y = -0.5f; y < 0.5f; y += 0.02f)
    {
        std::normal_distribution<float> dist_normal(0.0, unifor(gen));
        for (float z = 0.0f; z < 0.15f; z += 0.02f)
        {
            front_face.emplace_back((dist_normal(gen) + dist_chaos(gen)) * 0.5f, y, z);
        }
    }

    for (float x = 0.0f; x > -1.2f; x -= 0.02f)
    {
        std::normal_distribution<float> dist_normal(0.0, 0.02);
        for (float z = 0.0f; z < 0.15f; z += 0.02f)
        {
            left_side_face.emplace_back(x, -0.5f + (dist_normal(gen) + dist_chaos(gen)) * 0.5f, z);
        }
    }

    Eigen::Isometry3f T(Eigen::Isometry3f::Identity());
    T.translate(Eigen::Vector3f(0.0, 1.0, 0.0));
    pcl::transformPointCloud(left_side_face, right_side_face, T);
    cloud = front_face + left_side_face + right_side_face;
    T = Eigen::Isometry3f::Identity();
    T.rotate(Eigen::AngleAxisf(angle, Eigen::Vector3f::UnitZ()));
    pcl::transformPointCloud(cloud, angel_cloud, T);

    return angel_cloud;
}
}
