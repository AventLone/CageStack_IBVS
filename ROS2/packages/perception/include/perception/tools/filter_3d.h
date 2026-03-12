#pragma once
#include <opencv2/core/types.hpp>
#include <pcl/common/transforms.h>
#include <pcl_conversions/pcl_conversions.h>

#include "../types/common.hpp"

void getCloud(const SemanticCloud& semantic_cloud, const int label, pcl::PointCloud<pcl::PointXYZ>& target_cloud);

void getCloud(const SemanticCloud& semantic_cloud, pcl::PointCloud<pcl::PointXYZRGB>& colored_cloud);

template<class PointT>
void getCloud(const pcl::PointCloud<PointT>& src_cloud, typename pcl::PointCloud<PointT>::Ptr inliers,
              typename pcl::PointCloud<PointT>::Ptr outliers, const Eigen::Vector3f& base_position, const ROI& roi)
{
    const ROI roi_rebased{
                roi.min_x + base_position[0], roi.max_x + base_position[0],
                roi.min_y + base_position[1], roi.max_y + base_position[1],
                roi.min_z + base_position[2], roi.max_z + base_position[2]
            };

    const size_t src_size = src_cloud.size();

    if (inliers != nullptr)
    {
        inliers->reserve(src_size);
        if (outliers != nullptr)
        {
            outliers->reserve(src_size);
            for (const PointT& point : src_cloud)
            {
                if (roi_rebased.enclose(point))
                {
                    inliers->push_back(point);
                }
                else
                {
                    outliers->push_back(point);
                }
            }
        }
        else
        {
            for (const PointT& point : src_cloud)
            {
                if (roi_rebased.enclose(point))
                {
                    inliers->push_back(point);
                }
            }
        }
    }
    else if (outliers != nullptr)
    {
        outliers->reserve(src_size);
        for (const PointT& point : src_cloud)
        {
            if (!roi_rebased.enclose(point))
            {
                outliers->push_back(point);
            }
        }
    }
}

#include <pcl/io/ply_io.h>

template<class PointT>
void getCloud(const pcl::PointCloud<PointT>& src_cloud, typename pcl::PointCloud<PointT>::Ptr inliers,
              typename pcl::PointCloud<PointT>::Ptr outliers, const Eigen::Isometry3f& base_pose, const ROI& roi)
{
    const Eigen::Isometry3f base_pose_inv = base_pose.inverse();
    pcl::PointCloud<PointT> cloud_rebased;
    pcl::transformPointCloud(src_cloud, cloud_rebased, base_pose_inv);
    const size_t src_size = src_cloud.size();
    if (inliers != nullptr)
    {
        pcl::PointCloud<PointT> inliers_temp;
        inliers_temp.reserve(src_size);
        if (outliers != nullptr)
        {
            pcl::PointCloud<PointT> outliers_temp;
            outliers_temp.reserve(src_size);
            for (const PointT& point : cloud_rebased)
            {
                if (roi.enclose(point))
                {
                    inliers_temp.push_back(point);
                }
                else
                {
                    outliers_temp.push_back(point);
                }
            }
            pcl::transformPointCloud(inliers_temp, *inliers, base_pose);
            pcl::transformPointCloud(outliers_temp, *outliers, base_pose);
        }
        else
        {
            for (const PointT& point : cloud_rebased)
            {
                if (roi.enclose(point))
                {
                    inliers_temp.push_back(point);
                }
            }
            pcl::transformPointCloud(inliers_temp, *inliers, base_pose);
        }
    }
    else if (outliers != nullptr)
    {
        pcl::PointCloud<PointT> outliers_temp;
        outliers_temp.reserve(src_size);
        for (const PointT& point : cloud_rebased)
        {
            if (!roi.enclose(point))
            {
                outliers_temp.push_back(point);
            }
        }
        pcl::transformPointCloud(outliers_temp, *outliers, base_pose);
    }
}

template<class PointT>
pcl::PointCloud<PointT> removeGround(const pcl::PointCloud<PointT>& cloud, const float threshold = 0.01f)
{
    pcl::PointCloud<PointT> cloud_removed_ground;
    cloud_removed_ground.reserve(cloud.size());
    for (const auto& point : cloud)
    {
        if (point.z > threshold)
        {
            cloud_removed_ground.push_back(point);
        }
    }

    return cloud_removed_ground;
}

/* Check for free space */
template<class PointT>
bool checkSpace(const pcl::PointCloud<PointT>& cloud, const Eigen::Isometry3f& pose, const ROI& roi)
{
    pcl::PointCloud<PointT> cloud_rebased;
    pcl::transformPointCloud(cloud, cloud_rebased, pose.inverse());
    uint32_t points_count{};
    for (const auto& point : cloud_rebased)
    {
        if (roi.enclose(point))
        {
            ++points_count;
        }
    }

    if (points_count > 6)
    {
        return false;
    }

    return true;
}
