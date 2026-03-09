#pragma once
#include <opencv2/core/types.hpp>
#include <pcl/common/transforms.h>

#include "../types/common.hpp"

void getCloud(const SemanticCloud& semantic_cloud, const uint32_t label, pcl::PointCloud<pcl::PointXYZ>& target_cloud);

void getCloud(const SemanticCloud& semantic_cloud, pcl::PointCloud<pcl::PointXYZRGB>& colored_cloud);

template<class PointT>
void getCloud(const pcl::PointCloud<PointT>& src_cloud, typename pcl::PointCloud<PointT>::Ptr inliers,
              typename pcl::PointCloud<PointT>::Ptr outliers, const Eigen::Vector3f& origin_base, const ROI& roi)
{
    const ROI roi_rebased{
                roi.min_x + origin_base[0], roi.max_x + origin_base[0],
                roi.min_y + origin_base[1], roi.max_y + origin_base[1],
                roi.min_z + origin_base[2], roi.max_z + origin_base[2]
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
    pcl::PointCloud<PointT> transformed_cloud;
    pcl::transformPointCloud(cloud, transformed_cloud, pose);
    uint32_t points_count{};
    for (const auto& point : transformed_cloud)
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
