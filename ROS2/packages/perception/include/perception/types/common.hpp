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
};
