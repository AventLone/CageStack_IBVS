#pragma once
#include "./point_cloud.hpp"

using RawCloud = pcl::PointCloud<pcl::PointXYZ>;
using ColoredCloud = pcl::PointCloud<pcl::PointXYZRGB>;

using SemanticCloud = pcl::PointCloud<SemanticPoint>;
using SemanticCloudPtr = std::unique_ptr<SemanticCloud>;
