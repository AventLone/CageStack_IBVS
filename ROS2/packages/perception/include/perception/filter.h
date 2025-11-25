#pragma once
#include "perception/types/point_cloud.h"

void getCloud(const SemanticCloud& semantic_cloud, const uint32_t label, pcl::PointCloud<pcl::PointXYZ>& target_cloud);

void getCloud(const SemanticCloud& semantic_cloud, pcl::PointCloud<pcl::PointXYZRGB>& colored_cloud);
