#pragma once
#include "./point_cloud.hpp"
#include "./concurrentqueue.hpp"


using RawCloud = pcl::PointCloud<pcl::PointXYZ>;
using ColoredCloud = pcl::PointCloud<pcl::PointXYZRGB>;

using SemanticCloud = pcl::PointCloud<SemanticPoint>;
using SemanticCloudPtr = std::unique_ptr<SemanticCloud>;

template<class T>
using lockfree_queue = moodycamel::ConcurrentQueue<T>;
