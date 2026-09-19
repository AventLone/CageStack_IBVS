#pragma once
#include <Eigen/Core>
#include <vector>

struct PointXYZIT
{
    Eigen::Vector3f position;
    float intensity;
    double time_offset; // Seconds relative to scan begin; points need not be sorted.
};

struct StampedCloud
{
    double begin_time{}, end_time{};
    std::vector<PointXYZIT> points;
};
