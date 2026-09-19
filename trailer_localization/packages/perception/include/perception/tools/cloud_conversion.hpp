#pragma once
#include <algorithm>
#include <string_view>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include "perception/types/stamped_cloud.hpp"

// Hesai layout: FLOAT64 absolute per-point timestamp in seconds.
inline bool parseCloudMsg(const sensor_msgs::msg::PointCloud2& msg, StampedCloud& cloud)
{
    const auto has_field = [&msg](const std::string_view name, const uint8_t datatype)
        {
            for (const auto& field : msg.fields)
            {
                if (field.name == name)
                {
                    return field.datatype == datatype && field.count == 1;
                }
            }
            return false;
        };

    if (!has_field("x", sensor_msgs::msg::PointField::FLOAT32) ||
        !has_field("y", sensor_msgs::msg::PointField::FLOAT32) ||
        !has_field("z", sensor_msgs::msg::PointField::FLOAT32) ||
        !has_field("intensity", sensor_msgs::msg::PointField::FLOAT32) ||
        !has_field("timestamp", sensor_msgs::msg::PointField::FLOAT64))
    {
        return false;
    }

    cloud.points.clear();
    if (const auto size = static_cast<std::size_t>(msg.width) * msg.height;
        cloud.points.capacity() < size)
    {
        cloud.points.reserve(size);
    }

    sensor_msgs::PointCloud2ConstIterator<float> x(msg, "x");
    sensor_msgs::PointCloud2ConstIterator<float> y(msg, "y");
    sensor_msgs::PointCloud2ConstIterator<float> z(msg, "z");
    sensor_msgs::PointCloud2ConstIterator<float> intensity(msg, "intensity");
    sensor_msgs::PointCloud2ConstIterator<double> timestamp(msg, "timestamp");

    if (msg.width == 0 || msg.height == 0)
    {
        return false;
    }
    cloud.begin_time = *timestamp;
    cloud.end_time = *timestamp;

    for (; x != x.end(); ++x, ++y, ++z, ++timestamp, ++intensity)
    {
        cloud.begin_time = std::min(cloud.begin_time, *timestamp);
        cloud.end_time = std::max(cloud.end_time, *timestamp);
        cloud.points.emplace_back(Eigen::Vector3f{*x, *y, *z}, *intensity, *timestamp);
    }

    if (cloud.points.empty())
    {
        return false;
    }
    for (auto& point : cloud.points)
    {
        point.time_offset -= cloud.begin_time;
    }
    return true;
}

inline pcl::PointCloud<pcl::PointXYZ> getCloudXYZ(const std::vector<PointXYZIT>& src)
{
    pcl::PointCloud<pcl::PointXYZ> dst;
    dst.resize(src.size());
    for (std::size_t i = 0; i < src.size(); ++i)
    {
        dst.points[i].x = src[i].position.x();
        dst.points[i].y = src[i].position.y();
        dst.points[i].z = src[i].position.z();
    }
    return dst;
}

inline pcl::PointCloud<pcl::PointXYZI> getCloudXYZI(const std::vector<PointXYZIT>& src)
{
    pcl::PointCloud<pcl::PointXYZI> dst;
    dst.resize(src.size());
    for (std::size_t i = 0; i < src.size(); ++i)
    {
        dst.points[i].x = src[i].position.x();
        dst.points[i].y = src[i].position.y();
        dst.points[i].z = src[i].position.z();
        dst.points[i].intensity = src[i].intensity;
    }
    return dst;
}
