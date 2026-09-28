#pragma once
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <Eigen/Core>
#include <execution>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <rclcpp/time.hpp>

struct PointXYZIT
{
    Eigen::Vector3f position;
    float intensity;
    double time_offset;   // Time relative to scan begin [s].
};

struct StampedCloud
{
    double begin_time, end_time;   // Timestamp at scan begin and end.
    std::vector<PointXYZIT> points;
};

// static bool parseCloudMsg(const sensor_msgs::msg::PointCloud2& msg, StampedCloud& cloud)
// {
//     const auto has_field = [&msg](const std::string_view name, const uint8_t datatype)
//         {
//             for (const auto& field : msg.fields)
//             {
//                 if (field.name == name)
//                 {
//                     return field.datatype == datatype && field.count == 1;
//                 }
//             }
//             return false;
//         };
//
//     if (!has_field("x", sensor_msgs::msg::PointField::FLOAT32) ||
//         !has_field("y", sensor_msgs::msg::PointField::FLOAT32) ||
//         !has_field("z", sensor_msgs::msg::PointField::FLOAT32) ||
//         !has_field("intensity", sensor_msgs::msg::PointField::FLOAT32) ||
//         !has_field("timestamp", sensor_msgs::msg::PointField::FLOAT64))
//     {
//         return false;
//     }
//
//     cloud.points.clear();
//     if (const auto size = static_cast<std::size_t>(msg.width) * msg.height;
//         cloud.points.capacity() < size)
//     {
//         cloud.points.reserve(size);
//     }
//
//     sensor_msgs::PointCloud2ConstIterator<float> x(msg, "x");
//     sensor_msgs::PointCloud2ConstIterator<float> y(msg, "y");
//     sensor_msgs::PointCloud2ConstIterator<float> z(msg, "z");
//     sensor_msgs::PointCloud2ConstIterator<float> intensity(msg, "intensity");
//     sensor_msgs::PointCloud2ConstIterator<double> timestamp(msg, "timestamp");
//
//     cloud.begin_time = *timestamp;
//
//     for (; x != x.end(); ++x, ++y, ++z, ++timestamp, ++intensity)
//     {
//         cloud.points.emplace_back(Eigen::Vector3f{*x, *y, *z}, *intensity, *timestamp - cloud.begin_time);
//     }
//
//     if (cloud.points.empty())
//     {
//         return false;
//     }
//     cloud.end_time = cloud.begin_time + cloud.points.back().time_offset;
//     return true;
// }

static bool parseCloudMsg(const sensor_msgs::msg::PointCloud2& msg, StampedCloud& cloud)
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
        !has_field("time", sensor_msgs::msg::PointField::FLOAT32))
    {
        return false;
    }

    cloud.points.clear();

    const auto size = static_cast<std::size_t>(msg.width) * msg.height;

    cloud.points.reserve(size);

    sensor_msgs::PointCloud2ConstIterator<float> x(msg, "x");
    sensor_msgs::PointCloud2ConstIterator<float> y(msg, "y");
    sensor_msgs::PointCloud2ConstIterator<float> z(msg, "z");
    sensor_msgs::PointCloud2ConstIterator<float> intensity(msg, "intensity");
    sensor_msgs::PointCloud2ConstIterator<float> time(msg, "time");

    const double header_time = rclcpp::Time(msg.header.stamp).seconds();

    double min_offset = std::numeric_limits<double>::max();
    double max_offset = std::numeric_limits<double>::lowest();

    for (; x != x.end(); ++x, ++y, ++z, ++intensity, ++time)
    {
        const auto offset = static_cast<double>(*time);

        min_offset = std::min(min_offset, offset);
        max_offset = std::max(max_offset, offset);

        cloud.points.emplace_back(Eigen::Vector3f{*x, *y, *z}, *intensity, offset);
    }

    if (cloud.points.empty())
    {
        return false;
    }

    cloud.begin_time = header_time + min_offset;
    cloud.end_time   = header_time + max_offset;

    // 统一让 time_offset 相对真正的 scan begin
    for (auto& point : cloud.points)
    {
        point.time_offset -= min_offset;
    }

    return true;
}

static pcl::PointCloud<pcl::PointXYZ> getCloudXYZ(const std::vector<PointXYZIT>& src)
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

static pcl::PointCloud<pcl::PointXYZI> getCloudXYZI(const std::vector<PointXYZIT>& src)
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