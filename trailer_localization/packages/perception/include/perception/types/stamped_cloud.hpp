#pragma once
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <Eigen/Core>

struct PointXYZIT
{
    Eigen::Vector3f position;
    float intensity;
    double timestamp;
};

struct TimedCloud
{
    double begin_time{0.0}, end_time{0.0};
    std::vector<PointXYZIT> points;
};

static std::optional<TimedCloud> parseCloudMsg(const sensor_msgs::msg::PointCloud2& msg)
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
        return std::nullopt;
    }

    TimedCloud cloud;
    cloud.points.reserve(static_cast<std::size_t>(msg.width) * msg.height);

    sensor_msgs::PointCloud2ConstIterator<float> x(msg, "x");
    sensor_msgs::PointCloud2ConstIterator<float> y(msg, "y");
    sensor_msgs::PointCloud2ConstIterator<float> z(msg, "z");
    sensor_msgs::PointCloud2ConstIterator<float> intensity(msg, "intensity");
    sensor_msgs::PointCloud2ConstIterator<double> timestamp(msg, "timestamp");

    for (; x != x.end(); ++x, ++y, ++z, ++timestamp, ++intensity)
    {
        cloud.points.emplace_back(Eigen::Vector3f{*x, *y, *z}, *intensity, *timestamp);
    }

    if (!cloud.points.empty())
    {
        cloud.begin_time = cloud.points.front().timestamp;
        cloud.end_time = cloud.points.back().timestamp;
    }

    return cloud;
}