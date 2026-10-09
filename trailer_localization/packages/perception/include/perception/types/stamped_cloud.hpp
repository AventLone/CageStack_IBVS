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

    cloud.begin_time = *timestamp;

    for (; x != x.end(); ++x, ++y, ++z, ++timestamp, ++intensity)
    {
        cloud.points.emplace_back(Eigen::Vector3f{*x, *y, *z}, *intensity, *timestamp - cloud.begin_time);
    }

    if (cloud.points.empty())
    {
        return false;
    }
    cloud.end_time = cloud.begin_time + cloud.points.back().time_offset;
    return true;
}

// static bool parseCloudMsg(const sensor_msgs::msg::PointCloud2& msg, StampedCloud& cloud)
// {
//     const auto has_field = [&msg](const std::string_view name, const uint8_t datatype)
//     {
//         for (const auto& field : msg.fields)
//         {
//             if (field.name == name)
//             {
//                 return field.datatype == datatype && field.count == 1;
//             }
//         }
//         return false;
//     };
//
//     if (!has_field("x", sensor_msgs::msg::PointField::FLOAT32) ||
//         !has_field("y", sensor_msgs::msg::PointField::FLOAT32) ||
//         !has_field("z", sensor_msgs::msg::PointField::FLOAT32) ||
//         !has_field("intensity", sensor_msgs::msg::PointField::FLOAT32) ||
//         !has_field("time", sensor_msgs::msg::PointField::FLOAT32))
//     {
//         return false;
//     }
//
//     cloud.points.clear();
//
//     const auto size = static_cast<std::size_t>(msg.width) * msg.height;
//
//     cloud.points.reserve(size);
//
//     sensor_msgs::PointCloud2ConstIterator<float> x(msg, "x");
//     sensor_msgs::PointCloud2ConstIterator<float> y(msg, "y");
//     sensor_msgs::PointCloud2ConstIterator<float> z(msg, "z");
//     sensor_msgs::PointCloud2ConstIterator<float> intensity(msg, "intensity");
//     sensor_msgs::PointCloud2ConstIterator<float> time(msg, "time");
//
//     const double header_time = rclcpp::Time(msg.header.stamp).seconds();
//
//     double min_offset = std::numeric_limits<double>::max();
//     double max_offset = std::numeric_limits<double>::lowest();
//
//     for (; x != x.end(); ++x, ++y, ++z, ++intensity, ++time)
//     {
//         const auto offset = static_cast<double>(*time);
//
//         min_offset = std::min(min_offset, offset);
//         max_offset = std::max(max_offset, offset);
//
//         cloud.points.emplace_back(Eigen::Vector3f{*x, *y, *z}, *intensity, offset);
//     }
//
//     if (cloud.points.empty())
//     {
//         return false;
//     }
//
//     cloud.begin_time = header_time + min_offset;
//     cloud.end_time   = header_time + max_offset;
//
//     // 统一让 time_offset 相对真正的 scan begin
//     for (auto& point : cloud.points)
//     {
//         point.time_offset -= min_offset;
//     }
//
//     return true;
// }

// 按 PointCloud2 声明的字节序读取字段。
// 支持大小端转换，T 应为 float 或 uint32_t。
// template <typename T>
// static T readPointValue(const uint8_t* data, const bool is_bigendian)
// {
//     T value{};
//     std::memcpy(&value, data, sizeof(T));
//
//     static const bool host_bigendian = []
//         {
//             constexpr uint16_t marker = 0x0102;
//             uint8_t bytes[sizeof(marker)];
//             std::memcpy(bytes, &marker, sizeof(marker));
//             return bytes[0] == 0x01;
//         }();
//
//     if (is_bigendian != host_bigendian)
//     {
//         auto* bytes = reinterpret_cast<uint8_t*>(&value);
//         std::reverse(bytes, bytes + sizeof(T));
//     }
//
//     return value;
// }

// static bool parseCloudMsg(const sensor_msgs::msg::PointCloud2& msg, StampedCloud& cloud)
// {
//     using PointField = sensor_msgs::msg::PointField;
//
//     cloud.points.clear();
//     cloud.begin_time = 0.0;
//     cloud.end_time = 0.0;
//
//     if (msg.width == 0 || msg.height == 0 || msg.point_step == 0)
//     {
//         return false;
//     }
//
//     // 查找字段定义。
//     const auto find_field = [&msg](const std::string_view name) -> const PointField*
//     {
//         for (const auto& field : msg.fields)
//         {
//             if (field.name == name)
//             {
//                 return &field;
//             }
//         }
//         return nullptr;
//     };
//
//     const auto* x_field = find_field("x");
//     const auto* y_field = find_field("y");
//     const auto* z_field = find_field("z");
//     const auto* intensity_field = find_field("intensity");
//     const auto* timestamp_field = find_field("timestamp");
//
//     // 检查字段类型、数量和内存边界。
//     const auto valid_field = [&msg](const PointField* field, const uint8_t datatype,
//                                     const uint32_t count, const std::size_t element_size)
//     {
//         if (field == nullptr || field->datatype != datatype ||
//             field->count != count || field->offset > msg.point_step)
//         {
//             return false;
//         }
//
//         const std::size_t bytes = static_cast<std::size_t>(count) * element_size;
//         return bytes <= msg.point_step - field->offset;
//     };
//
//     if (!valid_field(x_field, PointField::FLOAT32, 1, sizeof(float)) ||
//         !valid_field(y_field, PointField::FLOAT32, 1, sizeof(float)) ||
//         !valid_field(z_field, PointField::FLOAT32, 1, sizeof(float)) ||
//         !valid_field(intensity_field, PointField::FLOAT32, 1, sizeof(float)) ||
//         !valid_field(timestamp_field, PointField::UINT32, 2, sizeof(uint32_t)))
//     {
//         return false;
//     }
//
//     if (const std::size_t min_row_size = static_cast<std::size_t>(msg.width) * msg.point_step;
//         msg.row_step < min_row_size)
//     {
//         return false;
//     }
//
//     if (const std::size_t required_size = static_cast<std::size_t>(msg.row_step) * msg.height;
//         msg.data.size() < required_size)
//     {
//         return false;
//     }
//
//     const auto get_timestamp_ns = [&msg, timestamp_field](const uint8_t* point) -> uint64_t
//     {
//         // Isaac Sim 6.0:
//         // timestamp[0] = 低 32 位
//         // timestamp[1] = 高 32 位
//         const uint8_t* timestamp_data = point + timestamp_field->offset;
//
//         const auto low = readPointValue<uint32_t>(timestamp_data, msg.is_bigendian);
//
//         const auto high = readPointValue<uint32_t>(timestamp_data + sizeof(uint32_t), msg.is_bigendian);
//
//         return (static_cast<uint64_t>(high) << 32) | static_cast<uint64_t>(low);
//     };
//
//     // 第一遍：获取整个扫描的最早、最晚点时间戳。
//     uint64_t min_timestamp_ns = std::numeric_limits<uint64_t>::max();
//     uint64_t max_timestamp_ns = 0;
//
//     for (uint32_t row = 0; row < msg.height; ++row)
//     {
//         const uint8_t* row_data = msg.data.data() + static_cast<std::size_t>(row) * msg.row_step;
//         for (uint32_t col = 0; col < msg.width; ++col)
//         {
//             const uint8_t* point = row_data + static_cast<std::size_t>(col) * msg.point_step;
//             const uint64_t timestamp_ns = get_timestamp_ns(point);
//
//             min_timestamp_ns = std::min(min_timestamp_ns, timestamp_ns);
//             max_timestamp_ns = std::max(max_timestamp_ns, timestamp_ns);
//         }
//     }
//
//     constexpr double NS_TO_SEC = 1e-9;
//
//     cloud.begin_time = static_cast<double>(min_timestamp_ns) * NS_TO_SEC;
//     cloud.end_time = static_cast<double>(max_timestamp_ns) * NS_TO_SEC;
//
//     // 第二遍：解析点云，并计算相对于扫描起点的时间偏移。
//     const std::size_t point_count = static_cast<std::size_t>(msg.width) * msg.height;
//
//     cloud.points.reserve(point_count);
//     for (uint32_t row = 0; row < msg.height; ++row)
//     {
//         const uint8_t* row_data = msg.data.data() + static_cast<std::size_t>(row) * msg.row_step;
//
//         for (uint32_t col = 0; col < msg.width; ++col)
//         {
//             const uint8_t* point = row_data + static_cast<std::size_t>(col) * msg.point_step;
//             const auto x = readPointValue<float>(point + x_field->offset, msg.is_bigendian);
//             const auto y = readPointValue<float>(point + y_field->offset, msg.is_bigendian);
//             const auto z = readPointValue<float>(point + z_field->offset, msg.is_bigendian);
//             const auto intensity = readPointValue<float>(point + intensity_field->offset, msg.is_bigendian);
//             const uint64_t timestamp_ns = get_timestamp_ns(point);
//
//             // 先做整数减法，再转换为 double，减少精度损失。
//             const double time_offset = static_cast<double>(timestamp_ns - min_timestamp_ns) * NS_TO_SEC;
//             cloud.points.push_back(PointXYZIT{Eigen::Vector3f{x, y, z}, intensity, time_offset});
//         }
//     }
//
//     return !cloud.points.empty();
// }

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