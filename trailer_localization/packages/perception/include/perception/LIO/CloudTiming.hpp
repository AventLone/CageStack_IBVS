#pragma once
#include <algorithm>
#include <cstring>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>
#include <string>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include "perception/LIO/ImuTypes.hpp"

namespace lio
{
struct CloudTimingConfig
{
    std::string time_field{"auto"};
    std::string time_mode{"auto"};        // auto, relative, absolute
    double time_scale{0.0};                // 0: infer from field; otherwise seconds/unit
    bool stamp_is_end{false};              // For relative timestamps only.
    bool allow_untimed_cloud{false};       // Explicit opt-in; all points are at scan end.
    double scan_duration{0.1};             // Used only for untimed clouds.
    double max_scan_duration{0.5};
};

struct TimedCloud
{
    double begin{0.0}, end{0.0};
    bool has_point_time{false};
    std::vector<PointXYZT> points;
    std::vector<float> intensities;
};

inline TimedCloud readTimedCloud(const sensor_msgs::msg::PointCloud2& msg, const CloudTimingConfig& config)
{
    using Field = sensor_msgs::msg::PointField;
    if (config.time_mode != "auto" && config.time_mode != "relative" && config.time_mode != "absolute")
        throw std::invalid_argument("time_mode must be auto, relative or absolute.");
    if (!std::isfinite(config.time_scale) || config.time_scale < 0.0 ||
        !std::isfinite(config.scan_duration) || config.scan_duration <= 0.0 ||
        !std::isfinite(config.max_scan_duration) || config.max_scan_duration < config.scan_duration)
        throw std::invalid_argument("Invalid LiDAR time scale or scan duration.");
    const auto field = [&msg](const std::string& name) -> const Field*
    {
        for (const auto& item : msg.fields) if (item.name == name) return &item;
        return nullptr;
    };
    const Field *x = field("x"), *y = field("y"), *z = field("z"), *intensity = field("intensity");
    const Field* time = nullptr;
    if (config.time_field == "auto")
    {
        for (const auto& name : {"timestamp", "time", "t", "offset_time"})
            if ((time = field(name))) break;
    }
    else time = field(config.time_field);
    if (!x || !y || !z) throw std::invalid_argument("PointCloud2 requires x/y/z fields.");
    if (!time && !config.allow_untimed_cloud)
        throw std::invalid_argument("No per-point time field: configure time_field or explicitly enable allow_untimed_cloud.");
    if (msg.width == 0 || msg.height == 0 || msg.point_step == 0 ||
        static_cast<std::uint64_t>(msg.row_step) < static_cast<std::uint64_t>(msg.width) * msg.point_step ||
        static_cast<std::uint64_t>(msg.row_step) * msg.height > msg.data.size())
        throw std::invalid_argument("Empty or malformed PointCloud2 layout.");
    const auto fieldSize = [](const Field& f) -> std::size_t
    {
        switch (f.datatype)
        {
            case Field::INT8: case Field::UINT8: return 1;
            case Field::INT16: case Field::UINT16: return 2;
            case Field::INT32: case Field::UINT32: case Field::FLOAT32: return 4;
            case Field::FLOAT64: return 8;
            default: throw std::invalid_argument("Unsupported PointCloud2 field datatype.");
        }
    };
    for (const auto* f : {x, y, z, intensity, time})
        if (f && (f->count != 1 || static_cast<std::uint64_t>(f->offset) + fieldSize(*f) > msg.point_step))
            throw std::invalid_argument("PointCloud2 scalar field exceeds point_step.");
    const std::uint16_t endian_probe = 1;
    const bool host_big_endian = *reinterpret_cast<const unsigned char*>(&endian_probe) == 0;
    const auto read = [&](const unsigned char* point, const Field& f)
    {
        unsigned char bytes[8]{};
        const std::size_t size = fieldSize(f);
        std::memcpy(bytes, point + f.offset, size);
        if (msg.is_bigendian != host_big_endian) std::reverse(bytes, bytes + size);
        // memcpy avoids alignment and strict-aliasing assumptions about PointCloud2.
        const auto value = [&bytes]<class T>() -> double
        {
            T result;
            std::memcpy(&result, bytes, sizeof(T));
            return static_cast<double>(result);
        };
        switch (f.datatype)
        {
            case Field::INT8: return value.template operator()<std::int8_t>();
            case Field::UINT8: return value.template operator()<std::uint8_t>();
            case Field::INT16: return value.template operator()<std::int16_t>();
            case Field::UINT16: return value.template operator()<std::uint16_t>();
            case Field::INT32: return value.template operator()<std::int32_t>();
            case Field::UINT32: return value.template operator()<std::uint32_t>();
            case Field::FLOAT32: return value.template operator()<float>();
            case Field::FLOAT64: return value.template operator()<double>();
            default: throw std::invalid_argument("Unsupported point datatype.");
        }
    };
    double scale = config.time_scale;
    if (scale == 0.0)
    {
        if (!time || time->name == "timestamp" || time->name == "time") scale = 1.0;
        else if (time->name == "t" || time->name == "offset_time") scale = 1e-9;
        else throw std::invalid_argument("Custom time fields require an explicit time_scale.");
    }
    const bool absolute = time && (config.time_mode == "absolute" ||
                          (config.time_mode == "auto" && time->name == "timestamp"));
    TimedCloud result;
    result.has_point_time = time != nullptr;
    double min_time = std::numeric_limits<double>::infinity(), max_time = -min_time;
    for (std::uint32_t row = 0; row < msg.height; ++row)
    {
        for (std::uint32_t column = 0; column < msg.width; ++column)
        {
            const auto* data = msg.data.data() + static_cast<std::size_t>(row) * msg.row_step +
                               static_cast<std::size_t>(column) * msg.point_step;
            const double t = time ? read(data, *time) * scale : config.scan_duration;
            if (!std::isfinite(t) || (!absolute && t < 0.0))
                throw std::invalid_argument("Non-finite or negative relative point time.");
            // Determine the full scan interval BEFORE spatial/intensity filtering.
            min_time = std::min(min_time, t);
            max_time = std::max(max_time, t);
            PointXYZT point{static_cast<float>(read(data, *x)), static_cast<float>(read(data, *y)),
                            static_cast<float>(read(data, *z)), t};
            if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z)) continue;
            result.points.push_back(point);
            result.intensities.push_back(intensity ? static_cast<float>(read(data, *intensity)) : 1.0f);
        }
    }
    const double stamp = static_cast<double>(msg.header.stamp.sec) + 1e-9 * msg.header.stamp.nanosec;
    if (absolute)
    {
        result.begin = min_time;
        result.end = max_time;
        for (auto& point : result.points) point.relative_time -= min_time;
    }
    else
    {
        result.begin = config.stamp_is_end ? stamp - max_time : stamp;
        result.end = result.begin + max_time;
    }
    if (!std::isfinite(result.begin) || !std::isfinite(result.end) || result.end <= result.begin ||
        result.end - result.begin > config.max_scan_duration)
        throw std::invalid_argument("Invalid scan duration; check point time units and time_mode.");
    return result;
}
}  // namespace lio
