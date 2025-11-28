#include "../../include/perception/tools/filter.h"
#include <random>

void getCloud(const SemanticCloud& semantic_cloud, const uint32_t label, pcl::PointCloud<pcl::PointXYZ>& target_cloud)
{
    for (const auto& point : semantic_cloud.points)
    {
        if (point.label == label)
        {
            target_cloud.emplace_back(point.x, point.y, point.y);
        }
    }
}

void getCloud(const SemanticCloud& semantic_cloud, pcl::PointCloud<pcl::PointXYZRGB>& colored_cloud)
{
    colored_cloud.header = semantic_cloud.header;
    colored_cloud.is_dense = semantic_cloud.is_dense;
    colored_cloud.points.reserve(semantic_cloud.points.size());

    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_int_distribution<int> dist(0, 255);

    static std::unordered_map<uint32_t, std::tuple<uint8_t, uint8_t, uint8_t>> color_map; // map label -> (r,g,b)
    for (const auto& p : semantic_cloud.points)
    {
        uint8_t r, g, b;
        if (const auto it = color_map.find(p.label); it == color_map.end())
        {
            r = static_cast<uint8_t>(dist(gen));
            g = static_cast<uint8_t>(dist(gen));
            b = static_cast<uint8_t>(dist(gen));
            color_map.emplace(p.label, std::make_tuple(r, g, b));
        }
        else
        {
            std::tie(r, g, b) = it->second;
        }

        pcl::PointXYZRGB q;
        q.x = p.x;
        q.y = p.y;
        q.z = p.z;
        q.r = r;
        q.g = g;
        q.b = b;
        colored_cloud.points.push_back(std::move(q));
    }

    colored_cloud.width = static_cast<uint32_t>(colored_cloud.points.size());
    colored_cloud.height = 1;
}
