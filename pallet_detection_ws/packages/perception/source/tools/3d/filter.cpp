#include "perception/tools/3d/filter.h"
#include <random>

void getCloud(const SemanticCloud& semantic_cloud, const int label, pcl::PointCloud<pcl::PointXYZ>& target_cloud)
{
    target_cloud.reserve(semantic_cloud.size());
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

    static std::unordered_map<int, std::tuple<uint8_t, uint8_t, uint8_t>> color_map; // map label -> (r,g,b)
    for (const auto& p : semantic_cloud.points)
    {
        uint8_t r, g, b;
        if (const auto it = color_map.find(p.label); it == color_map.end())
        {
            if (p.label == 0)
            {
                r = 255;
                g = 255;
                b = 255;
            }
            else
            {
                r = static_cast<uint8_t>(dist(gen));
                g = static_cast<uint8_t>(dist(gen));
                b = static_cast<uint8_t>(dist(gen));
            }
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
        colored_cloud.points.push_back(q);
    }

    colored_cloud.width = static_cast<uint32_t>(colored_cloud.points.size());
    colored_cloud.height = 1;
}

std::vector<RawCloud::Ptr> getInstanceClusters(const InstanceCloud& src_cloud, const int target_label)
{
    std::unordered_map<uint16_t, RawCloud::Ptr> cluster_map;

    for (const auto& point : src_cloud)
    {
        if (point.label == target_label)
        {
            if (cluster_map.find(point.instance) == cluster_map.end())
            {
                cluster_map[point.instance] = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
            }
            cluster_map[point.instance]->points.emplace_back(point.x, point.y, point.z);
        }
    }

    std::vector<RawCloud::Ptr> output_clusters;
    output_clusters.reserve(cluster_map.size());

    for (auto& [_, snd] : cluster_map)
    {
        snd->width = snd->points.size();
        snd->height = 1;
        snd->is_dense = true; // 假设没有 NaN 点

        output_clusters.push_back(std::move(snd));
    }

    return output_clusters;
}
