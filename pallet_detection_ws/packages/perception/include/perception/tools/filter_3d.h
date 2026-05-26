#pragma once
#include <opencv2/opencv.hpp>
#include <pcl/common/transforms.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/features/normal_3d.h>   // Core normal estimation header
#include <pcl/kdtree/kdtree_flann.h>

#include "../types/common.hpp"

void getCloud(const SemanticCloud& semantic_cloud, int label, pcl::PointCloud<pcl::PointXYZ>& target_cloud);

void getCloud(const SemanticCloud& semantic_cloud, pcl::PointCloud<pcl::PointXYZRGB>& colored_cloud);

template<class PointT>
void getCloud(const pcl::PointCloud<PointT> cloud, pcl::PointCloud<pcl::PointXYZRGB>& colored_cloud)
{
    const static auto get_random_color = []() -> std::tuple<uint8_t, uint8_t, uint8_t>
        {
            static cv::RNG rng(66);
            // 1. 在 HSV 空间创建 1x1 的像素点
            // H (色调): 0-180 随机
            // S (饱和度): 200-255 确保色彩鲜艳
            // V (亮度): 200-255 确保颜色明亮，不发暗
            const cv::Mat hsv(1, 1, CV_8UC3, cv::Scalar(rng.uniform(0, 180), rng.uniform(200, 255), rng.uniform(200, 255)));

            // 2. 将其转换回 BGR 空间
            cv::Mat bgr;
            cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);

            // 3. 读取转换后的颜色
            const cv::Vec3b p = bgr.at<cv::Vec3b>(0, 0);
            return {p[0], p[1], p[2]};
        };

    static std::unordered_map<int, std::tuple<uint8_t, uint8_t, uint8_t>> color_map; // map label -> (r,g,b)

    for (const auto& p : cloud)
    {
        uint8_t r, g, b;
        if (const auto it = color_map.find(p.label); it == color_map.end())
        {
            std::tie(r, g, b) = get_random_color();

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
}

inline void getColorCloudFromInstanceCloud(const InstanceCloud& cloud, pcl::PointCloud<pcl::PointXYZRGB>& colored_cloud)
{
    const static auto get_random_color = []() -> std::tuple<uint8_t, uint8_t, uint8_t>
        {
            static cv::RNG rng(66);
            // 1. 在 HSV 空间创建 1x1 的像素点
            // H (色调): 0-180 随机
            // S (饱和度): 200-255 确保色彩鲜艳
            // V (亮度): 200-255 确保颜色明亮，不发暗
            const cv::Mat hsv(1, 1, CV_8UC3, cv::Scalar(rng.uniform(0, 180), rng.uniform(200, 255), rng.uniform(200, 255)));

            // 2. 将其转换回 BGR 空间
            cv::Mat bgr;
            cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);

            // 3. 读取转换后的颜色
            const cv::Vec3b p = bgr.at<cv::Vec3b>(0, 0);
            return {p[0], p[1], p[2]};
        };

    static std::unordered_map<int, std::tuple<uint8_t, uint8_t, uint8_t>> color_map; // map label -> (r,g,b)

    for (const auto& p : cloud)
    {
        uint8_t r, g, b;
        if (const auto it = color_map.find(p.instance); it == color_map.end())
        {
            std::tie(r, g, b) = get_random_color();

            color_map.emplace(p.instance, std::make_tuple(r, g, b));
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
}

template<class PointT>
void getCloud(const pcl::PointCloud<PointT>& src_cloud, typename pcl::PointCloud<PointT>::Ptr inliers,
              typename pcl::PointCloud<PointT>::Ptr outliers, const Eigen::Vector3f& base_position, const ROI& roi)
{
    const ROI roi_rebased{
                roi.min_x + base_position[0], roi.max_x + base_position[0],
                roi.min_y + base_position[1], roi.max_y + base_position[1],
                roi.min_z + base_position[2], roi.max_z + base_position[2]
            };

    const size_t src_size = src_cloud.size();

    if (inliers != nullptr)
    {
        inliers->reserve(src_size);
        if (outliers != nullptr)
        {
            outliers->reserve(src_size);
            for (const PointT& point : src_cloud)
            {
                if (roi_rebased.enclose(point))
                {
                    inliers->push_back(point);
                }
                else
                {
                    outliers->push_back(point);
                }
            }
        }
        else
        {
            for (const PointT& point : src_cloud)
            {
                if (roi_rebased.enclose(point))
                {
                    inliers->push_back(point);
                }
            }
        }
    }
    else if (outliers != nullptr)
    {
        outliers->reserve(src_size);
        for (const PointT& point : src_cloud)
        {
            if (!roi_rebased.enclose(point))
            {
                outliers->push_back(point);
            }
        }
    }
}

template<class PointT>
void getCloud(const pcl::PointCloud<PointT>& src_cloud, typename pcl::PointCloud<PointT>::Ptr inliers,
              typename pcl::PointCloud<PointT>::Ptr outliers, const Eigen::Isometry3f& base_pose, const ROI& roi)
{
    const Eigen::Isometry3f base_pose_inv = base_pose.inverse();
    pcl::PointCloud<PointT> cloud_rebased;
    pcl::transformPointCloud(src_cloud, cloud_rebased, base_pose_inv);
    const size_t src_size = src_cloud.size();
    if (inliers != nullptr)
    {
        pcl::PointCloud<PointT> inliers_temp;
        inliers_temp.reserve(src_size);
        if (outliers != nullptr)
        {
            pcl::PointCloud<PointT> outliers_temp;
            outliers_temp.reserve(src_size);
            for (const PointT& point : cloud_rebased)
            {
                if (roi.enclose(point))
                {
                    inliers_temp.push_back(point);
                }
                else
                {
                    outliers_temp.push_back(point);
                }
            }
            pcl::transformPointCloud(inliers_temp, *inliers, base_pose);
            pcl::transformPointCloud(outliers_temp, *outliers, base_pose);
        }
        else
        {
            for (const PointT& point : cloud_rebased)
            {
                if (roi.enclose(point))
                {
                    inliers_temp.push_back(point);
                }
            }
            pcl::transformPointCloud(inliers_temp, *inliers, base_pose);
        }
    }
    else if (outliers != nullptr)
    {
        pcl::PointCloud<PointT> outliers_temp;
        outliers_temp.reserve(src_size);
        for (const PointT& point : cloud_rebased)
        {
            if (!roi.enclose(point))
            {
                outliers_temp.push_back(point);
            }
        }
        pcl::transformPointCloud(outliers_temp, *outliers, base_pose);
    }
}

/**
 * @brief 提取指定语义标签的所有实例聚类
 *
 * @param src_cloud  输入的目标点云
 * @param target_label 需要提取的指定语义标签（例如：车辆、行人等）
 * @return std::vector<pcl::PointCloud<InstancePoint>::Ptr> 包含所有独立实例点云的 vector
 */
std::vector<RawCloud::Ptr> getInstanceClusters(const InstanceCloud& src_cloud, int target_label);

template<class PointT>
void removeGround(const pcl::PointCloud<PointT>& cloud_in, pcl::PointCloud<PointT>& cloud_out, const float threshold = 0.01f)
{
    cloud_out.reserve(cloud_in.size());
    float min_z{std::numeric_limits<float>::max()};
    for (const auto& point : cloud_in)
    {
        min_z = std::min(min_z, point.z);
    }

    min_z += threshold;

    for (const auto& point : cloud_in)
    {
        if (point.z > min_z)
        {
            cloud_out.push_back(point);
        }
    }
}

/* Check for free space */
template<class PointT>
bool checkSpace(const pcl::PointCloud<PointT>& cloud, const Eigen::Isometry3f& pose, const ROI& roi)
{
    pcl::PointCloud<PointT> cloud_rebased;
    pcl::transformPointCloud(cloud, cloud_rebased, pose.inverse());

    uint32_t points_count{};
    for (const auto& point : cloud_rebased)
    {
        if (roi.enclose(point))
        {
            ++points_count;
        }
    }

    if (points_count > 9)
    {
        return false;
    }

    return true;
}

template<class PointT>
bool normalFilter(const typename pcl::PointCloud<PointT>::Ptr& src_cloud,
                  typename pcl::PointCloud<PointT>::Ptr cloud_in,
                  typename pcl::PointCloud<PointT>::Ptr cloud_out = nullptr,
                  const int direction = 0)
{
    const size_t N = src_cloud.size();
    if (N < 10)
    {
        return false;
    }

    // ===================== 2. Configure Normal Estimation =====================
    // auto normals = std::make_shared<pcl::PointCloud<pcl::Normal>>();
    pcl::PointCloud<pcl::Normal> normals;
    pcl::NormalEstimation<PointT, pcl::Normal> ne; // Normal estimation object

    // Set input point cloud
    ne.setInputCloud(src_cloud);

    // Create a KdTree for fast neighborhood search (REQUIRED)
    auto tree = std::make_shared<pcl::search::KdTree<PointT>>();
    ne.setSearchMethod(tree);

    // Key parameter: Search radius (defines the local neighborhood)
    // Too small = noisy normals; Too large = over-smoothing
    ne.setRadiusSearch(0.05); // 5cm neighborhood (adjust for your data!)

    // ===================== 3. Compute Normals =====================
    ne.compute(normals);

    constexpr float threshold = 0.66f;
    if (cloud_in != nullptr)
    {
        cloud_in->clear();
        cloud_in->reserve(N);
        if (cloud_out != nullptr)
        {
            cloud_out->clear();
            cloud_out->reserve(N);
            for (size_t i = 0; i < N; ++i)
            {
                if (normals[i].normal[direction] > threshold)
                {
                    cloud_in->push_back(src_cloud->points[i]);
                }
                else
                {
                    cloud_out->push_back(src_cloud->points[i]);
                }
            }
        }
        else
        {
            for (size_t i = 0; i < N; ++i)
            {
                if (normals[i].normal[direction] > threshold)
                {
                    cloud_in->push_back(src_cloud->points[i]);
                }
            }
        }
    }
    else if (cloud_out != nullptr)
    {
        cloud_out->clear();
        cloud_out->reserve(N);
        for (size_t i = 0; i < N; ++i)
        {
            if (normals[i].normal[direction] <= threshold)
            {
                cloud_out->push_back(src_cloud->points[i]);
            }
        }
    }

    return true;
}
