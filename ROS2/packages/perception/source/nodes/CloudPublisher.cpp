#include "perception/nodes/CloudPublisher.h"
#include <cv_bridge/cv_bridge.hpp>
#include <pcl/common/transforms.h>
#include <pcl_conversions/pcl_conversions.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <Eigen/Geometry>
#include "perception/tools/filter.h"

void CloudPublisher::imgsHandler(const ImgMsg::ConstSharedPtr& fork_semantics_msg, const ImgMsg::ConstSharedPtr& left_semantics_msg,
                                 const ImgMsg::ConstSharedPtr& right_semantics_msg, const ImgMsg::ConstSharedPtr& fork_depth_msg,
                                 const ImgMsg::ConstSharedPtr& left_depth_msg,
                                 const ImgMsg::ConstSharedPtr& right_depth_msg) const
{
    const auto fork_semantics_ptr = cv_bridge::toCvShare(fork_semantics_msg, "mono8");
    const auto left_semantics_ptr = cv_bridge::toCvShare(left_semantics_msg, "mono8");
    const auto right_semantics_ptr = cv_bridge::toCvShare(right_semantics_msg, "mono8");
    const auto fork_depth_ptr = cv_bridge::toCvShare(fork_depth_msg, "mono16");
    const auto left_depth_ptr = cv_bridge::toCvShare(left_depth_msg, "mono16");
    const auto right_depth_ptr = cv_bridge::toCvShare(right_depth_msg, "mono16");

    const cv::Mat& fork_semantics = fork_semantics_ptr->image;
    const cv::Mat& left_semantics = left_semantics_ptr->image;
    const cv::Mat& right_semantics = right_semantics_ptr->image;
    const cv::Mat& fork_depth = fork_depth_ptr->image;
    const cv::Mat& left_depth = left_depth_ptr->image;
    const cv::Mat& right_depth = right_depth_ptr->image;

    SemanticCloud semantic_cloud_from_fork, semantic_cloud_from_left, semantic_cloud_from_right;
    RawCloud cage_posts_cloud;
    for (int v = 0; v < fork_depth.rows; v += 6)
    {
        const auto* target_on_fork = fork_depth.ptr<uint16_t>(v);
        const auto* target_on_left = left_depth.ptr<uint16_t>(v);
        const auto* target_on_right = right_depth.ptr<uint16_t>(v);

        const auto* labels_on_fork = fork_semantics.ptr<uint8_t>(v);
        const auto* labels_on_left = left_semantics.ptr<uint8_t>(v);
        const auto* labels_on_right = right_semantics.ptr<uint8_t>(v);

        for (int u = 0; u < fork_depth.cols; u += 3)
        {
            if (target_on_fork[u] > 0)
            {
                if (const float depth = static_cast<float>(target_on_fork[u]) * 1.0e-3f; depth < depth_threshold)
                {
                    const float X = (static_cast<float>(u) - cx) * depth * fx_inv;
                    const float Y = (static_cast<float>(v) - cy) * depth * fy_inv;
                    const uint32_t label = labels_on_fork[u];
                    semantic_cloud_from_fork.emplace_back(depth, -X, -Y, label);

                    if (label == 20)
                    {
                        cage_posts_cloud.emplace_back(depth, -X, -Y);
                    }
                }
            }

            if (target_on_left[u] > 0)
            {
                if (const float depth = static_cast<float>(target_on_left[u]) * 1.0e-3f; depth < depth_threshold)
                {
                    const float X = (static_cast<float>(u) - cx) * depth * fx_inv;
                    const float Y = (static_cast<float>(v) - cy) * depth * fy_inv;
                    const uint32_t label = labels_on_left[u];
                    semantic_cloud_from_left.emplace_back(depth, -X, -Y, label);
                }
            }

            if (target_on_right[u] > 0)
            {
                if (const float depth = static_cast<float>(target_on_right[u]) * 1.0e-3f; depth < depth_threshold)
                {
                    const float X = (static_cast<float>(u) - cx) * depth * fx_inv;
                    const float Y = (static_cast<float>(v) - cy) * depth * fy_inv;
                    const uint32_t label = labels_on_right[u];
                    semantic_cloud_from_right.emplace_back(depth, -X, -Y, label);
                }
            }
        }
    }

    SemanticCloud semantic_cloud;

    if (!semantic_cloud_from_fork.empty())
    {
        SemanticCloud cloud_in_base;
        pcl::transformPointCloud(semantic_cloud_from_fork, cloud_in_base, mForkCameraExtrinsics);
        semantic_cloud = std::move(cloud_in_base);

        pcl::transformPointCloud(cage_posts_cloud, cage_posts_cloud, mForkCameraExtrinsics);
    }

    //
    // if (!target_cloud_on_left.empty())
    // {
    //     SemanticCloud cloud_in_base;
    //     pcl::transformPointCloud(target_cloud_on_left, cloud_in_base, mLeftCameraExtrinsics);
    //     semantic_cloud += cloud_in_base;
    // }
    //
    // if (!target_cloud_on_right.empty())
    // {
    //     SemanticCloud cloud_in_base;
    //     pcl::transformPointCloud(target_cloud_on_right, cloud_in_base, mRightCameraExtrinsics);
    //     semantic_cloud += cloud_in_base;
    // }

    // if (semantic_cloud.size() > 10)
    // {
    //     ColoredCloud colored_cloud;
    //     getCloud(semantic_cloud, colored_cloud);
    //
    //     sensor_msgs::msg::PointCloud2 semantic_cloud_msg, colored_cloud_msg;
    //     pcl::toROSMsg(semantic_cloud, semantic_cloud_msg);
    //     pcl::toROSMsg(colored_cloud, colored_cloud_msg);
    //
    //     semantic_cloud_msg.header.stamp = this->now();
    //     semantic_cloud_msg.header.frame_id = "map";
    //     colored_cloud_msg.header.stamp = this->now();
    //     colored_cloud_msg.header.frame_id = "map";
    //
    //     mSemanticCloudPub->publish(semantic_cloud_msg);
    //     mColoredCloudPub->publish(colored_cloud_msg);
    // }

    if (semantic_cloud.size() > 10)
    {
        ColoredCloud semantic_cloud_with_clolor;
        getCloud(semantic_cloud, semantic_cloud_with_clolor);

        sensor_msgs::msg::PointCloud2 semantic_cloud_msg, semantic_cloud_with_color_msg;
        pcl::toROSMsg(cage_posts_cloud, semantic_cloud_msg);
        pcl::toROSMsg(semantic_cloud_with_clolor, semantic_cloud_with_color_msg);

        semantic_cloud_msg.header.stamp = this->now();
        semantic_cloud_msg.header.frame_id = global_frame_id;
        semantic_cloud_with_color_msg.header.stamp = this->now();
        semantic_cloud_with_color_msg.header.frame_id = global_frame_id;

        mSemanticCloudPub->publish(semantic_cloud_msg);
        mColoredCloudPub->publish(semantic_cloud_with_color_msg);
    }
}
