#include "perception/nodes/TargetPosePublisher.h"
#include "perception/nodes/CloudPublisher.h"
#include <cv_bridge/cv_bridge.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <Eigen/Geometry>
#include <pcl_conversions/pcl_conversions.h>
#include "perception/tools/feature_detect_3d.hpp"
#include "perception/tools/filter.h"


void TargetPosePublisher::cloudHandler(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& cloud_msg) const
{
    // SemanticCloud semantic_cloud;
    // pcl::fromROSMsg(*cloud_msg, semantic_cloud);

    RawCloud cage_posts_cloud;
    pcl::fromROSMsg(*cloud_msg, cage_posts_cloud);

    // RawCloud cage_posts_cloud;
    // getCloud(semantic_cloud, 18, cage_posts_cloud);
    //
    // pcl::io::savePCDFile("cage_post.pcd", cage_posts_cloud);

    Eigen::Vector3f center = computeCenter(cage_posts_cloud);
    // pcl::compute3DCentroid(cage_posts_cloud, center);
    const float angle = computeAngleByPCA(cage_posts_cloud);

    tf2::Quaternion tf2_quat;
    tf2_quat.setRPY(0.0, 0.0, angle);
    geometry_msgs::msg::Quaternion geom_quat;
    tf2::convert(tf2_quat, geom_quat);

    geometry_msgs::msg::PoseStamped target_pose;
    target_pose.header.frame_id = global_frame_id;
    target_pose.header.stamp = this->now();
    target_pose.pose.position.x = center[0] - 1.39;
    target_pose.pose.position.y = center[1];
    target_pose.pose.position.z = center[2];
    target_pose.pose.orientation = geom_quat;
    mTargetPosePub->publish(target_pose);

    const Eigen::Vector3f target_size = getCloudSize(cage_posts_cloud);

    Eigen::Isometry2f T_1(Eigen::Isometry2f::Identity()), T_2(Eigen::Isometry2f::Identity());
    T_1.rotate(angle);
    T_1.pretranslate(center.head<2>());
    T_2.translate(Eigen::Vector2f(0.5f * target_size[1], 0.0f));
    Eigen::Isometry2f T_3 = T_1 * T_2;
    const Eigen::Vector2f mark_position = T_3.translation();

    visualization_msgs::msg::Marker target_marker_msg;
    target_marker_msg.header.frame_id = global_frame_id; // 或者 base_link/odom
    target_marker_msg.header.stamp = this->now();
    target_marker_msg.ns = "bbox";
    target_marker_msg.id = 0;
    target_marker_msg.type = visualization_msgs::msg::Marker::CUBE;
    target_marker_msg.action = visualization_msgs::msg::Marker::ADD;

    // Cube 尺寸（米）
    target_marker_msg.scale.x = target_size[1];
    target_marker_msg.scale.y = target_size[1];
    target_marker_msg.scale.z = target_size[2];

    // 半透明颜色 (r,g,b,a)
    target_marker_msg.color.r = 0.0f;
    target_marker_msg.color.g = 1.0f;
    target_marker_msg.color.b = 0.0f;
    target_marker_msg.color.a = 0.8f; // alpha<1 表示半透明

    target_marker_msg.pose.position.x = mark_position[0];
    target_marker_msg.pose.position.y = mark_position[1];
    target_marker_msg.pose.position.z = center[2];
    target_marker_msg.pose.orientation = geom_quat;

    mTargetBBoxPub->publish(target_marker_msg);
}
