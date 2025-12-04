#pragma once
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include "../types/common.hpp"

class TargetPosePublisher final : public rclcpp::Node
{
    const std::string global_frame_id = "map";

public:
    explicit TargetPosePublisher(const std::string& name = "Target_Pose_Publisher") : rclcpp::Node(name)
    {
        mTargetBBoxPub = create_publisher<visualization_msgs::msg::Marker>("target_bbox", rclcpp::SensorDataQoS().best_effort());
        mTargetPosePub = create_publisher<geometry_msgs::msg::PoseStamped>("target_pose", rclcpp::SensorDataQoS().reliable());

        mCloudSub = create_subscription<sensor_msgs::msg::PointCloud2>("semantic_cloud", rclcpp::SensorDataQoS().best_effort(),
                                                                       std::bind(&TargetPosePublisher::cloudHandler, this,
                                                                                 std::placeholders::_1));

        RCLCPP_INFO(get_logger(), "The node has been activated.");
    }

    ~TargetPosePublisher() override
    {
        RCLCPP_INFO(get_logger(), "The node has been shutdown.");
    }

private:
    /** Subscribers **/
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr mCloudSub;

    /** Publishers **/
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr mTargetPosePub;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr mTargetBBoxPub;

    void cloudHandler(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& cloud_msg) const;

    visualization_msgs::msg::Marker createBBox3D(const std::string& frame_id, const geometry_msgs::msg::Pose& pose,
                                                 const geometry_msgs::msg::Vector3& size)
    {
        visualization_msgs::msg::Marker marker;
        marker.header.frame_id = frame_id;
        // marker.header.stamp = node->now();
        marker.ns = "bbox_3d";
        marker.id = 0;
        marker.type = visualization_msgs::msg::Marker::LINE_LIST;
        marker.action = visualization_msgs::msg::Marker::ADD;

        // Set the scale of the lines (width)
        marker.scale.x = 0.02; // 2 cm line width

        // Set color
        marker.color.r = 0.0;
        marker.color.g = 1.0;
        marker.color.b = 0.0;
        marker.color.a = 1.0;

        // Define the cube dimensions and center
        constexpr double size_x = 1.0, size_y = 1.0, size_z = 1.0;
        double x = 0.0, y = 0.0, z = 0.0;

        // Compute the 8 vertices of the cube
        geometry_msgs::msg::Point p[8];

        p[0].x = x - size_x / 2;
        p[0].y = y - size_y / 2;
        p[0].z = z - size_z / 2;
        p[1].x = x + size_x / 2;
        p[1].y = y - size_y / 2;
        p[1].z = z - size_z / 2;
        p[2].x = x + size_x / 2;
        p[2].y = y + size_y / 2;
        p[2].z = z - size_z / 2;
        p[3].x = x - size_x / 2;
        p[3].y = y + size_y / 2;
        p[3].z = z - size_z / 2;
        p[4].x = x - size_x / 2;
        p[4].y = y - size_y / 2;
        p[4].z = z + size_z / 2;
        p[5].x = x + size_x / 2;
        p[5].y = y - size_y / 2;
        p[5].z = z + size_z / 2;
        p[6].x = x + size_x / 2;
        p[6].y = y + size_y / 2;
        p[6].z = z + size_z / 2;
        p[7].x = x - size_x / 2;
        p[7].y = y + size_y / 2;
        p[7].z = z + size_z / 2;

        // Now, we define the 12 edges (each edge by two points)
        // We'll push the points in pairs for each edge.

        // Bottom face (4 edges)
        marker.points.push_back(p[0]);
        marker.points.push_back(p[1]);
        marker.points.push_back(p[1]);
        marker.points.push_back(p[2]);
        marker.points.push_back(p[2]);
        marker.points.push_back(p[3]);
        marker.points.push_back(p[3]);
        marker.points.push_back(p[0]);

        // Top face (4 edges)
        marker.points.push_back(p[4]);
        marker.points.push_back(p[5]);
        marker.points.push_back(p[5]);
        marker.points.push_back(p[6]);
        marker.points.push_back(p[6]);
        marker.points.push_back(p[7]);
        marker.points.push_back(p[7]);
        marker.points.push_back(p[4]);

        // Vertical edges (4 edges)
        marker.points.push_back(p[0]);
        marker.points.push_back(p[4]);
        marker.points.push_back(p[1]);
        marker.points.push_back(p[5]);
        marker.points.push_back(p[2]);
        marker.points.push_back(p[6]);
        marker.points.push_back(p[3]);
        marker.points.push_back(p[7]);

        return marker;
    }
};
