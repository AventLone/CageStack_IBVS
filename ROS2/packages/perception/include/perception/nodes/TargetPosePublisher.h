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
    explicit TargetPosePublisher(const std::string& name, const rclcpp::NodeOptions& options) : rclcpp::Node(name, options)
    {
        this->declare_parameter("SensorTopic.SemanticCloud", "semantic_cloud");
        this->declare_parameter("TargetTopic.BBox", "target_bbox");
        this->declare_parameter("TargetTopic.Pose", "target_pose");

        const std::string semantic_cloud_topic = this->get_parameter("SensorTopic.SemanticCloud").as_string();
        const std::string target_bbox_topic = this->get_parameter("TargetTopic.BBox").as_string();
        const std::string target_pose_topic = this->get_parameter("TargetTopic.Pose").as_string();

        mCloudSub = create_subscription<sensor_msgs::msg::PointCloud2>(semantic_cloud_topic, rclcpp::SensorDataQoS().best_effort(),
                                                                       std::bind(&TargetPosePublisher::cloudHandler, this,
                                                                                 std::placeholders::_1));

        mTargetBBoxPub = create_publisher<visualization_msgs::msg::Marker>(target_bbox_topic, rclcpp::SensorDataQoS().best_effort());
        mTargetPosePub = create_publisher<geometry_msgs::msg::PoseStamped>(target_pose_topic, rclcpp::SensorDataQoS().reliable());

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
};
