#pragma once
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include "../types/common.hpp"

class PoseEstimation final : public rclcpp::Node
{
    const std::string global_frame_id = "map";

public:
    explicit PoseEstimation(const std::string& name, const rclcpp::NodeOptions& options) : rclcpp::Node(name, options)
    {
        initSubscriptions();
        initPublishers();
        RCLCPP_INFO(get_logger(), "The node has been activated.");
    }

    ~PoseEstimation() override
    {
        RCLCPP_INFO(get_logger(), "The node has been shutdown.");
    }

private:
    /* Flag */
    bool mHasGoal{false};
    Eigen::Vector2f mGoal;   // The position of the goal
    Eigen::Vector3f mGoodsDimensions;

    /** Subscribers **/
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr mCloudSub;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr mGoalSub;

    /** Publishers **/
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr mTargetPosePub;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr mTargetBBoxPub;

    void initSubscriptions();

    void initPublishers();

    void cloudHandler(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& cloud_msg) const;
};
