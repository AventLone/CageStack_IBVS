#pragma once
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include "../types/common.hpp"

class PoseEstimation final : public rclcpp::Node
{
    const std::string global_frame_id = "map";

public:
    explicit PoseEstimation(const std::string& name, const rclcpp::NodeOptions& options) : rclcpp::Node(name, options)
    {
        initSubscriptions();
        initPublishers();
        this->set_parameter(rclcpp::Parameter("use_sim_time", true));
        mTfBuffer = std::make_unique<tf2_ros::Buffer>(this->get_clock());
        mTfListener = std::make_shared<tf2_ros::TransformListener>(*mTfBuffer);
        RCLCPP_INFO(get_logger(), "The node has been activated.");
    }

    ~PoseEstimation() override
    {
        RCLCPP_INFO(get_logger(), "The node has been shutdown.");
    }

private:
    /* Flag */
    bool mHasGoal{false};
    Eigen::Vector2f mGoal; // The position of the goal
    Eigen::Vector3f mGoodsDimensions;

    /** Subscribers **/
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr mCloudSub;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr mGoalSub;

    /** Publishers **/
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr mTargetPosePub;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr mTargetBBoxPub;

    /**/
    std::shared_ptr<tf2_ros::TransformListener> mTfListener;
    std::unique_ptr<tf2_ros::Buffer> mTfBuffer;

    void initSubscriptions();

    void initPublishers();

    void cloudHandler(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& cloud_msg) const;
};
