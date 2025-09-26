#include "ackermann_control/AckermannControl.h"
#include "ackermann_control/QoS.h"
#include <tf2/LinearMath/Matrix3x3.h>


AckermannControl::AckermannControl(const std::string& name) : rclcpp::Node(name)
{
    mGoalSubscriber = this->create_subscription<tf2_msgs::msg::TFMessage>
            ("goal", gRobStateQoS, std::bind(&AckermannControl::goalMsgHandler, this, std::placeholders::_1));

    mGoalDisplayPub = this->create_publisher<visualization_msgs::msg::Marker>("goal_display", gRobStateQoS);
    mAckerDrivePub = this->create_publisher<ackermann_msgs::msg::AckermannDriveStamped>
            ("ackermann_cmd", gControlMsgQoS);

    RCLCPP_INFO(get_logger(), "The node has activated.");
}

void AckermannControl::goalMsgHandler(const tf2_msgs::msg::TFMessage::ConstSharedPtr& msg)
{
    std::vector<double> goal(3, 0.0);
    for (const auto& tf : msg->transforms)
    {
        if (tf.header.frame_id == "body" && tf.child_frame_id == "pallet")
        {
            const auto& T = tf.transform;

            goal[0] = T.translation.x;
            goal[1] = T.translation.y;

            tf2::Quaternion q(T.rotation.x, T.rotation.y,
                              T.rotation.z, T.rotation.w);
            double roll, pitch, yaw;
            tf2::Matrix3x3(q).getRPY(roll, pitch, yaw);
            goal[1] = yaw;

            visualization_msgs::msg::Marker goal_marker_msg;
            goal_marker_msg.header.frame_id = "SM_Forklift_C01_01"; // 或者 base_link/odom
            goal_marker_msg.header.stamp = this->now();
            goal_marker_msg.ns = "demo";
            goal_marker_msg.id = 0;
            goal_marker_msg.type = visualization_msgs::msg::Marker::CUBE;
            goal_marker_msg.action = visualization_msgs::msg::Marker::ADD;

            // Cube 尺寸（米）
            goal_marker_msg.scale.x = 1.0;
            goal_marker_msg.scale.y = 1.0;
            goal_marker_msg.scale.z = 0.8;

            // 半透明颜色 (r,g,b,a)
            goal_marker_msg.color.r = 0.0f;
            goal_marker_msg.color.g = 1.0f;
            goal_marker_msg.color.b = 0.0f;
            goal_marker_msg.color.a = 0.5f; // alpha<1 表示半透明

            goal_marker_msg.pose.position.x = T.translation.x;
            goal_marker_msg.pose.position.y = T.translation.y;
            goal_marker_msg.pose.position.z = T.translation.z + 0.4;
            goal_marker_msg.pose.orientation = T.rotation;

            mGoalDisplayPub->publish(goal_marker_msg);
        }
    }

    addGoal(std::move(goal));
}

