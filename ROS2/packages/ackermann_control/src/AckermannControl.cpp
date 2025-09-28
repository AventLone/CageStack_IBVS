#include "ackermann_control/AckermannControl.h"
#include "ackermann_control/QoS.h"
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>


AckermannControl::AckermannControl(const std::string& name) : rclcpp::Node(name)
{
    this->declare_parameter("dt", 0.08);

    this->get_parameter("dt", mDt);

    mVehicleStateSubscriber = this->create_subscription<sensor_msgs::msg::JointState>(
        "forklift/joint_states", gControlMsgQoS,
        [this](const sensor_msgs::msg::JointState::ConstSharedPtr& msg)
            {
                std::lock_guard lock(mStateMutex);
                mDriveWheelAngularVelocity = msg->velocity[0];
                mSteerAngle = msg->position[2];
            });
    mGoalSubscriber = this->create_subscription<tf2_msgs::msg::TFMessage>
            ("goal", gRobStateQoS, std::bind(&AckermannControl::goalMsgHandler, this, std::placeholders::_1));

    mAckerDrivePub = this->create_publisher<ackermann_msgs::msg::AckermannDriveStamped>
            ("ackermann_cmd", gControlMsgQoS);

    mGoalDisplayPub = this->create_publisher<visualization_msgs::msg::Marker>("goal_display", gRobStateQoS);
    mPathPub = this->create_publisher<nav_msgs::msg::Path>("state_path", gRobStateQoS);

    mDriveMsgTimer = this->create_wall_timer(std::chrono::milliseconds(1000 * static_cast<int64_t>(mDt)), [this]()
                                                 {
                                                     std::lock_guard lock(mDriveMsgsMutex);
                                                     if (!mAckerDriveMsgs.empty())
                                                     {
                                                         mAckerDrivePub->publish(mAckerDriveMsgs.front());
                                                         // RCLCPP_WARN(get_logger(), "CmdMsg, drive_velocity: %f, steer_angle: %f.",
                                                         //             mAckerDriveMsgs.front().drive.speed,
                                                         //             mAckerDriveMsgs.front().drive.steering_angle);
                                                         mAckerDriveMsgs.pop();
                                                     }
                                                     else
                                                     {
                                                         ackermann_msgs::msg::AckermannDriveStamped drive_msg;
                                                         drive_msg.drive.speed = 0.0f;
                                                         drive_msg.drive.steering_angle = 0.0f;
                                                         mAckerDrivePub->publish(drive_msg);
                                                     }
                                                 });

    mMpcThread = std::thread(&AckermannControl::mpcLoop, this);

    RCLCPP_INFO(get_logger(), "The node has activated.");
}

void AckermannControl::goalMsgHandler(const tf2_msgs::msg::TFMessage::ConstSharedPtr& msg)
{
    std::vector<double> goal(3, 0.0);
    for (const auto& tf : msg->transforms)
    {
        if (tf.header.frame_id == "body" && tf.child_frame_id == "pallet")
        {
            const geometry_msgs::msg::Transform& T = tf.transform;

            goal[0] = T.translation.x;
            goal[1] = T.translation.y;

            tf2::Quaternion q(T.rotation.x, T.rotation.y,
                              T.rotation.z, T.rotation.w);
            double roll, pitch, yaw;
            tf2::Matrix3x3(q).getRPY(roll, pitch, yaw);
            goal[2] = yaw;

            addGoal(std::move(goal));

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
            goal_marker_msg.color.a = 0.6f; // alpha<1 表示半透明

            goal_marker_msg.pose.position.x = T.translation.x;
            goal_marker_msg.pose.position.y = T.translation.y;
            goal_marker_msg.pose.position.z = T.translation.z + 0.4;
            goal_marker_msg.pose.orientation = T.rotation;

            mGoalDisplayPub->publish(goal_marker_msg);
        }
    }
}

void AckermannControl::mpcLoop()
{
    while (rclcpp::ok())
    {
        {
            std::scoped_lock lock(mGoalMutex, mStateMutex);
            if (mGoal.empty())
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                continue;
            }
            mNMPC.setGoalAndState(mGoal, {mDriveWheelAngularVelocity, mSteerAngle});
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }

        const auto [us, xs] = mNMPC.solve();

        std::queue<ackermann_msgs::msg::AckermannDriveStamped> temp_drive_msgs;

        double speed{}, steer_angle{}; {
            std::lock_guard lock(mStateMutex);
            speed = mDriveWheelAngularVelocity;
            steer_angle = mSteerAngle;
        }
        for (auto& u : us)
        {
            ackermann_msgs::msg::AckermannDriveStamped drive_msg;
            speed += mDt * u[0];
            steer_angle += mDt * u[1];
            drive_msg.drive.speed = speed;
            // drive_msg.drive.acceleration = u[0];
            drive_msg.drive.steering_angle = steer_angle;
            // drive_msg.drive.steering_angle_velocity = u[1];
            temp_drive_msgs.push(drive_msg);
        } {
            std::lock_guard lock(mDriveMsgsMutex);
            mAckerDriveMsgs = temp_drive_msgs;
        }

        nav_msgs::msg::Path state_path;
        state_path.header.frame_id = "SM_Forklift_C01_01";
        for (const auto& x : xs)
        {
            geometry_msgs::msg::PoseStamped pose;
            pose.header.frame_id = "SM_Forklift_C01_01";
            pose.header.stamp = this->now();
            // pose.pose.orientation.
            pose.pose.position.x = x[0];
            pose.pose.position.y = x[1];
            tf2::Quaternion orien;
            orien.setRPY(0.0, 0.0, x[2]);
            orien.normalize();
            pose.pose.orientation = tf2::toMsg(orien);
            state_path.poses.push_back(pose);
        }
        mPathPub->publish(state_path);

        // break;
    }
}

