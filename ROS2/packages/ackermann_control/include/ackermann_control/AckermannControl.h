#pragma once
#include <rclcpp/rclcpp.hpp>
#include <tf2_msgs/msg/tf_message.hpp>
#include <ackermann_msgs/msg/ackermann_drive_stamped.hpp>
#include <nav_msgs/msg/path.hpp>
// #include <nav_msgs/msg/goals.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <sensor_msgs/msg/joint_state.hpp>

#include "ackermann_control/NMPC.h"

class AckermannControl final : public rclcpp::Node
{
public:
    explicit AckermannControl(const std::string& name = "ackermann_control");

    ~AckermannControl() override
    {
        if (mMpcThread.joinable())
        {
            mMpcThread.join();
        }
        RCLCPP_INFO(get_logger(), "The node has been shutdown.");
    }

private:
    double mDt{};
    static constexpr int CONTROL_HORIZON{2};

    /* Self-State */
    double mDriveWheelAngularVelocity{};
    double mSteerAngle{};

    NMPC mNMPC;
    std::vector<double> mGoal;
    std::queue<ackermann_msgs::msg::AckermannDriveStamped> mAckerDriveMsgs;
    nav_msgs::msg::Path mPlanningPath;

    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr mVehicleStateSubscriber;
    rclcpp::Subscription<tf2_msgs::msg::TFMessage>::SharedPtr mGoalSubscriber;
    rclcpp::Publisher<ackermann_msgs::msg::AckermannDriveStamped>::SharedPtr mAckerDrivePub;

    /* Visualization */
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr mGoalDisplayPub;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr mPathPub;

    rclcpp::TimerBase::SharedPtr mDriveMsgTimer;

    std::thread mMpcThread;
    std::mutex mStateMutex, mGoalMutex, mDriveMsgsMutex, mStateSequenceMutex;

    void goalMsgHandler(const tf2_msgs::msg::TFMessage::ConstSharedPtr& msg);

    void addGoal(std::vector<double>&& goal)
    {
        std::lock_guard<std::mutex> lock(mGoalMutex);
        mGoal = std::move(goal);
    }

    void mpcLoop();
};
