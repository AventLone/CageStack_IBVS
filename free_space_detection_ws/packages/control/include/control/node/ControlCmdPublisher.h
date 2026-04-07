#pragma once
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <geometry_msgs/msg/pose2_d.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <nav_msgs/msg/path.hpp>
#include "../kinematics.hpp"


class ControlCmdPublisher final : public rclcpp::Node
{
public:
    explicit ControlCmdPublisher(const std::string& name = "controller") : rclcpp::Node(name)
    {
        initSolver();
        initSubscriptions();
        initPublishers();

        mWorker = std::thread(&ControlCmdPublisher::cmdPubLoop, this);
        RCLCPP_INFO(get_logger(), "The node has been activated.");
        RCLCPP_INFO_STREAM(get_logger(), "\n" << mControllerParams);
    }

    ~ControlCmdPublisher() override
    {
        //
        {
            std::unique_lock<std::mutex> lock(mGoalMutex);
            mIsShutdown = true;
        }
        mTriggerEvent.notify_one();
        if (mWorker.joinable())
        {
            mWorker.join();
        }
        RCLCPP_INFO(get_logger(), "The node has been shutdown.");
    }

private:
    bool mIsShutdown{false};
    std::queue<std::vector<double>> mGoalBuffer;
    std::thread mWorker;
    std::mutex mGoalMutex;
    std::condition_variable mTriggerEvent;

    BicycleController::Ptr mController;
    nmpc::Params mControllerParams{};

    rclcpp::Subscription<geometry_msgs::msg::Pose2D>::SharedPtr mGoalPoseSub;

    rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr mCmdPub;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr mPathPub;

    void initSolver();

    void initSubscriptions();

    void initPublishers();

    void cmdPubLoop();
};
