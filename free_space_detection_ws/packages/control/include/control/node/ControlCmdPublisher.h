#pragma once
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/path.hpp>
#include <tf2/LinearMath/Quaternion.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include "../kinematics.hpp"


class ControlCmdPublisher final : public rclcpp::Node
{
public:
    explicit ControlCmdPublisher(const std::string& name = "controller") : rclcpp::Node(name)
    {
        initSolver();
        initSubscriptions();
        initPublishers();        

        mCmdPubTimer = this->create_wall_timer(std::chrono::milliseconds(static_cast<int64_t>(mControllerParams.dt * 1000.0)),
                                               std::bind(&ControlCmdPublisher::cmdPubLoop, this));
        RCLCPP_INFO(get_logger(), "The node has been activated.");
        RCLCPP_INFO_STREAM(get_logger(), "\n" << mControllerParams);
    }

    ~ControlCmdPublisher() override
    {
        RCLCPP_INFO(get_logger(), "The node has been shutdown.");
    }

private:
    std::queue<std::vector<double>> mGoalBuffer;

    BicycleController::Ptr mController;
    nmpc::Params mControllerParams{};

    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr mGoalPoseSub;

    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr mCmdPub;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr mPathPub;

    rclcpp::TimerBase::SharedPtr mCmdPubTimer;

    void initSolver();

    void initSubscriptions();

    void initPublishers();

    void cmdPubLoop();
};
