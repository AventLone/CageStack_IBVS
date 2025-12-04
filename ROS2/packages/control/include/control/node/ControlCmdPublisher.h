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
    explicit ControlCmdPublisher(const std::string& name = "ControlCmdPublisher") : rclcpp::Node(name)
    {
        initSolver();

        mGoalPoseSub = this->create_subscription<geometry_msgs::msg::PoseStamped>
        ("target_pose", 10, [this](const geometry_msgs::msg::PoseStamped::ConstSharedPtr& pose_msg) -> void
             {
                 std::vector<double> goal(3, 0.0);
                 tf2::Quaternion tf2_quat;
                 tf2::convert(pose_msg->pose.orientation, tf2_quat);
                 goal[0] = pose_msg->pose.position.x;
                 goal[1] = pose_msg->pose.position.y;

                 const tf2::Matrix3x3 mat(tf2_quat);
                 double roll, pitch, yaw;
                 mat.getRPY(roll, pitch, yaw);
                 goal[2] = yaw;
                 mGoalBuffer.push(std::move(goal));
                 if (mGoalBuffer.size() > 1)
                 {
                     mGoalBuffer.pop();
                 }
             });

        mCmdPubTimer = this->create_wall_timer(std::chrono::milliseconds(static_cast<int64_t>(mControllerParams.dt * 1000.0)),
                                               std::bind(&ControlCmdPublisher::cmdPubLoop, this));

        RCLCPP_INFO(get_logger(), "The node has been activated.");
        // RCLCPP_INFO_STREAM(get_logger(), mControllerParams);
        std::cout << mControllerParams << std::endl;
    }

private:
    std::queue<std::vector<double>> mGoalBuffer;

    BicycleController::Ptr mController;
    nmpc::Params mControllerParams{};

    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr mGoalPoseSub;

    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr mCmdPub;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr mPathPub;

    rclcpp::TimerBase::SharedPtr mCmdPubTimer;

    void initSolver()
    {
        this->declare_parameter("NMPC.Horizon", 20);
        this->declare_parameter("NMPC.Dt", 0.1);

        this->declare_parameter("NMPC.InputLen", 2);
        this->declare_parameter("NMPC.StateLen", 4);
        this->declare_parameter("NMPC.OutputLen", 4);

        this->declare_parameter("NMPC.MaxAcc", 3.0);
        this->declare_parameter("NMPC.MaxSpeed", 1.2);

        this->declare_parameter("NMPC.MaxSteerSpeed", 0.5);
        this->declare_parameter("NMPC.MaxSteerAngle", 1.14);

        this->declare_parameter("NMPC.WheelBase", 1.37);
        this->declare_parameter("NMPC.WheelRadius", 0.11);

        this->declare_parameter("NMPC.WeightQ", std::vector<double>{360.0, 660.0, 370.0, 60.0});
        this->declare_parameter("NMPC.WeightF", std::vector<double>{4000.0, 4000.0, 4000.0});
        this->declare_parameter("NMPC.WeightR", std::vector<double>{0.01, 0.001});

        this->get_parameter("NMPC.Horizon", mControllerParams.horizon);
        this->get_parameter("NMPC.Dt", mControllerParams.dt);

        this->get_parameter("NMPC.InputLen", mControllerParams.input_len);
        this->get_parameter("NMPC.StateLen", mControllerParams.state_len);
        this->get_parameter("NMPC.OutputLen", mControllerParams.output_len);

        this->get_parameter("NMPC.MaxAcc", mControllerParams.max_acc);
        this->get_parameter("NMPC.MaxSpeed", mControllerParams.max_speed);

        this->get_parameter("NMPC.MaxSteerSpeed", mControllerParams.max_steer_speed);
        this->get_parameter("NMPC.MaxSteerAngle", mControllerParams.max_steer_angle);

        this->get_parameter("NMPC.WheelBase", mControllerParams.wheel_base);
        this->get_parameter("NMPC.WheelRadius", mControllerParams.wheel_radius);

        this->get_parameter("NMPC.WeightQ", mControllerParams.weight_Q);
        this->get_parameter("NMPC.WeightF", mControllerParams.weight_F);
        this->get_parameter("NMPC.WeightR", mControllerParams.weight_R);

        mController = std::make_unique<BicycleController>(mControllerParams);
    }

    void cmdPubLoop();
};
