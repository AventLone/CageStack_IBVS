#pragma once
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
// #include <geometry_msgs/msg/pose2_d.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <nav_msgs/msg/path.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include "../kinematics.hpp"


class ControlCmdPublisher final : public rclcpp::Node
{
    struct DataElement
    {
        std::vector<double> goal;
        double steer_angle;
    };

public:
    explicit ControlCmdPublisher(const std::string& name = "controller") : rclcpp::Node(name)
    {
        this->set_parameter(rclcpp::Parameter("use_sim_time", true));
        initSolver();
        initSubscriptions();
        initPublishers();
        RCLCPP_INFO(this->get_logger(), "Current Sim Time: %f", this->now().seconds());
        mWorker = std::thread(&ControlCmdPublisher::cmdPubLoop, this);

        RCLCPP_INFO(get_logger(), "The node has been activated.");
        RCLCPP_INFO_STREAM(get_logger(), "\n" << mControllerParams);
    }

    ~ControlCmdPublisher() override
    {
        //
        {
            std::lock_guard<std::mutex> lock(mBufferMutex);
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
    std::queue<DataElement> mDataBuffer;
    std::thread mWorker;
    std::mutex mBufferMutex;
    std::condition_variable mTriggerEvent;

    BicycleController::Ptr mController;
    nmpc::Params mControllerParams{};

    // rclcpp::Subscription<geometry_msgs::msg::Pose2D>::SharedPtr mGoalPoseSub;
    // rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr mSteerAngelSub;

    /*** Synchronized Subsribers ***/
    // using CloudMsg = sensor_msgs::msg::PointCloud2;
    // using JointStatesMsg = sensor_msgs::msg::JointState;
    // using SyncPolicy = message_filters::sync_policies::ApproximateTime<CloudMsg, JointStatesMsg>;
    using SyncPolicy = message_filters::sync_policies::ApproximateTime<geometry_msgs::msg::PoseStamped,
        sensor_msgs::msg::JointState>;
    message_filters::Subscriber<geometry_msgs::msg::PoseStamped> mGoalPoseSub;
    message_filters::Subscriber<sensor_msgs::msg::JointState> mSteerAngelSub;
    std::unique_ptr<message_filters::Synchronizer<SyncPolicy>> mSynchronizer;

    rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr mCmdPub;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr mPathPub;

    void initSolver();

    void initSubscriptions();

    void initPublishers();

    void dataHandler(const geometry_msgs::msg::PoseStamped::ConstSharedPtr& pose_msg,
                     const sensor_msgs::msg::JointState::ConstSharedPtr& joint_states_msg)
    {
        std::vector<double> goal(3, 0.0);
        goal[0] = pose_msg->pose.position.x;
        goal[1] = pose_msg->pose.position.y;
        tf2::Quaternion tf_q;
        tf2::fromMsg(pose_msg->pose.orientation, tf_q); // Convert ROS msg → tf2 quaternion

        // 3. Convert quaternion to roll/pitch/yaw (Euler angles)
        double roll, pitch, yaw;
        tf2::Matrix3x3(tf_q).getRPY(roll, pitch, yaw); // Order: X(roll), Y(pitch), Z(yaw)
        goal[2] = yaw;
        //
        {
            std::lock_guard<std::mutex> lock(mBufferMutex);
            while (!mDataBuffer.empty())
            {
                mDataBuffer.pop();
            }

            mDataBuffer.push({goal, joint_states_msg->position[2]});
        }

        mTriggerEvent.notify_one();
    }

    void cmdPubLoop();
};
