#pragma once
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
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
        mWorker = std::thread(&PoseEstimation::workerLoop, this);
        RCLCPP_INFO(get_logger(), "The node has been activated.");
    }

    ~PoseEstimation() override
    {
        //
        {
            std::unique_lock<std::mutex> lock(mBufferMutex);
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
    /* Received Data Buffer */
    bool mHasGoal{false}, mIsShutdown{false};
    std::mutex mBufferMutex;
    std::condition_variable mTriggerEvent;
    std::thread mWorker;
    std::queue<sensor_msgs::msg::PointCloud2> mCloudBuffer;

    Eigen::Vector3f mGoal; // The position of the goal
    geometry_msgs::msg::PoseStamped mGoalMsg;
    const Eigen::Vector3f mLoadDimensions{1.2f, 1.0f, 1.5f};

    /** Subscribers **/
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr mCloudSub;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr mGoalSub;

    /** Publishers **/
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr mTargetPosePub;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr mVisualizationPub;

    /**/
    std::shared_ptr<tf2_ros::TransformListener> mTfListener;
    std::unique_ptr<tf2_ros::Buffer> mTfBuffer;

    void initSubscriptions();

    void initPublishers();

    void cloudHandler(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& cloud_msg) const;

    void pushInBuffer(const sensor_msgs::msg::PointCloud2& msg)
    {
        std::lock_guard<std::mutex> lock(mBufferMutex);
        while (!mCloudBuffer.empty())
        {
            mCloudBuffer.pop();
        }
        mCloudBuffer.push(msg);
    }

    void workerLoop();
};
