#pragma once
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include "../types/common.hpp"
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>

class PoseEstimation final : public rclcpp::Node
{
    const std::string global_frame_id = "map";

    static geometry_msgs::msg::Quaternion toQuaternionMsg(const double yaw)
    {
        tf2::Quaternion tf2_quat;
        tf2_quat.setRPY(0.0, 0.0, yaw);
        geometry_msgs::msg::Quaternion geom_quat;
        tf2::convert(tf2_quat, geom_quat);
        return geom_quat;
    }

    struct BufferElement
    {
        sensor_msgs::msg::PointCloud2 cloud;
        /* Control input */
        float v, delta;
    };

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
        if (mLoopCount > 0)
        {
            RCLCPP_INFO(get_logger(), "The average latency of perception is %f ms", mTotalElapseTime / static_cast<double>(mLoopCount));
        }
        RCLCPP_INFO(get_logger(), "The node has been shutdown.");
    }

private:
    /* Received Data Buffer */
    bool mHasGoal{false}, mIsShutdown{false}, mUkfInit{false};
    std::mutex mBufferMutex;
    std::condition_variable mTriggerEvent;
    std::thread mWorker;
    std::queue<BufferElement> mDataBuffer;

    double mTotalElapseTime{};
    size_t mLoopCount{};

    // Eigen::Vector3f mGoal; // The position of the goal
    geometry_msgs::msg::PoseStamped mGoalMsg;
    const Eigen::Vector3f mLoadDimensions{1.2f, 1.0f, 1.5f};

    /** Subscribers **/
    // rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr mCloudSub;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr mGoalSub;

    /*** Synchronized Subsribers ***/
    using CloudMsg = sensor_msgs::msg::PointCloud2;
    using JointStatesMsg = sensor_msgs::msg::JointState;
    using SyncPolicy = message_filters::sync_policies::ApproximateTime<CloudMsg, JointStatesMsg>;
    message_filters::Subscriber<CloudMsg> mCloudSub;
    message_filters::Subscriber<JointStatesMsg> mJointStateSub;
    std::unique_ptr<message_filters::Synchronizer<SyncPolicy>> mSynchronizer;

    /** Publishers **/
    // rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr mTargetPosePub, mLoadPosePub;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr mSlotPosePub;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr mVisualizationPub;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr mRoiCloudPub;

    /**/
    std::shared_ptr<tf2_ros::TransformListener> mTfListener;
    std::unique_ptr<tf2_ros::Buffer> mTfBuffer;

    void initSubscriptions();

    void initPublishers();

    void dataHandler(const CloudMsg::ConstSharedPtr& cloud_msg,
                     const JointStatesMsg::ConstSharedPtr& joint_state_msg)
    {
        std::lock_guard<std::mutex> lock(mBufferMutex);
        while (!mDataBuffer.empty())
        {
            mDataBuffer.pop();
        }
        const auto v = static_cast<float>(joint_state_msg->velocity[3]);
        const auto delta = static_cast<float>(joint_state_msg->position[2]);
        BufferElement data{*cloud_msg, v, delta};
        mDataBuffer.push(std::move(data));

        this->mTriggerEvent.notify_one();
    }

    visualization_msgs::msg::Marker getCubeMarker(const char* frame_id, const char* ns, const int id,
                                                  const double cube_size_x, const double cube_size_y, const double cube_size_z,
                                                  const float color_r, const float color_g, const float color_b, const float color_a,
                                                  const Eigen::Vector3f& pose) const
    {
        visualization_msgs::msg::Marker marker;
        marker.header.frame_id = frame_id;
        marker.header.stamp = this->now();
        marker.ns = ns;
        marker.id = id;
        marker.type = visualization_msgs::msg::Marker::CUBE;
        marker.action = visualization_msgs::msg::Marker::ADD;
        marker.scale.x = cube_size_x;
        marker.scale.y = cube_size_y;
        marker.scale.z = cube_size_z;
        marker.color.r = color_r;
        marker.color.g = color_g;
        marker.color.b = color_b;
        marker.color.a = color_a;
        Eigen::Isometry2f T_1(Eigen::Isometry2f::Identity()), T_2(Eigen::Isometry2f::Identity());
        T_1.rotate(pose[2]);
        T_1.pretranslate(Eigen::Vector2f(pose.x(), pose.y()));
        T_2.translate(Eigen::Vector2f(-0.5f * cube_size_x, 0.0f));
        Eigen::Isometry2f T_3 = T_1 * T_2;

        marker.pose.position.x = T_3.translation()[0];
        marker.pose.position.y = T_3.translation()[1];
        marker.pose.orientation = toQuaternionMsg(pose[2]);

        return marker;
    }

    void workerLoop();

    /* Sub detection modules */
    bool estimateSlotPose(const RawCloud::Ptr& cloud, Eigen::Vector3f& slot_pose) const;
};
