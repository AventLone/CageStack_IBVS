#pragma once
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/pose2_d.hpp>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include "../types/common.hpp"
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <geometry_msgs/msg/pose_array.hpp>

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
    bool mHasGoal{false}, mIsShutdown{false};
    std::mutex mBufferMutex;
    std::condition_variable mTriggerEvent;
    std::thread mWorker;
    std::queue<InstanceCloudPtr> mCloudBuffer;

    double mTotalElapseTime{};
    size_t mLoopCount{};

    // Eigen::Vector3f mGoal; // The position of the goal
    geometry_msgs::msg::PoseStamped mGoalMsg;
    const Eigen::Vector3f mLoadDimensions{1.2f, 1.0f, 1.5f};

    /** Subscribers **/
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr mCloudSub;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr mGoalSub;

    /** Publishers **/
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr mTargetPosePub, mLoadPosePub, mSlotPosePub;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr mVisualizationPub;
    rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr mPosesPub;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr mRoiCloudPub;

    /**/
    std::shared_ptr<tf2_ros::TransformListener> mTfListener;
    std::unique_ptr<tf2_ros::Buffer> mTfBuffer;

    void initSubscriptions();

    void initPublishers();

    void pushInBuffer(InstanceCloudPtr&& data)
    {
        std::lock_guard<std::mutex> lock(mBufferMutex);
        while (!mCloudBuffer.empty())
        {
            mCloudBuffer.pop();
        }
        mCloudBuffer.push(std::move(data));
    }

    static geometry_msgs::msg::Pose targetToCubePose(const Eigen::Vector3f& target_pose, const Eigen::Vector3f& cube_dimensions)
    {
        Eigen::Isometry2f T_target(Eigen::Isometry2f::Identity());
        T_target.rotate(target_pose[2]);
        T_target.pretranslate(target_pose.head<2>());

        Eigen::Isometry2f T_target2cube(Eigen::Isometry2f::Identity());
        T_target2cube.translate(Eigen::Vector2f(-0.5 * cube_dimensions[0], 0.0f));

        Eigen::Isometry2f T_cube = T_target * T_target2cube;

        geometry_msgs::msg::Pose cube_pose;
        cube_pose.position.x = T_cube.translation()[0];
        cube_pose.position.y = T_cube.translation()[1];
        cube_pose.position.z = 0.5 * cube_dimensions[2];
        cube_pose.orientation = toQuaternionMsg(target_pose[2]);

        return cube_pose;
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
    bool estimateLoadPose(const RawCloud::Ptr& cloud, Eigen::Vector3f& load_pose) const;

    static bool estimateDimensionsAndPose(const RawCloud::Ptr& input_cloud, Eigen::Vector3f& pose, Eigen::Vector3f& dimensions);
};
