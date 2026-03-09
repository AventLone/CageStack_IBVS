#include "perception/nodes/PoseEstimation.h"
#include "perception/nodes/CloudPublisher.h"
#include <cv_bridge/cv_bridge.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <Eigen/Geometry>
#include <pcl_conversions/pcl_conversions.h>
#include "perception/tools/feature_detect_3d.hpp"
#include "perception/tools/filter_3d.h"
#include <tf2_eigen/tf2_eigen.hpp> // ROS 2 header
#include "perception/tools/filter_3d.h"

void PoseEstimation::initSubscriptions()
{
    const std::string param_name = "TopicName.Perception.SemanticCloud";
    this->declare_parameter(param_name, "/perception/semantic_cloud");
    const std::string semantic_cloud_topic = this->get_parameter(param_name).as_string();
    mCloudSub = create_subscription<sensor_msgs::msg::PointCloud2>(semantic_cloud_topic, rclcpp::SensorDataQoS(),
                                                                   [this](const sensor_msgs::msg::PointCloud2::ConstSharedPtr& msg)
                                                                       {
                                                                           this->pushInBuffer(*msg);
                                                                           this->mTriggerEvent.notify_one();
                                                                       });
    mGoalSub = create_subscription<geometry_msgs::msg::PoseStamped>(
        "/goal_pose", rclcpp::ServicesQoS(), [this](const geometry_msgs::msg::PoseStamped::ConstSharedPtr& goal_msg) -> void
            {
                RCLCPP_INFO(get_logger(), "Received a goal...");
                this->mGoalMsg = *goal_msg;
                //
                {
                    std::lock_guard<std::mutex> lock(mBufferMutex);
                    this->mHasGoal = true;
                }
                this->mTriggerEvent.notify_one();
            });
}

void PoseEstimation::initPublishers()
{
    const std::string params_prefix = "TopicName.Perception.Target";
    this->declare_parameters<std::string>(params_prefix, {
                                              {"Pose", "/perception/target/pose"},
                                              {"BBox", "/perception/visualization/cubes"}
                                          });
    std::map<std::string, std::string> target_topics;
    if (!this->get_parameters<std::string>(params_prefix, target_topics))
    {
        RCLCPP_ERROR(this->get_logger(), "Failed to get parameters, target topic names!");
    }
    mVisualizationPub = this->create_publisher<visualization_msgs::msg::MarkerArray>(target_topics["BBox"], rclcpp::SensorDataQoS());
    mTargetPosePub = this->create_publisher<geometry_msgs::msg::PoseStamped>(target_topics["Pose"], rclcpp::ServicesQoS());
}

void PoseEstimation::workerLoop()
{
    static constexpr std::string_view ns = "visualization";
    static constexpr std::string_view frame_id = "map";
    static constexpr float load_size_x = 1.2f;
    static constexpr float load_size_y = 1.0f;
    static constexpr float load_size_z = 1.5f;
    while (rclcpp::ok())
    {
        sensor_msgs::msg::PointCloud2 cloud_msg;
        visualization_msgs::msg::MarkerArray visualization_msg;
        //
        {
            std::unique_lock<std::mutex> lock(mBufferMutex);
            mTriggerEvent.wait(lock, [this]() -> bool { return (mHasGoal && !mCloudBuffer.empty()) || mIsShutdown; });
            if (mIsShutdown)
            {
                break;
            }
            cloud_msg = mCloudBuffer.front();
            mCloudBuffer.pop();
        }

        SemanticCloud::Ptr cloud = std::make_shared<SemanticCloud>();
        pcl::fromROSMsg(cloud_msg, *cloud);

        /* Stage 1. Detect the pose of the load on the forks */
        /* Step 1. Get the pose of the forks */
        Eigen::Isometry3f T_body2fork;
        try
        {
            // This returns the pose of 'fork' in 'body' coordinates
            const geometry_msgs::msg::TransformStamped tf_body2fork = mTfBuffer->lookupTransform("LOLA", "fork", tf2::TimePointZero);
            T_body2fork = tf2::transformToEigen(tf_body2fork).cast<float>();
        }
        catch (const tf2::TransformException& ex)
        {
            RCLCPP_ERROR(this->get_logger(), "Could not transform fork to body: %s", ex.what());
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }

        ROI fork_roi{-1.5f, 0.1f, -0.66f, 0.66f, -0.3f, 1.5f};
        ROI slot_space_roi{-1.0f, 1.0f, -0.9f, 0.9f, 0.0f, 1.5f};

        /* Visualize fork_roi */
        visualization_msgs::msg::Marker fork_roi_msg;
        fork_roi_msg.header.frame_id = "LOLA";
        fork_roi_msg.header.stamp = this->now();
        fork_roi_msg.ns = ns;
        fork_roi_msg.id = 0;
        fork_roi_msg.type = visualization_msgs::msg::Marker::CUBE;
        fork_roi_msg.action = visualization_msgs::msg::Marker::ADD;
        fork_roi_msg.scale.x = static_cast<double>(fork_roi.max_x - fork_roi.min_x);
        fork_roi_msg.scale.y = static_cast<double>(fork_roi.max_y - fork_roi.min_y);
        fork_roi_msg.scale.z = static_cast<double>(fork_roi.max_z - fork_roi.min_z);
        fork_roi_msg.color.r = 0.8;
        fork_roi_msg.color.g = 0.8;
        fork_roi_msg.color.b = 1.0;
        fork_roi_msg.color.a = 0.2;
        fork_roi_msg.pose.position.x = T_body2fork.translation()[0] - 0.5 * fork_roi_msg.scale.x;
        fork_roi_msg.pose.position.y = T_body2fork.translation()[1];
        fork_roi_msg.pose.position.z = T_body2fork.translation()[2] + 0.5 * fork_roi_msg.scale.z - 0.3;
        visualization_msg.markers.push_back(fork_roi_msg);

        /* Visualize slot_space_roi */
        visualization_msgs::msg::Marker slot_roi_msg;
        slot_roi_msg.header.frame_id = "LOLA";
        slot_roi_msg.header.stamp = this->now();
        slot_roi_msg.ns = ns;
        slot_roi_msg.id = 1;
        slot_roi_msg.type = visualization_msgs::msg::Marker::CUBE;
        slot_roi_msg.action = visualization_msgs::msg::Marker::ADD;
        slot_roi_msg.scale.x = static_cast<double>(slot_space_roi.max_x - slot_space_roi.min_x);
        slot_roi_msg.scale.y = static_cast<double>(slot_space_roi.max_y - slot_space_roi.min_y);
        slot_roi_msg.scale.z = static_cast<double>(slot_space_roi.max_z - slot_space_roi.min_z);
        slot_roi_msg.color.r = 0.8;
        slot_roi_msg.color.g = 1.0;
        slot_roi_msg.color.b = 0.8;
        slot_roi_msg.color.a = 0.2;
        slot_roi_msg.pose.position.x = mGoalMsg.pose.position.x;
        slot_roi_msg.pose.position.y = mGoalMsg.pose.position.y;
        slot_roi_msg.pose.position.z = 0.5 * slot_roi_msg.scale.z;
        visualization_msg.markers.push_back(slot_roi_msg);

        visualization_msgs::msg::Marker load_on_fork_msg;
        load_on_fork_msg.header.frame_id = "LOLA";
        load_on_fork_msg.header.stamp = this->now();
        load_on_fork_msg.ns = ns;
        load_on_fork_msg.id = 2;
        load_on_fork_msg.type = visualization_msgs::msg::Marker::CUBE;
        load_on_fork_msg.action = visualization_msgs::msg::Marker::ADD;
        load_on_fork_msg.scale.x = load_size_x;
        load_on_fork_msg.scale.y = load_size_y;
        load_on_fork_msg.scale.z = load_size_z;
        load_on_fork_msg.color.r = 0.0;
        load_on_fork_msg.color.g = 0.0;
        load_on_fork_msg.color.b = 0.6;
        load_on_fork_msg.color.a = 0.5;
        load_on_fork_msg.pose.position.x = T_body2fork.translation()[0] - 0.5 * load_size_x - 0.25;
        load_on_fork_msg.pose.position.y = T_body2fork.translation()[1];
        load_on_fork_msg.pose.position.z = T_body2fork.translation()[2] + 0.5 * load_size_z - 0.1;
        visualization_msg.markers.push_back(load_on_fork_msg);
        mVisualizationPub->publish(visualization_msg);

        SemanticCloud::Ptr cloud_on_forks = std::make_shared<SemanticCloud>();
        SemanticCloud::Ptr cloud_off_forks = std::make_shared<SemanticCloud>();
        getCloud(*cloud, cloud_on_forks, cloud_off_forks, T_body2fork.translation(), fork_roi);
        if (cloud_on_forks->size() < 10 || cloud_off_forks->size() < 10)
        {
            RCLCPP_ERROR(get_logger(), "cloud_on_forks or cloud_off_forks has too less points!");
            continue;
        }

        /* Slot pose estimate */
        SemanticCloud::Ptr slot_space_cloud = std::make_shared<SemanticCloud>();
        getCloud(*cloud_off_forks, slot_space_cloud, nullptr, mGoal, slot_space_roi);
        if (slot_space_cloud->size() < 10)
        {
            RCLCPP_ERROR(get_logger(), "slot_space_cloud has too less points!");
            continue;
        }
        SemanticCloud slot_space_cloud_without_ground = removeGround(*slot_space_cloud);
        Eigen::Isometry3d goal_pose{};
        tf2::fromMsg(mGoalMsg.pose, goal_pose);

        constexpr ROI load_roi{-0.6f, 0.6f, -0.5f, 0.5f, 0.0f, 1.5f};

        if (checkSpace(slot_space_cloud_without_ground, goal_pose.cast<float>(), load_roi))
        {
            RCLCPP_WARN(get_logger(), "Didn't detect anything!");
            continue;
        }


        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}
