#pragma once
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <opencv2/opencv.hpp>
#include "../types/common.hpp"
#include "../tools/OrthographicProjector.hpp"
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <std_msgs/msg/string.hpp>

class CloudBuild final : public rclcpp::Node
{
    static constexpr float cx = 960.0f;
    static constexpr float cy = 600.0f;
    static constexpr float fx = 1920.0f;
    static constexpr float fy = fx;
    static constexpr float fx_inv = 1.0 / fx;
    static constexpr float fy_inv = 1.0 / fy;

    static constexpr float depth_threshold = 8.0f;

    const std::string global_frame_id = "LOLA";

    std::unordered_map<std::string, int> mSemanticLabels;

public:
    explicit CloudBuild(const std::string& name, const rclcpp::NodeOptions& options) : 
        rclcpp::Node(name, options),
        mT_fork2camera(Eigen::Isometry3f::Identity())
    {
        initSubscritions();
        initPublishers();

        // mLeftCameraExtrinsics.rotate(Eigen::AngleAxisf(M_PIf / 18.0f, Eigen::Vector3f::UnitZ()));
        // mLeftCameraExtrinsics.pretranslate(Eigen::Vector3f(-0.4f, 0.6f, 1.0f));

        // mRightCameraExtrinsics.rotate(Eigen::AngleAxisf(-M_PIf / 18.0f, Eigen::Vector3f::UnitZ()));
        // mRightCameraExtrinsics.pretranslate(Eigen::Vector3f(-0.4f, -0.6f, 1.0f));
        mTfBuffer = std::make_unique<tf2_ros::Buffer>(this->get_clock());
        mTfListener = std::make_shared<tf2_ros::TransformListener>(*mTfBuffer);

        mT_fork2camera.rotate(Eigen::AngleAxisf(M_PIf, Eigen::Vector3f::UnitZ()));
        mT_fork2camera.pretranslate(Eigen::Vector3f(-0.1f, 0.0f, -0.1f));


        // Force the node to use simulation time
        this->set_parameter(rclcpp::Parameter("use_sim_time", true));
        // Now, this->now() will return Isaac Sim's time
        RCLCPP_INFO(this->get_logger(), "Current Sim Time: %f", this->now().seconds());
        RCLCPP_INFO(get_logger(), "The node has been activated.");
    }

    ~CloudBuild() override
    {
        RCLCPP_INFO(get_logger(), "The node has been shutdown.");
    }

private:
    std::shared_ptr<tf2_ros::TransformListener> mTfListener;
    std::unique_ptr<tf2_ros::Buffer> mTfBuffer;
    // Eigen::Isometry3f mLeftCameraExtrinsics, mRightCameraExtrinsics;
    Eigen::Isometry3f mT_fork2camera;

    /*** Synchronized Subsribers ***/
    using ImgMsg = sensor_msgs::msg::Image;
    using StrMsg = std_msgs::msg::String;
    using SyncPolicy = message_filters::sync_policies::ApproximateTime<ImgMsg, ImgMsg>;
    message_filters::Subscriber<ImgMsg> mDepthSub, mSemanticSub;

    std::unique_ptr<message_filters::Synchronizer<SyncPolicy>> mSynchronizer;

    /* Subsription */
    rclcpp::Subscription<StrMsg>::SharedPtr mSemanticLabelsSub;

    /* Publishers */
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr mSemanticCloudPub, mColoredCloudPub;

    void initSubscritions();

    void initPublishers();

    void imgsHandler(const ImgMsg::ConstSharedPtr& depth_msg, const ImgMsg::ConstSharedPtr& semantics_msg) const;

    void semanticLabelsHandler(const StrMsg::ConstSharedPtr& semantic_labels_msg);
};
