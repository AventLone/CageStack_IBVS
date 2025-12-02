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


class CloudPublisher final : public rclcpp::Node
{
    static constexpr float fx = 251.66666f;
    static constexpr float fy = fx;
    static constexpr float fx_inv = 1.0 / fx;
    static constexpr float fy_inv = 1.0 / fy;
    static constexpr float cx = 480.0f;
    static constexpr float cy = 393.0f;

    static constexpr float depth_threshold = 10.0f;

    const std::string global_frame_id = "map";

public:
    explicit CloudPublisher(const std::string& name = "cloud_publisher") : rclcpp::Node(name),
                                                                           mForkCameraExtrinsics(Eigen::Isometry3f::Identity()),
                                                                           mLeftCameraExtrinsics(Eigen::Isometry3f::Identity()),
                                                                           mRightCameraExtrinsics(Eigen::Isometry3f::Identity())
    {
        mForkSemanticsSub.subscribe(this, "fork_semantics");
        mLeftSemanticSub.subscribe(this, "left_semantics");
        mRightSemanticSub.subscribe(this, "right_semantics");
        mForkDepthSub.subscribe(this, "fork_depth");
        mLeftDepthSub.subscribe(this, "left_depth");
        mRightDepthSub.subscribe(this, "right_depth");

        mSynchronizer = std::make_unique<message_filters::Synchronizer<SyncPolicy>>(SyncPolicy(2),
                                                                                    mForkSemanticsSub, mLeftSemanticSub,
                                                                                    mRightSemanticSub,
                                                                                    mForkDepthSub, mLeftDepthSub, mRightDepthSub);
        // 设置更小的时间容差（单位：秒）
        mSynchronizer->setMaxIntervalDuration(rclcpp::Duration(0, 1000000)); // 10ms 容差
        mSynchronizer->registerCallback(std::bind(&CloudPublisher::imgsHandler, this,
                                                  std::placeholders::_1, std::placeholders::_2, std::placeholders::_3,
                                                  std::placeholders::_4, std::placeholders::_5, std::placeholders::_6));

        mSemanticCloudPub = create_publisher<sensor_msgs::msg::PointCloud2>("semantic_cloud", rclcpp::SensorDataQoS().best_effort());
        mColoredCloudPub = create_publisher<sensor_msgs::msg::PointCloud2>("colored_cloud", rclcpp::SensorDataQoS().best_effort());

        // mForkCameraExtrinsics.rotate(Eigen::AngleAxisf(M_PIf * 5.0f / 18.0f, Eigen::Vector3f::UnitY()));
        // mForkCameraExtrinsics.pretranslate(Eigen::Vector3f(0.5f, 0.0f, 0.7f));
        mForkCameraExtrinsics.translate(Eigen::Vector3f(0.3f, 0.0, 1.5f));

        mLeftCameraExtrinsics.rotate(Eigen::AngleAxisf(M_PIf / 18.0f, Eigen::Vector3f::UnitZ()));
        mLeftCameraExtrinsics.pretranslate(Eigen::Vector3f(-0.4f, 0.6f, 1.0f));

        mRightCameraExtrinsics.rotate(Eigen::AngleAxisf(-M_PIf / 18.0f, Eigen::Vector3f::UnitZ()));
        mRightCameraExtrinsics.pretranslate(Eigen::Vector3f(-0.4f, -0.6f, 1.0f));

        RCLCPP_INFO(get_logger(), "The node has been activated.");
    }

    ~CloudPublisher() override
    {
        RCLCPP_INFO(get_logger(), "The node has been shutdown.");
    }

private:
    Eigen::Isometry3f mForkCameraExtrinsics, mLeftCameraExtrinsics, mRightCameraExtrinsics;

    /*** Synchronized Subsribers ***/
    using ImgMsg = sensor_msgs::msg::Image;
    using SyncPolicy = message_filters::sync_policies::ApproximateTime<ImgMsg, ImgMsg, ImgMsg, ImgMsg, ImgMsg, ImgMsg>;
    message_filters::Subscriber<sensor_msgs::msg::Image> mForkSemanticsSub,
            mLeftSemanticSub,
            mRightSemanticSub,
            mForkDepthSub,
            mLeftDepthSub,
            mRightDepthSub;
    std::unique_ptr<message_filters::Synchronizer<SyncPolicy>> mSynchronizer;

    /* Publishers */
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr mSemanticCloudPub, mColoredCloudPub;

    void imgsHandler(const ImgMsg::ConstSharedPtr& fork_semantics_msg,
                     const ImgMsg::ConstSharedPtr& left_semantics_msg,
                     const ImgMsg::ConstSharedPtr& right_semantics_msg,
                     const ImgMsg::ConstSharedPtr& fork_depth_msg,
                     const ImgMsg::ConstSharedPtr& left_depth_msg,
                     const ImgMsg::ConstSharedPtr& right_depth_msg) const;
};
