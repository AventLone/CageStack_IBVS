#pragma once
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include "perception/types/common.hpp"
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <std_msgs/msg/string.hpp>
#include <tf2_eigen/tf2_eigen.hpp> // ROS 2 header
#include "perception/tools/2d/rfdetr_segmentor.h"
#include "perception/tools/2d/SingleInstanceTracker.h"
#include "perception/tools/2d/OpticalFlowTracking.h"
#include "perception/tools/2d/FeatureMatchTracking.h"

class CloudBuild final : public rclcpp::Node
{
    static constexpr float cx = 487.64654541015625f;
    static constexpr float cy = 366.9764709472656f;
    static constexpr float fx = 366.9764709472656f;
    static constexpr float fy = fx;
    static constexpr float fx_inv = 1.0 / fx;
    static constexpr float fy_inv = 1.0 / fy;

    static constexpr float depth_threshold = 3.6f;

    const std::string global_frame_id = "LOLA";

    struct ImgSet
    {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        Eigen::Isometry3f T_body2fork;
        cv::Mat depth_img, rgb_img;
    };

    struct InstanceData
    {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        Eigen::Isometry3f T_body2fork;
        cv::Mat depth_img;
        std::vector<Instance> instances;
    };

public:
    explicit CloudBuild(const std::string& name, const rclcpp::NodeOptions& options) : rclcpp::Node(name, options),
                                                                                       mT_fork2camera(Eigen::Isometry3f::Identity())
    {
        initSubscritions();
        initPublishers();

        mSegmentor = std::make_unique<RfDetrSeg>(
            "/media/avent/DATA/pretrained_weights/instance_segmentation/rfdetr-seg-medium-20260618.plan");
        // mSegmentor = std::make_unique<RfDetrSeg>("/home/avent/Desktop/rfdetr-seg-medium.plan");

        // mTfBuffer = std::make_unique<tf2_ros::Buffer>(this->get_clock());
        // mTfListener = std::make_shared<tf2_ros::TransformListener>(*mTfBuffer);

        // geometry_msgs::msg::Transform trans;
        // trans.translation.x = 0.0033704949666389793;
        // trans.translation.y = -0.060000312545999565;
        // trans.translation.z = 0.651563465680261;
        // trans.rotation.x = 0.5852007777496758;
        // trans.rotation.y = 0.5851465593085956;
        // trans.rotation.z = -0.39693472679331315;
        // trans.rotation.w = -0.39697150592454955;
        // mT_fork2camera = tf2::transformToEigen(trans).cast<float>();

        mT_fork2camera.rotate(Eigen::AngleAxisf(M_PIf, Eigen::Vector3f::UnitZ()));
        mT_fork2camera.prerotate(Eigen::AngleAxisf(-M_PIf / 7.0f, Eigen::Vector3f::UnitY()));
        mT_fork2camera.pretranslate(Eigen::Vector3f(-0.35f, -0.03f, 0.56f));

        // Force the node to use simulation time
        this->set_parameter(rclcpp::Parameter("use_sim_time", true));
        // Now, this->now() will return Isaac Sim's time
        RCLCPP_INFO(this->get_logger(), "Current Sim Time: %f", this->now().seconds());
        mSegWorker = std::thread(&CloudBuild::segmentLoop, this);
        mMainWorker = std::thread(&CloudBuild::workerLoop, this);
        RCLCPP_INFO(get_logger(), "The node has been activated.");
    }

    ~CloudBuild() override
    {
        //
        {
            std::unique_lock lock(mImgBufferMutex);
            mIsShutdown = true;
        }
        mTriggerSegEvent.notify_one();
        mTriggerCloudEvent.notify_one();
        if (mMainWorker.joinable())
        {
            mMainWorker.join();
        }
        if (mSegWorker.joinable())
        {
            mSegWorker.join();
        }
        RCLCPP_INFO(get_logger(), "The node has been shutdown.");
    }

private:
    /* Received Data Buffer */
    bool mIsShutdown{false};
    std::mutex mImgBufferMutex, mInstanceBufferMutex;
    std::condition_variable mTriggerSegEvent, mTriggerCloudEvent;
    std::thread mMainWorker, mSegWorker;
    std::queue<ImgSet> mImgsBuffer;
    std::queue<InstanceData> mInstanceBuffer;

    // std::shared_ptr<tf2_ros::TransformListener> mTfListener;
    // std::unique_ptr<tf2_ros::Buffer> mTfBuffer;
    Eigen::Isometry3f mT_fork2camera;

    std::unique_ptr<RfDetrSeg> mSegmentor;
    // SingleInstanceTracker mInstanceTracker{36}; // Set “MAX LOST FRAMES” to 16
    // OpticalFlowTracking mOpticalTracker;
    FeatureMatchingTracking mFeatureTracker;

    /*** Synchronized Subsribers ***/
    using ImgMsg = sensor_msgs::msg::Image;
    using SyncPolicy = message_filters::sync_policies::ApproximateTime<ImgMsg, ImgMsg>;
    message_filters::Subscriber<ImgMsg> mDepthSub, mSemanticSub;
    std::unique_ptr<message_filters::Synchronizer<SyncPolicy>> mSynchronizer;

    /* Publishers */
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr mColoredCloudPub, mInstanceCloudPub;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr mFilteredImagePub, mSegImagePub;

    void initParams();

    void initSubscritions();

    void initPublishers();

    void imgsHandler(const ImgMsg::ConstSharedPtr& depth_msg, const ImgMsg::ConstSharedPtr& rgb_msg);

    void pushInBuffer(ImgSet&& img_set)
    {
        std::lock_guard lock(mImgBufferMutex);
        while (!mImgsBuffer.empty())
        {
            mImgsBuffer.pop();
        }
        mImgsBuffer.push(std::move(img_set));
    }

    void segmentLoop();

    void workerLoop();
};
