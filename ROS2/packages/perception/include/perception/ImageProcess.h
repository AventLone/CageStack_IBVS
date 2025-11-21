#pragma once
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <opencv2/opencv.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <visualization_msgs/msg/marker.hpp>

class ImageProcess final : public rclcpp::Node
{
    using CloudXYZ = pcl::PointCloud<pcl::PointXYZ>;
    using CloudPtr = std::unique_ptr<CloudXYZ>;

    struct ImageSet
    {
        using Ptr = std::unique_ptr<ImageSet>;
        cv::Mat fork_semantics, fork_depth, left_semantics, left_depth, right_semantics, right_depth;
    };


    std::queue<ImageSet::Ptr> mImgsBuffer;
    std::queue<std::unique_ptr<CloudXYZ>> mTargetCloudBuffer;
    std::mutex mImgsBufferMutex, mTargeCloudBufferMutex;
    std::thread mCloudPubLoopThread, mTargetPosePubLoopThread;

    Eigen::Isometry3f mForkCameraExtrinsics, mLeftCameraExtrinsics, mRightCameraExtrinsics;

public:
    explicit ImageProcess();

    ~ImageProcess() override
    {
        if (mCloudPubLoopThread.joinable())
        {
            mCloudPubLoopThread.join();
        }
        if (mTargetPosePubLoopThread.joinable())
        {
            mTargetPosePubLoopThread.join();
        }
        RCLCPP_INFO(get_logger(), "The node has been shutdown.");
    }

private:
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr mTargetCloudPub;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr mTargetBBoxPub;

    /*** Synchronized Subsribers ***/
    using ImgMsg = sensor_msgs::msg::Image;
    using SyncPolicy = message_filters::sync_policies::ApproximateTime<ImgMsg, ImgMsg, ImgMsg, ImgMsg, ImgMsg, ImgMsg>;
    message_filters::Subscriber<sensor_msgs::msg::Image>
            mForkSemanticsSub,
            mLeftSemanticSub,
            mRightSemanticSub,
            mForkDepthSub,
            mLeftDepthSub,
            mRightDepthSub;
    std::unique_ptr<message_filters::Synchronizer<SyncPolicy>> mSynchronizer;

    void imgsHandler(const ImgMsg::ConstSharedPtr& fork_semantics,
                     const ImgMsg::ConstSharedPtr& left_semantics,
                     const ImgMsg::ConstSharedPtr& right_semantics,
                     const ImgMsg::ConstSharedPtr& fork_depth,
                     const ImgMsg::ConstSharedPtr& left_depth,
                     const ImgMsg::ConstSharedPtr& right_depth);

    ImageSet::Ptr getImgSet()
    {
        std::lock_guard<std::mutex> lock(mImgsBufferMutex);
        if (mImgsBuffer.empty())
        {
            return nullptr;
        }
        ImageSet::Ptr img_set = std::move(mImgsBuffer.front());
        mImgsBuffer.pop();
        return img_set;
    }

    void pushTargetCloud(std::unique_ptr<CloudXYZ>&& cloud)
    {
        std::lock_guard<std::mutex> lock(mTargeCloudBufferMutex);
        mTargetCloudBuffer.push(std::move(cloud));
    }

    std::unique_ptr<CloudXYZ> getTargetCloud()
    {
        std::lock_guard<std::mutex> lock(mTargeCloudBufferMutex);
        if (mTargetCloudBuffer.empty())
        {
            return nullptr;
        }
        std::unique_ptr<CloudXYZ> target_cloud = std::move(mTargetCloudBuffer.front());
        mTargetCloudBuffer.pop();
        return target_cloud;
    }

    void cloudPubLoop();

    void targetPosePubLoop();
};
