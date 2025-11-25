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
#include "perception/OrthographicProjector.hpp"
#include "perception/types/point_cloud.h"
#include "perception/concurrentqueue.h"

class ImageProcess final : public rclcpp::Node
{
    template<class T>
    using lockfree_queue = moodycamel::ConcurrentQueue<T>;
    using CloudXYZ = pcl::PointCloud<pcl::PointXYZ>;
    using CloudPtr = std::unique_ptr<CloudXYZ>;

    struct ImageSet
    {
        using Ptr = std::unique_ptr<ImageSet>;
        cv::Mat fork_semantics, fork_depth, left_semantics, left_depth, right_semantics, right_depth;
    };

    OrthographicProjector<pcl::PointXYZ> mCloudProjector;

    lockfree_queue<ImageSet::Ptr> mImgsBuffer{1024};
    lockfree_queue<SemanticCloudPtr> mCloudBuffer{1024};
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

    void cloudPubLoop();

    void targetPosePubLoop();
};
