#include "perception/ImageProcess.h"
#include <cv_bridge/cv_bridge.hpp>
#include <pcl/common/transforms.h>
#include <pcl_conversions/pcl_conversions.h>
#include <Eigen/Geometry>

ImageProcess::ImageProcess() : rclcpp::Node("image_process"),
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

    mSynchronizer = std::make_unique<message_filters::Synchronizer<SyncPolicy>>(SyncPolicy(5),
                                                                                mForkSemanticsSub, mLeftSemanticSub, mRightSemanticSub,
                                                                                mForkDepthSub, mLeftDepthSub, mRightDepthSub);
    mSynchronizer->registerCallback(std::bind(&ImageProcess::imgsHandler, this,
                                              std::placeholders::_1, std::placeholders::_2, std::placeholders::_3,
                                              std::placeholders::_4, std::placeholders::_5, std::placeholders::_6));

    mTargetCloudPub = create_publisher<sensor_msgs::msg::PointCloud2>("cloud", rclcpp::SensorDataQoS().best_effort());
    mTargetBBoxPub = create_publisher<visualization_msgs::msg::Marker>("target_bbox", rclcpp::SensorDataQoS().best_effort());

    mForkCameraExtrinsics.rotate(Eigen::AngleAxisf(M_PIf * 5.0f / 18.0f, Eigen::Vector3f::UnitY()));
    mForkCameraExtrinsics.pretranslate(Eigen::Vector3f(0.5f, 0.0f, 0.7f));

    mLeftCameraExtrinsics.rotate(Eigen::AngleAxisf(M_PIf / 18.0f, Eigen::Vector3f::UnitZ()));
    mLeftCameraExtrinsics.pretranslate(Eigen::Vector3f(-0.4f, 0.6f, 1.0f));

    mRightCameraExtrinsics.rotate(Eigen::AngleAxisf(-M_PIf / 18.0f, Eigen::Vector3f::UnitZ()));
    mRightCameraExtrinsics.pretranslate(Eigen::Vector3f(-0.4f, -0.6f, 1.0f));

    mCloudPubLoopThread = std::thread(&ImageProcess::cloudPubLoop, this);
    RCLCPP_INFO(get_logger(), "The node has been activated.");
}

void ImageProcess::imgsHandler(const ImgMsg::ConstSharedPtr& fork_semantics, const ImgMsg::ConstSharedPtr& left_semantics,
                               const ImgMsg::ConstSharedPtr& right_semantics, const ImgMsg::ConstSharedPtr& fork_depth,
                               const ImgMsg::ConstSharedPtr& left_depth,
                               const ImgMsg::ConstSharedPtr& right_depth)
{
    const cv_bridge::CvImagePtr fork_semantics_ptr = cv_bridge::toCvCopy(fork_semantics, "mono8");
    const cv_bridge::CvImagePtr left_semantics_ptr = cv_bridge::toCvCopy(left_semantics, "mono8");
    const cv_bridge::CvImagePtr right_semantics_ptr = cv_bridge::toCvCopy(right_semantics, "mono8");
    const cv_bridge::CvImagePtr fork_depth_ptr = cv_bridge::toCvCopy(fork_depth, "mono16");
    const cv_bridge::CvImagePtr left_depth_ptr = cv_bridge::toCvCopy(left_depth, "mono16");
    const cv_bridge::CvImagePtr right_depth_ptr = cv_bridge::toCvCopy(right_depth, "mono16");

    auto imgs = std::make_unique<ImageSet>();
    imgs->fork_semantics = std::move(fork_semantics_ptr->image);
    imgs->left_semantics = std::move(left_semantics_ptr->image);
    imgs->right_semantics = std::move(right_semantics_ptr->image);
    imgs->fork_depth = std::move(fork_depth_ptr->image);
    imgs->left_depth = std::move(left_depth_ptr->image);
    imgs->right_depth = std::move(right_depth_ptr->image);

    std::lock_guard<std::mutex> lock(mImgsBufferMutex);
    mImgsBuffer.push(std::move(imgs));
}

void ImageProcess::cloudPubLoop()
{
    constexpr float fx = 251.66666f;
    constexpr float fy = fx;
    constexpr float fx_inv = 1.0 / fx;
    constexpr float fy_inv = 1.0 / fy;
    constexpr float cx = 480.0f;
    constexpr float cy = 393.0f;

    while (rclcpp::ok())
    {
        // const auto begin_time = std::chrono::high_resolution_clock::now();
        ImageSet::Ptr img_set = getImgSet();
        if (img_set == nullptr)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }

        img_set->fork_depth.setTo(0, img_set->fork_semantics != 3);
        img_set->left_depth.setTo(0, img_set->left_semantics != 3);
        img_set->right_depth.setTo(0, img_set->right_semantics != 2);

        CloudXYZ target_cloud_on_fork, target_cloud_on_left, target_cloud_on_right;
        for (int v = 0; v < img_set->fork_depth.rows; v += 6)
        {
            const uint16_t* target_on_fork = img_set->fork_depth.ptr<uint16_t>(v);
            const uint16_t* target_on_left = img_set->left_depth.ptr<uint16_t>(v);
            const uint16_t* target_on_right = img_set->right_depth.ptr<uint16_t>(v);

            for (int u = 0; u < img_set->fork_depth.cols; u += 6)
            {
                if (target_on_fork[u] > 0)
                {
                    if (const float depth = static_cast<float>(target_on_fork[u]) * 1.0e-3f; depth < 10.0f)
                    {
                        const float X = (static_cast<float>(u) - cx) * depth * fx_inv;
                        const float Y = (static_cast<float>(v) - cy) * depth * fy_inv;
                        target_cloud_on_fork.emplace_back(depth, -X, -Y);
                    }
                }

                if (target_on_left[u] > 0)
                {
                    if (const float depth = static_cast<float>(target_on_left[u]) * 1.0e-3f; depth < 10.0f)
                    {
                        const float X = (static_cast<float>(u) - cx) * depth * fx_inv;
                        const float Y = (static_cast<float>(v) - cy) * depth * fy_inv;
                        target_cloud_on_left.emplace_back(depth, -X, -Y);
                    }
                }

                if (target_on_right[u] > 0)
                {
                    if (const float depth = static_cast<float>(target_on_right[u]) * 1.0e-3f; depth < 10.0f)
                    {
                        const float X = (static_cast<float>(u) - cx) * depth * fx_inv;
                        const float Y = (static_cast<float>(v) - cy) * depth * fy_inv;
                        target_cloud_on_right.emplace_back(depth, -X, -Y);
                    }
                }
            }
        }

        auto cloud = std::make_unique<CloudXYZ>();

        if (!target_cloud_on_fork.empty())
        {
            CloudXYZ cloud_in_base;
            pcl::transformPointCloud(target_cloud_on_fork, cloud_in_base, mForkCameraExtrinsics);
            *cloud = std::move(cloud_in_base);
        }

        if (!target_cloud_on_left.empty())
        {
            CloudXYZ cloud_in_base;
            pcl::transformPointCloud(target_cloud_on_left, cloud_in_base, mLeftCameraExtrinsics);
            *cloud += cloud_in_base;
        }

        if (!target_cloud_on_right.empty())
        {
            CloudXYZ cloud_in_base;
            pcl::transformPointCloud(target_cloud_on_right, cloud_in_base, mRightCameraExtrinsics);
            *cloud += cloud_in_base;
        }

        if (!cloud->empty())
        {
            sensor_msgs::msg::PointCloud2 cloud_msg;
            pcl::toROSMsg(*cloud, cloud_msg);
            pushTargetCloud(std::move(cloud));
            cloud_msg.header.stamp = now();
            cloud_msg.header.frame_id = "map";
            mTargetCloudPub->publish(cloud_msg);
        }

        // const auto elapse_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        //     std::chrono::high_resolution_clock::now() - begin_time).count();
        //
        // RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1000, "Elapse: %ld.", elapse_time);
    }
}

void ImageProcess::targetPosePubLoop()
{
    while (rclcpp::ok())
    {
        std::unique_ptr<CloudXYZ> target_cloud = getTargetCloud();
        if (target_cloud == nullptr)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }
        cv::Mat top_view;
    }
}
