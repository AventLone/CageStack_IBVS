#include "perception/ImageProcess.h"
#include <cv_bridge/cv_bridge.hpp>
#include <pcl/common/transforms.h>
#include <pcl_conversions/pcl_conversions.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <Eigen/Geometry>
#include "perception/features_detect_2d.h"
#include "perception/feature_detect_3d.hpp"
#include "perception/filter.h"

ImageProcess::ImageProcess() : rclcpp::Node("image_process"),
                               mCloudProjector(View::TOP, 0.005f),
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
    mTargetPosePub = create_publisher<geometry_msgs::msg::PoseStamped>("target_pose", rclcpp::SensorDataQoS().reliable());

    mForkCameraExtrinsics.rotate(Eigen::AngleAxisf(M_PIf * 5.0f / 18.0f, Eigen::Vector3f::UnitY()));
    mForkCameraExtrinsics.pretranslate(Eigen::Vector3f(0.5f, 0.0f, 0.7f));

    mLeftCameraExtrinsics.rotate(Eigen::AngleAxisf(M_PIf / 18.0f, Eigen::Vector3f::UnitZ()));
    mLeftCameraExtrinsics.pretranslate(Eigen::Vector3f(-0.4f, 0.6f, 1.0f));

    mRightCameraExtrinsics.rotate(Eigen::AngleAxisf(-M_PIf / 18.0f, Eigen::Vector3f::UnitZ()));
    mRightCameraExtrinsics.pretranslate(Eigen::Vector3f(-0.4f, -0.6f, 1.0f));

    mCloudPubLoopThread = std::thread(&ImageProcess::cloudPubLoop, this);
    mTargetPosePubLoopThread = std::thread(&ImageProcess::targetPosePubLoop, this);
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

    // std::lock_guard<std::mutex> lock(mImgsBufferMutex);
    // mImgsBuffer.push(std::move(imgs));
    mImgsBuffer.enqueue(std::move(imgs));
}

void ImageProcess::cloudPubLoop()
{
    constexpr float fx = 251.66666f;
    constexpr float fy = fx;
    constexpr float fx_inv = 1.0 / fx;
    constexpr float fy_inv = 1.0 / fy;
    constexpr float cx = 480.0f;
    constexpr float cy = 393.0f;

    constexpr float depth_threshold = 10.0f;

    bool has_logged = false;

    while (rclcpp::ok())
    {
        ImageSet::Ptr img_set;
        if (!mImgsBuffer.try_dequeue(img_set))
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }

        SemanticCloud target_cloud_on_fork, target_cloud_on_left, target_cloud_on_right;
        for (int v = 0; v < img_set->fork_depth.rows; v += 25)
        {
            const uint16_t* target_on_fork = img_set->fork_depth.ptr<uint16_t>(v);
            const uint16_t* target_on_left = img_set->left_depth.ptr<uint16_t>(v);
            const uint16_t* target_on_right = img_set->right_depth.ptr<uint16_t>(v);

            const uint8_t* labels_on_fork = img_set->fork_semantics.ptr<uint8_t>(v);
            const uint8_t* labels_on_left = img_set->left_semantics.ptr<uint8_t>(v);
            const uint8_t* labels_on_right = img_set->right_semantics.ptr<uint8_t>(v);

            for (int u = 0; u < img_set->fork_depth.cols; u += 6)
            {
                if (target_on_fork[u] > 0)
                {
                    if (const float depth = static_cast<float>(target_on_fork[u]) * 1.0e-3f; depth < depth_threshold)
                    {
                        const float X = (static_cast<float>(u) - cx) * depth * fx_inv;
                        const float Y = (static_cast<float>(v) - cy) * depth * fy_inv;
                        const uint32_t label = labels_on_fork[u];
                        SemanticPoint point{};
                        point.x = depth;
                        point.y = -X;
                        point.z = -Y;
                        point.label = label;
                        target_cloud_on_fork.push_back(point);
                    }
                }

                if (target_on_left[u] > 0)
                {
                    if (const float depth = static_cast<float>(target_on_left[u]) * 1.0e-3f; depth < depth_threshold)
                    {
                        const float X = (static_cast<float>(u) - cx) * depth * fx_inv;
                        const float Y = (static_cast<float>(v) - cy) * depth * fy_inv;
                        const uint32_t label = labels_on_left[u];
                        SemanticPoint point{};
                        point.x = depth;
                        point.y = -X;
                        point.z = -Y;
                        point.label = label;
                        target_cloud_on_left.push_back(point);
                    }
                }

                if (target_on_right[u] > 0)
                {
                    if (const float depth = static_cast<float>(target_on_right[u]) * 1.0e-3f; depth < depth_threshold)
                    {
                        const float X = (static_cast<float>(u) - cx) * depth * fx_inv;
                        const float Y = (static_cast<float>(v) - cy) * depth * fy_inv;
                        const uint32_t label = labels_on_right[u];
                        SemanticPoint point{};
                        point.x = depth;
                        point.y = -X;
                        point.z = -Y;
                        point.label = label;

                        target_cloud_on_right.push_back(point);
                    }
                }
            }
        }

        auto cloud = std::make_unique<SemanticCloud>();

        if (!target_cloud_on_fork.empty())
        {
            SemanticCloud cloud_in_base;
            pcl::transformPointCloud(target_cloud_on_fork, cloud_in_base, mForkCameraExtrinsics);
            *cloud = std::move(cloud_in_base);
        }

        if (!target_cloud_on_left.empty())
        {
            SemanticCloud cloud_in_base;
            pcl::transformPointCloud(target_cloud_on_left, cloud_in_base, mLeftCameraExtrinsics);
            *cloud += cloud_in_base;
        }

        if (!target_cloud_on_right.empty())
        {
            SemanticCloud cloud_in_base;
            pcl::transformPointCloud(target_cloud_on_right, cloud_in_base, mRightCameraExtrinsics);
            *cloud += cloud_in_base;
        }

        if (!cloud->empty())
        {
            pcl::PointCloud<pcl::PointXYZRGB> colored_cloud;
            getCloud(*cloud, colored_cloud);

            sensor_msgs::msg::PointCloud2 cloud_msg;
            pcl::toROSMsg(colored_cloud, cloud_msg);
            mCloudBuffer.enqueue(std::move(cloud));
            cloud_msg.header.stamp = now();
            cloud_msg.header.frame_id = "map";
            mTargetCloudPub->publish(cloud_msg);
        }
    }
}

void ImageProcess::targetPosePubLoop()
{
    while (rclcpp::ok())
    {
        SemanticCloudPtr cloud;
        if (!mCloudBuffer.try_dequeue(cloud))
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }
        pcl::PointCloud<pcl::PointXYZ> cage_posts_cloud;
        getCloud(*cloud, 17, cage_posts_cloud);
        Eigen::Vector4f centroid;
        pcl::compute3DCentroid(cage_posts_cloud, centroid);
        float angle = computeAngleByPCA(cage_posts_cloud);

        tf2::Quaternion tf2_quat;
        tf2_quat.setRPY(0.0, 0.0, angle);
        geometry_msgs::msg::Quaternion geom_quat;
        tf2::convert(tf2_quat, geom_quat);

        geometry_msgs::msg::PoseStamped target_pose;
        target_pose.header.frame_id = "map";
        target_pose.header.stamp = this->now();
        target_pose.pose.position.x = centroid[0];
        target_pose.pose.position.y = centroid[1];
        target_pose.pose.position.z = centroid[2];
        target_pose.pose.orientation = geom_quat;
        mTargetPosePub->publish(target_pose);

        const Eigen::Vector3f target_size = getCloudSize(cage_posts_cloud);

        Eigen::Isometry2f T_1(Eigen::Isometry2f::Identity()), T_2(Eigen::Isometry2f::Identity());
        T_1.rotate(angle);
        T_1.pretranslate(centroid.head<2>());
        T_2.translate(Eigen::Vector2f(0.5f * target_size[1], 0.0f));
        Eigen::Isometry2f T_3 = T_1 * T_2;
        const Eigen::Vector2f mark_position = T_3.translation();

        visualization_msgs::msg::Marker target_marker_msg;
        target_marker_msg.header.frame_id = "map"; // 或者 base_link/odom
        target_marker_msg.header.stamp = this->now();
        target_marker_msg.ns = "demo";
        target_marker_msg.id = 0;
        target_marker_msg.type = visualization_msgs::msg::Marker::CUBE;
        target_marker_msg.action = visualization_msgs::msg::Marker::ADD;

        // Cube 尺寸（米）
        target_marker_msg.scale.x = target_size[1];
        target_marker_msg.scale.y = target_size[1];
        target_marker_msg.scale.z = target_size[2];

        // 半透明颜色 (r,g,b,a)
        target_marker_msg.color.r = 0.0f;
        target_marker_msg.color.g = 1.0f;
        target_marker_msg.color.b = 0.0f;
        target_marker_msg.color.a = 0.8f; // alpha<1 表示半透明

        target_marker_msg.pose.position.x = mark_position[0];
        target_marker_msg.pose.position.y = mark_position[1];
        target_marker_msg.pose.position.z = centroid[2];
        target_marker_msg.pose.orientation = geom_quat;

        mTargetBBoxPub->publish(target_marker_msg);
    }
}
