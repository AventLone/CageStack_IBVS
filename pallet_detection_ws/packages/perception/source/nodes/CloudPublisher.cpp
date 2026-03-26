#include "perception/nodes/CloudPublisher.h"
#include <cv_bridge/cv_bridge.hpp>
#include <pcl/common/transforms.h>
#include <pcl_conversions/pcl_conversions.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <Eigen/Geometry>
#include <tf2_eigen/tf2_eigen.hpp> // ROS 2 header
#include "perception/tools/filter_3d.h"
#include <random>


void CloudBuild::initSubscritions()
{
    const std::string params_prefix = "TopicName.Sensor.Camera";
    // this->declare_parameters<std::string>(params_prefix, {{"Mid.Depth", "/sensors/camera/mid/depth"},
    //                                                       {"Mid.Semantics", "/sensors/camera/mid/semantics"}});
    this->declare_parameters<std::string>(params_prefix, {
                                              {"Mid.Depth", "/zed/zed_node/depth/depth_registered"},
                                              {"Mid.Semantics", "/zed/zed_node/rgb/image_rect_color"}
                                          });

    std::map<std::string, std::string> sensor_topics;
    if (!this->get_parameters<std::string>(params_prefix, sensor_topics))
    {
        RCLCPP_ERROR(get_logger(), "Failed to get parameters, sensor topic names!");
    }

    mDepthSub.subscribe(this, sensor_topics["Mid.Depth"]);
    mSemanticSub.subscribe(this, sensor_topics["Mid.Semantics"]);

    mSynchronizer = std::make_unique<message_filters::Synchronizer<SyncPolicy>>(SyncPolicy(2),
                                                                                mDepthSub, mSemanticSub);

    // 设置更小的时间容差（单位：秒）
    mSynchronizer->setMaxIntervalDuration(rclcpp::Duration(0, 1000000)); // 10ms 容差
    mSynchronizer->registerCallback(std::bind(&CloudBuild::imgsHandler,
                                              this, std::placeholders::_1, std::placeholders::_2));
}

void CloudBuild::initPublishers()
{
    const std::string params_prefix = "TopicName.Perception";
    this->declare_parameters<std::string>(params_prefix, {
                                              {"SemanticCloud", "/perception/semantic_cloud"},
                                              {"ColoredCloud", "/perception/colored_cloud"}
                                          });

    std::map<std::string, std::string> cloud_topics;
    if (!this->get_parameters<std::string>(params_prefix, cloud_topics))
    {
        RCLCPP_FATAL(get_logger(), "Failed to get parameters, cloud topic names!");
    }
    mColoredCloudPub = create_publisher<sensor_msgs::msg::PointCloud2>(cloud_topics["ColoredCloud"],
                                                                       rclcpp::SensorDataQoS());
    mFilteredImagePub = create_publisher<sensor_msgs::msg::Image>("/filtered_rgb", rclcpp::SensorDataQoS());
}

void CloudBuild::imgsHandler(const ImgMsg::ConstSharedPtr& depth_msg,
                             const ImgMsg::ConstSharedPtr& rgb_msg)
{
    /* Get the pose of the forks */
    // Eigen::Isometry3f T_body2fork;
    // try
    // {
    //     // This returns the pose of 'fork' in 'body' coordinates
    //     const geometry_msgs::msg::TransformStamped tf_body2fork =
    //             mTfBuffer->lookupTransform("LOLA", "fork", tf2::TimePointZero);
    //     T_body2fork = tf2::transformToEigen(tf_body2fork).cast<float>();
    // }
    // catch (const tf2::TransformException& ex)
    // {
    //     RCLCPP_ERROR(this->get_logger(), "Could not transform fork to body: %s", ex.what());
    //     return;
    // }

    const auto depth_ptr = cv_bridge::toCvShare(depth_msg, sensor_msgs::image_encodings::TYPE_32FC1);
    const auto rgb_ptr = cv_bridge::toCvShare(rgb_msg, sensor_msgs::image_encodings::BGRA8);

    ImgSet img_set{};
    // img_set.T_body2fork = T_body2fork;
    img_set.T_body2fork = Eigen::Isometry3f::Identity();
    img_set.depth_img = depth_ptr->image.clone();
    cv::cvtColor(rgb_ptr->image, img_set.semantic_img, cv::COLOR_BGRA2RGB);
    // img_set.semantic_img = rgb_ptr->image.clone();

    pushInBuffer(std::move(img_set));
    mTriggerEvent.notify_one();
}

void CloudBuild::workerLoop()
{
    while (rclcpp::ok())
    {
        sensor_msgs::msg::PointCloud2 cloud_msg;
        ImgSet img_set;
        //
        {
            std::unique_lock<std::mutex> lock(mBufferMutex);
            mTriggerEvent.wait(lock, [this]() -> bool { return !mImgsBuffer.empty() || mIsShutdown; });
            if (mIsShutdown)
            {
                break;
            }
            img_set = mImgsBuffer.front();
            mImgsBuffer.pop();
        }
        cv::Mat mask = cv::Mat::zeros(img_set.depth_img.size(), CV_8UC1);
        mask.setTo(255, img_set.depth_img > 3.6f);

        ColoredCloud rgb_cloud_camera; // This is the semantic cloud in the camera coordinate system.
        rgb_cloud_camera.reserve(img_set.depth_img.total() / 2);
        constexpr int skip_step = 2;
        for (int v = 0; v < img_set.depth_img.rows; v += skip_step)
        {
            const auto* depth_ptr = img_set.depth_img.ptr<float>(v);
            const auto* rgb_ptr = img_set.semantic_img.ptr<cv::Vec3b>(v);

            for (int u = 0; u < img_set.depth_img.cols; u += skip_step)
            {
                const float depth = depth_ptr[u];
                if (depth < 0.1f || depth > depth_threshold)
                {
                    continue;
                }

                const float x = (static_cast<float>(u) - cx) * depth * fx_inv;
                if (constexpr float y_thresh = 2.0f; std::abs(x) > y_thresh)
                {
                    continue;
                }
                const float y = (static_cast<float>(v) - cy) * depth * fy_inv;

                const cv::Vec3b& rgb_pixel = rgb_ptr[u];
                rgb_cloud_camera.emplace_back(rgb_pixel[0], rgb_pixel[1], rgb_pixel[2],
                                              depth, -x, -y);
            }
        }

        img_set.semantic_img.setTo(cv::Vec3b(0, 0, 0), mask);
        cv_bridge::CvImage cv_image;
        cv_image.header.stamp = this->now(); // ROS timestamp (now)
        cv_image.header.frame_id = "LOLA"; // ROS frame ID
        cv_image.encoding = sensor_msgs::image_encodings::RGB8; // Explicit RGB encoding
        cv_image.image = img_set.semantic_img; // Bind OpenCV mat

        // 2. Convert to ROS Image message (efficient copy/memory management)
        mFilteredImagePub->publish(*cv_image.toImageMsg());

        if (rgb_cloud_camera.size() < 10)
        {
            continue;
        }

        ColoredCloud rgb_cloud_base; // This is the semantic cloud in the base link coordinate system.
        // const Eigen::Isometry3f T_body2camera = img_set.T_body2fork * mT_fork2camera;
        static const Eigen::Isometry3f T_body2camera = mT_fork2camera;
        pcl::transformPointCloud(rgb_cloud_camera, rgb_cloud_base, T_body2camera);

        sensor_msgs::msg::PointCloud2 rgb_cloud_msg;
        pcl::toROSMsg(rgb_cloud_base, rgb_cloud_msg);

        rgb_cloud_msg.header.stamp = this->now();
        rgb_cloud_msg.header.frame_id = global_frame_id;

        mColoredCloudPub->publish(rgb_cloud_msg);

        // std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}
