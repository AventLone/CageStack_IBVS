#include "perception/nodes/CloudPublisher.h"
#include <cv_bridge/cv_bridge.hpp>
#include <pcl/common/transforms.h>
#include <pcl_conversions/pcl_conversions.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <Eigen/Geometry>
#include <tf2_eigen/tf2_eigen.hpp> // ROS 2 header
#include "perception/tools/2d/rfdetr_segmentor.h"
#include "perception/tools/3d/filter.h"
#include <random>

enum
{
    PALLET = 0,
    STORAGE_CAGE = 1,
    GOODS = 2
};

void CloudBuild::initSubscritions()
{
    const std::string params_prefix = "TopicName.Sensor.Camera";
    this->declare_parameters<std::string>(params_prefix, {
                                              {"Mid.Depth", "/zed/zed_node/depth/depth_registered"},
                                              {"Mid.Rgb", "/zed/zed_node/rgb/image_rect_color"}
                                              // {"Mid.Rgb", "/zed/zed_node/rgb/color/rect/image"}
                                          });

    std::map<std::string, std::string> sensor_topics;
    if (!this->get_parameters<std::string>(params_prefix, sensor_topics))
    {
        RCLCPP_ERROR(get_logger(), "Failed to get parameters, sensor topic names!");
    }

    mDepthSub.subscribe(this, sensor_topics["Mid.Depth"]);
    mSemanticSub.subscribe(this, sensor_topics["Mid.Rgb"]);

    mSynchronizer = std::make_unique<message_filters::Synchronizer<SyncPolicy>>(SyncPolicy(2),
                                                                                mDepthSub, mSemanticSub);

    // 设置更小的时间容差（单位：秒）
    mSynchronizer->setMaxIntervalDuration(rclcpp::Duration(0, 10 * 100000)); // 10ms 容差
    mSynchronizer->registerCallback(std::bind(&CloudBuild::imgsHandler, this, std::placeholders::_1, std::placeholders::_2));
}

void CloudBuild::initPublishers()
{
    const std::string params_prefix = "TopicName.Perception";
    this->declare_parameters<std::string>(params_prefix, {
                                              {"SemanticCloud", "/perception/instance_cloud"},
                                              {"ColoredCloud", "/perception/colored_cloud"}
                                          });

    std::map<std::string, std::string> cloud_topics;
    if (!this->get_parameters<std::string>(params_prefix, cloud_topics))
    {
        RCLCPP_FATAL(get_logger(), "Failed to get parameters, cloud topic names!");
    }
    mInstanceCloudPub = create_publisher<sensor_msgs::msg::PointCloud2>("/perception/instance_cloud", rclcpp::SensorDataQoS());
    mColoredCloudPub = create_publisher<sensor_msgs::msg::PointCloud2>(cloud_topics["ColoredCloud"], rclcpp::SensorDataQoS());
    mFilteredImagePub = create_publisher<sensor_msgs::msg::Image>("/filtered_rgb", rclcpp::SensorDataQoS());
    mSegImagePub = create_publisher<sensor_msgs::msg::Image>("/seg_rgb", rclcpp::SensorDataQoS());
}

void CloudBuild::imgsHandler(const ImgMsg::ConstSharedPtr& depth_msg, const ImgMsg::ConstSharedPtr& rgb_msg)
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

    static const int cols_half = rgb_ptr->image.cols / 2;
    static const int rows_half = rgb_ptr->image.rows / 2;
    static const int bound_left = cols_half - rows_half;
    static const int bound_right = cols_half + rows_half;

    ImgSet img_set{};
    // img_set.T_body2fork = T_body2fork;
    img_set.T_body2fork = Eigen::Isometry3f::Identity();
    depth_ptr->image(cv::Range::all(), cv::Range(bound_left, bound_right)).copyTo(img_set.depth_img);
    cv::cvtColor(rgb_ptr->image(cv::Range::all(), cv::Range(bound_left, bound_right)), img_set.rgb_img, cv::COLOR_BGRA2RGB);

    pushInBuffer(std::move(img_set));
    mTriggerSegEvent.notify_one();
}

void CloudBuild::segmentLoop()
{
    const std::unordered_map<int, std::string> label_dict{{0, "pallet"}, {1, "storage_cage"}, {2, "goods"}};
    while (rclcpp::ok())
    {
        ImgSet img_set;
        //
        {
            std::unique_lock<std::mutex> lock(mImgBufferMutex);
            mTriggerSegEvent.wait(lock, [this]() -> bool { return !mImgsBuffer.empty() || mIsShutdown; });
            if (mIsShutdown)
            {
                break;
            }
            img_set = std::move(mImgsBuffer.front());
            mImgsBuffer.pop();
        }

        static const auto choose_front_instances = [](std::vector<Instance> instances, int front_num) -> std::vector<Instance>
            {
                std::sort(instances.begin(), instances.end(), [](const Instance& a, const Instance& b) -> bool
                              {
                                  return a.bbox.y > b.bbox.y;
                              });

                front_num = std::min(front_num, static_cast<int>(instances.size()));
                return {std::make_move_iterator(instances.begin()), std::make_move_iterator(instances.begin() + front_num)};
            };

        std::vector<Instance> results = choose_front_instances(mSegmentor->seg(img_set.rgb_img, 0.6, 0.3, 8), 5);
        // std::vector<Instance> results = mSegmentor->seg(img_set.rgb_img, 0.5, 0.3, 8);

        cv::Mat visualization;
        if (results.empty())
        {
            RCLCPP_INFO(get_logger(), "results is empty!");
            visualization = std::move(img_set.rgb_img);
        }
        else
        {
            visualizeInstanceSeg(img_set.rgb_img, visualization, results, label_dict);
        }

        // Create a cv_bridge object
        cv_bridge::CvImage img_bridge;
        img_bridge.header.stamp = this->now(); // Optional: set a timestamp
        img_bridge.header.frame_id = "camera_frame";
        img_bridge.encoding = sensor_msgs::image_encodings::RGB8; // e.g., "bgr8"
        img_bridge.image = visualization;

        // Convert to ROS message
        auto ros_image = std::make_unique<sensor_msgs::msg::Image>();
        img_bridge.toImageMsg(*ros_image);
        mSegImagePub->publish(std::move(ros_image));

        InstanceData instance_data;
        instance_data.instances = std::move(results);
        instance_data.depth_img = std::move(img_set.depth_img);
        instance_data.T_body2fork = std::move(img_set.T_body2fork);

        std::lock_guard<std::mutex> lock(mInstanceBufferMutex);
        while (!mInstanceBuffer.empty())
        {
            mImgsBuffer.pop();
        }
        mInstanceBuffer.push(std::move(instance_data));
        mTriggerCloudEvent.notify_one();
    }
}

void CloudBuild::workerLoop()
{
    constexpr int offset_x = 960 / 2 - 600 / 2;
    while (rclcpp::ok())
    {
        InstanceData instance_data;
        //
        {
            std::unique_lock<std::mutex> lock(mInstanceBufferMutex);
            mTriggerCloudEvent.wait(lock, [this]() -> bool { return !mInstanceBuffer.empty() || mIsShutdown; });
            if (mIsShutdown)
            {
                break;
            }
            instance_data = std::move(mInstanceBuffer.front());
            mInstanceBuffer.pop();
        }

        const auto& depth_img = instance_data.depth_img;
        const auto& instances = instance_data.instances;

        std::vector<std::vector<InstancePoint>> local_clouds(instances.size());
        ColoredCloud color_cloud;
        color_cloud.reserve(depth_img.cols * depth_img.rows);
        for (size_t i = 0; i < instances.size(); ++i)
        {
            static constexpr int step = 2;
            const auto& instance = instances[i];
            if (instance.mask.empty())
                continue;

            // 获取当前实例在全图中的 ROI 矩形区域
            cv::Rect roi = instance.bbox & cv::Rect(0, 0, depth_img.cols, depth_img.rows);
            if (roi.width <= 0 || roi.height <= 0)
                continue;

            // 提取深度图和 Mask 的局部区域（无内存拷贝，仅创建视图）
            cv::Mat depth_roi = depth_img(roi);
            cv::Mat mask_roi = instance.mask;

            // 确保 Mask 的大小与 ROI 一致（有时 Mask 只有 ROI 大小，有时是全图大小）
            if (mask_roi.size() != roi.size())
            {
                mask_roi = mask_roi(roi);
            }

            std::vector<InstancePoint>& current_local = local_clouds[i];
            current_local.reserve(roi.width * roi.height / 4);
            for (int v_roi = 0; v_roi < roi.height; v_roi += step)
            {
                const int v_global = roi.y + v_roi; // 全局图像坐标 v

                const auto* depth_ptr = depth_roi.ptr<float>(v_roi);
                const uint8_t* mask_ptr = mask_roi.ptr<uint8_t>(v_roi);

                for (int u_roi = 0; u_roi < roi.width; u_roi += step)
                {
                    // 仅当 mask 激活且深度值有效时处理
                    if (const float depth = depth_ptr[u_roi]; mask_ptr[u_roi] > 0 && depth > 1.0 && depth < 5.0f)
                    {
                        const int u_global = roi.x + u_roi + offset_x;

                        const float x = (static_cast<float>(u_global) - cx) * depth * fx_inv;
                        const float y = (static_cast<float>(v_global) - cy) * depth * fy_inv;

                        current_local.emplace_back(depth, -x, -y,
                                                   static_cast<uint16_t>(instance.class_id),
                                                   static_cast<uint16_t>(i));
                    }
                }
            }
        }

        // 4. 合并所有局部点云（计算总大小后一次性分配，效率最高）
        size_t total_points = 0;
        for (const auto& lc : local_clouds)
        {
            total_points += lc.size();
        }

        InstanceCloud cloud;
        cloud.points.resize(total_points);
        cloud.width = total_points;
        cloud.height = 1;
        cloud.is_dense = false;

        size_t offset = 0;
        for (const auto& lc : local_clouds)
        {
            if (!lc.empty())
            {
                std::memcpy(&cloud.points[offset], lc.data(), lc.size() * sizeof(InstancePoint));
                offset += lc.size();
            }
        }

        // ColoredCloud rgb_cloud_base; // This is the semantic cloud in the base link coordinate system.
        InstanceCloud cloud_base;
        // const Eigen::Isometry3f T_body2camera = img_set.T_body2fork * mT_fork2camera;
        static const Eigen::Isometry3f T_body2camera = mT_fork2camera;
        pcl::transformPointCloud(cloud, cloud_base, T_body2camera);

        ColoredCloud colored_cloud;
        getColorCloudFromInstanceCloud(cloud_base, colored_cloud);
        sensor_msgs::msg::PointCloud2 rgb_cloud_msg;
        pcl::toROSMsg(colored_cloud, rgb_cloud_msg);
        rgb_cloud_msg.header.stamp = this->now();
        rgb_cloud_msg.header.frame_id = global_frame_id;
        mColoredCloudPub->publish(rgb_cloud_msg);

        auto instance_cloud_msg = std::make_unique<sensor_msgs::msg::PointCloud2>();
        pcl::toROSMsg(cloud_base, *instance_cloud_msg);
        instance_cloud_msg->header.stamp = this->now();
        instance_cloud_msg->header.frame_id = global_frame_id;
        mInstanceCloudPub->publish(std::move(instance_cloud_msg));
    }
}
