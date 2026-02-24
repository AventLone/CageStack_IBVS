#include "perception/nodes/CloudPublisher.h"
#include <cv_bridge/cv_bridge.hpp>
#include <pcl/common/transforms.h>
#include <pcl_conversions/pcl_conversions.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <Eigen/Geometry>
#include <tf2_eigen/tf2_eigen.hpp> // ROS 2 header
#include "perception/tools/filter_3d.h"


static std::unordered_map<std::string, int> parseSemanticLabels(const std_msgs::msg::String::ConstSharedPtr& msg)
{
    const std::string msg_string = msg->data;

    std::unordered_map<std::string, int> out;
    const size_t n = msg_string.size();
    // find first '{'
    const size_t pos = msg_string.find('{');
    if (pos == std::string::npos)
        return out;
    size_t i = pos + 1;

    while (i < n)
    {
        // find next quote for the id token (either " or ')
        size_t q1 = msg_string.find_first_of("\"'", i);
        if (q1 == std::string::npos)
            break;
        char quote = msg_string[q1];

        // find closing quote for id
        size_t q2 = msg_string.find(quote, q1 + 1);
        if (q2 == std::string::npos)
            break;
        std::string idStr = msg_string.substr(q1 + 1, q2 - q1 - 1);

        // try parse integer id
        int id = 0;
        try { id = std::stoi(idStr); }
        catch (...)
        {
            i = q2 + 1;
            continue;
        }

        // find `"class"` (allow single or double quoted)
        size_t classKey = msg_string.find("\"class\"", q2);
        if (classKey == std::string::npos)
            classKey = msg_string.find("'class'", q2);
        if (classKey == std::string::npos)
            break;

        // find colon after "class"
        size_t colon = msg_string.find(':', classKey);
        if (colon == std::string::npos)
            break;

        // find start quote of the class value
        size_t vq1 = msg_string.find_first_of("\"'", colon + 1);
        if (vq1 == std::string::npos)
            break;
        char vquote = msg_string[vq1];

        // find end quote of the class value
        size_t vq2 = msg_string.find(vquote, vq1 + 1);
        if (vq2 == std::string::npos)
            break;
        std::string cls = msg_string.substr(vq1 + 1, vq2 - vq1 - 1);

        // store mapping: class name -> id
        out[cls] = id;

        // advance i past this object
        i = vq2 + 1;
    }

    return out;
}

void CloudBuild::initSubscritions()
{
    const std::string params_prefix = "TopicName.Sensor.Camera";
    // this->declare_parameters<std::string>(params_prefix, {{"Mid.Depth", "/sensors/camera/mid/depth"},
    //                                                       {"Mid.Semantics", "/sensors/camera/mid/semantics"}});
    this->declare_parameters<std::string>(params_prefix, {{"Mid.Depth", "/forkheel_camera/depth"},
                                                          {"Mid.Semantics", "/forkheel_camera/semantic_segmentation"}});

    std::map<std::string, std::string> sensor_topics;
    if (!this->get_parameters<std::string>(params_prefix, sensor_topics))
    {
        RCLCPP_FATAL(get_logger(), "Failed to get parameters, sensor topic names!");
    }

    mDepthSub.subscribe(this, sensor_topics["Mid.Depth"]);
    mSemanticSub.subscribe(this, sensor_topics["Mid.Semantics"]);

    mSynchronizer = std::make_unique<message_filters::Synchronizer<SyncPolicy>>(SyncPolicy(2), mDepthSub, mSemanticSub);

    // 设置更小的时间容差（单位：秒）
    mSynchronizer->setMaxIntervalDuration(rclcpp::Duration(0, 1000000)); // 10ms 容差
    mSynchronizer->registerCallback(std::bind(&CloudBuild::imgsHandler, this, std::placeholders::_1, std::placeholders::_2));

    mSemanticLabelsSub = this->create_subscription<StrMsg>("semantic_labels", rclcpp::SensorDataQoS().best_effort(),
                std::bind(&CloudBuild::semanticLabelsHandler, this, std::placeholders::_1));
}

void CloudBuild::initPublishers()
{
    const std::string params_prefix = "TopicName.Perception";
    this->declare_parameters<std::string>(params_prefix, {{"SemanticCloud", "/perception/semantic_cloud"},
                                                          {"ColoredCloud", "/perception/colored_cloud"}});

    std::map<std::string, std::string> cloud_topics;
    if (!this->get_parameters<std::string>(params_prefix, cloud_topics))
    {
        RCLCPP_FATAL(get_logger(), "Failed to get parameters, cloud topic names!");
    }

    mSemanticCloudPub = create_publisher<sensor_msgs::msg::PointCloud2>(cloud_topics["SemanticCloud"], rclcpp::SensorDataQoS().best_effort());
    mColoredCloudPub = create_publisher<sensor_msgs::msg::PointCloud2>(cloud_topics["ColoredCloud"], rclcpp::SensorDataQoS().best_effort());
}

void CloudBuild::imgsHandler(const ImgMsg::ConstSharedPtr& depth_msg, const ImgMsg::ConstSharedPtr& semantics_msg) const
{
    // geometry_msgs::msg::TransformStamped T_body2fork;
    Eigen::Isometry3f T_body2fork;
    try
    {
        // This returns the pose of 'fork' in 'body' coordinates
        geometry_msgs::msg::TransformStamped tf_body2fork = mTfBuffer->lookupTransform("LOLA", "fork", tf2::TimePointZero);
        T_body2fork = tf2::transformToEigen(tf_body2fork).cast<float>();
    }
    catch (const tf2::TransformException& ex)
    {
        RCLCPP_ERROR(this->get_logger(), "Could not transform fork to body: %s", ex.what());
        return;
    }

    const auto depth_ptr = cv_bridge::toCvShare(depth_msg, sensor_msgs::image_encodings::TYPE_32FC1);
    const auto semantics_ptr = cv_bridge::toCvShare(semantics_msg, sensor_msgs::image_encodings::TYPE_32SC1);

    const cv::Mat& semantic_image = semantics_ptr->image;
    const cv::Mat& depth_image = depth_ptr->image;

    int pallet_label{}, ramp_label{}, trailer_label{}, goods_label{};
    auto it = mSemanticLabels.find("pallet");
    if(it!=mSemanticLabels.end())
    {
        pallet_label = it->second;
    }
    it = mSemanticLabels.find("ramp");
    if(it!=mSemanticLabels.end())
    {
        ramp_label = it->second;
    }
    it = mSemanticLabels.find("trailer");
    if(it!=mSemanticLabels.end())
    {
        trailer_label = it->second;
    }
    it = mSemanticLabels.find("goods");
    if(it!=mSemanticLabels.end())
    {
        goods_label = it->second;
    }

    SemanticCloud semantic_cloud_camera;   // This is the semantic cloud in the camera coordinate system.
    constexpr int skip_step = 3;
    for (int v = 0; v < depth_image.rows; v += skip_step)
    {
        const auto* depth_ptr = depth_image.ptr<float>(v);
        const auto* label_ptr = semantic_image.ptr<int>(v);

        for (int u = 0; u < depth_image.cols; u += skip_step)
        {
            if (const float depth = depth_ptr[u]; depth > 0.1f && depth < depth_threshold)
            {
                int label{};
                if (label_ptr[u] == 0)
                {
                    label = 0;
                }
                else if (label_ptr[u] == pallet_label)
                {
                    label = 1;
                }
                else if (label_ptr[u] == ramp_label)
                {
                    label = 2;
                }
                else if (label_ptr[u] == trailer_label)
                {
                    label = 3;
                }
                else if (label_ptr[u] == goods_label)
                {
                    label = 4;
                }
                else
                {
                    label = 0;
                }

                const float x = (static_cast<float>(u) - cx) * depth * fx_inv;
                const float y = (static_cast<float>(v) - cy) * depth * fy_inv;
                semantic_cloud_camera.emplace_back(depth, -x, -y, label);
            }
        }
    }

    if (semantic_cloud_camera.size() < 10)
    {
        return;
    }

    SemanticCloud semantic_cloud_base;   // This is the semantic cloud in the base link coordinate system.
    const Eigen::Isometry3f T_body2camera = T_body2fork * mT_fork2camera;
    pcl::transformPointCloud(semantic_cloud_camera, semantic_cloud_base, T_body2camera);

    ColoredCloud colored_cloud;
    getCloud(semantic_cloud_base, colored_cloud);

    sensor_msgs::msg::PointCloud2 semantic_cloud_msg, colored_cloud_msg;
    pcl::toROSMsg(semantic_cloud_base, semantic_cloud_msg);
    pcl::toROSMsg(colored_cloud, colored_cloud_msg);

    semantic_cloud_msg.header.stamp = this->now();
    semantic_cloud_msg.header.frame_id = global_frame_id;
    colored_cloud_msg.header.stamp = this->now();
    colored_cloud_msg.header.frame_id = global_frame_id;

    mSemanticCloudPub->publish(semantic_cloud_msg);
    mColoredCloudPub->publish(colored_cloud_msg);
}

void CloudBuild::semanticLabelsHandler(const StrMsg::ConstSharedPtr& semantic_labels_msg)
{
    const std::unordered_map<std::string, int> semantic_labels = parseSemanticLabels(semantic_labels_msg);
    auto it = semantic_labels.find("pallet");
    if(it != semantic_labels.end())
    {
        const int pallet_label = it->second;
        mSemanticLabels["pallet"] = pallet_label;
    }

    it = semantic_labels.find("ramp");
    if(it != semantic_labels.end())
    {
        const int ramp_label = it->second;
        mSemanticLabels["ramp"] = ramp_label;
    }

    it = semantic_labels.find("goods");
    if(it != semantic_labels.end())
    {
        const int goods_label = it->second;
        mSemanticLabels["ramp"] = goods_label;
    }

    it = semantic_labels.find("trailer");
    if(it != semantic_labels.end())
    {
        const int trailer_label = it->second;
        mSemanticLabels["trailer"] = trailer_label;
    }
}
