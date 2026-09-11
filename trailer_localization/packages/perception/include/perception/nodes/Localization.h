#pragma once
#include <pcl/point_cloud.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/twist_stamped.hpp>
#include <nav_msgs/msg/path.hpp>
#include <pcl/point_types.h>
#include <tf2_eigen/tf2_eigen.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <vector>
#include "perception/GICP/SparsityAwareGICP.h"
#include "perception/types/common.hpp"

class Localization : public rclcpp::Node
{
    static constexpr float MAP_RESOLUTION = 0.1f;
public:
    explicit Localization(const std::string& node_name) : Node(node_name), mTfBuffer(this->get_clock()), mTfListener(mTfBuffer)
    {
        /* Lookup transform */
        // while (rclcpp::ok())
        // {
        //     try
        //     {
        //         T_truck2lidar = tf2::transformToEigen(mTfBuffer.lookupTransform("LOLA", "JT128", tf2::TimePointZero)).cast<float>();
        //         break;
        //     }
        //     catch (const tf2::TransformException& ex)
        //     {
        //         RCLCPP_ERROR(this->get_logger(), "Could not transform fork to body: %s", ex.what());
        //     }
        // }

        SparsityAwareGICP::Config config{};
        config.voxel_size = MAP_RESOLUTION * 6;
        config.max_fitness_score = 0.1f;  // 平均意义下的点位误差尺度 10 cm
        mGicp.setConfig(config);

        mIntensityThreshold = static_cast<float>(declare_parameter<double>("intensity_threshold", -1.0));
        mIntensityKeepRatio = std::clamp(static_cast<float>(declare_parameter<double>("intensity_keep_ratio", 0.9)), 0.01f, 1.0f);

        initSubscribers();
        initPublisher();
        mWorker = std::thread(&Localization::workerLoop, this);
        RCLCPP_INFO(get_logger(), "The node has been activated.");
    }

    ~Localization() override
    {
        {
            std::lock_guard lock(mScanBufferMutex);
            mIsShutdown = true;
        }
        mTrigger.notify_one();
        if (mWorker.joinable())
        {
            mWorker.join();
        }
        RCLCPP_INFO(get_logger(), "This node has been shutdown.");
    }

private:
    /* Subscribers */
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr mLidarScanSub;
    rclcpp::Subscription<geometry_msgs::msg::TwistStamped>::SharedPtr mWheelOdomSub;

    /* Publishers */
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr mProcessedScanVisPub, mVoxelMapPub;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr mBasePosePub;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr mBasePosePathPub;

    /* Data Buffers */
    std::queue<sensor_msgs::msg::PointCloud2> mScanBuffer;

    /* Multi-thread utilities */
    bool mIsShutdown{false};
    std::thread mWorker;
    std::mutex mScanBufferMutex;
    // std::mutex mFusionMutex;
    std::condition_variable mTrigger;

    /* TF tree utilities */
    tf2_ros::Buffer mTfBuffer;
    tf2_ros::TransformListener mTfListener;
    Eigen::Isometry3f T_truck2lidar;

    /* Trailer voxel map and estimated pose */
    pcl::PointCloud<pcl::PointXYZ>::Ptr mTrailerVoxelMap;
    Eigen::Isometry3f mBasePose{Eigen::Isometry3f::Identity()};   // Pose of the truck
    nav_msgs::msg::Path mBasePosePath;
    ROI mTrailerRoi{};
    SparsityAwareGICP mGicp;
    std::vector<double> mGicpDurationsMs;
    std::size_t mGicpDurationCount{0};
    float mIntensityThreshold{-1.0f};
    float mIntensityKeepRatio{0.6f};
    bool mIntensityAnalyzed{false};

    void initSubscribers()
    {
        // mLidarScanSub = create_subscription<sensor_msgs::msg::PointCloud2>("/iv_points", rclcpp::SensorDataQoS(),
        mLidarScanSub = create_subscription<sensor_msgs::msg::PointCloud2>("/hesai/pandar", 10,
            [this](const sensor_msgs::msg::PointCloud2::ConstSharedPtr& scan_msg)
                {
                    {
                       std::lock_guard lock(mScanBufferMutex);
                       while (!mScanBuffer.empty())
                       {
                           mScanBuffer.pop();
                       }
                       mScanBuffer.push(*scan_msg);
                    }
                    mTrigger.notify_one();
                });
    }

    void initPublisher()
    {
        mProcessedScanVisPub = create_publisher<sensor_msgs::msg::PointCloud2>("/scan_vis", rclcpp::SensorDataQoS());
        mVoxelMapPub = create_publisher<sensor_msgs::msg::PointCloud2>("/voxel_map", rclcpp::SensorDataQoS());
        mBasePosePub = create_publisher<geometry_msgs::msg::PoseStamped>("/base_pose", rclcpp::SensorDataQoS());
        mBasePosePathPub = create_publisher<nav_msgs::msg::Path>("/base_pose_path", rclcpp::SensorDataQoS());
    }

    /* Transform LiDAR scan to base truck frame */
    // void transformLidarScan(const pcl::PointCloud<pcl::PointXYZ>& src_scan, const rclcpp::Time& stamp,
    //                         pcl::PointCloud<pcl::PointXYZ>& dst_scan) const
    // {
    //     const auto T_truck2lidar = tf2::transformToEigen(mTfBuffer.lookupTransform("LOLA", "JT128",
    //                                                      stamp, tf2::durationFromSec(0.05))).cast<float>();
    //     pcl::transformPointCloud(src_scan, dst_scan, T_truck2lidar);
    // }

    void makeTemplate(const pcl::PointCloud<pcl::PointXYZ>& src_scan);

    bool alignICP(const pcl::PointCloud<pcl::PointXYZ>::Ptr& current_scan);

    void recordGicpDuration(double duration_ms);

    void updateVoxelMap(const pcl::PointCloud<pcl::PointXYZ>& scan_in_truck);

    void workerLoop();
};