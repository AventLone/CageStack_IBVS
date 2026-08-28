#pragma once
#include <pcl/point_cloud.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <pcl/point_types.h>
#include <tf2_eigen/tf2_eigen.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <pcl/common/transforms.h>
#include <pcl/registration/icp.h>
// #include <pcl/registration/gicp.h>
// #include <pcl/registration/transformation_estimation_2D.h>
// #include <pcl/registration/impl/icp.hpp>
// #include <pcl/registration/impl/gicp.hpp>
#include <pcl/filters/voxel_grid.h>
#include "perception/types/common.hpp"

class TrailerLocalization : public rclcpp::Node
{
public:
    explicit TrailerLocalization(const std::string& node_name) : Node(node_name), mTfBuffer(this->get_clock()), mTfListener(mTfBuffer)
    {
        initSubscribers();
        initPublisher();
        mWorker = std::thread(&TrailerLocalization::workerLoop, this);
        RCLCPP_INFO(get_logger(), "The node has been activated.");
    }

    ~TrailerLocalization() override
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
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr mImuSub;

    /* Publishers */
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr mProcessedScanVisPub;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr mTrailerPosePub;

    /* Data Buffers */
    std::queue<sensor_msgs::msg::PointCloud2> mScanBuffer;

    /* Multi-thread utilities */
    bool mIsShutdown{false};
    std::thread mWorker;
    std::mutex mScanBufferMutex;
    std::condition_variable mTrigger;

    /* TF tree utilities */
    tf2_ros::Buffer mTfBuffer;
    tf2_ros::TransformListener mTfListener;

    /* ICP template, voxel map and estimated pose */
    pcl::PointCloud<pcl::PointXYZ>::Ptr mTrailerTemplate;
    pcl::PointCloud<pcl::PointXYZ>::Ptr mTrailerVoxelMap;
    Eigen::Isometry3f mTrailerPose{Eigen::Isometry3f::Identity()};
    ROI mTrailerRoi{};

    void initSubscribers()
    {
        mLidarScanSub = create_subscription<sensor_msgs::msg::PointCloud2>("/sim_scan", rclcpp::SensorDataQoS(),
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
        mTrailerPosePub = create_publisher<geometry_msgs::msg::PoseStamped>("/trailer_pose", rclcpp::SensorDataQoS());
    }

    /* Transform LiDAR scan to base truck frame */
    void transformLidarScan(const pcl::PointCloud<pcl::PointXYZ>& src_scan,
                            const rclcpp::Time& stamp,
                            pcl::PointCloud<pcl::PointXYZ>& dst_scan) const
    {
        const auto T_truck2lidar = tf2::transformToEigen(mTfBuffer.lookupTransform("LOLA", "JT128",
                                                         stamp, tf2::durationFromSec(0.05))).cast<float>();
        pcl::transformPointCloud(src_scan, dst_scan, T_truck2lidar);
    }

    void makeTemplate(const pcl::PointCloud<pcl::PointXYZ>& src_scan);

    bool alignICP(const pcl::PointCloud<pcl::PointXYZ>::Ptr& current_scan, Eigen::Isometry3f& out_pose);

    void updateVoxelMap(const pcl::PointCloud<pcl::PointXYZ>& scan_in_truck);

    void workerLoop();
};