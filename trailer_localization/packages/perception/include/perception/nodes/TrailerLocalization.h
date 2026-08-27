#pragma once
#include <pcl/point_cloud.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <pcl/point_types.h>
#include <tf2_eigen/tf2_eigen.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <pcl/common/transforms.h>

class TrailerLocalization : public rclcpp::Node
{
public:
    explicit TrailerLocalization(const std::string& node_name) : Node(node_name), mTfBuffer(this->get_clock()), mTfListener(mTfBuffer)
        // mTrailerTemplate(std::make_shared<pcl::PointCloud<pcl::PointXYZ>>())
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

    /* Publishers */
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr mProcessedScanVisPub;

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

    /* ICP template */
    pcl::PointCloud<pcl::PointXYZ>::Ptr mTrailerTemplate;

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

    void makeTemplate(const pcl::PointCloud<pcl::PointXYZ>::Ptr& src_scan);

    void workerLoop();
};