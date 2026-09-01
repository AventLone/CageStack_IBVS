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
#include <pcl/filters/voxel_grid.h>
#include "perception/LIO/SparsityAwareGICP.hpp"
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
    // rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr mImuSub;

    /* Publishers */
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr mProcessedScanVisPub;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr mTrailerPosePub;

    /* Data Buffers */
    std::queue<sensor_msgs::msg::PointCloud2> mScanBuffer;
    // std::deque<std::pair<double, IMUSample>> mImuBuffer;

    /* Multi-thread utilities */
    bool mIsShutdown{false};
    std::thread mWorker;
    std::mutex mScanBufferMutex;
    // std::mutex mImuBufferMutex;
    // std::mutex mLioMutex;
    std::condition_variable mTrigger;

    /* TF tree utilities */
    tf2_ros::Buffer mTfBuffer;
    tf2_ros::TransformListener mTfListener;

    /* ICP template, voxel map and estimated pose */
    pcl::PointCloud<pcl::PointXYZ>::Ptr mTrailerTemplate;
    pcl::PointCloud<pcl::PointXYZ>::Ptr mTrailerVoxelMap;
    Eigen::Isometry3f mTrailerPose{Eigen::Isometry3f::Identity()};
    ROI mTrailerRoi{};
    // IteratedESKF mLioFilter;
    perception::lio::SparsityAwareGICP mGicp;
    // double mFilterTime{-1.0};

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

        // mImuSub = create_subscription<sensor_msgs::msg::Imu>("/imu", rclcpp::SensorDataQoS(),
        //     [this](const sensor_msgs::msg::Imu::ConstSharedPtr& imu_msg)
        //     {
        //         IMUSample sample;
        //         sample.gyro = Eigen::Vector3f(
        //             static_cast<float>(imu_msg->angular_velocity.x),
        //             static_cast<float>(imu_msg->angular_velocity.y),
        //             static_cast<float>(imu_msg->angular_velocity.z));
        //         sample.acceleration = Eigen::Vector3f(
        //             static_cast<float>(imu_msg->linear_acceleration.x),
        //             static_cast<float>(imu_msg->linear_acceleration.y),
        //             static_cast<float>(imu_msg->linear_acceleration.z));

        //         const rclcpp::Time stamp(imu_msg->header.stamp);
        //         const double imu_time = stamp.seconds();
        //         {
        //             std::lock_guard lock(mImuBufferMutex);
        //             mImuBuffer.emplace_back(imu_time, sample);
        //             while (mImuBuffer.size() > 4000)
        //             {
        //                 mImuBuffer.pop_front();
        //             }
        //         }

        //         {
        //             std::lock_guard lock(mLioMutex);
        //             if (mFilterTime < 0.0)
        //             {
        //                 mFilterTime = imu_time;
        //             }
        //             else if (imu_time > mFilterTime)
        //             {
        //                 mLioFilter.predict(sample, static_cast<float>(imu_time - mFilterTime));
        //                 mFilterTime = imu_time;
        //             }
        //             else
        //             {
        //                 return;
        //             }
        //         }
        //         publishLioPose(stamp);
        //     });

    }

    void initPublisher()
    {
        mProcessedScanVisPub = create_publisher<sensor_msgs::msg::PointCloud2>("/scan_vis", rclcpp::SensorDataQoS());
        mTrailerPosePub = create_publisher<geometry_msgs::msg::PoseStamped>("/trailer_pose", rclcpp::SensorDataQoS());
    }

    /* Transform LiDAR scan to base truck frame */
    void transformLidarScan(const pcl::PointCloud<pcl::PointXYZ>& src_scan, const rclcpp::Time& stamp,
                            pcl::PointCloud<pcl::PointXYZ>& dst_scan) const
    {
        const auto T_truck2lidar = tf2::transformToEigen(mTfBuffer.lookupTransform("LOLA", "JT128",
                                                         stamp, tf2::durationFromSec(0.05))).cast<float>();
        pcl::transformPointCloud(src_scan, dst_scan, T_truck2lidar);
    }

    void makeTemplate(const pcl::PointCloud<pcl::PointXYZ>& src_scan);

    bool alignICP(const pcl::PointCloud<pcl::PointXYZ>::Ptr& current_scan, Eigen::Isometry3f& out_pose) const;

    void updateVoxelMap(const pcl::PointCloud<pcl::PointXYZ>& scan_in_truck);

    // bool updateLio(const Eigen::Isometry3f& lidar_pose, const rclcpp::Time& scan_stamp);

    // void publishLioPose(const rclcpp::Time& stamp);

    void workerLoop();
};