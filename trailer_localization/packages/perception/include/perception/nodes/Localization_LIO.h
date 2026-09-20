#pragma once
#include <pcl/point_cloud.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/path.hpp>
#include <pcl/point_types.h>
#include <tf2_eigen/tf2_eigen.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include "perception/GICP/GICP.h"
#include "perception/types/common.hpp"
#include "perception/LIO/ESKF.h"
#include "perception/LIO/ImuProcessor.h"

class Localization_LIO : public rclcpp::Node
{
    static constexpr float MAP_RESOLUTION = 0.1f;
public:
    explicit Localization_LIO(const std::string& node_name) : Node(node_name), mTfBuffer(this->get_clock()), mTfListener(mTfBuffer)
    {
        /* Lookup transform */
        Sophus::SE3d T_il;
        while (rclcpp::ok())
        {
            try
            {
                // const auto transform = tf2::transformToEigen(mTfBuffer.lookupTransform("LOLA", "JT128", tf2::TimePointZero));
                const auto transform = tf2::transformToEigen(mTfBuffer.lookupTransform("imu", "PandarXT-32", tf2::TimePointZero));
                T_il = Sophus::SE3d(transform.rotation(), transform.translation());
                mT_il = transform.cast<float>();
                break;
            }
            catch (const tf2::TransformException& ex)
            {
                RCLCPP_ERROR(this->get_logger(), "Could not transform fork to body: %s", ex.what());
            }
        }

        mImuProcessor = std::make_unique<lio::ImuProcessor>(T_il);

        GICP::Config config{};
        config.voxel_size = MAP_RESOLUTION * 5;
        config.max_fitness_score = 0.1f;  // 平均意义下的点位误差尺度 10 cm
        mGicp.setConfig(config);

        mIntensityThreshold = static_cast<float>(declare_parameter<double>("intensity_threshold", -1.0));
        mIntensityKeepRatio = std::clamp(static_cast<float>(declare_parameter<double>("intensity_keep_ratio", 0.6)), 0.01f, 1.0f);

        initSubscribers();
        initPublisher();
        mLidarWorker = std::thread(&Localization_LIO::lidarWorkerLoop, this);
        RCLCPP_INFO(get_logger(), "The node has been activated.");
    }

    ~Localization_LIO() override
    {
        {
            std::lock_guard lock(mScanBufferMutex);
            mIsShutdown = true;
        }
        mLidarTrigger.notify_one();
        if (mLidarWorker.joinable())
        {
            mLidarWorker.join();
        }
        RCLCPP_INFO(get_logger(), "This node has been shutdown.");
    }

private:
    /* Subscribers */
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr mLidarScanSub;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr mImuSub;

    /* Publishers */
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr mProcessedScanVisPub, mVoxelMapPub;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr mBasePosePub;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr mBasePosePathPub;

    /* Data Buffers */
    std::deque<sensor_msgs::msg::PointCloud2::ConstSharedPtr> mScanBuffer;
    std::deque<lio::ImuData> mImuBuffer;

    /* Multi-thread utilities */
    bool mIsShutdown{false};
    std::thread mLidarWorker, mImuWorker;
    std::mutex mScanBufferMutex, mImuBufferMutex;
    std::condition_variable mLidarTrigger, mImuTrigger;

    /* TF tree utilities */
    tf2_ros::Buffer mTfBuffer;
    tf2_ros::TransformListener mTfListener;
    Eigen::Isometry3f T_truck2lidar;

    /* Trailer voxel map and estimated pose */
    pcl::PointCloud<pcl::PointXYZ>::Ptr mMap;
    // Eigen::Isometry3d mBasePose{Eigen::Isometry3d::Identity()};   // Pose of the truck
    nav_msgs::msg::Path mBasePosePath;

    /* LIO */
    Eigen::Isometry3f mT_il;   // Extrinsic from LiDAR to IMU
    GICP mGicp;
    lio::ImuProcessor::Ptr mImuProcessor;
    lio::ESKF mESKF;
    float mIntensityThreshold{-1.0f};
    float mIntensityKeepRatio{0.6f};
    bool mInitialized{false};

    void initSubscribers()
    {
        // mLidarScanSub = create_subscription<sensor_msgs::msg::PointCloud2>("/iv_points", rclcpp::SensorDataQoS(),
        mLidarScanSub = create_subscription<sensor_msgs::msg::PointCloud2>("/hesai/pandar", 10,
        // mLidarScanSub = create_subscription<sensor_msgs::msg::PointCloud2>("/iv_points", 10,
            [this](sensor_msgs::msg::PointCloud2::ConstSharedPtr scan_msg)
                {
                    {
                        std::lock_guard lock(mScanBufferMutex);
                        if (mScanBuffer.size() > 4)
                        {
                            mScanBuffer.pop_front();
                        }
                        mScanBuffer.push_back(std::move(scan_msg));
                    }
                    mLidarTrigger.notify_one();
                });

        mImuSub = create_subscription<sensor_msgs::msg::Imu>("/alphasense/imu", rclcpp::SensorDataQoS().keep_last(2000),
            [this](const sensor_msgs::msg::Imu::ConstSharedPtr& msg)
                {
                    const lio::ImuData imu{rclcpp::Time(msg->header.stamp).seconds(),
                            Eigen::Vector3d(msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z),
                            Eigen::Vector3d(msg->linear_acceleration.x, msg->linear_acceleration.y, msg->linear_acceleration.z)};

                    std::lock_guard lock(mImuBufferMutex);
                    while (mImuBuffer.size() > 2000)
                    {
                        mImuBuffer.pop_front();
                    }
                    mImuBuffer.push_back(imu);
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

    pcl::PointCloud<pcl::PointXYZ>::Ptr denoiseAndDownsample(const pcl::PointCloud<pcl::PointXYZI>& src) const;
    pcl::PointCloud<pcl::PointXYZ>::Ptr denoiseAndDownsample(const StampedCloud& src) const;

    bool alignScanToMap(const pcl::PointCloud<pcl::PointXYZ>::Ptr& current_scan, double timestamp);

    void updateVoxelMap(const pcl::PointCloud<pcl::PointXYZ>& scan);

    void lidarWorkerLoop();
};