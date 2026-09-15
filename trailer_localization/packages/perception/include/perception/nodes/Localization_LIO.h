#pragma once
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/path.hpp>
#include "perception/LIO/CloudTiming.hpp"
#include "perception/LIO/LidarMeasurement.h"

class Localization_LIO : public rclcpp::Node
{
public:
    explicit Localization_LIO(const std::string& node_name);
    ~Localization_LIO() override;

private:
    // Callbacks only enqueue; this single worker owns the filter and map.
    std::mutex mBufferMutex;
    std::condition_variable mTrigger;
    std::deque<sensor_msgs::msg::PointCloud2::ConstSharedPtr> mScanBuffer;
    std::deque<lio::ImuData> mImuBuffer;
    bool mIsShutdown{false};
    double mLastImuTime{-std::numeric_limits<double>::infinity()};
    double mImuTimeOffset{0.0};
    std::thread mLidarWorker;

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr mLidarScanSub;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr mImuSub;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr mProcessedScanVisPub, mVoxelMapPub;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr mBasePosePub;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr mBasePosePathPub;

    lio::CloudTimingConfig mTimingConfig;
    SparseVoxel::Config mMapConfig;
    std::unique_ptr<lio::ESKF> mEskf;
    std::unique_ptr<lio::LidarMeasurement> mLidarMeasurement;
    std::unique_ptr<SparseVoxel> mMap;
    nav_msgs::msg::Path mBasePosePath;
    std::string mWorldFrame, mImuFrame, mLidarFrame;
    Sophus::SE3d mTi2b;  // Base -> IMU; T_WB = T_WI * T_IB.
    double mScanResolution{0.1};
    double mImuWaitTimeout{1.0};
    double mInitializationDuration{1.0};
    float mIntensityThreshold{-1.0f};
    float mIntensityKeepRatio{0.6f};
    bool mIntensityAnalyzed{false};

    void lidarWorkerLoop();
    bool collectImu(double scan_begin, double scan_end, std::vector<lio::ImuData>& imu);
    void publish(const pcl::PointCloud<pcl::PointXYZ>& scan, double timestamp);
};
