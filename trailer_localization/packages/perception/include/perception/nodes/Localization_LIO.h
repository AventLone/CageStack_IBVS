#pragma once
#include <condition_variable>
#include <deque>
#include <mutex>
#include <optional>
#include <thread>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/path.hpp>
#include "perception/GICP/GICP.h"
#include "perception/LIO/ImuProcessor.h"

class Localization_LIO : public rclcpp::Node
{
public:
    explicit Localization_LIO(const std::string& node_name);
    ~Localization_LIO() override;

private:
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr mLidarScanSub;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr mImuSub;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr mProcessedScanVisPub, mVoxelMapPub;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr mBasePosePub;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr mBasePosePathPub;

    // One lock protects sensor queues and the live filter. GICP runs outside it.
    std::mutex mMutex;
    std::condition_variable mTrigger;
    bool mShutdown{false};
    sensor_msgs::msg::PointCloud2::ConstSharedPtr mPendingScan;
    std::deque<lio::ImuData> mImuBuffer;
    std::optional<lio::ESKF> mInitialFilter, mLiveFilter;
    std::thread mLidarWorker;
    lio::ESKF::Config mFilterConfig;

    Sophus::SE3d mT_il, mT_ib; // LiDAR -> IMU, base -> IMU
    std::string mMapFrame;
    double mInitDuration;
    double mMapResolution;
    double mKeyframeTranslation, mKeyframeRotation;
    float mIntensityThreshold, mIntensityKeepRatio;
    double mSelfFilterHalfWidth;
    int mPathMaxPoses;
    bool mIntensityAnalyzed{false};
    lio::ESKF::MeasurementCov mLidarCov{lio::ESKF::MeasurementCov::Zero()};

    // LiDAR worker owns map, registration and scan-time path exclusively.
    GICP mGicp;
    pcl::PointCloud<pcl::PointXYZ>::Ptr mMap{new pcl::PointCloud<pcl::PointXYZ>};
    Sophus::SE3d mLastKeyframe;
    nav_msgs::msg::Path mBasePosePath;

    void onImu(const sensor_msgs::msg::Imu& msg);
    void lidarWorkerLoop();
    bool alignICP(const pcl::PointCloud<pcl::PointXYZ>& scan, lio::ESKF& filter);
    void updateVoxelMap(const pcl::PointCloud<pcl::PointXYZ>& scan, const Sophus::SE3d& T_wl);
    void publishScan(const pcl::PointCloud<pcl::PointXYZ>& scan, const lio::ESKF& filter);
    geometry_msgs::msg::PoseStamped basePose(const lio::ESKF& filter) const;
};
