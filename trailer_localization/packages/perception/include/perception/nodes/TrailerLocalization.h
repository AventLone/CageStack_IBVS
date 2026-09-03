#pragma once
#include <pcl/point_cloud.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/twist_stamped.hpp>
#include <pcl/point_types.h>
#include <tf2_eigen/tf2_eigen.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <algorithm>
#include <numeric>
#include <vector>
#include "perception/LIO/SparsityAwareGICP.hpp"
// #include "perception/kalman_filter/EKF.hpp"
#include "perception/types/common.hpp"

class TrailerLocalization : public rclcpp::Node
{
public:
    explicit TrailerLocalization(const std::string& node_name) : Node(node_name), mTfBuffer(this->get_clock()), mTfListener(mTfBuffer)
    {
        /* Lookup transform */
        while (rclcpp::ok())
        {
            try
            {
                T_truck2lidar = tf2::transformToEigen(mTfBuffer.lookupTransform("LOLA", "JT128", tf2::TimePointZero)).cast<float>();
                break;
            }
            catch (const tf2::TransformException& ex)
            {
                RCLCPP_ERROR(this->get_logger(), "Could not transform fork to body: %s", ex.what());
            }
        }

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
    // rclcpp::Subscription<geometry_msgs::msg::TwistStamped>::SharedPtr mWheelOdomSub;

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
    // std::mutex mFusionMutex;
    std::condition_variable mTrigger;

    /* TF tree utilities */
    tf2_ros::Buffer mTfBuffer;
    tf2_ros::TransformListener mTfListener;
    Eigen::Isometry3f T_truck2lidar;

    /* Trailer voxel map and estimated pose */
    pcl::PointCloud<pcl::PointXYZ>::Ptr mTrailerVoxelMap;
    Eigen::Isometry3f mTrailerPose{Eigen::Isometry3f::Identity()};
    ROI mTrailerRoi{};
    // perception::kalman::AckermannLidarEKF mFusionFilter;
    perception::lio::SparsityAwareGICP mGicp;
    std::vector<double> mGicpDurationsMs;
    std::size_t mGicpDurationCount{0};
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

        // mWheelOdomSub = create_subscription<geometry_msgs::msg::TwistStamped>("/wheel_odometry", rclcpp::SensorDataQoS(),
        //     [this](const geometry_msgs::msg::TwistStamped::ConstSharedPtr& odometry_msg)
        //     {
        //         perception::kalman::AckermannMeasurement measurement;
        //         // Temporary topic contract: linear.x is wheel speed [m/s], angular.z is steering angle [rad].
        //         measurement.speed = static_cast<float>(odometry_msg->twist.linear.x);
        //         measurement.steering_angle = static_cast<float>(odometry_msg->twist.angular.z);
        //
        //         const double odometry_time = rclcpp::Time(odometry_msg->header.stamp).seconds();
        //         std::lock_guard lock(mFusionMutex);
        //         if (!mFusionFilter.initialized())
        //         {
        //             return;
        //         }
        //         if (mFilterTime >= 0.0 && odometry_time > mFilterTime)
        //         {
        //             mFusionFilter.predict(measurement, static_cast<float>(odometry_time - mFilterTime));
        //             mTrailerPose = mFusionFilter.pose().inverse();
        //         }
        //         mFilterTime = odometry_time;
        //
        //         geometry_msgs::msg::PoseStamped pose_msg;
        //         pose_msg.header = odometry_msg->header;
        //         pose_msg.header.frame_id = "LOLA";
        //         pose_msg.pose = tf2::toMsg(Eigen::Isometry3d(mTrailerPose.cast<double>()));
        //         mTrailerPosePub->publish(pose_msg);
        //     });
    }

    void initPublisher()
    {
        mProcessedScanVisPub = create_publisher<sensor_msgs::msg::PointCloud2>("/scan_vis", rclcpp::SensorDataQoS());
        mTrailerPosePub = create_publisher<geometry_msgs::msg::PoseStamped>("/trailer_pose", rclcpp::SensorDataQoS());
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

    bool alignICP(const pcl::PointCloud<pcl::PointXYZ>::Ptr& current_scan, Eigen::Isometry3f& out_pose);

    void recordGicpDuration(double duration_ms);

    void updateVoxelMap(const pcl::PointCloud<pcl::PointXYZ>& scan_in_truck);

    // bool updateLio(const Eigen::Isometry3f& lidar_pose, const rclcpp::Time& scan_stamp);

    // void publishLioPose(const rclcpp::Time& stamp);

    void workerLoop();
};