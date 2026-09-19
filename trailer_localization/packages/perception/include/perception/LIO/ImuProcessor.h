#pragma once
// #include <algorithm>
// #include <stdexcept>
// #include <vector>
#include <sophus/se3.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <rclcpp/time.hpp>
#include "perception/types/imu.hpp"
#include "perception/types/stamped_cloud.hpp"

namespace lio
{
static ImuData fromMsg(const sensor_msgs::msg::Imu& imu_msg)
{
    return ImuData{.timestamp = rclcpp::Time(imu_msg.header.stamp).seconds(),
                   .gyro = Eigen::Vector3d(imu_msg.angular_velocity.x,
                                           imu_msg.angular_velocity.y,
                                           imu_msg.angular_velocity.z),
                   .accel = Eigen::Vector3d(imu_msg.linear_acceleration.x,
                                            imu_msg.linear_acceleration.y,
                                            imu_msg.linear_acceleration.z)};
}

class ImuProcessor
{
public:
    /**
     * T_il: LiDAR's pose in IMU frame
     * p_i = T_il * p_l
     */
    explicit ImuProcessor(const Sophus::SE3d& T_il) : mT_il(T_il)
    {
    }

    /**
     * Process one LiDAR scan.
     *
     * 1. Integrate IMU from scan_begin to scan_end.
     * 2. Save the IMU trajectory during this scan.
     * 3. Deskew all LiDAR points to scan_end.
     * 4. Update state to scan_end.
     *
     * Requirement:
     *   state.timestamp == scan_begin
     * IMU data must cover:
     *   [scan_begin, scan_end]
     */
    void process(const std::vector<ImuData>& imu_data, StampedCloud& stamped_cloud, ImuState& state)
    {
        if (imu_data.size() < 2)
        {
            throw std::runtime_error("Insufficient IMU measurements.");
        }

        if (constexpr double time_epsilon = 1e-6;
            std::abs(state.timestamp - stamped_cloud.begin_time) > time_epsilon)
        {
            throw std::runtime_error("IMU state timestamp must equal scan_begin.");
        }

        if (imu_data.front().timestamp > stamped_cloud.begin_time || imu_data.back().timestamp < stamped_cloud.end_time)
        {
            throw std::runtime_error("IMU measurements do not cover LiDAR scan.");
        }

        const auto samples = buildImuSequence(imu_data, stamped_cloud.begin_time, stamped_cloud.end_time);
        propagate(samples, state);
        deskew(stamped_cloud);
    }

    static std::vector<ImuData> buildImuSequence(const std::vector<ImuData>& imu_data, double begin, double end);

    /**
     * Propagate nominal IMU state.
     * This only propagates: R, p, v
     * Bias is assumed constant during one scan.
     * Bias covariance/random walk should be handled by the IESKF.
     */
    void propagate(const std::vector<ImuData>& imu_data, ImuState& state);


private:
    struct PoseState
    {
        double timestamp = 0.0;
        Sophus::SO3d R_wi;
        Eigen::Vector3d p_wi = Eigen::Vector3d::Zero();
        Eigen::Vector3d v_wi = Eigen::Vector3d::Zero();
    };

    const Sophus::SE3d mT_il;   // LiDAR -> IMU extrinsic.

    std::vector<PoseState> mTrajectory;    // IMU trajectory of current LiDAR scan.

    static Sophus::SE3d toSE3(const PoseState& state)
    {
        return {state.R_wi, state.p_wi};
    }

    /**
     * Linear interpolation of raw IMU measurements.
     */
    static ImuData interpolateImu(const ImuData& imu0, const ImuData& imu1, const double timestamp)
    {
        const double dt = imu1.timestamp - imu0.timestamp;

        if (dt <= 0.0)
        {
            return imu0;
        }

        const double alpha = std::clamp((timestamp - imu0.timestamp) / dt, 0.0, 1.0);

        ImuData result;
        result.timestamp = timestamp;
        result.gyro = (1.0 - alpha) * imu0.gyro + alpha * imu1.gyro;
        result.accel = (1.0 - alpha) * imu0.accel + alpha * imu1.accel;

        return result;
    }

    static ImuData imuAt(const std::vector<ImuData>& imu_data, const double timestamp)
    {
        const auto iter = std::lower_bound(imu_data.begin(), imu_data.end(), timestamp, [](const ImuData& imu, const double time)
                {
                    return imu.timestamp < time;
                });

        if (iter == imu_data.begin())
        {
            return *iter;
        }

        if (iter == imu_data.end())
        {
            return imu_data.back();
        }

        if (std::abs(iter->timestamp - timestamp) < 1e-9)
        {
            return *iter;
        }

        return interpolateImu(*(iter - 1), *iter, timestamp);
    }

    /**
     * Deskew LiDAR points to scan_end.
     * For a point captured at time t:
     *   p_L_end =
     *       T_WL(end)^-1
     *       T_WL(t)
     *       p_L(t)
     * where: T_WL = T_WI * T_IL
     */
    void deskew(StampedCloud& stamped_cloud) const;

    /**
     * Get interpolated IMU pose at arbitrary timestamp.
     */
    Sophus::SE3d poseAt(double timestamp) const;
};
}  // namespace lio