#pragma once
#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <vector>
#include <sophus/se3.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <rclcpp/time.hpp>
namespace lio
{
struct ImuData
{
    double timestamp = 0.0;

    // Raw IMU measurements.
    Eigen::Vector3d gyro = Eigen::Vector3d::Zero();   // rad/s
    Eigen::Vector3d accel = Eigen::Vector3d::Zero();  // m/s^2
};

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

struct PointXYZT
{
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
    double relative_time = 0.0;   // Time relative to scan begin [s].
};

/**
 * Nominal IMU state.
 *
 * Frame convention:
 *
 *   R_WI : IMU -> World
 *   p_WI : IMU position in World
 *   v_WI : IMU velocity in World
 *
 * IMU biases are part of the state because they should normally
 * be estimated by IESKF rather than treated as fixed parameters.
 */
struct ImuState
{
    double timestamp = 0.0;

    Sophus::SO3d R_WI;
    Eigen::Vector3d p_WI = Eigen::Vector3d::Zero();
    Eigen::Vector3d v_WI = Eigen::Vector3d::Zero();

    Eigen::Vector3d gyro_bias = Eigen::Vector3d::Zero();
    Eigen::Vector3d accel_bias = Eigen::Vector3d::Zero();

    Eigen::Vector3d gravity = Eigen::Vector3d(0.0, 0.0, -9.81);
};

class ImuProcessor
{
public:
    /**
     * T_IL:
     *   LiDAR -> IMU
     *   p_I = T_IL * p_L
     */
    explicit ImuProcessor(const Sophus::SE3d& T_IL) : mTi2l(T_IL)
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
    void process(const std::vector<ImuData>& imu_data, std::vector<PointXYZT>& points,
                 const double scan_begin, const double scan_end, ImuState& state)
    {
        if (imu_data.size() < 2)
        {
            throw std::runtime_error("Insufficient IMU measurements.");
        }

        if (constexpr double kTimeEpsilon = 1e-6;
            std::abs(state.timestamp - scan_begin) > kTimeEpsilon)
        {
            throw std::runtime_error("IMU state timestamp must equal scan_begin.");
        }

        if (imu_data.front().timestamp > scan_begin || imu_data.back().timestamp < scan_end)
        {
            throw std::runtime_error("IMU measurements do not cover LiDAR scan.");
        }

        const auto samples = buildImuSequence(imu_data, scan_begin, scan_end);
        propagate(samples, state);
        deskew(points, scan_begin, scan_end);
    }

    /**
     * Propagate nominal IMU state.
     * This only propagates: R, p, v
     * Bias is assumed constant during one scan.
     * Bias covariance/random walk should be handled by the IESKF.
     */
    void propagate(const std::vector<ImuData>& imu_data, ImuState& state)
    {
        if (imu_data.size() < 2)
        {
            return;
        }

        mTrajectory.clear();
        mTrajectory.reserve(imu_data.size());

        mTrajectory.push_back(
            PoseState{
                .timestamp = state.timestamp,
                .R_WI = state.R_WI,
                .p_WI = state.p_WI,
                .v_WI = state.v_WI,
            });

        for (std::size_t i = 0; i + 1 < imu_data.size(); ++i)
        {
            const ImuData& imu0 = imu_data[i];
            const ImuData& imu1 = imu_data[i + 1];

            const double dt = imu1.timestamp - imu0.timestamp;

            if (dt <= 0.0)
            {
                continue;
            }

            /*
             * Bias corrected angular velocity.
             * Midpoint integration:
             *   ω = (ω0 + ω1) / 2 - bg
             */
            const Eigen::Vector3d omega = 0.5 * (imu0.gyro + imu1.gyro) - state.gyro_bias;

            /*
             * Bias corrected specific force.
             * Accelerometer measures:
             *   f = R_IW (a_W - g_W)
             * therefore: a_W = R_WI * f + g_W
             */
            const Eigen::Vector3d specific_force = 0.5 * (imu0.accel + imu1.accel)- state.accel_bias;

            /*
             * Orientation at middle of the interval.
             *
             * This gives better acceleration integration
             * than simply using R_k.
             */
            const Sophus::SO3d R_mid = state.R_WI * Sophus::SO3d::exp(omega * (0.5 * dt));

            const Eigen::Vector3d accel_world = R_mid * specific_force + state.gravity;

            /*
             * Position / velocity integration.
             */
            state.p_WI += state.v_WI * dt + 0.5 * accel_world * dt * dt;
            state.v_WI += accel_world * dt;

            /*
             * Rotation integration.
             */
            state.R_WI = state.R_WI * Sophus::SO3d::exp(omega * dt);
            state.timestamp = imu1.timestamp;

            mTrajectory.push_back(
                PoseState{
                    .timestamp = state.timestamp,
                    .R_WI = state.R_WI,
                    .p_WI = state.p_WI,
                    .v_WI = state.v_WI,
                });
        }
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
    void deskew(std::vector<PointXYZT>& points, const double scan_begin, const double scan_end) const
    {
        if (mTrajectory.empty())
        {
            throw std::runtime_error("IMU trajectory is empty.");
        }

        const Sophus::SE3d T_WI_end = poseAt(scan_end);
        const Sophus::SE3d T_WL_end = T_WI_end * mTi2l;
        const Sophus::SE3d T_Lend_W = T_WL_end.inverse();

        for (auto& [x, y, z, relative_time] : points)
        {
            const double point_time = std::clamp(scan_begin + relative_time, scan_begin, scan_end);
            const Sophus::SE3d T_WI_t = poseAt(point_time);
            const Sophus::SE3d T_WL_t = T_WI_t * mTi2l;
            const Eigen::Vector3d p_L(x, y, z);
            const Eigen::Vector3d p_deskewed = T_Lend_W * (T_WL_t * p_L);

            x = static_cast<float>(p_deskewed.x());
            y = static_cast<float>(p_deskewed.y());
            z = static_cast<float>(p_deskewed.z());
        }
    }

    /**
     * Get interpolated IMU pose at arbitrary timestamp.
     */
    Sophus::SE3d poseAt(const double timestamp) const
    {
        if (mTrajectory.empty())
        {
            throw std::runtime_error("IMU trajectory is empty.");
        }

        if (timestamp <= mTrajectory.front().timestamp)
        {
            return toSE3(mTrajectory.front());
        }

        if (timestamp >= mTrajectory.back().timestamp)
        {
            return toSE3(mTrajectory.back());
        }

        const auto iter = std::lower_bound(mTrajectory.begin(), mTrajectory.end(), timestamp,
            [](const PoseState& state, const double time)
                {
                    return state.timestamp < time;
                });

        const PoseState& s1 = *iter;
        const PoseState& s0 = *(iter - 1);

        const double dt = s1.timestamp - s0.timestamp;

        const double alpha = (timestamp - s0.timestamp) / dt;

        /* Translation interpolation */
        const Eigen::Vector3d position = (1.0 - alpha) * s0.p_WI + alpha * s1.p_WI;

        /*
         * SO(3) interpolation.
         * Instead of directly doing quaternion SLERP:
         *   R = R0 * Exp(alpha * Log(R0^-1 R1))
         * This fits naturally with Sophus.
         */
        const Sophus::SO3d delta_R = s0.R_WI.inverse() * s1.R_WI;
        const Sophus::SO3d rotation = s0.R_WI * Sophus::SO3d::exp(alpha * delta_R.log());

        return Sophus::SE3d(rotation, position);
    }

private:
    struct PoseState
    {
        double timestamp = 0.0;
        Sophus::SO3d R_WI;
        Eigen::Vector3d p_WI = Eigen::Vector3d::Zero();
        Eigen::Vector3d v_WI = Eigen::Vector3d::Zero();
    };

    Sophus::SE3d mTi2l;   // LiDAR -> IMU extrinsic.

    std::vector<PoseState> mTrajectory;    // IMU trajectory of current LiDAR scan.

    static Sophus::SE3d toSE3(const PoseState& state)
    {
        return Sophus::SE3d(state.R_WI, state.p_WI);
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
     * Create:
     * scan_begin
     *     ↓
     * IMU
     * IMU
     * IMU
     *     ↓
     * scan_end
     *
     * so integration begins and ends exactly at
     * LiDAR timestamps.
     */
    static std::vector<ImuData> buildImuSequence(const std::vector<ImuData>& imu_data, const double scan_begin, const double scan_end)
    {
        std::vector<ImuData> result;
        result.reserve(imu_data.size() + 2);
        result.push_back(imuAt(imu_data, scan_begin));

        for (const auto& imu : imu_data)
        {
            if (imu.timestamp > scan_begin && imu.timestamp < scan_end)
            {
                result.push_back(imu);
            }
        }

        result.push_back(imuAt(imu_data, scan_end));
        return result;
    }
};
}  // namespace lio