#pragma once
#include <sophus/se3.hpp>

namespace lio
{
struct ImuData
{
    double timestamp = 0.0;

    // Raw IMU measurements.
    Eigen::Vector3d gyro = Eigen::Vector3d::Zero();   // rad/s
    Eigen::Vector3d accel = Eigen::Vector3d::Zero();  // m/s^2
};

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

}  // namespace lio
