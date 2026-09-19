#pragma once
#include <sophus/se3.hpp>

namespace lio
{
struct ImuData
{
    double timestamp = 0.0;

    /* Raw IMU measurements */
    Eigen::Vector3d gyro = Eigen::Vector3d::Zero();   // rad/s
    Eigen::Vector3d accel = Eigen::Vector3d::Zero();  // m/s^2
};

/**
 * Nominal IMU state.
 * Frame convention:
 *   R_wi : IMU orientation in World
 *   p_wi : IMU position in World
 *   v_wi : IMU velocity in World
 * IMU biases are part of the state because they should normally
 * be estimated by IESKF rather than treated as fixed parameters.
 */
struct ImuState
{
    double timestamp = 0.0;

    Sophus::SO3d R_wi;
    Eigen::Vector3d p_wi = Eigen::Vector3d::Zero();
    Eigen::Vector3d v_wi = Eigen::Vector3d::Zero();

    /* Fixed after initialization. */
    Eigen::Vector3d gyro_bias = Eigen::Vector3d::Zero();
    Eigen::Vector3d accel_bias = Eigen::Vector3d::Zero();

    Eigen::Vector3d gravity = Eigen::Vector3d(0.0, 0.0, -9.81);    // Fixed in world frame.
};
}  // namespace lio
