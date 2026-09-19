#pragma once
#include <sophus/se3.hpp>
#include "perception/types/imu.hpp"

namespace lio
{
/**
 * @brief Template type for covariance matrices
 * @tparam Type The vector type for which to generate a covariance (usually a state or measurement type)
 */
template<class Type>
using CovarianceT = Eigen::Matrix<typename Type::Scalar, Type::RowsAtCompileTime, Type::RowsAtCompileTime>;

/**
 * @class Jacobian
 * @brief Template type of jacobian of VecA to VecB
 */
template<class VecA, class VecB>
using Jacobian = Eigen::Matrix<typename VecA::Scalar, VecA::RowsAtCompileTime, VecB::RowsAtCompileTime>;

constexpr double DEG2RAD = 1.0 / 180.0 * M_PI;

/**
 * 9-state manifold ESKF
 * Local error state: dx = [dp, dtheta, dv]
 * Right rotation perturbation:
 *   R_true = R * Exp(dtheta)
 * Nominal state:
 *   p, R, v
 * bg, ba and gravity are calibrated during initialization and then fixed.
 */
class ESKF
{
    using StateT       = Eigen::Matrix<double, 9, 1>;   // [dp, dtheta, dv]
    using MeasurementT = Sophus::SE3d::Tangent;   // [p, R]
    using MotionNoiseT = Eigen::Matrix<double, 6, 1>;   // [n_a, n_g]

    using StateCov       = CovarianceT<StateT>;
    using MeasurementCov = CovarianceT<MeasurementT>;
    using MotionNoiseCov = CovarianceT<MotionNoiseT>;

    using StateJacobian = Jacobian<StateT, StateT>;
    using NoiseJacobian = Jacobian<StateT, MotionNoiseT>;
    using MeasurementJacobian = Jacobian<MeasurementT, StateT>;

public:
    ESKF()
    {
        /* State covariance, dx = [dp, dtheta, dv] */
        constexpr double initial_position_std = 0.01;             // m
        constexpr double initial_rotation_std = 0.5 * DEG2RAD;    // rad
        constexpr double initial_velocity_std = 0.10;             // m/s
        mP.block<3, 3>(0, 0).diagonal().setConstant(initial_position_std * initial_position_std);
        mP.block<3, 3>(3, 3).diagonal().setConstant(initial_rotation_std * initial_rotation_std);
        mP.block<3, 3>(6, 6).diagonal().setConstant(initial_velocity_std * initial_velocity_std);

        /* IMU noise covariance. n = [na, ng] */
        constexpr double gyro_noise_std = 0.02 * DEG2RAD;  // rad/s
        constexpr double accel_noise_std = 0.03;           // m/s^2
        mQ.block<3, 3>(0, 0).diagonal().setConstant(accel_noise_std * accel_noise_std);
        mQ.block<3, 3>(3, 3).diagonal().setConstant(gyro_noise_std * gyro_noise_std);

        /* Default LiDAR odometry measurement noise. z = [p, R] */
        constexpr double lidar_position_std = 0.03;            // m
        constexpr double lidar_rotation_std = 0.3 * DEG2RAD;  // rad
        mV.block<3, 3>(0, 0).diagonal().setConstant(lidar_position_std * lidar_position_std);
        mV.block<3, 3>(3, 3).diagonal().setConstant(lidar_rotation_std * lidar_rotation_std);
    }

    bool initialize(const std::vector<ImuData>& samples);   // IMU 状态，Voxel Map 初始化
    void predict(const ImuData& imu_data);                     // 预测

    void observe(const Sophus::SE3d& pose, const double timestamp)
    {
        observe(pose, mV, timestamp);
    }

    void observe(const Sophus::SE3d& pose, const MeasurementCov& measurement_cov, double timestamp);   // 观测更新

    [[nodiscard]] const ImuState& state() const
    {
        return mState;
    }

    [[nodiscard]] Eigen::Isometry3d pose() const
    {
        const Sophus::SE3d transform(mState.R_wi, mState.p_wi);
        return Eigen::Isometry3d(transform.matrix());
    }

private:
    bool mInitialized{false};

    ImuState mState;                           // Nominal state，需要靠初始化确定
    StateCov mP{StateCov::Zero()};         // State Covariance

    MotionNoiseCov mQ{MotionNoiseCov::Zero()};           // Motion Noise
    MeasurementCov mV{MeasurementCov::Zero()};           // Measure Noise

    static Eigen::Matrix3d rightJacobianSO3(const Eigen::Vector3d& phi)
    {
        const double theta2 = phi.squaredNorm();
        const Eigen::Matrix3d Phi = Sophus::SO3d::hat(phi);

        if (theta2 < 1e-10)
        {
            return Eigen::Matrix3d::Identity() - 0.5 * Phi + (1.0 / 6.0) * Phi * Phi;
        }

        const double theta = std::sqrt(theta2);

        return Eigen::Matrix3d::Identity() - (1.0 - std::cos(theta)) / theta2 * Phi +
            (theta - std::sin(theta)) / (theta2 * theta) * Phi * Phi;
    }

    /* 经过预测更新，修正了误差状态的估计，然后需要把误差状态归入名义状态 */
    void update(const StateT& dx)
    {
        mState.p_wi += dx.block<3, 1>(0, 0);
        mState.R_wi *= Sophus::SO3d::exp(dx.block<3, 1>(3, 0));
        mState.v_wi += dx.block<3, 1>(6, 0);
    }
};
}
