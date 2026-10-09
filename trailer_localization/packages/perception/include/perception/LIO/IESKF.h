#pragma once
#include <pcl/point_cloud.h>
#include <sophus/se3.hpp>
#include "perception/types/nav_state.hpp"
#include "perception/types/iVox.h"

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
 * 15-state manifold ESKF
 * Local error state: dx = [dp, dtheta, dv, db_a, db_g]
 * Right rotation perturbation:
 *   R_true = R * Exp(dtheta)
 * Nominal state:
 *   p, R, v, b_a, b_g
 * Biases are calibrated during initialization and estimated during updates.
 * Gravity is fixed in the world frame.
 */
class IESKF
{
    using StateT       = Eigen::Matrix<double, 15, 1>;   // [dp, dtheta, dv, dba, dbg]
    using MeasurementT = Sophus::SE3d::Tangent;         // [p, R]
    using MotionNoiseT = Eigen::Matrix<double, 6, 1>;   // [n_a, n_g]

    using StateCov       = CovarianceT<StateT>;
    using MeasurementCov = CovarianceT<MeasurementT>;
    using MotionNoiseCov = CovarianceT<MotionNoiseT>;

    using StateJacobian = Jacobian<StateT, StateT>;
    using NoiseJacobian = Jacobian<StateT, MotionNoiseT>;
    using MeasurementJacobian = Jacobian<MeasurementT, StateT>;

public:
    IESKF()
    {
        /* State covariance, dx = [dp, dtheta, dv, dba, dbg] */
        constexpr double initial_position_std = 0.01;             // m
        constexpr double initial_rotation_std = 0.5 * DEG2RAD;    // rad
        constexpr double initial_velocity_std = 0.10;             // m/s
        constexpr double initial_accel_bias_std = 0.10;
        constexpr double initial_gyro_bias_std = 0.10 * DEG2RAD;
        mP.block<3, 3>(0, 0).diagonal().setConstant(initial_position_std * initial_position_std);
        mP.block<3, 3>(3, 3).diagonal().setConstant(initial_rotation_std * initial_rotation_std);
        mP.block<3, 3>(6, 6).diagonal().setConstant(initial_velocity_std * initial_velocity_std);
        mP.block<3, 3>(9, 9).diagonal().setConstant(initial_accel_bias_std * initial_accel_bias_std);
        mP.block<3, 3>(12, 12).diagonal().setConstant(initial_gyro_bias_std * initial_gyro_bias_std);

        /* IMU noise covariance. n = [na, ng] */
        constexpr double accel_noise_std = 0.00002;           // m/s^2
        constexpr double gyro_noise_std = 0.00001 * DEG2RAD;  // rad/s
        mQ.block<3, 3>(0, 0).diagonal().setConstant(accel_noise_std * accel_noise_std);
        mQ.block<3, 3>(3, 3).diagonal().setConstant(gyro_noise_std * gyro_noise_std);

        /* Default LiDAR odometry measurement noise. z = [p, R] */
        // constexpr double lidar_position_std = 0.001;            // m
        // constexpr double lidar_rotation_std = 0.01 * DEG2RAD;   // rad
        // mV.block<3, 3>(0, 0).diagonal().setConstant(lidar_position_std * lidar_position_std);
        // mV.block<3, 3>(3, 3).diagonal().setConstant(lidar_rotation_std * lidar_rotation_std);

        mH.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
        mH.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity();

        mVoxelMap = std::make_unique<IVox>();
    }

    void setTimestamp(const double stamp)
    {
        mState.timestamp = stamp;
    }

    bool initialize(const std::vector<ImuData>& samples);   // IMU 状态初始化
    void initialize(const pcl::PointCloud<pcl::PointXYZ>& scan) const  // Voxel Map 初始化
    {
        mVoxelMap->initialize(scan);
    }

    void predict(const std::vector<ImuData>& imu_datas)     // 预测
    {
        for (const auto& imu_data : imu_datas)
        {
            predict(imu_data);
        }
    }

    void observe(const pcl::PointCloud<pcl::PointXYZ>& scan);   // 观测更新

    [[nodiscard]] const NavState& state() const
    {
        return mState;
    }

    [[nodiscard]] const StateCov& covariance() const
    {
        return mP;
    }

    [[nodiscard]] Eigen::Isometry3d pose() const
    {
        const Sophus::SE3d transform(mState.R_wi, mState.p_wi);
        return Eigen::Isometry3d(transform.matrix());
    }

private:
    bool mInitialized{false};

    NavState mState;                       // Nominal state，需要靠初始化确定
    StateCov mP{StateCov::Zero()};         // State Covariance

    MotionNoiseCov mQ{MotionNoiseCov::Zero()};           // Motion Noise
    // MeasurementCov mV{MeasurementCov::Zero()};           // Measure Noise

    MeasurementJacobian mH{MeasurementJacobian::Zero()};  // Observe Matrix

    IVox::Ptr mVoxelMap;

    void predict(const ImuData& imu_data);                     // 预测

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

    static Eigen::Matrix3d rightJacobianInverseSO3(const Eigen::Vector3d& phi)
    {
        const double theta2 = phi.squaredNorm();
        const Eigen::Matrix3d Phi = Sophus::SO3d::hat(phi);

        if (theta2 < 1e-10)
        {
            return Eigen::Matrix3d::Identity() + 0.5 * Phi + (1.0 / 12.0) * Phi * Phi;
        }

        const double theta = std::sqrt(theta2);
        return Eigen::Matrix3d::Identity() + 0.5 * Phi + (1.0 / theta2 - (1.0 + std::cos(theta)) / (2.0 * theta * std::sin(theta))) * Phi * Phi;
    }

    static StateT stateDifference(const NavState& state, const NavState& prior)
    {
        StateT dx = StateT::Zero();

        dx.segment<3>(0) = state.p_wi - prior.p_wi;
        dx.segment<3>(3) = (prior.R_wi.inverse() * state.R_wi).log();
        dx.segment<3>(6) = state.v_wi - prior.v_wi;
        dx.segment<3>(9) = state.accel_bias - prior.accel_bias;
        dx.segment<3>(12) = state.gyro_bias - prior.gyro_bias;

        return dx;
    }

    /* 经过预测更新，修正了误差状态的估计，然后需要把误差状态归入名义状态 */
    void update(const StateT& dx)
    {
        mState.p_wi += dx.segment<3>(0);
        mState.R_wi *= Sophus::SO3d::exp(dx.segment<3>(3));
        mState.v_wi += dx.segment<3>(6);
        mState.accel_bias += dx.segment<3>(9);
        mState.gyro_bias += dx.segment<3>(12);
    }
};
}
