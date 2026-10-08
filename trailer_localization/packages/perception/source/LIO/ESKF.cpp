#include "perception/LIO/ESKF.h"
#include <iostream>
#include <format>

namespace lio
{
bool ESKF::initialize(const std::vector<ImuData>& samples)
{
    if (samples.size() < 20 || samples.back().timestamp - samples.front().timestamp < 2.0)
    {
        std::cerr << "Too few IMU samples or too short time." << std::endl;
        return false;
    }

    Eigen::Vector3d gyro = Eigen::Vector3d::Zero();
    Eigen::Vector3d accel = Eigen::Vector3d::Zero();
    for (const auto& imu : samples)
    {
        gyro += imu.gyro;
        accel += imu.accel;
    }
    gyro /= static_cast<double>(samples.size());
    accel /= static_cast<double>(samples.size());

    double gyro_variance = 0.0;
    double accel_variance = 0.0;
    for (std::size_t i = 0; i < samples.size(); ++i)
    {
        if (constexpr double max_gap = 0.05; i > 0 && samples[i].timestamp - samples[i - 1].timestamp > max_gap)
        {
            std::cerr << "Time gap is too big!" << std::endl;
            return false;
        }
        gyro_variance += (samples[i].gyro - gyro).squaredNorm();
        accel_variance += (samples[i].accel - accel).squaredNorm();
    }

    if (gyro.norm() > 0.1 || gyro_variance / static_cast<double>(samples.size()) > 0.0025 ||
        accel_variance / static_cast<double>(samples.size()) > 0.25 || std::abs(accel.norm() - 9.81) > 0.5)
    {
        std::cerr << "Covariance is too big!" << std::endl;
        return false;
    }

    /* Set initial state */
    mState.timestamp = samples.back().timestamp;
    mState.R_wi = Sophus::SO3d(Eigen::Quaterniond::FromTwoVectors(accel.normalized(), Eigen::Vector3d::UnitZ()));
    mState.gyro_bias = gyro;
    mState.accel_bias = accel + mState.R_wi.inverse() * mState.gravity;

    mInitialized = true;
    return true;
}

void ESKF::predict(const ImuData& imu_data)
{
    if (!mInitialized)
    {
        throw std::invalid_argument("Initialize ESKF before prediction.");
    }

    // const double dt = imu_data.timestamp - mState.timestamp;
    const double dt = imu_data.timestamp - mState.timestamp;
    if (dt > 0.05 || dt < 0)
    {
        // 时间间隔不对，可能是第一个IMU数据，没有历史信息
        // LOG(INFO) << "skip this imu because dt_ = " << dt;
        mState.timestamp = imu_data.timestamp;
        return;
    }

    auto& p = mState.p_wi;
    auto& R = mState.R_wi;
    auto& v = mState.v_wi;

    const Eigen::Vector3d accel = imu_data.accel - mState.accel_bias;
    const Eigen::Vector3d gyro = imu_data.gyro - mState.gyro_bias;
    const auto& g = mState.gravity;

    /*
     * Save R_k.
     *
     * Nominal propagation and its Jacobian must use the same
     * linearization point.
     */
    const Sophus::SO3d R_old = R;
    const Eigen::Vector3d phi = gyro * dt;
    const Sophus::SO3d dR = Sophus::SO3d::exp(phi);
    const Eigen::Matrix3d accel_hat = Sophus::SO3d::hat(accel);
    const Eigen::Vector3d accel_world = R_old * accel + g;

    p += v * dt + 0.5 * accel_world * dt * dt;
    v += accel_world * dt;
    R *= dR;

    /*
     * ------------------------------------------------------------
     * Error-state transition
     * ------------------------------------------------------------
     * dx = [dp, dtheta, dv]
     * Right perturbation: R_true = R * Exp(dtheta)
     */
    StateJacobian F = StateJacobian::Identity();
    F.block<3, 3>(0, 3) = -0.5 * R_old.matrix() * accel_hat * dt * dt;
    F.block<3, 3>(0, 6) = Eigen::Matrix3d::Identity() * dt;   //dp / dv
    F.block<3, 3>(3, 3) = dR.inverse().matrix();
    F.block<3, 3>(6, 3) = -R_old.matrix() * accel_hat * dt;   //dv / dtheta

    NoiseJacobian G = NoiseJacobian::Zero();
    G.block<3, 3>(0, 0) = -0.5 * R_old.matrix() * dt * dt;   // dp / n_a
    G.block<3, 3>(3, 3) = -rightJacobianSO3(phi) * dt;       // dtheta / n_g
    G.block<3, 3>(6, 0) = -R_old.matrix() * dt;              // dv / n_a

    mP = F * mP * F.transpose() + G * mQ * G.transpose();   // Covariance propagation
    mP = 0.5 * (mP + mP.transpose());                       // Suppress accumulated numerical asymmetry

    mState.timestamp = imu_data.timestamp;
}

void ESKF::observe(const Sophus::SE3d& pose, const MeasurementCov& measurement_cov)
{
    if (!mInitialized)
    {
        return;
    }

    MeasurementT residual;
    residual.head<3>() = pose.translation() - mState.p_wi;   // Translation residual

    /*
     * Right-invariant local rotation residual: R_meas = R_pred * Exp(dtheta)
     * therefore: dtheta = Log(R_pred^-1 R_meas)
     */
    residual.tail<3>() = (mState.R_wi.inverse() * pose.so3()).log();

    const MeasurementCov S = mH * mP * mH.transpose() + measurement_cov;
    const auto PHt = mP * mH.transpose();

    /* K = P H^T S^-1, Avoid explicit inverse() */
    const Eigen::Matrix<double, 9, 6> K = S.ldlt().solve(PHt.transpose()).transpose();
    const StateT dx = K * residual;

    /* Joseph-form covariance update */
    const StateJacobian I = StateJacobian::Identity();
    const StateJacobian IKH = I - K * mH;

    mP = IKH * mP * IKH.transpose() + K * measurement_cov * K.transpose();
    update(dx);   // Inject local correction into nominal state

    /*
     * Covariance reset after SO(3) error injection.
     * For right perturbation: G_reset ~= I - 1/2 [dtheta]x
     */
    StateJacobian reset = StateJacobian::Identity();
    const Eigen::Vector3d dtheta = dx.segment<3>(3);
    reset.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity() - 0.5 * Sophus::SO3d::hat(dtheta);
    mP = reset * mP * reset.transpose();
    mP = 0.5 * (mP + mP.transpose());
}
}