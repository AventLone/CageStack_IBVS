#include "perception/LIO/ESKF.h"

namespace lio
{
ESKF::ESKF(const Config& config) : mConfig(config)
{
    mP.block<3, 3>(0, 0).diagonal().setConstant(config.initial_position_std * config.initial_position_std);
    mP.block<3, 3>(3, 3).diagonal().setConstant(config.initial_rotation_std * config.initial_rotation_std);
    mP.block<3, 3>(6, 6).diagonal().setConstant(config.initial_velocity_std * config.initial_velocity_std);
    mQ.block<3, 3>(0, 0).diagonal().setConstant(config.accel_noise_density * config.accel_noise_density);
    mQ.block<3, 3>(3, 3).diagonal().setConstant(config.gyro_noise_density * config.gyro_noise_density);
}

void ESKF::initialize(const std::vector<ImuData>& samples)
{
    Eigen::Vector3d gyro = Eigen::Vector3d::Zero();
    Eigen::Vector3d accel = Eigen::Vector3d::Zero();
    for (const auto& sample : samples)
    {
        gyro += sample.gyro;
        accel += sample.accel;
    }
    gyro /= static_cast<double>(samples.size());
    accel /= static_cast<double>(samples.size());

    mState = ImuState{};
    mState.R_wi = Sophus::SO3d(Eigen::Quaterniond::FromTwoVectors(accel, Eigen::Vector3d::UnitZ()));
    mState.gyro_bias = gyro;
    // A single static pose cannot separate tilt from all accelerometer biases.
    // This removes only the gravity-magnitude discrepancy along measured gravity.
    mState.accel_bias = accel + mState.R_wi.inverse() * mState.gravity;
    mLastImu = samples.back();
    mState.timestamp = mLastImu.timestamp;
}

void ESKF::predict(const ImuData& next)
{
    const double dt = next.timestamp - mState.timestamp;
    const Eigen::Vector3d accel = 0.5 * (mLastImu.accel + next.accel) - mState.accel_bias;
    const Eigen::Vector3d gyro = 0.5 * (mLastImu.gyro + next.gyro) - mState.gyro_bias;
    const Eigen::Vector3d phi = gyro * dt;
    const Sophus::SO3d half_rotation = Sophus::SO3d::exp(0.5 * phi);
    const Sophus::SO3d rotation_step = Sophus::SO3d::exp(phi);
    const Eigen::Matrix3d R = mState.R_wi.matrix();
    const Eigen::Matrix3d R_mid = R * half_rotation.matrix();
    const Eigen::Vector3d accel_world = R_mid * accel + mState.gravity;

    // Jacobians of the same midpoint integration used by the nominal state.
    const Eigen::Matrix3d A = -R * Sophus::SO3d::hat(half_rotation * accel);
    const Eigen::Matrix3d B = R_mid * Sophus::SO3d::hat(accel) *
                              Sophus::SO3d::leftJacobian(-0.5 * phi) * (0.5 * dt);
    StateCov F = StateCov::Identity();
    F.block<3, 3>(0, 3) = 0.5 * A * dt * dt;
    F.block<3, 3>(0, 6) = Eigen::Matrix3d::Identity() * dt;
    F.block<3, 3>(3, 3) = rotation_step.inverse().matrix();
    F.block<3, 3>(6, 3) = A * dt;

    Eigen::Matrix<double, 9, 6> G = Eigen::Matrix<double, 9, 6>::Zero();
    G.block<3, 3>(0, 0) = -0.5 * R_mid * dt * dt;
    G.block<3, 3>(0, 3) = 0.5 * B * dt * dt;
    G.block<3, 3>(3, 3) = -Sophus::SO3d::leftJacobian(-phi) * dt;
    G.block<3, 3>(6, 0) = -R_mid * dt;
    G.block<3, 3>(6, 3) = B * dt;
    mP = (F * mP * F.transpose() + G * (mQ / dt) * G.transpose()).eval();
    // Integrated white acceleration: Q_pp = q_a * dt^3 / 3, not dt^3 / 4.
    mP.block<3, 3>(0, 0) += R_mid * mQ.block<3, 3>(0, 0) * R_mid.transpose() * (dt * dt * dt / 12.0);
    mP = (0.5 * (mP + mP.transpose())).eval();

    mState.p_wi += mState.v_wi * dt + 0.5 * accel_world * dt * dt;
    mState.v_wi += accel_world * dt;
    mState.R_wi *= rotation_step;
    mState.timestamp = next.timestamp;
    mLastImu = next;
}

bool ESKF::observe(const Sophus::SE3d& pose, const MeasurementCov& covariance)
{
    Eigen::Matrix<double, 6, 1> residual;
    residual.head<3>() = pose.translation() - mState.p_wi;
    residual.tail<3>() = (mState.R_wi.inverse() * pose.so3()).log();
    // H = [I_6, 0]; position/orientation correlations also correct velocity.
    const MeasurementCov S = mP.topLeftCorner<6, 6>() + covariance;
    const Eigen::LDLT<MeasurementCov> solver(S);
    if (mConfig.innovation_gate > 0.0 && residual.dot(solver.solve(residual)) > mConfig.innovation_gate)
    {
        return false;
    }
    const Eigen::Matrix<double, 9, 6> K = solver.solve(mP.leftCols<6>().transpose()).transpose();
    const Eigen::Matrix<double, 9, 1> dx = K * residual;
    StateCov IKH = StateCov::Identity();
    IKH.leftCols<6>() -= K;
    mP = (IKH * mP * IKH.transpose() + K * covariance * K.transpose()).eval();

    mState.p_wi += dx.head<3>();
    mState.R_wi *= Sophus::SO3d::exp(dx.segment<3>(3));
    mState.v_wi += dx.tail<3>();
    StateCov reset = StateCov::Identity();
    reset.block<3, 3>(3, 3) = Sophus::SO3d::leftJacobian(-dx.segment<3>(3));
    mP = (reset * mP * reset.transpose()).eval();
    mP = (0.5 * (mP + mP.transpose())).eval();
    return true;
}
} // namespace lio
