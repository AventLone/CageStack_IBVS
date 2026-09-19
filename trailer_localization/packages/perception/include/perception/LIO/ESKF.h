#pragma once
#include <Eigen/Cholesky>
#include <sophus/se3.hpp>
#include <vector>
#include "perception/types/imu.hpp"

namespace lio
{
// Error state [dp_world, dtheta_imu, dv_world], R_true = R * Exp(dtheta).
// Biases and gravity are calibrated once; they are not filter states.
class ESKF
{
public:
    using StateCov = Eigen::Matrix<double, 9, 9>;
    using MeasurementCov = Eigen::Matrix<double, 6, 6>;

    struct Config
    {
        double initial_position_std{0.01};
        double initial_rotation_std{0.00872664626}; // 0.5 deg
        double initial_velocity_std{0.1};
        double accel_noise_density{0.03};          // m/s^2 / sqrt(Hz)
        double gyro_noise_density{0.00034906585};  // rad/s / sqrt(Hz)
        double innovation_gate{22.458};            // chi-square(6), 99.9%; <= 0 disables
    };

    ESKF() : ESKF(Config{}) {}
    explicit ESKF(const Config& config);

    // Caller supplies a stationary initialization interval, then ordered IMU.
    void initialize(const std::vector<ImuData>& samples);
    void predict(const ImuData& next);

    // Pose and covariance must refer to the current filter timestamp.
    // Covariance coordinates: world translation, right-local IMU rotation.
    bool observe(const Sophus::SE3d& pose, const MeasurementCov& covariance);

    const ImuState& state() const { return mState; }
    const StateCov& covariance() const { return mP; }
    Sophus::SE3d pose() const { return {mState.R_wi, mState.p_wi}; }

private:
    Config mConfig;
    ImuState mState;
    ImuData mLastImu;
    StateCov mP{StateCov::Zero()};
    MeasurementCov mQ{MeasurementCov::Zero()}; // continuous noise, [accel, gyro]
};
} // namespace lio
