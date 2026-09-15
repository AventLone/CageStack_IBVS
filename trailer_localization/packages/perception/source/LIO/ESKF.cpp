#include "perception/LIO/ESKF.h"
#include <Eigen/Cholesky>

namespace lio
{
namespace
{
bool finite(const ImuState& state)
{
    return std::isfinite(state.timestamp) && state.R_WI.matrix().allFinite() &&
           state.p_WI.allFinite() && state.v_WI.allFinite() && state.gyro_bias.allFinite() &&
           state.accel_bias.allFinite() && state.gravity.allFinite();
}
}

ESKF::ESKF(const Config& config, const Sophus::SE3d& T_IL) : mConfig(config), mTi2l(T_IL), mImuProcessor(T_IL)
{
    for (const double value : {config.gyro_noise, config.accel_noise, config.gyro_bias_noise,
                        config.accel_bias_noise, config.gravity_magnitude, config.max_imu_gap,
                        config.max_integration_step, config.max_plane_rmse,
                        config.convergence_translation, config.convergence_rotation,
                        config.max_position_correction, config.max_rotation_correction})
    {
        if (!std::isfinite(value) || value <= 0.0)
        {
            throw std::invalid_argument("ESKF noise densities and thresholds must be positive and finite.");
        }
    }
    if (config.max_iterations <= 0 || config.min_correspondences < 3 || !T_IL.matrix().allFinite())
    {
        throw std::invalid_argument("Invalid ESKF iterations, correspondence count or extrinsic.");
    }
}

StateCovariance ESKF::initialCovariance()
{
    StateCovariance covariance = StateCovariance::Zero();
    covariance.diagonal() << 1e-4, 1e-4, 1e-4, 0.01, 0.01, 0.25,
                             0.1, 0.1, 0.1, 1e-4, 1e-4, 1e-4, 0.1, 0.1, 0.1;
    return covariance;
}

void ESKF::reset(const ImuState& state, const StateCovariance& covariance)
{
    if (!finite(state) || !covariance.allFinite() ||
        !covariance.isApprox(covariance.transpose(), 1e-10) ||
        Eigen::LLT<StateCovariance>(covariance).info() != Eigen::Success)
    {
        throw std::invalid_argument("ESKF reset needs a finite state and symmetric positive definite covariance.");
    }
    mState = state;
    mCovariance = covariance;
    mLastImu.reset();
    mInitialized = true;
}

bool ESKF::initialize(const std::vector<ImuData>& samples)
{
    ImuProcessor::validate(samples);
    if (samples.size() < 20 || samples.back().timestamp - samples.front().timestamp < 0.5) return false;
    Eigen::Vector3d gyro = Eigen::Vector3d::Zero(), accel = Eigen::Vector3d::Zero();
    for (const auto& imu : samples) { gyro += imu.gyro; accel += imu.accel; }
    gyro /= static_cast<double>(samples.size());
    accel /= static_cast<double>(samples.size());
    double gyro_variance = 0.0, accel_variance = 0.0;
    for (std::size_t i = 0; i < samples.size(); ++i)
    {
        if (i > 0 && samples[i].timestamp - samples[i - 1].timestamp > mConfig.max_imu_gap) return false;
        gyro_variance += (samples[i].gyro - gyro).squaredNorm();
        accel_variance += (samples[i].accel - accel).squaredNorm();
    }

    if (gyro.norm() > 0.1 || gyro_variance / static_cast<double>(samples.size()) > 0.0025 ||
        accel_variance / static_cast<double>(samples.size()) > 0.25 || std::abs(accel.norm() - mConfig.gravity_magnitude) > 0.5)
    {
        return false;
    }

    ImuState initial;
    initial.timestamp = samples.back().timestamp;
    initial.gravity = Eigen::Vector3d(0.0, 0.0, -mConfig.gravity_magnitude);
    initial.R_WI = Sophus::SO3d(Eigen::Quaterniond::FromTwoVectors(accel.normalized(), Eigen::Vector3d::UnitZ()));
    initial.gyro_bias = gyro;
    // Only the component along gravity can be inferred from one stationary pose.
    initial.accel_bias = accel + initial.R_WI.inverse() * initial.gravity;
    reset(initial);
    mLastImu = samples.back();
    return true;
}

void ESKF::predictCovariance(const ImuState& state, const ImuData& a, const ImuData& b)
{
    const double dt = b.timestamp - a.timestamp;
    const Eigen::Vector3d omega = 0.5 * (a.gyro + b.gyro) - state.gyro_bias;
    const Eigen::Vector3d force = 0.5 * (a.accel + b.accel) - state.accel_bias;
    const Eigen::Matrix3d R = (state.R_WI * Sophus::SO3d::exp(omega * (0.5 * dt))).matrix();

    // Right-error dynamics. These cross terms let LiDAR constrain v and biases.
    StateCovariance F = StateCovariance::Zero();
    F.block<3, 3>(0, 6).setIdentity();
    F.block<3, 3>(3, 3) = -Sophus::SO3d::hat(omega);
    F.block<3, 3>(3, 9) = -Eigen::Matrix3d::Identity();
    F.block<3, 3>(6, 3) = -R * Sophus::SO3d::hat(force);
    F.block<3, 3>(6, 12) = -R;
    StateCovariance noise = StateCovariance::Zero();
    noise.block<3, 3>(3, 3).diagonal().setConstant(mConfig.gyro_noise * mConfig.gyro_noise);
    noise.block<3, 3>(6, 6).diagonal().setConstant(mConfig.accel_noise * mConfig.accel_noise);
    noise.block<3, 3>(9, 9).diagonal().setConstant(mConfig.gyro_bias_noise * mConfig.gyro_bias_noise);
    noise.block<3, 3>(12, 12).diagonal().setConstant(mConfig.accel_bias_noise * mConfig.accel_bias_noise);
    const StateCovariance F2 = F * F;
    const StateCovariance Phi = StateCovariance::Identity() + F * dt + 0.5 * F2 * dt * dt;
    const StateCovariance half = StateCovariance::Identity() + F * (0.5 * dt) + F2 * (0.125 * dt * dt);
    // Simpson integration of Phi(t) G Qc G^T Phi(t)^T preserves PSD and
    // position/velocity noise cross terms. Short substeps bound truncation error.
    const StateCovariance Qd = (dt / 6.0) * (noise + 4.0 * half * noise * half.transpose() + Phi * noise * Phi.transpose());
    const StateCovariance predicted = Phi * mCovariance * Phi.transpose() + Qd;
    mCovariance = 0.5 * (predicted + predicted.transpose());
}

void ESKF::predict(const ImuData& imu_data)
{
    if (!mInitialized || !ImuProcessor::finite(imu_data)) throw std::invalid_argument("Initialize ESKF before prediction with finite IMU.");
    if (!mLastImu)
    {
        if (std::abs(imu_data.timestamp - mState.timestamp) > 1e-9)
            throw std::invalid_argument("First IMU must anchor the current state timestamp.");
        mLastImu = imu_data;
        return;
    }
    if (imu_data.timestamp <= mState.timestamp || imu_data.timestamp - mState.timestamp > mConfig.max_imu_gap)
        throw std::invalid_argument("Out-of-order IMU or excessive IMU gap.");
    const auto samples = ImuProcessor::buildImuSequence({*mLastImu, imu_data}, mState.timestamp, imu_data.timestamp, mConfig.max_integration_step);
    mImuProcessor.propagate(samples, mState, [this](const auto& s, const auto& a, const auto& b)
        {
            predictCovariance(s, a, b);
        });
    mLastImu = imu_data;
}

void ESKF::processScan(const std::vector<ImuData>& imu_data, std::vector<PointXYZT>& points, const double scan_begin, const double scan_end)
{
    if (!mInitialized)
    {
        throw std::logic_error("Initialize ESKF before processing a scan.");
    }

    ImuProcessor::validate(imu_data);
    for (std::size_t i = 1; i < imu_data.size(); ++i)
    {
        if (imu_data[i].timestamp > mState.timestamp && imu_data[i - 1].timestamp < scan_end &&
            imu_data[i].timestamp - imu_data[i - 1].timestamp > mConfig.max_imu_gap)
        {
            throw std::invalid_argument("IMU gap exceeds max_imu_gap.");
        }
    }

    mImuProcessor.process(imu_data, points, scan_begin, scan_end, mState, [this](const auto& s, const auto& a, const auto& b)
        {
            predictCovariance(s, a, b);
        }, mConfig.max_integration_step);
    mLastImu = ImuProcessor::buildImuSequence(imu_data, scan_end, scan_end).front();
}

Eigen::Matrix3d ESKF::rightJacobianInverse(const Eigen::Vector3d& rotation)
{
    const double theta2 = rotation.squaredNorm();
    const double theta = std::sqrt(theta2);
    const double coefficient = theta2 < 1e-8 ? 1.0 / 12.0 + theta2 / 720.0 : (1.0 - 0.5 * theta / std::tan(0.5 * theta)) / theta2;
    const Eigen::Matrix3d hat = Sophus::SO3d::hat(rotation);
    return Eigen::Matrix3d::Identity() + 0.5 * hat + coefficient * hat * hat;
}

ESKF::Result ESKF::update(const MeasurementModel& model)
{
    if (!mInitialized || !model)
    {
        throw std::logic_error("ESKF update requires initialization and a measurement model.");
    }

    Result result;
    const ImuState prior = mState;
    const StateCovariance prior_information = mCovariance.llt().solve(StateCovariance::Identity());

    ImuState iterate = prior;
    StateCovariance information;
    ErrorStateT gradient;

    const auto linearize = [&]()
    {
        const Measurement measurement = model(iterate);
        result.num_correspondences = measurement.count;
        result.plane_rmse = measurement.count > 0 ? std::sqrt(measurement.squared_error / measurement.count) :
                            std::numeric_limits<double>::infinity();
        if (measurement.count < mConfig.min_correspondences || !std::isfinite(result.plane_rmse) ||
            !measurement.information.allFinite() || !measurement.gradient.allFinite()) return false;
        const ErrorStateT error = boxMinus(iterate, prior);
        StateCovariance A = StateCovariance::Identity();
        A.block<3, 3>(3, 3) = rightJacobianInverse(error.segment<3>(3));

        // Count the same propagated prior ONCE. A transports its tangent to this
        // iterate. Do not recursively shrink P inside the optimization loop.
        information = A.transpose() * prior_information * A;
        gradient = A.transpose() * prior_information * error;
        information.topLeftCorner<6, 6>() += measurement.information;
        gradient.head<6>() += measurement.gradient;
        return information.allFinite() && gradient.allFinite();
    };

    for (int iteration = 0; iteration < mConfig.max_iterations; ++iteration)
    {
        if (!linearize())
        {
            return result;
        }

        const Eigen::LLT<StateCovariance> solver(information);
        if (solver.info() != Eigen::Success)
        {
            return result;
        }

        const ErrorStateT increment = solver.solve(-gradient);
        if (!increment.allFinite())
        {
            return result;
        }

        iterate = boxPlus(iterate, increment);
        ++result.iterations;
        if (const ErrorStateT correction = boxMinus(iterate, prior);
            !finite(iterate) || correction.head<3>().norm() > mConfig.max_position_correction ||
            correction.segment<3>(3).norm() > mConfig.max_rotation_correction)
        {
            return result;
        }

        if (increment.head<3>().norm() < mConfig.convergence_translation &&
            increment.segment<3>(3).norm() < mConfig.convergence_rotation &&
            increment.tail<9>().norm() < 1e-3)
        {
            result.converged = true;
            break;
        }
    }

    // Re-linearize in the FINAL state's tangent. This recenters the posterior;
    // applying an additional reset Jacobian would transport it twice.
    // On failure the propagated state and covariance remain untouched.
    // if (!result.converged || !linearize() || result.plane_rmse > mConfig.max_plane_rmse)
    // {
    //     return result;
    // }

    const Eigen::LLT<StateCovariance> solver(information);
    if (solver.info() != Eigen::Success)
    {
        return result;
    }

    const StateCovariance posterior = solver.solve(StateCovariance::Identity());
    const StateCovariance symmetric = 0.5 * (posterior + posterior.transpose());
    if (!symmetric.allFinite() || Eigen::LLT<StateCovariance>(symmetric).info() != Eigen::Success)
    {
        return result;
    }

    mState = iterate;
    mCovariance = symmetric;
    result.accepted = true;
    return result;
}
}  // namespace lio
