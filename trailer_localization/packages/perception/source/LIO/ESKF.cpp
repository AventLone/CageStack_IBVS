#include "perception/LIO/ESKF.h"
#include <Eigen/Cholesky>
#include <algorithm>
#include <execution>

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
                        config.max_integration_step, config.damping_factor, config.max_fitness_score,
                        config.convergence_translation, config.convergence_rotation,
                        config.max_position_correction, config.max_rotation_correction,
                        config.lidar_noise, static_cast<double>(config.max_correspondence_distance),
                        static_cast<double>(config.map_voxel_size)})
    {
        if (!std::isfinite(value) || value <= 0.0)
        {
            throw std::invalid_argument("ESKF noise densities and thresholds must be positive and finite.");
        }
    }
    if (config.max_iterations <= 0 || config.min_correspondences < 3 || config.max_map_voxels == 0 ||
        !std::isfinite(config.cauchy_kernel_scale) || config.cauchy_kernel_scale < 0.0 ||
        !T_IL.matrix().allFinite())
    {
        throw std::invalid_argument("Invalid ESKF iterations, correspondence count or extrinsic.");
    }
    mVoxelConfig.voxel_size = config.map_voxel_size;
    mVoxelConfig.max_voxels_num = config.max_map_voxels;
    mVoxelConfig.estimate_covariances = true;
    mMap = std::make_unique<SparseVoxel>(mVoxelConfig);
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
    mMap->clear();
    mInitialized = true;
}

void ESKF::clear() noexcept
{
    mInitialized = false;
    mLastImu.reset();
    mMap->clear();
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

ImuState ESKF::boxPlus(const ImuState& state, const ErrorStateT& increment)
{
    ImuState result = state;
    result.p_WI += increment.segment<3>(0);
    result.R_WI *= Sophus::SO3d::exp(increment.segment<3>(3));
    result.v_WI += increment.segment<3>(6);
    result.gyro_bias += increment.segment<3>(9);
    result.accel_bias += increment.segment<3>(12);
    return result;
}

ErrorStateT ESKF::boxMinus(const ImuState& state, const ImuState& reference)
{
    ErrorStateT result;
    result << state.p_WI - reference.p_WI, (reference.R_WI.inverse() * state.R_WI).log(),
              state.v_WI - reference.v_WI, state.gyro_bias - reference.gyro_bias,
              state.accel_bias - reference.accel_bias;
    return result;
}

Eigen::Matrix3d ESKF::rightJacobianInverse(const Eigen::Vector3d& rotation)
{
    const double theta2 = rotation.squaredNorm();
    const double theta = std::sqrt(theta2);
    const double coefficient = theta2 < 1e-8 ? 1.0 / 12.0 + theta2 / 720.0 : (1.0 - 0.5 * theta / std::tan(0.5 * theta)) / theta2;
    const Eigen::Matrix3d hat = Sophus::SO3d::hat(rotation);
    return Eigen::Matrix3d::Identity() + 0.5 * hat + coefficient * hat * hat;
}

void ESKF::findCorrespondences(const std::vector<const PointWithCovariance*>& source,
                               const ImuState& state,
                               std::vector<Correspondence>& correspondences) const
{
    const Sophus::SE3d T_WL = Sophus::SE3d(state.R_WI, state.p_WI) * mTi2l;
    std::transform(std::execution::par, source.begin(), source.end(), correspondences.begin(),
        [this, &T_WL](const PointWithCovariance* source_point)
        {
            const Eigen::Vector3d transformed = T_WL * source_point->position.cast<double>();
            constexpr int adjacent_voxels = 1;
            const SparseVoxel::Neighbor nearest = mMap->nearestNeighbor(
                transformed.cast<float>(), mConfig.max_correspondence_distance, adjacent_voxels);
            return Correspondence{nearest.point, transformed};
        });
}

bool ESKF::buildAndSolve(const std::vector<const PointWithCovariance*>& source,
                         const std::vector<Correspondence>& correspondences,
                         const ImuState& prior,
                         const StateCovariance& prior_information,
                         const ImuState& iterate,
                         StateCovariance& information,
                         ErrorStateT& increment,
                         std::size_t& num_correspondences,
                         double& squared_error_sum) const
{
    PoseInformation lidar_information = PoseInformation::Zero();
    PoseGradient lidar_gradient = PoseGradient::Zero();
    num_correspondences = 0;
    squared_error_sum = 0.0;

    const Eigen::Matrix3d R_WI = iterate.R_WI.matrix();
    const Eigen::Matrix3d R_WL = R_WI * mTi2l.rotationMatrix();
    const double inverse_variance = 1.0 / (mConfig.lidar_noise * mConfig.lidar_noise);
    const double kernel_scale2 = mConfig.cauchy_kernel_scale * mConfig.cauchy_kernel_scale;

    for (std::size_t index = 0; index < correspondences.size(); ++index)
    {
        const auto& [target_point, transformed_position] = correspondences[index];
        if (target_point == nullptr) continue;

        const PointWithCovariance& source_point = *source[index];
        Eigen::Matrix3d covariance = Eigen::Matrix3d::Identity();
        if (source_point.covariance_valid)
        {
            if (target_point->covariance_valid)
            {
                covariance = target_point->covariance.cast<double>();
            }
            covariance += R_WL * source_point.covariance.cast<double>() * R_WL.transpose();
        }

        const Eigen::Matrix3d precision = covariance.inverse();
        if (!precision.allFinite()) continue;

        const Eigen::Vector3d residual = transformed_position - target_point->position.cast<double>();
        const Eigen::Vector3d precision_residual = precision * residual;
        const double mahalanobis_error = residual.dot(precision_residual);
        const double weight = mConfig.cauchy_kernel_scale > 0.0 ?
            1.0 / (1.0 + mahalanobis_error / kernel_scale2) : 1.0;

        // ESKF error: additive world translation and right IMU rotation.
        // p_I includes the LiDAR-to-IMU lever arm, so this is the direct
        // right-perturbation Jacobian without a separate pose-frame mapping.
        const Eigen::Vector3d p_I = mTi2l * source_point.position.cast<double>();
        Eigen::Matrix<double, 3, 6> jacobian;
        jacobian.leftCols<3>().setIdentity();
        jacobian.rightCols<3>() = -R_WI * Sophus::SO3d::hat(p_I);
        lidar_information.noalias() += inverse_variance * jacobian.transpose() * weight * precision * jacobian;
        lidar_gradient.noalias() += inverse_variance * jacobian.transpose() * weight * precision_residual;
        squared_error_sum += residual.squaredNorm();
        ++num_correspondences;
    }

    if (num_correspondences < mConfig.min_correspondences || !std::isfinite(squared_error_sum) ||
        !lidar_information.allFinite() || !lidar_gradient.allFinite()) return false;

    const ErrorStateT error = boxMinus(iterate, prior);
    StateCovariance prior_jacobian = StateCovariance::Identity();
    prior_jacobian.block<3, 3>(3, 3) = rightJacobianInverse(error.segment<3>(3));
    information = prior_jacobian.transpose() * prior_information * prior_jacobian;
    ErrorStateT gradient = prior_jacobian.transpose() * prior_information * error;
    information.topLeftCorner<6, 6>() += lidar_information;
    gradient.head<6>() += lidar_gradient;
    if (!information.allFinite() || !gradient.allFinite()) return false;

    StateCovariance damped_information = information;
    damped_information.topLeftCorner<6, 6>().diagonal().array() += mConfig.damping_factor;
    const Eigen::LLT<StateCovariance> solver(damped_information);
    if (solver.info() != Eigen::Success) return false;
    increment = solver.solve(-gradient);
    return solver.info() == Eigen::Success && increment.allFinite();
}

void ESKF::insertCloud(const pcl::PointCloud<pcl::PointXYZ>& cloud, const bool initialize)
{
    const Sophus::SE3d T_WL = Sophus::SE3d(mState.R_WI, mState.p_WI) * mTi2l;
    pcl::PointCloud<pcl::PointXYZ> world;
    world.reserve(cloud.size());
    for (const auto& point : cloud)
    {
        const Eigen::Vector3d transformed = T_WL * Eigen::Vector3d(point.x, point.y, point.z);
        world.emplace_back(static_cast<float>(transformed.x()),
                           static_cast<float>(transformed.y()),
                           static_cast<float>(transformed.z()));
    }
    if (initialize) mMap->initialize(world);
    else mMap->insert(world);
}

ESKF::Result ESKF::update(const pcl::PointCloud<pcl::PointXYZ>& cloud)
{
    if (!mInitialized) throw std::logic_error("Initialize ESKF before a LiDAR update.");

    Result result;
    if (cloud.empty()) return result;

    SparseVoxel source_index(mVoxelConfig);
    source_index.initialize(cloud);
    if (source_index.pointCount() < mConfig.min_correspondences) return result;
    if (mMap->empty())
    {
        insertCloud(cloud, true);
        result.accepted = true;
        result.converged = true;
        result.map_initialized = true;
        return result;
    }

    const std::vector<const PointWithCovariance*> source = source_index.points();
    std::vector<Correspondence> correspondences(source.size());
    const ImuState prior = mState;
    const Eigen::LLT<StateCovariance> prior_solver(mCovariance);
    if (prior_solver.info() != Eigen::Success) return result;
    const StateCovariance prior_information = prior_solver.solve(StateCovariance::Identity());

    ImuState iterate = prior;
    StateCovariance information;
    ErrorStateT increment;
    double squared_error_sum = 0.0;

    for (int iteration = 0; iteration < mConfig.max_iterations; ++iteration)
    {
        findCorrespondences(source, iterate, correspondences);
        if (!buildAndSolve(source, correspondences, prior, prior_information, iterate,
                           information, increment, result.num_correspondences, squared_error_sum)) break;
        result.fitness_score = squared_error_sum / static_cast<double>(result.num_correspondences);

        iterate = boxPlus(iterate, increment);
        ++result.iterations;
        if (const ErrorStateT correction = boxMinus(iterate, prior);
            !finite(iterate) || correction.head<3>().norm() > mConfig.max_position_correction ||
            correction.segment<3>(3).norm() > mConfig.max_rotation_correction)
        {
            return result;
        }

        if (increment.head<3>().norm() < mConfig.convergence_translation &&
            increment.segment<3>(3).norm() < mConfig.convergence_rotation)
        {
            result.converged = true;
            break;
        }
    }

    if (!result.converged && result.iterations > 0 && std::isfinite(result.fitness_score))
    {
        result.converged = true;
    }

    if (!result.converged) return result;

    // Rebuild correspondences and the undamped information matrix at the final
    // state. The solved increment is intentionally discarded here.
    findCorrespondences(source, iterate, correspondences);
    if (!buildAndSolve(source, correspondences, prior, prior_information, iterate,
                       information, increment, result.num_correspondences, squared_error_sum)) return result;
    result.fitness_score = squared_error_sum / static_cast<double>(result.num_correspondences);
    if (result.fitness_score > mConfig.max_fitness_score) return result;

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
    insertCloud(cloud, false);
    result.accepted = true;
    return result;
}
}  // namespace lio
