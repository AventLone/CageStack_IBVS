#include "perception/LIO/IESKF.h"
#include <array>
#include <numeric>
#include <execution>
#include <iostream>

namespace lio
{
bool IESKF::initialize(const std::vector<ImuData>& samples)
{
    if (samples.size() < 30 || samples.back().timestamp - samples.front().timestamp < 2.0)
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
    for (const auto& sample : samples)
    {
        gyro_variance += (sample.gyro - gyro).squaredNorm();
        accel_variance += (sample.accel - accel).squaredNorm();
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

void IESKF::predict(const ImuData& imu_data)
{
    if (!mInitialized || !std::isfinite(imu_data.timestamp) ||
        !imu_data.accel.allFinite() || !imu_data.gyro.allFinite())
    {
        return;
    }

    const double dt = imu_data.timestamp - mState.timestamp;
    if (dt <= 0.0)
    {
        return;
    }
    if (dt > 0.05)
    {
        mState.timestamp = imu_data.timestamp;
        return;
    }

    const Eigen::Vector3d accel = imu_data.accel - mState.accel_bias;
    const Eigen::Vector3d gyro = imu_data.gyro - mState.gyro_bias;
    const Eigen::Vector3d phi = gyro * dt;
    const Sophus::SO3d half_rotation = Sophus::SO3d::exp(0.5 * phi);
    const Sophus::SO3d rotation_increment = half_rotation * half_rotation;
    const Eigen::Matrix3d rotation_mid = (mState.R_wi * half_rotation).matrix();
    const Eigen::Vector3d accel_world = rotation_mid * accel + mState.gravity;
    const double half_dt = 0.5 * dt;

    mState.p_wi += mState.v_wi * dt + half_dt * dt * accel_world;
    mState.v_wi += dt * accel_world;
    mState.R_wi *= rotation_increment;

    const Eigen::Matrix3d rotated_accel_hat = rotation_mid * Sophus::SO3d::hat(accel);
    const Eigen::Matrix3d velocity_rotation = -dt * rotated_accel_hat * half_rotation.inverse().matrix();
    const Eigen::Matrix3d velocity_accel_bias = -dt * rotation_mid;
    const Eigen::Matrix3d velocity_gyro_bias = half_dt * dt * rotated_accel_hat * rightJacobianSO3(0.5 * phi);
    const Eigen::Matrix3d position_rotation = half_dt * velocity_rotation;
    const Eigen::Matrix3d position_accel_bias = half_dt * velocity_accel_bias;
    const Eigen::Matrix3d position_gyro_bias = half_dt * velocity_gyro_bias;
    const Eigen::Matrix3d rotation_transition = rotation_increment.inverse().matrix();
    const Eigen::Matrix3d rotation_gyro_bias = -dt * rightJacobianSO3(phi);

    StateCov left = mP;
    left.topRows<3>().noalias() += position_rotation * mP.middleRows<3>(3) +
        dt * mP.middleRows<3>(6) + position_accel_bias * mP.middleRows<3>(9) +
        position_gyro_bias * mP.bottomRows<3>();
    left.middleRows<3>(3).noalias() = rotation_transition * mP.middleRows<3>(3) +
        rotation_gyro_bias * mP.bottomRows<3>();
    left.middleRows<3>(6).noalias() += velocity_rotation * mP.middleRows<3>(3) +
        velocity_accel_bias * mP.middleRows<3>(9) + velocity_gyro_bias * mP.bottomRows<3>();

    StateCov propagated = left;
    propagated.leftCols<3>().noalias() += left.middleCols<3>(3) * position_rotation.transpose() +
        dt * left.middleCols<3>(6) + left.middleCols<3>(9) * position_accel_bias.transpose() +
        left.rightCols<3>() * position_gyro_bias.transpose();
    propagated.middleCols<3>(3).noalias() = left.middleCols<3>(3) * rotation_transition.transpose() +
        left.rightCols<3>() * rotation_gyro_bias.transpose();
    propagated.middleCols<3>(6).noalias() += left.middleCols<3>(3) * velocity_rotation.transpose() +
        left.middleCols<3>(9) * velocity_accel_bias.transpose() +
        left.rightCols<3>() * velocity_gyro_bias.transpose();

    Eigen::Matrix<double, 9, 6> noise = Eigen::Matrix<double, 9, 6>::Zero();
    noise.block<3, 3>(0, 0) = position_accel_bias;
    noise.block<3, 3>(0, 3) = position_gyro_bias;
    noise.block<3, 3>(3, 3) = rotation_gyro_bias;
    noise.block<3, 3>(6, 0) = velocity_accel_bias;
    noise.block<3, 3>(6, 3) = velocity_gyro_bias;
    propagated.topLeftCorner<9, 9>().noalias() += noise * mQ * noise.transpose();

    constexpr double accel_bias_random_walk_std = 1e-4;
    constexpr double gyro_bias_random_walk_std = 0.001 * DEG2RAD;
    propagated.block<3, 3>(9, 9).diagonal().array() += dt * accel_bias_random_walk_std * accel_bias_random_walk_std;
    propagated.block<3, 3>(12, 12).diagonal().array() += dt * gyro_bias_random_walk_std * gyro_bias_random_walk_std;
    mP = (0.5 * (propagated + propagated.transpose())).eval();
    mState.timestamp = imu_data.timestamp;
}

void IESKF::observe(const pcl::PointCloud<pcl::PointXYZ>& scan)
{
    if (!mInitialized || scan.empty() || mVoxelMap->empty())
    {
        return;
    }

    using PoseCov = Eigen::Matrix<double, 6, 6>;
    using PoseError = Eigen::Matrix<double, 6, 1>;
    using ConditionalProjection = Eigen::Matrix<double, 9, 6>;

    const NavState prior_state = mState;
    const StateCov prior_cov = mP;
    const Eigen::LLT<PoseCov> prior_solver(prior_cov.topLeftCorner<6, 6>());
    if (prior_solver.info() != Eigen::Success)
    {
        return;
    }
    const PoseCov prior_information = prior_solver.solve(PoseCov::Identity());
    const ConditionalProjection conditional_projection = prior_cov.bottomLeftCorner<9, 6>() * prior_information;
    const Eigen::Matrix<double, 9, 9> conditional_covariance = prior_cov.bottomRightCorner<9, 9>() -
        conditional_projection * prior_cov.topRightCorner<6, 9>();

    constexpr int max_iterations = 4;
    constexpr int num_neighbors = 5;
    constexpr int min_neighbors = 3;
    constexpr float max_correspondence_distance = 1.0f;
    constexpr float plane_threshold = 0.1f;
    constexpr double max_point_plane_distance = 0.2;
    constexpr double huber_distance = 0.05;
    constexpr double lidar_point_std = 0.001;
    constexpr double lidar_point_variance = lidar_point_std * lidar_point_std;
    constexpr double position_convergence = 1e-4;
    constexpr double rotation_convergence = 1e-4;
    constexpr double rematch_position = 0.02;
    constexpr double rematch_rotation = 0.005;

    struct ObservationPoint
    {
        Eigen::Vector3d point;
        Eigen::Vector3d normal{Eigen::Vector3d::Zero()};
        Eigen::Vector3d center{Eigen::Vector3d::Zero()};
        double variance{lidar_point_variance};
        bool valid{false};
    };

    std::vector<ObservationPoint> points;
    points.reserve(scan.size());
    for (const auto& point : scan)
    {
        if (point.getVector3fMap().allFinite())
        {
            points.push_back({point.getVector3fMap().cast<double>()});
        }
    }
    if (points.size() < 10)
    {
        return;
    }
    std::vector<std::size_t> point_indices(points.size());
    std::iota(point_indices.begin(), point_indices.end(), std::size_t{0});

    struct LidarContribution
    {
        PoseCov information{PoseCov::Zero()};
        PoseError gradient{PoseError::Zero()};
        int effective_points{0};
    };

    const auto reduceContribution = [](LidarContribution lhs, const LidarContribution& rhs)
    {
        lhs.information += rhs.information;
        lhs.gradient += rhs.gradient;
        lhs.effective_points += rhs.effective_points;
        return lhs;
    };
    const auto restorePrior = [&]
    {
        mState = prior_state;
        mP = prior_cov;
    };

    PoseCov final_pose_covariance = PoseCov::Zero();
    ConditionalProjection final_projection = conditional_projection;
    Eigen::Vector3d final_dtheta = Eigen::Vector3d::Zero();
    Eigen::Vector3d correspondence_position = mState.p_wi;
    Sophus::SO3d correspondence_rotation = mState.R_wi;
    bool rematch = true;

    for (int iteration = 0; iteration < max_iterations; ++iteration)
    {
        const StateT prior_error = stateDifference(mState, prior_state);
        PoseCov prior_jacobian = PoseCov::Identity();
        prior_jacobian.block<3, 3>(3, 3) = rightJacobianInverseSO3(prior_error.segment<3>(3));
        PoseCov information = prior_jacobian.transpose() * prior_information * prior_jacobian;
        PoseError gradient = prior_jacobian.transpose() * prior_information * prior_error.head<6>();
        const Eigen::Matrix3d R = mState.R_wi.matrix();
        const Eigen::Vector3d p = mState.p_wi;
        if (rematch)
        {
            correspondence_position = p;
            correspondence_rotation = mState.R_wi;
        }

        const LidarContribution lidar = std::transform_reduce(std::execution::par, point_indices.begin(), point_indices.end(),
            LidarContribution{}, reduceContribution, [&](std::size_t index)
            {
                ObservationPoint& point = points[index];
                LidarContribution contribution;
                const Eigen::Vector3d point_w = R * point.point + p;
                if (rematch)
                {
                    point.valid = false;
                    std::array<IVox::Neighbor, num_neighbors> neighbors;
                    const std::size_t count = mVoxelMap->nearestNeighbors(point_w.cast<float>(),
                        std::span<IVox::Neighbor>(neighbors), max_correspondence_distance);
                    if (static_cast<int>(count) < min_neighbors)
                    {
                        return contribution;
                    }
                    const LocalPlane plane = estimatePlane(std::span<const IVox::Neighbor>(neighbors.data(), count),
                        plane_threshold);
                    if (!plane.valid)
                    {
                        return contribution;
                    }
                    point.normal = plane.normal.cast<double>();
                    point.center = plane.center.cast<double>();
                    point.variance = lidar_point_variance + static_cast<double>(plane.normal_variance);
                    point.valid = true;
                }
                if (!point.valid)
                {
                    return contribution;
                }

                const double residual = point.normal.dot(point_w - point.center);
                const double absolute_residual = std::abs(residual);
                if (absolute_residual > max_point_plane_distance)
                {
                    return contribution;
                }
                PoseError jacobian;
                jacobian.head<3>() = point.normal;
                jacobian.tail<3>() = point.point.cross(R.transpose() * point.normal);
                const double robust_weight = absolute_residual <= huber_distance ? 1.0 : huber_distance / absolute_residual;
                const double weight = robust_weight / point.variance;
                contribution.information.selfadjointView<Eigen::Upper>().rankUpdate(jacobian, weight);
                contribution.gradient = (weight * residual) * jacobian;
                contribution.effective_points = 1;
                return contribution;
            });

        if (lidar.effective_points < 10)
        {
            restorePrior();
            return;
        }

        information += lidar.information.selfadjointView<Eigen::Upper>();
        gradient += lidar.gradient;
        const Eigen::LLT<PoseCov> solver(information);
        if (solver.info() != Eigen::Success)
        {
            restorePrior();
            return;
        }
        StateT dx;
        dx.head<6>() = -solver.solve(gradient);
        dx.tail<9>() = conditional_projection * (prior_error.head<6>() + prior_jacobian * dx.head<6>()) -
            prior_error.tail<9>();
        if (!dx.allFinite())
        {
            restorePrior();
            return;
        }
        update(dx);
        final_pose_covariance = solver.solve(PoseCov::Identity());
        final_projection.noalias() = conditional_projection * prior_jacobian;
        final_dtheta = dx.segment<3>(3);

        if (dx.segment<3>(0).norm() < position_convergence &&
            dx.segment<3>(3).norm() < rotation_convergence)
        {
            if (rematch)
            {
                break;
            }
            rematch = true;
        }
        else
        {
            rematch = (mState.p_wi - correspondence_position).norm() > rematch_position ||
                (correspondence_rotation.inverse() * mState.R_wi).log().norm() > rematch_rotation;
        }
    }

    StateCov posterior_cov;
    posterior_cov.topLeftCorner<6, 6>() = final_pose_covariance;
    posterior_cov.bottomLeftCorner<9, 6>().noalias() = final_projection * final_pose_covariance;
    posterior_cov.topRightCorner<6, 9>() = posterior_cov.bottomLeftCorner<9, 6>().transpose();
    posterior_cov.bottomRightCorner<9, 9>() = conditional_covariance;
    posterior_cov.bottomRightCorner<9, 9>().noalias() +=
        posterior_cov.bottomLeftCorner<9, 6>() * final_projection.transpose();

    const Eigen::Matrix3d reset_rotation = rightJacobianSO3(final_dtheta);
    posterior_cov.middleRows<3>(3) = (reset_rotation * posterior_cov.middleRows<3>(3)).eval();
    posterior_cov.middleCols<3>(3) = (posterior_cov.middleCols<3>(3) * reset_rotation.transpose()).eval();
    mP = (0.5 * (posterior_cov + posterior_cov.transpose())).eval();

    pcl::PointCloud<pcl::PointXYZ> scan_world;
    scan_world.resize(points.size());
    const Eigen::Matrix3d R_final = mState.R_wi.matrix();
    const Eigen::Vector3d& p_final = mState.p_wi;
    std::transform(std::execution::par, points.begin(), points.end(), scan_world.begin(),
        [&](const ObservationPoint& point)
        {
            const Eigen::Vector3f point_w = (R_final * point.point + p_final).cast<float>();
            return pcl::PointXYZ(point_w[0], point_w[1], point_w[2]);
        });
    mVoxelMap->insert(scan_world);
}
}