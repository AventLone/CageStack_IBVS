#include "perception/LIO/IESKF.h"
#include <numeric>
#include <execution>
#include <iostream>
#include <format>

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
    const double dt = imu_data.timestamp - mState.timestamp;
    if (dt > 0.05 || dt < 0)
    {
        // 时间间隔不对，可能是第一个IMU数据，没有历史信息
        mState.timestamp = imu_data.timestamp;
        return;
    }

    auto& p = mState.p_wi;
    auto& R = mState.R_wi;
    auto& v = mState.v_wi;

    const Eigen::Vector3d accel = imu_data.accel - mState.accel_bias;
    const Eigen::Vector3d gyro = imu_data.gyro - mState.gyro_bias;
    const Eigen::Vector3d& g = mState.gravity;


    /* Linearize at R_k */
    const Sophus::SO3d R_old = R;
    const Eigen::Vector3d phi = gyro * dt;
    const Sophus::SO3d dR = Sophus::SO3d::exp(phi);

    const Eigen::Matrix3d accel_hat = Sophus::SO3d::hat(accel);
    const Eigen::Matrix3d Rm = R_old.matrix();
    const Eigen::Vector3d accel_world = R_old * accel + g;

    /* Nominal state propagation */
    p += v * dt + 0.5 * accel_world * dt * dt;
    v += accel_world * dt;
    R *= dR;

    StateJacobian F = StateJacobian::Identity();
    F.block<3, 3>(0, 3) = -0.5 * Rm * accel_hat * dt * dt;
    F.block<3, 3>(0, 6) = Eigen::Matrix3d::Identity() * dt;
    F.block<3, 3>(0, 9) = -0.5 * Rm * dt * dt;
    F.block<3, 3>(3, 3) = dR.inverse().matrix();
    F.block<3, 3>(3, 12) = -rightJacobianSO3(phi) * dt;
    F.block<3, 3>(6, 3) = -Rm * accel_hat * dt;   // dv
    F.block<3, 3>(6, 9) = -Rm * dt;    // accel bias affects velocity

    /*
     * --------------------------------------------------
     * IMU measurement noise
     * n = [na, ng]
     * --------------------------------------------------
     */
    NoiseJacobian G = NoiseJacobian::Zero();
    G.block<3, 3>(0, 0) = -0.5 * Rm * dt * dt;
    G.block<3, 3>(3, 3) = -rightJacobianSO3(phi) * dt;
    G.block<3, 3>(6, 0) = -Rm * dt;

    mP = F * mP * F.transpose() + G * mQ * G.transpose();   // Covariance propagation
    // mP.block<3, 3>(9, 9) += mAccelBiasRandomWalk * dt;
    // mP.block<3, 3>(12, 12) += mGyroBiasRandomWalk * dt;

    mP = 0.5 * (mP + mP.transpose());   // 强制把协方差矩阵恢复成对称矩阵

    mState.timestamp = imu_data.timestamp;
}

void IESKF::observe(const pcl::PointCloud<pcl::PointXYZ>& scan)
{
    if (!mInitialized || scan.empty())
    {
        return;
    }

    /*
     * IMU propagated prior.
     * All IEKF iterations must use the same prior covariance,
     * otherwise the same LiDAR measurement would be counted
     * multiple times.
     */
    const NavState prior_state = mState;
    const StateCov prior_cov = mP;
    const StateCov prior_information = prior_cov.ldlt().solve(StateCov::Identity());

    constexpr int max_iterations = 4;

    constexpr int num_neighbors = 5;
    constexpr int min_neighbors = 3;

    constexpr float max_correspondence_distance = 1.0f;
    constexpr float plane_threshold = 0.1f;

    constexpr double max_point_plane_distance = 0.2;

    constexpr double lidar_point_std = 0.001;
    constexpr double lidar_point_variance = lidar_point_std * lidar_point_std;

    constexpr double position_convergence = 1e-4;
    constexpr double rotation_convergence = 1e-4;

    /*
     * LiDAR directly observes only:
     *
     * [dp, dtheta]
     *
     * so each point only needs a 6x6 contribution.
     */
    struct LidarContribution
    {
        Eigen::Matrix<double, 6, 6> information{Eigen::Matrix<double, 6, 6>::Zero()};
        Eigen::Matrix<double, 6, 1> gradient{Eigen::Matrix<double, 6, 1>::Zero()};

        int effective_points{0};
    };

    const auto reduceContribution = [](LidarContribution lhs, const LidarContribution& rhs)
        {
            lhs.information += rhs.information;
            lhs.gradient += rhs.gradient;
            lhs.effective_points += rhs.effective_points;
            return lhs;
        };

    StateCov final_information = prior_information;
    Eigen::Vector3d final_dtheta = Eigen::Vector3d::Zero();
    for (int iteration = 0; iteration < max_iterations; ++iteration)
    {
        /*
         * --------------------------------------------------
         * Prior term
         * --------------------------------------------------
         * Current iterate relative to the IMU prediction.
         * dx_prior = [dp, dtheta, dv, dba, dbg]
         */
        StateT prior_error = StateT::Zero();
        prior_error.segment<3>(0) = mState.p_wi - prior_state.p_wi;
        prior_error.segment<3>(3) = (prior_state.R_wi.inverse() * mState.R_wi).log();
        prior_error.segment<3>(6) = mState.v_wi - prior_state.v_wi;
        prior_error.segment<3>(9) = mState.accel_bias - prior_state.accel_bias;
        prior_error.segment<3>(12) = mState.gyro_bias - prior_state.gyro_bias;

        /*
         * Jacobian of the prior error w.r.t. the current
         * right perturbation.
         */
        StateJacobian J_prior = StateJacobian::Identity();
        J_prior.block<3, 3>(3, 3) = rightJacobianInverseSO3(prior_error.segment<3>(3));
        StateCov information = J_prior.transpose() * prior_information * J_prior;
        StateT gradient = J_prior.transpose() * prior_information * prior_error;

        /*
         * Current iteration state.
         * Copy them outside the parallel lambda so every
         * worker uses exactly the same linearization point.
         */
        const Eigen::Matrix3d R = mState.R_wi.matrix();
        const Eigen::Vector3d p = mState.p_wi;

        /*
         * --------------------------------------------------
         * Parallel LiDAR observation construction
         * --------------------------------------------------
         */
        const LidarContribution lidar = std::transform_reduce(std::execution::par, scan.begin(), scan.end(),
                LidarContribution{}, reduceContribution, [&](const pcl::PointXYZ& point)
                {
                    LidarContribution contribution;

                    /*
                     * Point in IMU frame.
                     *
                     * If cloud is in LiDAR frame, apply
                     * T_il here first.
                     */
                    const Eigen::Vector3d point_i = point.getVector3fMap().cast<double>();

                    /* Transform into world frame: p_w = R_wi p_i + p_wi */
                    const Eigen::Vector3d point_w = R * point_i + p;
                    const auto neighbors = mVoxelMap->nearestNeighbors(point_w.cast<float>(), num_neighbors, max_correspondence_distance);
                    if (static_cast<int>(neighbors.size()) < min_neighbors)
                    {
                        return contribution;
                    }

                    const LocalPlane plane = estimatePlane(neighbors, plane_threshold);
                    if (!plane.valid)
                    {
                        return contribution;
                    }

                    const Eigen::Vector3d normal = plane.normal.cast<double>();
                    const Eigen::Vector3d center = plane.center.cast<double>();

                    const double residual = normal.dot(point_w - center);   // Point-to-plane residual: r = n^T (R p_i + p - center)
                    if (std::abs(residual) > max_point_plane_distance)
                    {
                        return contribution;
                    }

                    /*
                     * --------------------------------------------------
                     * Measurement Jacobian
                     * --------------------------------------------------
                     * Right perturbation:
                     * R_true = R Exp(dtheta)
                     * R Exp(dtheta) p
                     * ~= Rp - R[p]x dtheta
                     * Therefore: H6 = [ n^T, -n^T R [p_i]x ]
                     */
                    Eigen::Matrix<double, 1, 6> H;
                    H.block<1, 3>(0, 0) = normal.transpose();
                    H.block<1, 3>(0, 3) = -normal.transpose() * R * Sophus::SO3d::hat(point_i);

                    /*
                     * Plane roughness contributes to the
                     * measurement uncertainty.
                     */
                    const double variance = lidar_point_variance + static_cast<double>(plane.normal_variance);
                    const double weight = 1.0 / variance;

                    /*
                     * Information contribution:
                     * Λ_i = H_i^T R_i^-1 H_i
                     * g_i = H_i^T R_i^-1 r_i
                     */
                    contribution.information.noalias() = weight * H.transpose() * H;
                    contribution.gradient.noalias() = H.transpose() * (weight * residual);
                    contribution.effective_points = 1;

                    return contribution;
                });

        if (lidar.effective_points < 10)
        {
            mState = prior_state;
            mP = prior_cov;
            return;
        }

        /*
         * LiDAR directly touches only [p, theta].
         *
         * Bias / velocity are updated through the
         * cross-covariance contained in the prior.
         */
        information.block<6, 6>(0, 0) += lidar.information;
        gradient.segment<6>(0) += lidar.gradient;

        /*
         * --------------------------------------------------
         * Iterated update
         *
         * information * dx = -gradient
         * --------------------------------------------------
         */
        const Eigen::LDLT<StateCov> ldlt(information);
        const StateT dx = -ldlt.solve(gradient);
        update(dx);   // Inject correction into the current iterate.

        final_information = information;
        final_dtheta = dx.segment<3>(3);

        /*
         * Only pose convergence matters for the geometric
         * LiDAR iteration.
         */
        if (dx.segment<3>(0).norm() < position_convergence &&
            dx.segment<3>(3).norm() < rotation_convergence)
        {
            break;
        }
    }

    StateCov posterior_cov = final_information.ldlt().solve(StateCov::Identity());   // Posterior covariance

    /* Covariance reset after final SO(3) injection */
    StateJacobian reset = StateJacobian::Identity();
    reset.block<3, 3>(3, 3) = rightJacobianSO3(final_dtheta);
    mP = reset * posterior_cov * reset.transpose();
    mP = 0.5 *(mP + mP.transpose());

    /*
     * --------------------------------------------------
     * Transform corrected scan to world frame.
     * This transformation can also be parallelized.
     * --------------------------------------------------
     */
    pcl::PointCloud<pcl::PointXYZ> scan_world;
    scan_world.resize(scan.size());

    const Eigen::Matrix3d R_final = mState.R_wi.matrix();
    const Eigen::Vector3d& p_final = mState.p_wi;
    std::transform(std::execution::par, scan.begin(), scan.end(), scan_world.begin(),
        [&](const pcl::PointXYZ& point)
        {
            const Eigen::Vector3d point_i = point.getVector3fMap().cast<double>();
            const Eigen::Vector3f point_w = (R_final * point_i + p_final).cast<float>();
            return pcl::PointXYZ(point_w[0], point_w[1], point_w[2]);
        });

    mVoxelMap->insert(scan_world);
}
}