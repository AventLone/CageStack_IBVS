#include "perception/GICP/GICP.h"
#include <algorithm>
#include <execution>

std::unique_ptr<SparseVoxel> GICP::createSourceIndex(
    const pcl::PointCloud<pcl::PointXYZ>& source) const
{
    auto source_index = std::make_unique<SparseVoxel>(mSparseVoxelConfig);
    source_index->initialize(source);
    return source_index;
}

void GICP::findCorrespondences(const std::vector<const PointWithCovariance*>& source,
                                            const SparseVoxel& target,
                                            const Sophus::SE3d& source_to_target,
                                            std::vector<Correspondence>& correspondences) const
{
    std::transform(std::execution::par, source.begin(), source.end(), correspondences.begin(),
        [&target, &source_to_target, this](const PointWithCovariance* source_point)
        {
            const Eigen::Vector3f transformed_position = source_to_target.cast<float>() * source_point->position;
            constexpr int adjacent_voxels = 1;
            const SparseVoxel::Neighbor nearest = target.nearestNeighbor(
                transformed_position, mConfig.max_correspondence_distance, adjacent_voxels);
            return Correspondence{nearest.point, transformed_position.cast<double>()};
        });
}

GICP::Linearization GICP::linearize(const SparseVoxel& source,
                                    const SparseVoxel& target,
                                    const Sophus::SE3d& source_to_target) const
{
    Linearization result;
    const std::vector<const PointWithCovariance*> source_points = source.points();
    if (source_points.empty() || target.empty())
    {
        return result;
    }

    std::vector<Correspondence> correspondences(source_points.size());
    findCorrespondences(source_points, target, source_to_target, correspondences);
    const Eigen::Matrix3d rotation = source_to_target.rotationMatrix();

    for (std::size_t index = 0; index < correspondences.size(); ++index)
    {
        const auto& [target_point, transformed_position] = correspondences[index];
        if (target_point == nullptr)
        {
            continue;
        }

        const PointWithCovariance& source_point = *source_points[index];

        Eigen::Matrix3d covariance = Eigen::Matrix3d::Identity();
        if (source_point.covariance_valid)
        {
            if (target_point->covariance_valid)
            {
                covariance = target_point->covariance.cast<double>();
            }
            covariance += rotation * source_point.covariance.cast<double>() * rotation.transpose();
        }

        const Eigen::Matrix3d precision = covariance.inverse();
        if (!precision.allFinite())
        {
            continue;
        }

        const Eigen::Vector3d residual = transformed_position - target_point->position.cast<double>();
        const Eigen::Vector3d precision_residual = precision * residual;

        const double mahalanobis_error = residual.dot(precision_residual);
        const double kernel_scale2 = mConfig.cauchy_kernel_scale * mConfig.cauchy_kernel_scale;
        const double weight = mConfig.cauchy_kernel_scale > 0.0 ? 1.0 / (1.0 + mahalanobis_error / kernel_scale2) : 1.0;

        Eigen::Matrix<double, 3, 6> jacobian;
        // Right perturbation: T' = T * Exp(delta_rho_L, delta_theta_L).
        // Both increments are expressed in the source/LiDAR frame.
        jacobian.leftCols<3>() = rotation;
        jacobian.rightCols<3>() = -rotation * Sophus::SO3d::hat(source_point.position.cast<double>());
        result.hessian.noalias() += jacobian.transpose() * weight * precision * jacobian;
        result.gradient.noalias() += jacobian.transpose() * weight * precision_residual;
        result.squared_error_sum += residual.squaredNorm();
        ++result.num_correspondences;
    }
    return result;
}

GICP::Result GICP::align(const pcl::PointCloud<pcl::PointXYZ>& source,
                                                   const Eigen::Isometry3d& initial_guess) const
{
    if (source.empty())
    {
        throw std::invalid_argument("source cloud is empty!");
    }
    if (!hasTarget())
    {
        throw std::runtime_error("Target was not initialized before you call this method!");
    }

    Result result;
    result.transform = initial_guess;

    const auto source_index = createSourceIndex(source);
    result.num_source_points = source_index->pointCount();
    result.num_target_points = mTarget->pointCount();
    if (source_index->empty() || mTarget->empty())
    {
        return result;
    }

    Sophus::SE3d source_to_target(initial_guess.rotation(), initial_guess.translation());

    for (int iteration = 0; iteration < mConfig.max_iterations; ++iteration)
    {
        const Linearization system = linearize(*source_index, *mTarget, source_to_target);
        result.num_correspondences = system.num_correspondences;
        result.fitness_score = system.fitnessScore();
        if (system.num_correspondences == 0 || !std::isfinite(system.squared_error_sum))
        {
            break;
        }

        Hessian damped_hessian = system.hessian;
        damped_hessian.diagonal().array() += mConfig.damping_factor;
        const Eigen::LDLT<Hessian> decomposition(damped_hessian);
        if (decomposition.info() != Eigen::Success) break;
        const Sophus::SE3d::Tangent right_increment = decomposition.solve(-system.gradient);
        if (decomposition.info() != Eigen::Success || !right_increment.allFinite()) break;

        source_to_target = source_to_target * Sophus::SE3d::exp(right_increment);

        ++result.iterations;
        result.transform = Eigen::Isometry3d(source_to_target.matrix());

        if (right_increment.head<3>().norm() < mConfig.convergence_translation &&
            right_increment.tail<3>().norm() < mConfig.convergence_rotation)
        {
            result.converged = true;
            break;
        }
    }

    if (!result.converged && result.iterations > 0 && std::isfinite(result.fitness_score))
    {
        result.converged = true;
    }
    return result;
}
