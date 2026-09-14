#include "perception/GICP/SparsityAwareGICP.h"
#include <algorithm>
#include <execution>

void SparsityAwareGICP::initializeTarget(const pcl::PointCloud<pcl::PointXYZ>& target)
{
    auto target_index = std::make_unique<SparseVoxel>(mSparseVoxelConfig);
    target_index->initialize(target);
    mTarget = std::move(target_index);
}

void SparsityAwareGICP::insertTargetPoints(const pcl::PointCloud<pcl::PointXYZ>& points)
{
    if (points.empty())
    {
        return;
    }
    if (!hasTarget())
    {
        initializeTarget(points);
        return;
    }
    mTarget->insert(points);
}

void SparsityAwareGICP::findCorrespondences(const std::vector<const PointWithCovariance*>& source,
                                            const SparseVoxel& target,
                                            const Sophus::SE3f& source_to_target,
                                            std::vector<Correspondence>& correspondences) const
{
    std::transform(std::execution::par, source.begin(), source.end(), correspondences.begin(),
        [&target, &source_to_target, this](const PointWithCovariance* source_point)
        {
            const Eigen::Vector3f transformed_position = source_to_target * source_point->position;
            constexpr int adjacent_voxels = 1;
            const SparseVoxel::Neighbor nearest = target.nearestNeighbor(
                transformed_position, mConfig.max_correspondence_distance, adjacent_voxels);
            return Correspondence{nearest.point, transformed_position};
        });
}

bool SparsityAwareGICP::buildAndSolve(const std::vector<const PointWithCovariance*>& source,
                   const std::vector<Correspondence>& correspondences, Sophus::SE3f& source_to_target,
                   std::size_t& num_correspondences, float& fitness_score, Sophus::SE3f::Tangent& left_increment) const
{
    Eigen::Matrix<float, 6, 6> hessian = Eigen::Matrix<float, 6, 6>::Zero();
    Eigen::Matrix<float, 6, 1> gradient = Eigen::Matrix<float, 6, 1>::Zero();

    float squared_error_sum = 0.0f;
    int valid_count = 0;
    const Eigen::Matrix3f rotation = source_to_target.rotationMatrix();

    for (std::size_t index = 0; index < correspondences.size(); ++index)
    {
        const auto& [target_point, transformed_position] = correspondences[index];
        if (target_point == nullptr)
        {
            continue;
        }

        const PointWithCovariance& source_point = *source[index];

        Eigen::Matrix3f covariance = Eigen::Matrix3f::Identity();
        if (source_point.covariance_valid)
        {
            covariance = target_point->covariance_valid ? target_point->covariance : Eigen::Matrix3f::Identity();
            covariance += rotation * source_point.covariance * rotation.transpose();
        }

        const Eigen::Matrix3f precision = covariance.inverse();
        if (!precision.allFinite())
        {
            continue;
        }

        const Eigen::Vector3f residual = transformed_position - target_point->position;
        const Eigen::Vector3f precision_residual = precision * residual;

        const float mahalanobis_error = residual.dot(precision_residual);
        const float kernel_scale2 = mConfig.cauchy_kernel_scale * mConfig.cauchy_kernel_scale;
        const float weight = mConfig.cauchy_kernel_scale > 0.0f ? 1.0f / (1.0f + mahalanobis_error / kernel_scale2) : 1.0f;

        Eigen::Matrix<float, 3, 6> jacobian;
        jacobian.leftCols<3>().setIdentity();
        jacobian.rightCols<3>() = -Sophus::SO3f::hat(transformed_position);
        hessian.noalias() += jacobian.transpose() * weight * precision * jacobian;
        gradient.noalias() += jacobian.transpose() * weight * precision_residual;
        squared_error_sum += residual.squaredNorm();
        ++valid_count;
    }

    num_correspondences = static_cast<std::size_t>(valid_count);
    fitness_score = valid_count > 0 ? squared_error_sum / static_cast<float>(valid_count) : std::numeric_limits<float>::infinity();
    if (valid_count == 0 || !std::isfinite(squared_error_sum))
    {
        return false;
    }

    hessian.diagonal().array() += mConfig.damping_factor;
    const Eigen::LDLT<Eigen::Matrix<float, 6, 6>> decomposition(hessian);
    if (decomposition.info() != Eigen::Success)
    {
        return false;
    }

    left_increment = decomposition.solve(-gradient);
    if (decomposition.info() != Eigen::Success || !left_increment.allFinite())
    {
        return false;
    }

    source_to_target = Sophus::SE3f::exp(left_increment) * source_to_target;
    return true;
}

SparsityAwareGICP::Result SparsityAwareGICP::align(const pcl::PointCloud<pcl::PointXYZ>& source,
                                                   const Eigen::Isometry3f& initial_guess) const
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

    SparseVoxel source_index(mSparseVoxelConfig);
    source_index.initialize(source);
    result.num_source_points = source_index.pointCount();
    result.num_target_points = mTarget->pointCount();
    if (source_index.empty() || mTarget->empty())
    {
        return result;
    }

    const std::vector<const PointWithCovariance*> source_points = source_index.points();
    std::vector<Correspondence> correspondences(source_points.size());

    Sophus::SE3f source_to_target(initial_guess.rotation(), initial_guess.translation());

    for (int iteration = 0; iteration < mConfig.max_iterations; ++iteration)
    {
        findCorrespondences(source_points, *mTarget, source_to_target, correspondences);

        Sophus::SE3f::Tangent left_increment;
        if (!buildAndSolve(source_points, correspondences, source_to_target,
                           result.num_correspondences, result.fitness_score, left_increment))
        {
            break;
        }

        ++result.iterations;
        result.transform = Eigen::Isometry3f(source_to_target.matrix());

        if (left_increment.head<3>().norm() < mConfig.convergence_translation &&
            left_increment.tail<3>().norm() < mConfig.convergence_rotation)
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
