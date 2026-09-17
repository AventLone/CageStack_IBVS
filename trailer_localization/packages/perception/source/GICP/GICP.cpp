#include "perception/GICP/GICP.h"
#include <algorithm>
#include <execution>

void GICP::initializeTarget(const pcl::PointCloud<pcl::PointXYZ>& target)
{
    auto target_index = std::make_unique<SparseVoxel>(mSparseVoxelConfig);
    target_index->initialize(target);
    mTarget = std::move(target_index);
}

void GICP::insertTargetPoints(const pcl::PointCloud<pcl::PointXYZ>& points)
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
            const SparseVoxel::Neighbor nearest = target.nearestNeighbor(transformed_position,
                                                                         mConfig.max_correspondence_distance,
                                                                         adjacent_voxels);
            return Correspondence{nearest.point, transformed_position.cast<double>()};
        });
}

std::optional<std::pair<std::size_t, double>> GICP::buildAndSolve(const std::vector<const PointWithCovariance*>& source,
                                                                  const std::vector<Correspondence>& correspondences,
                                                                  Sophus::SE3d& source_to_target,
                                                                  Sophus::SE3d::Tangent& left_increment) const
{
    double fitness_score{};

    Eigen::Matrix<double, 6, 6> hessian = Eigen::Matrix<double, 6, 6>::Zero();
    Eigen::Matrix<double, 6, 1> gradient = Eigen::Matrix<double, 6, 1>::Zero();

    double squared_error_sum = 0.0f;
    int valid_count = 0;
    const Eigen::Matrix3d rotation = source_to_target.rotationMatrix();

    for (std::size_t index = 0; index < correspondences.size(); ++index)
    {
        const auto& [target_point, transformed_position] = correspondences[index];
        if (target_point == nullptr)
        {
            continue;
        }

        const PointWithCovariance& source_point = *source[index];

        Eigen::Matrix3d covariance = Eigen::Matrix3d::Identity();
        if (source_point.covariance_valid)
        {
            // covariance = target_point->covariance_valid ? target_point->covariance : Eigen::Matrix3f::Identity();
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
        const double weight = mConfig.cauchy_kernel_scale > 0.0f ? 1.0f / (1.0f + mahalanobis_error / kernel_scale2) : 1.0f;

        Eigen::Matrix<double, 3, 6> jacobian;
        jacobian.leftCols<3>().setIdentity();
        jacobian.rightCols<3>() = -Sophus::SO3d::hat(transformed_position.cast<double>());
        hessian.noalias() += jacobian.transpose() * weight * precision * jacobian;
        gradient.noalias() += jacobian.transpose() * weight * precision_residual;
        squared_error_sum += residual.squaredNorm();
        ++valid_count;
    }

    auto num_correspondences = static_cast<std::size_t>(valid_count);
    fitness_score = valid_count > 0 ? squared_error_sum / static_cast<double>(valid_count) : std::numeric_limits<double>::infinity();
    if (valid_count == 0 || !std::isfinite(squared_error_sum))
    {
        return std::nullopt;
    }

    hessian.diagonal().array() += mConfig.damping_factor;
    const Eigen::LDLT<Eigen::Matrix<double, 6, 6>> decomposition(hessian);
    if (decomposition.info() != Eigen::Success)
    {
        return std::nullopt;
    }

    left_increment = decomposition.solve(-gradient);
    if (decomposition.info() != Eigen::Success || !left_increment.allFinite())
    {
        return std::nullopt;
    }

    source_to_target = Sophus::SE3d::exp(left_increment) * source_to_target;
    return std::make_pair(num_correspondences, fitness_score);
}

GICP::Result GICP::align(const pcl::PointCloud<pcl::PointXYZ>& source, const Eigen::Isometry3d& initial_guess) const
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

    Sophus::SE3d source_to_target(initial_guess.rotation(), initial_guess.translation());

    for (int iteration = 0; iteration < mConfig.max_iterations; ++iteration)
    {
        findCorrespondences(source_points, *mTarget, source_to_target, correspondences);

        Sophus::SE3d::Tangent left_increment;
        if (const auto calculation = buildAndSolve(source_points, correspondences, source_to_target, left_increment);
            calculation.has_value())
        {
            result.num_correspondences = calculation.value().first;
            result.fitness_score = calculation.value().second;
        }
        else
        {
            break;
        }

        ++result.iterations;
        result.transform = Eigen::Isometry3d(source_to_target.matrix());

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
