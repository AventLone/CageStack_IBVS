#include "perception/GICP/GICP.h"
#include <algorithm>
#include <execution>

void GICP::insertTargetPoints(const pcl::PointCloud<pcl::PointXYZ>& points)
{
    if (points.empty())
    {
        return;
    }
    mTarget.insert(points);
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
            const SparseVoxel::Neighbor nearest = target.nearestNeighbor(transformed_position, mConfig.max_correspondence_distance);
            return Correspondence{nearest.point, transformed_position.cast<double>()};
        });
}

std::optional<std::pair<std::size_t, double>> GICP::findCorrespondencesAndSolve(const std::vector<const PointWithCovariance*>& source,
                                                                                Sophus::SE3d& source_to_target,
                                                                                Sophus::SE3d::Tangent& left_increment,
                                                                                Eigen::Matrix<double, 6, 6>& hessian_) const
{
    struct Accumulator
    {
        Eigen::Matrix<double, 6, 6> hessian = Eigen::Matrix<double, 6, 6>::Zero();
        Eigen::Matrix<double, 6, 1> gradient = Eigen::Matrix<double, 6, 1>::Zero();

        double squared_error_sum = 0.0;
        std::size_t valid_count = 0;

        Accumulator& operator+=(const Accumulator& other)
        {
            hessian += other.hessian;
            gradient += other.gradient;
            squared_error_sum += other.squared_error_sum;
            valid_count += other.valid_count;
            return *this;
        }
    };

    // For covariance transformation and optimization.
    const Eigen::Matrix3d rotation = source_to_target.rotationMatrix();


    const Accumulator accumulator = std::transform_reduce(std::execution::par, source.begin(), source.end(), Accumulator{},
            [](Accumulator lhs, const Accumulator& rhs)   // Reduction
            {
                lhs += rhs;
                return lhs;
            },
            [&](const PointWithCovariance* source_point)   // Transform: source point -> correspondence -> normal equation contribution
            {
                Accumulator local;

                // ---------------------------------------------------------
                // 1. Transform source point
                // ---------------------------------------------------------
                const Eigen::Vector3d transformed_position = source_to_target * source_point->position.cast<double>();

                // ---------------------------------------------------------
                // 2. Find correspondence
                // ---------------------------------------------------------
                const SparseVoxel::Neighbor nearest = mTarget.nearestNeighbor(transformed_position.cast<float>(), mConfig.max_correspondence_distance);
                const PointWithCovariance* target_point = nearest.point;
                if (target_point == nullptr)
                {
                    return local;
                }

                // ---------------------------------------------------------
                // 3. GICP covariance
                // ---------------------------------------------------------
                Eigen::Matrix3d covariance = Eigen::Matrix3d::Identity();
                // This preserves the behavior of your original code.
                if (source_point->covariance_valid)
                {
                    if (target_point->covariance_valid)
                    {
                        covariance = target_point->covariance.cast<double>();
                    }
                    covariance.noalias() += rotation * source_point->covariance.cast<double>() * rotation.transpose();
                }

                const Eigen::LDLT<Eigen::Matrix3d> covariance_ldlt(covariance);

                if (covariance_ldlt.info() != Eigen::Success)
                {
                    return local;
                }

                // ---------------------------------------------------------
                // 4. Residual
                // ---------------------------------------------------------
                const Eigen::Vector3d residual = transformed_position - target_point->position.cast<double>();
                const Eigen::Vector3d precision_residual = covariance_ldlt.solve(residual);
                if (!precision_residual.allFinite())
                {
                    return local;
                }

                // ---------------------------------------------------------
                // 5. Jacobian
                // Left perturbation:
                // T' = exp(delta_xi) * T
                // J = [ I  -hat(p) ]
                // ---------------------------------------------------------
                Eigen::Matrix<double, 3, 6> jacobian;
                jacobian.leftCols<3>().setIdentity();
                jacobian.rightCols<3>() = -Sophus::SO3d::hat(transformed_position);
                const  Eigen::Matrix<double, 3, 6> precision_jacobian = covariance_ldlt.solve(jacobian);   // covariance^-1 * J

                local.hessian.noalias() = jacobian.transpose() * precision_jacobian;
                local.gradient.noalias() = jacobian.transpose() * precision_residual;
                local.squared_error_sum = residual.squaredNorm();
                local.valid_count = 1;

                return local;
            });

    // ---------------------------------------------------------------------
    // Solve
    // ---------------------------------------------------------------------
    if (accumulator.valid_count == 0 || !std::isfinite(accumulator.squared_error_sum))
    {
        return std::nullopt;
    }

    hessian_ = accumulator.hessian;

    const Eigen::LDLT< Eigen::Matrix<double, 6, 6>> decomposition(hessian_);
    if (decomposition.info() != Eigen::Success)
    {
        return std::nullopt;
    }

    left_increment = decomposition.solve(-accumulator.gradient);
    if (decomposition.info() != Eigen::Success || !left_increment.allFinite())
    {
        return std::nullopt;
    }

    // Left update
    source_to_target = Sophus::SE3d::exp(left_increment) * source_to_target;
    const double fitness_score = accumulator.squared_error_sum / static_cast<double>(accumulator.valid_count);

    return std::make_pair(accumulator.valid_count, fitness_score);
}

GICP::Result GICP::align(const pcl::PointCloud<pcl::PointXYZ>& source, const Eigen::Isometry3d& initial_guess) const
{
    if (source.empty())
    {
        throw std::invalid_argument("source cloud is empty!");
    }

    Result result;
    result.transform = initial_guess;

    SparseVoxel source_index;
    source_index.insert(source);
    result.num_source_points = source_index.pointCount();
    result.num_target_points = mTarget.pointCount();
    if (source_index.empty() || mTarget.empty())
    {
        return result;
    }

    const std::vector<const PointWithCovariance*> source_points = source_index.points();

    Sophus::SE3d source_to_target(initial_guess.rotation(), initial_guess.translation());
    Eigen::Matrix<double, 6, 6> hessian;
    for (int iteration = 0; iteration < mConfig.max_iterations; ++iteration)
    {
        Sophus::SE3d::Tangent left_increment;
        if (const auto calculation = findCorrespondencesAndSolve(source_points, source_to_target, left_increment, hessian);
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

        if (left_increment.head<3>().norm() < mConfig.convergence_translation &&
            left_increment.tail<3>().norm() < mConfig.convergence_rotation)
        {
            result.converged = true;
            break;
        }
    }
    result.transform = Eigen::Isometry3d(source_to_target.matrix());

    const Eigen::Matrix<double, 6, 6> gicp_cov_left = hessian.ldlt().solve(Eigen::Matrix<double, 6, 6>::Identity());
    Eigen::Matrix<double, 6, 6> A = Eigen::Matrix<double, 6, 6>::Zero();
    A.topLeftCorner<3, 3>().setIdentity();
    A.topRightCorner<3, 3>() = -Sophus::SO3d::hat(source_to_target.translation());
    A.bottomRightCorner<3, 3>() = source_to_target.rotationMatrix().transpose();
    result.measure_covariance.noalias() = A * gicp_cov_left * A.transpose();

    if (!result.converged && result.iterations > 0 && std::isfinite(result.fitness_score))
    {
        result.converged = true;
    }
    return result;
}
