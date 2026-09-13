#include "perception/GICP/SparsityAwareGICP.h"
#include <algorithm>

void SparsityAwareGICP::initializeTarget(const pcl::PointCloud<pcl::PointXYZ>& target)
{
    VoxelPointLayout target_layout =  mProcesser.process(target);
    if (mConfig.max_target_voxels == 0 || target_layout.voxels.size() > mConfig.max_target_voxels)
    {
        throw std::invalid_argument("Initial target exceeds max_target_voxels or the voxel budget is zero");
    }
    std::vector<PointWithCovariance> target_points = estimateCovariances(target_layout);
    mTarget = std::make_unique<TargetCache>(target_layout, std::move(target_points));
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
    if (mTarget->voxels.size() >= mConfig.max_target_voxels)
    {
        return;
    }

    std::vector<SparsePoint> sparse_points = mProcesser.makeSparseCloud(points);
    std::erase_if(sparse_points, [this](const SparsePoint& point)
                      {
                          return mTarget->occupied_voxels.contains(point.key);
                      });
    if (sparse_points.empty())
    {
        return;
    }

    VoxelPointLayout new_layout = Preprocesser::makeVoxelPointLayout(sparse_points);
    if (const std::size_t available_voxels = mConfig.max_target_voxels - mTarget->voxels.size();
        new_layout.voxels.size() > available_voxels)
    {
        return;
    }

    std::vector<PointWithCovariance> new_points = estimateCovariances(new_layout);
    const int point_offset = static_cast<int>(mTarget->points.size());
    const int voxel_offset = static_cast<int>(mTarget->voxels.size());
    for (auto& [start, count] : new_layout.voxels)
    {
        start += point_offset;
    }
    mTarget->points.insert(mTarget->points.end(), new_points.begin(), new_points.end());
    mTarget->voxels.insert(mTarget->voxels.end(), new_layout.voxels.begin(), new_layout.voxels.end());
    for (int index = 0; index < static_cast<int>(new_layout.voxel_keys.size()); ++index)
    {
        mTarget->voxel_map.emplace(new_layout.voxel_keys[index], voxel_offset + index);
    }
    mTarget->occupied_voxels.insert(new_layout.voxel_keys.begin(), new_layout.voxel_keys.end());
}

std::vector<PointWithCovariance> SparsityAwareGICP::estimateCovariances(VoxelPointLayout layout) const
{
    std::unordered_map<VoxelKey, int, VoxelKeyHash> voxel_map;
    voxel_map.reserve(layout.voxel_keys.size());
    for (std::size_t index = 0; index < layout.voxel_keys.size(); ++index)
    {
        voxel_map.emplace(layout.voxel_keys[index], index);
    }

    const int min_neighbors = std::max(3, mConfig.min_covariance_neighbors);
    const int max_neighbors = std::max(min_neighbors, mConfig.max_covariance_neighbors);
    const float regularization = std::max(1.0e-6f, mConfig.covariance_regularization);
    std::vector<PointWithCovariance> points = std::move(layout.points);

    for (int index = 0; index < static_cast<int>(points.size()); ++index)
    {
        constexpr int voxel_radius = 1;    // 协方差邻域沿每个轴扩展的体素数 r

        const PointWithCovariance point = points[index];
        const auto [i, j, k] = pointToVoxel(point.position, mConfig.voxel_size);
        std::vector<std::pair<float, int>> neighbors;
        neighbors.reserve(max_neighbors + 1);

        for (int dx = -voxel_radius; dx <= voxel_radius; ++dx)
        {
            for (int dy = -voxel_radius; dy <= voxel_radius; ++dy)
            {
                for (int dz = -voxel_radius; dz <= voxel_radius; ++dz)
                {
                    const auto found = voxel_map.find(VoxelKey{i + dx, j + dy, k + dz});

                    if (found == voxel_map.end())
                    {
                        continue;
                    }

                    const auto [start, count] = layout.voxels[found->second];
                    for (int offset = 0; offset < count; ++offset)
                    {
                        const int candidate_index = start + offset;
                        neighbors.emplace_back((points[candidate_index].position - point.position).squaredNorm(), candidate_index);
                    }
                }
            }
        }

        if (static_cast<int>(neighbors.size()) > max_neighbors)
        {
            std::ranges::nth_element(neighbors, neighbors.begin() + max_neighbors);
            neighbors.resize(max_neighbors);
        }
        if (static_cast<int>(neighbors.size()) < min_neighbors)
        {
            continue;
        }

        Eigen::Vector3f mean = Eigen::Vector3f::Zero();
        for (const auto& [distance, neighbor_index] : neighbors)
        {
            (void)distance;
            mean += points[neighbor_index].position;
        }
        mean /= static_cast<float>(neighbors.size());

        Eigen::Matrix3f covariance = Eigen::Matrix3f::Zero();
        for (const auto& [distance, neighbor_index] : neighbors)
        {
            (void)distance;
            const Eigen::Vector3f offset = points[neighbor_index].position - mean;
            covariance.noalias() += offset * offset.transpose();
        }
        covariance /= static_cast<float>(neighbors.size() - 1);
        covariance.diagonal().array() += regularization;

        if (const float determinant = covariance.determinant();
            !std::isfinite(determinant) || std::abs(determinant) <= 1.0e-12f)
        {
            continue;
        }
        const float inverse_norm = covariance.inverse().norm();
        if (!std::isfinite(inverse_norm) || inverse_norm <= 0.0f)
        {
            continue;
        }
        points[index].covariance = covariance * inverse_norm;
        points[index].covariance_valid = true;
    }
    return points;
}

void SparsityAwareGICP::findCorrespondences(const std::vector<PointWithCovariance>& source, const std::vector<PointWithCovariance>& target,
                                            const std::vector<VoxelEntry>& target_voxels,
                                            const std::unordered_map<VoxelKey, int, VoxelKeyHash>& target_voxel_map,
                                            const Sophus::SE3f& source_to_target,
                                            std::vector<Correspondence>& correspondences) const
{
    const float max_distance2 = mConfig.max_correspondence_distance * mConfig.max_correspondence_distance;

    for (int index = 0; index < static_cast<int>(source.size()); ++index)
    {
        const Eigen::Vector3f transformed_position = source_to_target * source[index].position;
        const auto [i, j, k] = pointToVoxel(transformed_position, mConfig.voxel_size);
        float best_distance2 = max_distance2;
        int best_target = -1;

        constexpr int adjacent_voxels = 1;
        for (int dx = -adjacent_voxels; dx <= adjacent_voxels; ++dx)
        {
            for (int dy = -adjacent_voxels; dy <= adjacent_voxels; ++dy)
            {
                for (int dz = -adjacent_voxels; dz <= adjacent_voxels; ++dz)
                {
                    const auto found = target_voxel_map.find(VoxelKey{i + dx, j + dy, k + dz});
                    if (found == target_voxel_map.end())
                    {
                        continue;
                    }

                    const auto [start, count] = target_voxels[found->second];
                    for (int offset = 0; offset < count; ++offset)
                    {
                        const int target_index = start + offset;
                        if (const float distance2 = (transformed_position - target[target_index].position).squaredNorm();
                            distance2 < best_distance2)
                        {
                            best_distance2 = distance2;
                            best_target = target_index;
                        }
                    }
                }
            }
        }
        correspondences[index] = Correspondence{best_target, transformed_position};
    }
}

bool SparsityAwareGICP::buildAndSolve(const std::vector<PointWithCovariance>& source, const std::vector<PointWithCovariance>& target,
                   const std::vector<Correspondence>& correspondences, Sophus::SE3f& source_to_target,
                   std::size_t& num_correspondences, float& fitness_score, Sophus::SE3f::Tangent& left_increment) const
{
    Eigen::Matrix<float, 6, 6> hessian = Eigen::Matrix<float, 6, 6>::Zero();
    Eigen::Matrix<float, 6, 1> gradient = Eigen::Matrix<float, 6, 1>::Zero();
    float squared_error_sum = 0.0f;
    int valid_count = 0;
    const Eigen::Matrix3f rotation = source_to_target.rotationMatrix();

    for (int index = 0; index < static_cast<int>(correspondences.size()); ++index)
    {
        const auto& [target_index, transformed_position] = correspondences[index];
        if (target_index < 0)
        {
            continue;
        }

        const PointWithCovariance& source_point = source[index];
        const PointWithCovariance& target_point = target[target_index];
        const Eigen::Vector3f residual = transformed_position - target_point.position;
        Eigen::Matrix3f covariance = Eigen::Matrix3f::Identity();
        if (source_point.covariance_valid)
        {
            covariance = target_point.covariance_valid ? target_point.covariance : Eigen::Matrix3f::Identity();
            covariance += rotation * source_point.covariance * rotation.transpose();
        }
        const Eigen::Matrix3f precision = covariance.inverse();
        if (!precision.allFinite())
        {
            continue;
        }

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

    const VoxelPointLayout source_layout = mProcesser.process(source);
    result.num_source_points = source_layout.points.size();
    result.num_target_points = mTarget->points.size();
    if (source_layout.points.empty() || mTarget->points.empty())
    {
        return result;
    }

    const std::vector<PointWithCovariance> source_points = estimateCovariances(source_layout);
    std::vector<Correspondence> correspondences(source_points.size());

    Sophus::SE3f source_to_target(initial_guess.rotation(), initial_guess.translation());

    for (int iteration = 0; iteration < mConfig.max_iterations; ++iteration)
    {
        findCorrespondences(source_points, mTarget->points, mTarget->voxels, mTarget->voxel_map, source_to_target, correspondences);

        Sophus::SE3f::Tangent left_increment;
        if (!buildAndSolve(source_points, mTarget->points, correspondences, source_to_target,
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
