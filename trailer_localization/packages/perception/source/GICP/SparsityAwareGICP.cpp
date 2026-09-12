#include "perception/GICP/SparsityAwareGICP.h"
#include <sophus/se3.hpp>
#include <algorithm>
#include <unordered_set>


namespace
{
struct VoxelKey
{
    int i{0};
    int j{0};
    int k{0};

    bool operator==(const VoxelKey& other) const noexcept
    {
        return i == other.i && j == other.j && k == other.k;
    }

    bool operator<(const VoxelKey& other) const noexcept
    {
        if (i != other.i)
        {
            return i < other.i;
        }
        if (j != other.j)
        {
            return j < other.j;
        }
        return k < other.k;
    }
};

struct VoxelKeyHash
{
    std::size_t operator()(const VoxelKey& key) const noexcept
    {
        const auto x = static_cast<std::size_t>(key.i);
        const auto y = static_cast<std::size_t>(key.j);
        const auto z = static_cast<std::size_t>(key.k);
        return (x * 73856093ULL) ^ (y * 19349663ULL) ^ (z * 83492791ULL);
    }
};

struct SparsePoint
{
    Eigen::Vector3f position{Eigen::Vector3f::Zero()};
    VoxelKey key{};
};

struct CpuPoint
{
    Eigen::Vector3f position{Eigen::Vector3f::Zero()};
    Eigen::Matrix3f covariance{Eigen::Matrix3f::Identity()};
    bool covariance_valid{false};
};

struct VoxelEntry
{
    int start{0};
    int count{0};
};

struct VoxelPointLayout
{
    std::vector<CpuPoint> points;
    std::vector<VoxelEntry> voxels;
    std::vector<VoxelKey> voxel_keys;
};

struct Correspondence
{
    int target_index{-1};
    Eigen::Vector3f transformed_position{Eigen::Vector3f::Zero()};
};

using OccupiedVoxels = std::unordered_map<VoxelKey, std::vector<std::size_t>, VoxelKeyHash>;

VoxelKey pointToVoxel(const Eigen::Vector3f& point, const float voxel_size)
{
    return VoxelKey{static_cast<int>(std::floor(point.x() / voxel_size)),
                    static_cast<int>(std::floor(point.y() / voxel_size)),
                    static_cast<int>(std::floor(point.z() / voxel_size))};
}

std::vector<SparsePoint> makeSparseCloud(const pcl::PointCloud<pcl::PointXYZ>& cloud, const SparsityAwareGICP::Config& config)
{
    std::vector<SparsePoint> sparse_cloud;
    sparse_cloud.reserve(cloud.size());
    OccupiedVoxels occupied_voxels;
    const int max_points_per_voxel = std::max(1, config.max_points_per_voxel);
    const float min_spacing_square = std::pow(std::max(0.001f, config.voxel_size * 0.1f), 2.0f);

    for (const auto& pcl_point : cloud)
    {
        if (!std::isfinite(pcl_point.x) || !std::isfinite(pcl_point.y) || !std::isfinite(pcl_point.z))
        {
            continue;
        }

        SparsePoint point;
        point.position = Eigen::Vector3f(pcl_point.x, pcl_point.y, pcl_point.z);
        point.key = pointToVoxel(point.position, config.voxel_size);

        auto& voxel_points = occupied_voxels[point.key];
        if (static_cast<int>(voxel_points.size()) >= max_points_per_voxel)
        {
            continue;
        }

        if (const bool too_close = std::ranges::any_of(std::as_const(voxel_points), [&](const std::size_t index)
                                                       {
                                                           return (sparse_cloud[index].position - point.position).squaredNorm() < min_spacing_square;
                                                       });
            too_close)
        {
            continue;
        }

        voxel_points.push_back(sparse_cloud.size());
        sparse_cloud.push_back(point);
    }
    return sparse_cloud;
}

VoxelPointLayout makeVoxelPointLayout(const std::vector<SparsePoint>& points)
{
    VoxelPointLayout layout;
    layout.points.reserve(points.size());
    std::vector<int> original_indices(points.size());
    std::iota(original_indices.begin(), original_indices.end(), 0);
    std::ranges::sort(original_indices, [&](const int first, const int second)
                          {
                              if (points[first].key == points[second].key)
                              {
                                  return first < second;
                              }
                              return points[first].key < points[second].key;
                          });

    VoxelKey current_key;
    bool have_current_key = false;
    for (int index = 0; index < static_cast<int>(original_indices.size()); ++index)
    {
        const auto& [position, key] = points[original_indices[index]];
        layout.points.push_back(CpuPoint{.position = position});
        if (!have_current_key || !(key == current_key))
        {
            layout.voxels.push_back(VoxelEntry{index, 1});
            layout.voxel_keys.push_back(key);
            current_key = key;
            have_current_key = true;
        }
        else
        {
            ++layout.voxels.back().count;
        }
    }
    return layout;
}

std::vector<CpuPoint> estimateCovariances(VoxelPointLayout layout, const SparsityAwareGICP::Config& config)
{
    std::unordered_map<VoxelKey, int, VoxelKeyHash> voxel_map;
    voxel_map.reserve(layout.voxel_keys.size());
    for (int index = 0; index < static_cast<int>(layout.voxel_keys.size()); ++index)
    {
        voxel_map.emplace(layout.voxel_keys[index], index);
    }

    const int min_neighbors = std::max(3, config.min_covariance_neighbors);
    const int max_neighbors = std::max(min_neighbors, config.max_covariance_neighbors);
    const int voxel_radius = std::max(0, config.covariance_voxel_radius);
    const float regularization = std::max(1.0e-6f, config.covariance_regularization);
    std::vector<CpuPoint> points = std::move(layout.points);

    for (int index = 0; index < static_cast<int>(points.size()); ++index)
    {
        const CpuPoint point = points[index];
        const auto [i, j, k] = pointToVoxel(point.position, config.voxel_size);
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

void findCorrespondences(const std::vector<CpuPoint>& source, const std::vector<CpuPoint>& target,
                         const std::vector<VoxelEntry>& target_voxels,
                         const std::unordered_map<VoxelKey, int, VoxelKeyHash>& target_voxel_map,
                         const Eigen::Isometry3f& transform, const SparsityAwareGICP::Config& config,
                         std::vector<Correspondence>& correspondences)
{
    // 对应搜索沿每个轴扩展的体素数 r，要求 >= 0；查询 (2*r + 1)^3 个体素。
    // 增大可覆盖更大的初始位姿误差，但搜索开销和误匹配风险增加；减小更快但易漏匹配。
    constexpr int adjacent_voxels = 1;

    const float max_distance2 = config.max_correspondence_distance * config.max_correspondence_distance;
    for (int index = 0; index < static_cast<int>(source.size()); ++index)
    {
        const Eigen::Vector3f transformed_position = transform * source[index].position;
        const auto [i, j, k] = pointToVoxel(transformed_position, config.voxel_size);
        float best_distance2 = max_distance2;
        int best_target = -1;

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

bool buildAndSolve(const std::vector<CpuPoint>& source, const std::vector<CpuPoint>& target,
                   const std::vector<Correspondence>& correspondences, const SparsityAwareGICP::Config& config,
                   Eigen::Isometry3f& transform, std::size_t& num_correspondences, float& fitness_score,
                   Sophus::SE3f::Tangent& delta)
{
    Eigen::Matrix<float, 6, 6> hessian = Eigen::Matrix<float, 6, 6>::Zero();
    Eigen::Matrix<float, 6, 1> gradient = Eigen::Matrix<float, 6, 1>::Zero();
    float squared_error_sum = 0.0f;
    int valid_count = 0;
    const Eigen::Matrix3f rotation = transform.rotation();

    for (int index = 0; index < static_cast<int>(correspondences.size()); ++index)
    {
        const auto& [target_index, transformed_position] = correspondences[index];
        if (target_index < 0)
        {
            continue;
        }

        const CpuPoint& source_point = source[index];
        const CpuPoint& target_point = target[target_index];
        const Eigen::Vector3f residual = transformed_position - target_point.position;
        Eigen::Matrix3f covariance = Eigen::Matrix3f::Identity();
        if (source_point.covariance_valid)
        {
            covariance = target_point.covariance_valid ? target_point.covariance : Eigen::Matrix3f::Identity();
            covariance += rotation * source_point.covariance * rotation.transpose();
        }
        const Eigen::Matrix3f precision = covariance.inverse();
        if (!precision.array().isFinite().all())
        {
            continue;
        }

        const Eigen::Vector3f precision_residual = precision * residual;
        const float mahalanobis_error = residual.dot(precision_residual);
        const float kernel_scale2 = config.cauchy_kernel_scale * config.cauchy_kernel_scale;
        const float weight = config.cauchy_kernel_scale > 0.0f ? 1.0f / (1.0f + mahalanobis_error / kernel_scale2) : 1.0f;
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

    hessian.diagonal().array() += config.damping_factor;
    const Eigen::LDLT<Eigen::Matrix<float, 6, 6>> decomposition(hessian);
    if (decomposition.info() != Eigen::Success)
    {
        return false;
    }
    delta = decomposition.solve(-gradient);
    if (decomposition.info() != Eigen::Success || !delta.array().isFinite().all())
    {
        return false;
    }
    const Sophus::SE3f updated_pose = Sophus::SE3f::exp(delta) *
                                      Sophus::SE3f(transform.rotation(), transform.translation());
    transform.linear() = updated_pose.rotationMatrix();
    transform.translation() = updated_pose.translation();
    return true;
}
} // namespace

struct SparsityAwareGICP::TargetCache
{
    std::unordered_set<VoxelKey, VoxelKeyHash> occupied_voxels;
    std::vector<CpuPoint> points;
    std::vector<VoxelEntry> voxels;
    std::unordered_map<VoxelKey, int, VoxelKeyHash> voxel_map;

    TargetCache(const VoxelPointLayout& layout, std::vector<CpuPoint> estimated_points)
        : points(std::move(estimated_points)), voxels(layout.voxels.begin(), layout.voxels.end())
    {
        occupied_voxels.insert(layout.voxel_keys.begin(), layout.voxel_keys.end());
        voxel_map.reserve(layout.voxel_keys.size());
        for (int index = 0; index < static_cast<int>(layout.voxel_keys.size()); ++index)
        {
            voxel_map.emplace(layout.voxel_keys[index], index);
        }
    }
};

SparsityAwareGICP::SparsityAwareGICP() : SparsityAwareGICP(Config{})
{}

SparsityAwareGICP::SparsityAwareGICP(const Config& config) : mConfig(config)
{}

SparsityAwareGICP::~SparsityAwareGICP() = default;
SparsityAwareGICP::SparsityAwareGICP(SparsityAwareGICP&&) noexcept = default;
SparsityAwareGICP& SparsityAwareGICP::operator=(SparsityAwareGICP&&) noexcept = default;

void SparsityAwareGICP::initializeTarget(const pcl::PointCloud<pcl::PointXYZ>& target)
{
    const std::vector<SparsePoint> target_sparse = makeSparseCloud(target, mConfig);
    VoxelPointLayout target_layout = makeVoxelPointLayout(target_sparse);
    if (mConfig.max_target_voxels == 0 || target_layout.voxels.size() > mConfig.max_target_voxels)
    {
        throw std::invalid_argument("Initial target exceeds max_target_voxels or the voxel budget is zero");
    }
    std::vector<CpuPoint> target_points = estimateCovariances(target_layout, mConfig);
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

    std::vector<SparsePoint> sparse_points = makeSparseCloud(points, mConfig);
    std::erase_if(sparse_points, [this](const SparsePoint& point)
                      {
                          return mTarget->occupied_voxels.contains(point.key);
                      });
    if (sparse_points.empty())
    {
        return;
    }

    VoxelPointLayout new_layout = makeVoxelPointLayout(sparse_points);
    if (const std::size_t available_voxels = mConfig.max_target_voxels - mTarget->voxels.size();
        new_layout.voxels.size() > available_voxels)
    {
        return;
    }

    std::vector<CpuPoint> new_points = estimateCovariances(new_layout, mConfig);
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

void SparsityAwareGICP::clearTarget() noexcept
{
    mTarget.reset();
}

bool SparsityAwareGICP::hasTarget() const noexcept
{
    return mTarget != nullptr;
}

SparsityAwareGICP::Result SparsityAwareGICP::align(const pcl::PointCloud<pcl::PointXYZ>& source,
                                                            const Eigen::Isometry3f& initial_guess) const
{
    Result result;
    result.transform = initial_guess;
    if (source.empty())
    {
        throw std::invalid_argument("source cloud is empty!");
    }
    if (!hasTarget())
    {
        throw std::runtime_error("Target was not initialized before you call this method!");
    }

    const std::vector<SparsePoint> source_sparse = makeSparseCloud(source, mConfig);
    const VoxelPointLayout source_layout = makeVoxelPointLayout(source_sparse);
    result.num_source_points = source_layout.points.size();
    result.num_target_points = mTarget->points.size();
    if (source_layout.points.empty() || mTarget->points.empty())
    {
        return result;
    }

    const std::vector<CpuPoint> source_points = estimateCovariances(source_layout, mConfig);
    std::vector<Correspondence> correspondences(source_points.size());
    Eigen::Isometry3f transform = initial_guess;
    for (int iteration = 0; iteration < mConfig.max_iterations; ++iteration)
    {
        findCorrespondences(source_points, mTarget->points, mTarget->voxels, mTarget->voxel_map,
                            transform, mConfig, correspondences);
        Sophus::SE3f::Tangent delta;
        if (!buildAndSolve(source_points, mTarget->points, correspondences, mConfig, transform,
                           result.num_correspondences, result.fitness_score, delta))
        {
            break;
        }

        ++result.iterations;
        result.transform = transform;
        if (delta.head<3>().norm() < mConfig.convergence_translation &&
            delta.tail<3>().norm() < mConfig.convergence_rotation)
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