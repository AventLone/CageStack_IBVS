#include "perception/GICP/SparsityAwareGICP.h"
#include "perception/GICP/kernels.cuh"
#include <cuco/static_map.cuh>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace
{
struct HostVoxelKey
{
    int x{0};
    int y{0};
    int z{0};

    bool operator==(const HostVoxelKey& other) const noexcept
    {
        return x == other.x && y == other.y && z == other.z;
    }

    bool operator<(const HostVoxelKey& other) const noexcept
    {
        if (x != other.x)
        {
            return x < other.x;
        }
        if (y != other.y)
        {
            return y < other.y;
        }
        return z < other.z;
    }
};

struct HostVoxelKeyHash
{
    std::size_t operator()(const HostVoxelKey& key) const noexcept
    {
        const auto x = static_cast<std::size_t>(key.x);
        const auto y = static_cast<std::size_t>(key.y);
        const auto z = static_cast<std::size_t>(key.z);
        return (x * 73856093ULL) ^ (y * 19349663ULL) ^ (z * 83492791ULL);
    }
};

using OccupiedVoxels = std::unordered_map<HostVoxelKey, std::vector<std::size_t>, HostVoxelKeyHash>;

struct SparsePoint
{
    Eigen::Vector3f position{Eigen::Vector3f::Zero()};   // 点云中原始点的空间坐标（x，y，z）
    HostVoxelKey key{};                                  // 离散体素网格坐标 （i, j, k)
};

HostVoxelKey pointToVoxel(const Eigen::Vector3f& point, const float voxel_size)
{
    return HostVoxelKey{static_cast<int>(std::floor(point.x() / voxel_size)),
                        static_cast<int>(std::floor(point.y() / voxel_size)),
                        static_cast<int>(std::floor(point.z() / voxel_size))};
}

std::int64_t packVoxelKey(const HostVoxelKey& key)
{
    return cuda_func::packVoxelKey(key.x, key.y, key.z);
}

std::vector<SparsePoint> makeSparseCloud(const pcl::PointCloud<pcl::PointXYZ>& cloud, const SparsityAwareGICP::Config& config)
{
    std::vector<SparsePoint> sparse_cloud;
    sparse_cloud.reserve(cloud.size());
    OccupiedVoxels occupied_voxels;
    const int max_points_per_voxel = std::max(1, config.max_points_per_voxel);
    // const float min_spacing_square = std::pow(std::max(0.001f, config.min_point_spacing), 2.0f);
    const float min_spacing_square = std::pow(std::max(0.001f, config.voxel_size * 0.1f), 2.0f);


    // Retain several well-spaced samples per voxel instead of replacing each
    // voxel with one centroid, preserving local geometry for point covariances.
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

        /* The ponit would not be pushed in sparse_cloud if it's too close to any of the existing points */
        if (const bool too_close = std::any_of(voxel_points.cbegin(), voxel_points.cend(), [&](const std::size_t index)
            {
                return (sparse_cloud[index].position - point.position).squaredNorm() < min_spacing_square;
            }); too_close)
        {
            continue;
        }

        voxel_points.push_back(sparse_cloud.size());
        sparse_cloud.push_back(point);
    }
    return sparse_cloud;
}

TargetLayout makeTargetLayout(const std::vector<SparsePoint>& points)
{
    TargetLayout layout;
    layout.points.reserve(points.size());
    std::vector<int> original_indices(points.size());
    std::iota(original_indices.begin(), original_indices.end(), 0);
    // Sorting by voxel makes every voxel's points contiguous, so a hash lookup
    // only needs to return one compact start/count entry.
    std::sort(original_indices.begin(), original_indices.end(), [&](const int first, const int second)
        {
            if (points[first].key == points[second].key)
            {
                return first < second;
            }
            return points[first].key < points[second].key;
        });

    HostVoxelKey current_key;
    bool have_current_key = false;
    for (int ordered_index = 0; ordered_index < static_cast<int>(original_indices.size()); ++ordered_index)
    {
        const int original_index = original_indices[ordered_index];
        const auto& [position, key] = points[original_index];
        DevicePoint device_point;
        device_point.x = position.x();
        device_point.y = position.y();
        device_point.z = position.z();
        layout.points.push_back(device_point);

        if (!have_current_key || !(key == current_key))
        {
            layout.voxels.push_back(DeviceVoxelEntry{ordered_index, 1});
            layout.voxel_keys.push_back(packVoxelKey(key));
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

} // namespace

static std::vector<cuco::pair<std::int64_t, int>> makeVoxelPairs(const TargetLayout& layout)
{
    std::vector<cuco::pair<std::int64_t, int>> pairs;
    pairs.reserve(layout.voxel_keys.size());
    for (int index = 0; index < static_cast<int>(layout.voxel_keys.size()); ++index)
    {
        pairs.emplace_back(layout.voxel_keys[index], index);
    }
    return pairs;
}

struct SparsityAwareGICP::TargetCache
{
    std::unordered_set<std::int64_t> occupied_voxels;
    thrust::device_vector<DevicePoint> points;
    thrust::device_vector<DeviceVoxelEntry> voxels;
    DeviceVoxelMap voxel_map;

    explicit TargetCache(const TargetLayout& layout, thrust::device_vector<DevicePoint> device_points, const std::size_t max_target_voxels)
        : points(std::move(device_points)),
          voxels(layout.voxels.begin(), layout.voxels.end()),
          voxel_map(std::max<std::size_t>(2, max_target_voxels * 2),
                    cuco::empty_key{std::numeric_limits<std::int64_t>::min()},
                    cuco::empty_value{-1},
                    cuda::std::equal_to<std::int64_t>{},
                    cuco::linear_probing<1, cuco::default_hash_function<std::int64_t>>{})
    {
        occupied_voxels.insert(layout.voxel_keys.begin(), layout.voxel_keys.end());
        const auto host_voxel_pairs = makeVoxelPairs(layout);
        thrust::device_vector<cuco::pair<std::int64_t, int>> voxel_pairs(host_voxel_pairs.begin(), host_voxel_pairs.end());
        voxel_map.insert(voxel_pairs.begin(), voxel_pairs.end());
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
    TargetLayout target_layout = makeTargetLayout(target_sparse);
    if (mConfig.max_target_voxels == 0 || target_layout.voxels.size() > mConfig.max_target_voxels)
    {
        throw std::invalid_argument("Initial target exceeds max_target_voxels or the voxel budget is zero");
    }
    thrust::device_vector<DevicePoint> device_points = cuda_func::estimateCovariances(target_layout, mConfig);
    mTarget = std::make_unique<TargetCache>(target_layout, std::move(device_points), mConfig.max_target_voxels);
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
    sparse_points.erase(std::remove_if(sparse_points.begin(), sparse_points.end(), [this](const SparsePoint& point)
        {
            return mTarget->occupied_voxels.find(packVoxelKey(point.key)) != mTarget->occupied_voxels.end();
        }), sparse_points.end());

    if (sparse_points.empty())
    {
        return;
    }

    TargetLayout new_layout = makeTargetLayout(sparse_points);
    if (const std::size_t available_voxels = mConfig.max_target_voxels - mTarget->voxels.size();
        new_layout.voxels.size() > available_voxels)
    {
        return;
    }

    thrust::device_vector<DevicePoint> device_points = cuda_func::estimateCovariances(new_layout, mConfig);
    const int point_offset = static_cast<int>(mTarget->points.size());
    const int voxel_offset = static_cast<int>(mTarget->voxels.size());
    for (auto& [start, count] : new_layout.voxels)
    {
        start += point_offset;
    }

    std::vector<cuco::pair<std::int64_t, int>> host_voxel_pairs;
    host_voxel_pairs.reserve(new_layout.voxel_keys.size());
    for (int index = 0; index < static_cast<int>(new_layout.voxel_keys.size()); ++index)
    {
        host_voxel_pairs.emplace_back(new_layout.voxel_keys[index], voxel_offset + index);
    }

    mTarget->points.insert(mTarget->points.end(), device_points.begin(), device_points.end());
    mTarget->voxels.insert(mTarget->voxels.end(), new_layout.voxels.begin(), new_layout.voxels.end());
    thrust::device_vector<cuco::pair<std::int64_t, int>> voxel_pairs(host_voxel_pairs.begin(), host_voxel_pairs.end());
    mTarget->voxel_map.insert(voxel_pairs.begin(), voxel_pairs.end());

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

    std::vector<SparsePoint> source_sparse = makeSparseCloud(source, mConfig);
    TargetLayout source_layout = makeTargetLayout(source_sparse);

    result.num_source_points = source_layout.points.size();
    result.num_target_points = mTarget->points.size();
    if (source_layout.points.empty() || mTarget->points.empty())
    {
        return result;
    }

    const thrust::device_vector<DevicePoint> device_source = cuda_func::estimateCovariances(source_layout, mConfig);
    thrust::device_vector<DeviceCorrespondence> device_correspondences(source_layout.points.size());

    const Eigen::Matrix3f initial_rotation = initial_guess.rotation();
    const Eigen::Vector3f initial_translation = initial_guess.translation();
    const std::array<float, 12> initial_transform{initial_rotation(0, 0), initial_rotation(0, 1), initial_rotation(0, 2),
                                                   initial_rotation(1, 0), initial_rotation(1, 1), initial_rotation(1, 2),
                                                   initial_rotation(2, 0), initial_rotation(2, 1), initial_rotation(2, 2),
                                                   initial_translation.x(), initial_translation.y(), initial_translation.z()};
    thrust::device_vector<float> device_transform(initial_transform.begin(), initial_transform.end());
    thrust::device_vector<DeviceAlignmentState> device_state(1, DeviceAlignmentState{});

    constexpr int block_size = LINEAR_SYSTEM_BLOCK_SIZE;
    const int grid_size = std::max(1, std::min(1024, static_cast<int>((source_layout.points.size() + block_size - 1) / block_size)));
    thrust::device_vector<float> device_partials(static_cast<std::size_t>(grid_size * LINEAR_SYSTEM_SIZE));

    for (int iteration = 0; iteration < mConfig.max_iterations; ++iteration)
    {
        cuda_func::findCorrespondences(device_source, mTarget->points, mTarget->voxels, mTarget->voxel_map,
                                       device_correspondences, device_transform, mConfig, device_state);
        cuda_func::buildLinearSystem(source_layout.points.size(), device_source, mTarget->points, device_correspondences,
                                     device_transform, mConfig, device_partials, device_state);
        cuda_func::solveAndUpdate(device_partials, grid_size, mConfig, device_transform, device_state);
    }

    thrust::host_vector<DeviceAlignmentState> host_state = device_state;
    thrust::host_vector<float> host_transform = device_transform;
    const auto& [fitness_score, num_correspondences, iterations, status] = host_state.front();
    result.iterations = iterations;
    result.num_correspondences = static_cast<std::size_t>(std::max(0, num_correspondences));
    result.fitness_score = fitness_score;
    result.transform.matrix() << host_transform[0], host_transform[1], host_transform[2], host_transform[9],
                                 host_transform[3], host_transform[4], host_transform[5], host_transform[10],
                                 host_transform[6], host_transform[7], host_transform[8], host_transform[11],
                                 0.0f, 0.0f, 0.0f, 1.0f;
    result.converged = status == AlignmentStatus::Converged;
    if (!result.converged && result.iterations > 0)
    {
        result.converged = std::isfinite(result.fitness_score);
    }
    return result;
}