#include "perception/LIO/SparsityAwareGICP.hpp"

#include <cuco/static_map.cuh>
#include <cuda/iterator>
#include <cuda/std/functional>
#include <cuda_runtime.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace perception::lio
{
Eigen::Isometry3f sophusExpUpdate(Eigen::Matrix<float, 6, 1> delta, bool constrain_to_se2);

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
        const auto x = static_cast<std::uint64_t>(static_cast<std::uint32_t>(key.x));
        const auto y = static_cast<std::uint64_t>(static_cast<std::uint32_t>(key.y));
        const auto z = static_cast<std::uint64_t>(static_cast<std::uint32_t>(key.z));
        return static_cast<std::size_t>((x * 73856093ULL) ^ (y * 19349663ULL) ^ (z * 83492791ULL));
    }
};

struct SparsePoint
{
    Eigen::Vector3f position{Eigen::Vector3f::Zero()};
    Eigen::Matrix3f covariance{Eigen::Matrix3f::Identity()};
    HostVoxelKey key{};
    bool covariance_valid{false};
};

struct DevicePoint
{
    float x{0.0f};
    float y{0.0f};
    float z{0.0f};
    float covariance[9]{};
    int covariance_valid{0};
};

struct DeviceVoxelEntry
{
    int x{0};
    int y{0};
    int z{0};
    int start{0};
    int count{0};
};

struct DeviceCorrespondence
{
    int source_index{-1};
    int target_index{-1};
    float transformed_x{0.0f};
    float transformed_y{0.0f};
    float transformed_z{0.0f};
    float distance_squared{0.0f};
    int valid{0};
};

struct TargetLayout
{
    std::vector<DevicePoint> points;
    std::vector<int> original_indices;
    std::vector<DeviceVoxelEntry> voxels;
    std::vector<std::int64_t> voxel_keys;
};

struct DeviceLinearSystem
{
    Eigen::Matrix<float, 6, 6> hessian{Eigen::Matrix<float, 6, 6>::Zero()};
    Eigen::Matrix<float, 6, 1> gradient{Eigen::Matrix<float, 6, 1>::Zero()};
    std::size_t valid_count{0};
    float mean_squared_error{std::numeric_limits<float>::infinity()};
};

HostVoxelKey pointToVoxel(const Eigen::Vector3f& point, const float voxel_size)
{
    return HostVoxelKey{
        static_cast<int>(std::floor(point.x() / voxel_size)),
        static_cast<int>(std::floor(point.y() / voxel_size)),
        static_cast<int>(std::floor(point.z() / voxel_size))};
}

__host__ __device__ std::int64_t packVoxelKey(const int x, const int y, const int z)
{
    constexpr std::int64_t coordinate_offset = 1 << 20;
    constexpr std::int64_t coordinate_mask = (1 << 21) - 1;
    const std::int64_t packed_x = (static_cast<std::int64_t>(x) + coordinate_offset) & coordinate_mask;
    const std::int64_t packed_y = (static_cast<std::int64_t>(y) + coordinate_offset) & coordinate_mask;
    const std::int64_t packed_z = (static_cast<std::int64_t>(z) + coordinate_offset) & coordinate_mask;
    return (packed_x << 42) | (packed_y << 21) | packed_z;
}

std::int64_t packVoxelKey(const HostVoxelKey& key)
{
    return packVoxelKey(key.x, key.y, key.z);
}

std::vector<SparsePoint> makeSparseCloud(const pcl::PointCloud<pcl::PointXYZ>& cloud, const SparsityAwareGICPConfig& config)
{
    std::vector<SparsePoint> points;
    points.reserve(cloud.size());
    std::unordered_map<HostVoxelKey, std::vector<std::size_t>, HostVoxelKeyHash> occupied_voxels;
    const int max_points_per_voxel = std::max(1, config.max_points_per_voxel);
    const float min_spacing2 = std::max(0.0f, config.min_point_spacing) * std::max(0.0f, config.min_point_spacing);

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

        const bool too_close = std::any_of(voxel_points.cbegin(), voxel_points.cend(), [&](const std::size_t index) {
            return (points[index].position - point.position).squaredNorm() < min_spacing2;
        });
        if (too_close)
        {
            continue;
        }

        voxel_points.push_back(points.size());
        points.push_back(point);
    }
    return points;
}

void estimateCovariances(std::vector<SparsePoint>& points, const SparsityAwareGICPConfig& config)
{
    std::unordered_map<HostVoxelKey, std::vector<std::size_t>, HostVoxelKeyHash> voxel_index;
    voxel_index.reserve(points.size());
    for (std::size_t index = 0; index < points.size(); ++index)
    {
        voxel_index[points[index].key].push_back(index);
    }

    const int min_neighbors = std::max(3, config.min_covariance_neighbors);
    const int max_neighbors = std::max(min_neighbors, config.max_covariance_neighbors);
    const float regularization = std::max(1.0e-6f, config.covariance_regularization);
    const auto fallback_covariance = Eigen::Matrix3f::Identity();
    std::vector<std::pair<float, std::size_t>> candidates;
    for (auto& point : points)
    {
        candidates.clear();
        for (int dx = -config.covariance_voxel_radius; dx <= config.covariance_voxel_radius; ++dx)
        {
            for (int dy = -config.covariance_voxel_radius; dy <= config.covariance_voxel_radius; ++dy)
            {
                for (int dz = -config.covariance_voxel_radius; dz <= config.covariance_voxel_radius; ++dz)
                {
                    const HostVoxelKey neighbor_key{point.key.x + dx, point.key.y + dy, point.key.z + dz};
                    const auto search = voxel_index.find(neighbor_key);
                    if (search == voxel_index.end())
                    {
                        continue;
                    }
                    for (const std::size_t neighbor_index : search->second)
                    {
                        candidates.emplace_back((points[neighbor_index].position - point.position).squaredNorm(), neighbor_index);
                    }
                }
            }
        }

        if (static_cast<int>(candidates.size()) < min_neighbors)
        {
            point.covariance = fallback_covariance;
            point.covariance_valid = false;
            continue;
        }

        const int neighbor_count = std::min<int>(max_neighbors, static_cast<int>(candidates.size()));
        if (neighbor_count < static_cast<int>(candidates.size()))
        {
            std::nth_element(candidates.begin(), candidates.begin() + neighbor_count, candidates.end(),
                             [](const auto& first, const auto& second) { return first.first < second.first; });
        }

        Eigen::Vector3f mean = Eigen::Vector3f::Zero();
        for (int i = 0; i < neighbor_count; ++i)
        {
            mean += points[candidates[i].second].position;
        }
        mean /= static_cast<float>(neighbor_count);

        Eigen::Matrix3f covariance = Eigen::Matrix3f::Zero();
        for (int i = 0; i < neighbor_count; ++i)
        {
            const Eigen::Vector3f centered = points[candidates[i].second].position - mean;
            covariance += centered * centered.transpose();
        }
        covariance /= static_cast<float>(neighbor_count - 1);

        covariance += Eigen::Matrix3f::Identity() * regularization;
        const Eigen::Matrix3f inverse_covariance = covariance.inverse();
        const float frobenius_norm = inverse_covariance.norm();
        if (!std::isfinite(frobenius_norm) || frobenius_norm <= 0.0f)
        {
            point.covariance = fallback_covariance;
            point.covariance_valid = false;
            continue;
        }

        point.covariance = covariance * frobenius_norm;
        point.covariance_valid = point.covariance.allFinite();
    }
}

std::vector<DevicePoint> toDevicePoints(const std::vector<SparsePoint>& points)
{
    std::vector<DevicePoint> device_points(points.size());
    std::transform(points.cbegin(), points.cend(), device_points.begin(), [](const SparsePoint& point) {
        DevicePoint device_point;
        device_point.x = point.position.x();
        device_point.y = point.position.y();
        device_point.z = point.position.z();
        device_point.covariance_valid = point.covariance_valid ? 1 : 0;
        for (int row = 0; row < 3; ++row)
        {
            for (int col = 0; col < 3; ++col)
            {
                device_point.covariance[row * 3 + col] = point.covariance(row, col);
            }
        }
        return device_point;
    });
    return device_points;
}

TargetLayout makeTargetLayout(const std::vector<SparsePoint>& points)
{
    TargetLayout layout;
    layout.points.reserve(points.size());
    layout.original_indices.resize(points.size());
    std::iota(layout.original_indices.begin(), layout.original_indices.end(), 0);
    std::sort(layout.original_indices.begin(), layout.original_indices.end(), [&](const int first, const int second) {
        if (points[first].key == points[second].key)
        {
            return first < second;
        }
        return points[first].key < points[second].key;
    });

    HostVoxelKey current_key;
    bool have_current_key = false;
    for (int ordered_index = 0; ordered_index < static_cast<int>(layout.original_indices.size()); ++ordered_index)
    {
        const int original_index = layout.original_indices[ordered_index];
        const auto& point = points[original_index];
        DevicePoint device_point;
        device_point.x = point.position.x();
        device_point.y = point.position.y();
        device_point.z = point.position.z();
        device_point.covariance_valid = point.covariance_valid ? 1 : 0;
        for (int row = 0; row < 3; ++row)
        {
            for (int col = 0; col < 3; ++col)
            {
                device_point.covariance[row * 3 + col] = point.covariance(row, col);
            }
        }
        layout.points.push_back(device_point);

        if (!have_current_key || !(point.key == current_key))
        {
            layout.voxels.push_back(DeviceVoxelEntry{point.key.x, point.key.y, point.key.z, ordered_index, 1});
            layout.voxel_keys.push_back(packVoxelKey(point.key));
            current_key = point.key;
            have_current_key = true;
        }
        else
        {
            ++layout.voxels.back().count;
        }
    }
    return layout;
}

Eigen::Isometry3f expUpdate(Eigen::Matrix<float, 6, 1> delta, const bool constrain_to_se2)
{
    return sophusExpUpdate(delta, constrain_to_se2);
}

template<class VoxelMapRef>
__global__ void findCorrespondencesKernel(const DevicePoint* source_points, const int num_source_points,
                                          const DevicePoint* target_points, const DeviceVoxelEntry* target_voxel_entries,
                                          VoxelMapRef target_voxels,
                                          DeviceCorrespondence* correspondences,
                                          const float voxel_size, const int adjacent_voxels,
                                          const float max_correspondence_distance2, const float* transform)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    while (index < num_source_points)
    {
        const DevicePoint source = source_points[index];
        const float transformed_x = transform[0] * source.x + transform[1] * source.y + transform[2] * source.z + transform[9];
        const float transformed_y = transform[3] * source.x + transform[4] * source.y + transform[5] * source.z + transform[10];
        const float transformed_z = transform[6] * source.x + transform[7] * source.y + transform[8] * source.z + transform[11];

        const int base_x = static_cast<int>(floorf(transformed_x / voxel_size));
        const int base_y = static_cast<int>(floorf(transformed_y / voxel_size));
        const int base_z = static_cast<int>(floorf(transformed_z / voxel_size));

        float best_distance2 = max_correspondence_distance2;
        int best_target = -1;
        for (int dx = -adjacent_voxels; dx <= adjacent_voxels; ++dx)
        {
            for (int dy = -adjacent_voxels; dy <= adjacent_voxels; ++dy)
            {
                for (int dz = -adjacent_voxels; dz <= adjacent_voxels; ++dz)
                {
                    const auto found = target_voxels.find(packVoxelKey(base_x + dx, base_y + dy, base_z + dz));
                    if (found == target_voxels.end())
                    {
                        continue;
                    }
                    const DeviceVoxelEntry voxel = target_voxel_entries[found->second];
                    for (int i = 0; i < voxel.count; ++i)
                    {
                        const int target_index = voxel.start + i;
                        const DevicePoint target = target_points[target_index];
                        const float diff_x = transformed_x - target.x;
                        const float diff_y = transformed_y - target.y;
                        const float diff_z = transformed_z - target.z;
                        const float distance2 = diff_x * diff_x + diff_y * diff_y + diff_z * diff_z;
                        if (distance2 < best_distance2)
                        {
                            best_distance2 = distance2;
                            best_target = target_index;
                        }
                    }
                }
            }
        }

        correspondences[index] = DeviceCorrespondence{index, best_target, transformed_x, transformed_y, transformed_z,
                                                      best_distance2, best_target >= 0 ? 1 : 0};
        index += stride;
    }
}

__device__ Eigen::Matrix3f covarianceMatrix(const DevicePoint& point)
{
    Eigen::Matrix3f covariance;
    for (int row = 0; row < 3; ++row)
    {
        for (int col = 0; col < 3; ++col)
        {
            covariance(row, col) = point.covariance[row * 3 + col];
        }
    }
    return covariance;
}

__device__ Eigen::Matrix3f rotationMatrix(const float* transform)
{
    Eigen::Matrix3f rotation;
    rotation << transform[0], transform[1], transform[2],
                transform[3], transform[4], transform[5],
                transform[6], transform[7], transform[8];
    return rotation;
}

__device__ Eigen::Matrix3f skewMatrix(const Eigen::Vector3f& vector)
{
    Eigen::Matrix3f skew;
    skew << 0.0f, -vector.z(), vector.y(), vector.z(), 0.0f, -vector.x(), -vector.y(), vector.x(), 0.0f;
    return skew;
}

__device__ bool matrixAllFinite(const Eigen::Matrix3f& matrix)
{
    for (int row = 0; row < 3; ++row)
    {
        for (int col = 0; col < 3; ++col)
        {
            if (!isfinite(matrix(row, col)))
            {
                return false;
            }
        }
    }
    return true;
}

__global__ void buildLinearSystemKernel(const DevicePoint* source_points, const DevicePoint* target_points,
                                        const DeviceCorrespondence* correspondences, const int num_correspondences,
                                        const float* transform, const float cauchy_kernel_scale, float* partials)
{
    constexpr int linear_system_size = 44;
    __shared__ float block_sum[linear_system_size];
    if (threadIdx.x < linear_system_size)
    {
        block_sum[threadIdx.x] = 0.0f;
    }
    __syncthreads();

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    while (index < num_correspondences)
    {
        if (const DeviceCorrespondence correspondence = correspondences[index]; correspondence.valid != 0)
        {
            const DevicePoint source = source_points[correspondence.source_index];
            const DevicePoint target = target_points[correspondence.target_index];
            const Eigen::Vector3f transformed_source(correspondence.transformed_x, correspondence.transformed_y, correspondence.transformed_z);
            const Eigen::Vector3f target_position(target.x, target.y, target.z);
            const Eigen::Vector3f residual = transformed_source - target_position;

            Eigen::Matrix3f covariance = Eigen::Matrix3f::Identity();
            if (source.covariance_valid != 0)
            {
                const Eigen::Matrix3f rotation = rotationMatrix(transform);
                const Eigen::Matrix3f target_covariance = target.covariance_valid != 0 ? covarianceMatrix(target) : Eigen::Matrix3f::Identity();
                covariance = target_covariance + rotation * covarianceMatrix(source) * rotation.transpose();
            }

            const Eigen::Matrix3f precision = covariance.inverse();
            if (matrixAllFinite(precision))
            {
                const Eigen::Vector3f precision_residual = precision * residual;
                const float mahalanobis_error = residual.dot(precision_residual);
                const float kernel_scale2 = cauchy_kernel_scale * cauchy_kernel_scale;
                const float weight = cauchy_kernel_scale > 0.0f ? 1.0f / (1.0f + mahalanobis_error / kernel_scale2) : 1.0f;

                Eigen::Matrix<float, 3, 6> jacobian;
                jacobian.template block<3, 3>(0, 0) = Eigen::Matrix3f::Identity();
                jacobian.template block<3, 3>(0, 3) = -skewMatrix(transformed_source);

                const Eigen::Matrix<float, 6, 6> local_hessian = jacobian.transpose() * weight * precision * jacobian;
                const Eigen::Matrix<float, 6, 1> local_gradient = jacobian.transpose() * weight * precision_residual;
                for (int row = 0; row < 6; ++row)
                {
                    for (int col = 0; col < 6; ++col)
                    {
                        atomicAdd(&block_sum[row * 6 + col], local_hessian(row, col));
                    }
                }
                for (int i = 0; i < 6; ++i)
                {
                    atomicAdd(&block_sum[36 + i], local_gradient(i));
                }
                atomicAdd(&block_sum[42], 1.0f);
                atomicAdd(&block_sum[43], residual.squaredNorm());
            }
        }
        index += stride;
    }

    __syncthreads();
    if (threadIdx.x < linear_system_size)
    {
        partials[blockIdx.x * linear_system_size + threadIdx.x] = block_sum[threadIdx.x];
    }
}

template<class VoxelMapRef>
void findCorrespondencesCuda(const std::vector<SparsePoint>& source_points,
                             const thrust::device_vector<DevicePoint>& device_source,
                             const thrust::device_vector<DevicePoint>& device_target,
                             const thrust::device_vector<DeviceVoxelEntry>& device_voxels,
                             VoxelMapRef target_voxels,
                             thrust::device_vector<DeviceCorrespondence>& device_correspondences,
                             thrust::device_vector<float>& device_transform,
                             const Eigen::Isometry3f& transform,
                             const SparsityAwareGICPConfig& config)
{
    const Eigen::Matrix3f rotation = transform.rotation();
    const Eigen::Vector3f translation = transform.translation();
    const std::array<float, 12> transform_array{
        rotation(0, 0), rotation(0, 1), rotation(0, 2),
        rotation(1, 0), rotation(1, 1), rotation(1, 2),
        rotation(2, 0), rotation(2, 1), rotation(2, 2),
        translation.x(), translation.y(), translation.z()};
    thrust::copy(transform_array.begin(), transform_array.end(), device_transform.begin());

    constexpr int block_size = 256;
    const int grid_size = std::max(1, std::min(1024, static_cast<int>((source_points.size() + block_size - 1) / block_size)));
    const float max_distance2 = config.max_correspondence_distance * config.max_correspondence_distance;
    findCorrespondencesKernel<<<grid_size, block_size>>>(thrust::raw_pointer_cast(device_source.data()), static_cast<int>(device_source.size()),
                                                         thrust::raw_pointer_cast(device_target.data()),
                                                         thrust::raw_pointer_cast(device_voxels.data()), target_voxels,
                                                         thrust::raw_pointer_cast(device_correspondences.data()), config.voxel_size,
                                                         config.adjacent_voxels, max_distance2,
                                                         thrust::raw_pointer_cast(device_transform.data()));
    if (cudaDeviceSynchronize() != cudaSuccess)
    {
        throw std::runtime_error("CUDA GICP correspondence search failed");
    }
}

DeviceLinearSystem buildLinearSystemCuda(const std::size_t num_source_points,
                                         const thrust::device_vector<DevicePoint>& device_source,
                                         const thrust::device_vector<DevicePoint>& device_target,
                                         const thrust::device_vector<DeviceCorrespondence>& device_correspondences,
                                         const thrust::device_vector<float>& device_transform,
                                         const SparsityAwareGICPConfig& config)
{
    constexpr int block_size = 256;
    constexpr int linear_system_size = 44;
    const int grid_size = std::max(1, std::min(1024, static_cast<int>((num_source_points + block_size - 1) / block_size)));

    thrust::device_vector<float> device_partials(static_cast<std::size_t>(grid_size * linear_system_size));
    thrust::fill(device_partials.begin(), device_partials.end(), 0.0f);

    buildLinearSystemKernel<<<grid_size, block_size>>>(thrust::raw_pointer_cast(device_source.data()),
                                                       thrust::raw_pointer_cast(device_target.data()),
                                                       thrust::raw_pointer_cast(device_correspondences.data()),
                                                       static_cast<int>(num_source_points),
                                                       thrust::raw_pointer_cast(device_transform.data()),
                                                       config.cauchy_kernel_scale,
                                                       thrust::raw_pointer_cast(device_partials.data()));
    if (cudaDeviceSynchronize() != cudaSuccess)
    {
        throw std::runtime_error("CUDA GICP linear-system reduction failed");
    }

    thrust::host_vector<float> partials = device_partials;

    DeviceLinearSystem linear_system;
    linear_system.hessian.setZero();
    linear_system.gradient.setZero();
    float valid_count = 0.0f;
    float squared_error_sum = 0.0f;
    for (int block = 0; block < grid_size; ++block)
    {
        const int offset = block * linear_system_size;
        for (int row = 0; row < 6; ++row)
        {
            for (int col = 0; col < 6; ++col)
            {
                linear_system.hessian(row, col) += partials[offset + row * 6 + col];
            }
        }
        for (int row = 0; row < 6; ++row)
        {
            linear_system.gradient(row) += partials[offset + 36 + row];
        }
        valid_count += partials[offset + 42];
        squared_error_sum += partials[offset + 43];
    }

    linear_system.valid_count = static_cast<std::size_t>(valid_count);
    linear_system.mean_squared_error = valid_count > 0.0f ? squared_error_sum / valid_count : std::numeric_limits<float>::infinity();
    return linear_system;
}

} // namespace

SparsityAwareGICP::SparsityAwareGICP(SparsityAwareGICPConfig config) : mConfig(config)
{}

const SparsityAwareGICPConfig& SparsityAwareGICP::config() const noexcept
{
    return mConfig;
}

void SparsityAwareGICP::setConfig(const SparsityAwareGICPConfig& config) noexcept
{
    mConfig = config;
}

SparsityAwareGICPResult SparsityAwareGICP::align(const pcl::PointCloud<pcl::PointXYZ>& source,
                                                 const pcl::PointCloud<pcl::PointXYZ>& target,
                                                 const Eigen::Isometry3f& initial_guess) const
{
    SparsityAwareGICPResult result;
    result.transform = initial_guess;

    if (source.empty() || target.empty())
    {
        return result;
    }

    std::vector<SparsePoint> source_sparse = makeSparseCloud(source, mConfig);
    std::vector<SparsePoint> target_sparse = makeSparseCloud(target, mConfig);
    estimateCovariances(source_sparse, mConfig);
    estimateCovariances(target_sparse, mConfig);

    result.num_source_points = source_sparse.size();
    result.num_target_points = target_sparse.size();
    if (source_sparse.empty() || target_sparse.empty())
    {
        return result;
    }

    const TargetLayout target_layout = makeTargetLayout(target_sparse);
    const std::vector<DevicePoint> source_device_points = toDevicePoints(source_sparse);
    thrust::device_vector<DevicePoint> device_source(source_device_points.begin(), source_device_points.end());
    thrust::device_vector<DevicePoint> device_target(target_layout.points.begin(), target_layout.points.end());
    thrust::device_vector<DeviceVoxelEntry> device_voxels(target_layout.voxels.begin(), target_layout.voxels.end());
    thrust::device_vector<DeviceCorrespondence> device_correspondences(source_sparse.size());
    thrust::device_vector<float> device_transform(12);

    constexpr std::int64_t empty_key = std::numeric_limits<std::int64_t>::min();
    constexpr int empty_value = -1;
    constexpr double load_factor = 0.5;
    const std::size_t map_capacity = std::max<std::size_t>(2, static_cast<std::size_t>(std::ceil(target_layout.voxels.size() / load_factor)));
    cuco::static_map target_voxel_map{
        map_capacity,
        cuco::empty_key{empty_key},
        cuco::empty_value{empty_value},
        cuda::std::equal_to<std::int64_t>{},
        cuco::linear_probing<1, cuco::default_hash_function<std::int64_t>>{}};

    thrust::device_vector<std::int64_t> device_voxel_keys(target_layout.voxel_keys.begin(), target_layout.voxel_keys.end());
    thrust::device_vector<int> device_voxel_indices(target_layout.voxels.size());
    thrust::sequence(device_voxel_indices.begin(), device_voxel_indices.end(), 0);
    auto voxel_pairs = cuda::make_transform_iterator(
        cuda::counting_iterator<std::size_t>{0},
        cuda::proclaim_return_type<cuco::pair<std::int64_t, int>>(
            [keys = device_voxel_keys.begin(), values = device_voxel_indices.begin()] __device__(const auto index) {
                return cuco::pair<std::int64_t, int>{keys[index], values[index]};
            }));
    target_voxel_map.insert(voxel_pairs, voxel_pairs + target_layout.voxels.size());
    const auto target_voxel_ref = target_voxel_map.ref(cuco::find);

    for (int iteration = 0; iteration < mConfig.max_iterations; ++iteration)
    {
        findCorrespondencesCuda(source_sparse, device_source, device_target, device_voxels, target_voxel_ref, device_correspondences,
                                device_transform, result.transform, mConfig);
        DeviceLinearSystem linear_system = buildLinearSystemCuda(source_sparse.size(), device_source, device_target,
                                                                 device_correspondences, device_transform, mConfig);
        if (linear_system.valid_count == 0 || !linear_system.hessian.allFinite() || !linear_system.gradient.allFinite())
        {
            break;
        }

        linear_system.hessian += Eigen::Matrix<float, 6, 6>::Identity() * mConfig.damping_factor;
        Eigen::Matrix<float, 6, 1> delta = linear_system.hessian.ldlt().solve(-linear_system.gradient);
        if (!delta.allFinite())
        {
            break;
        }
        if (mConfig.constrain_to_se2)
        {
            delta.y() = 0.0f;
            delta.z() = 0.0f;
            delta(3) = 0.0f;
            delta(4) = 0.0f;
        }

        result.transform = expUpdate(delta, mConfig.constrain_to_se2) * result.transform;
        result.iterations = iteration + 1;
        result.num_correspondences = linear_system.valid_count;
        result.fitness_score = linear_system.mean_squared_error;

        const float translation_step = delta.head<3>().norm();
        const float rotation_step = delta.tail<3>().norm();
        if (translation_step < mConfig.convergence_translation && rotation_step < mConfig.convergence_rotation)
        {
            result.converged = true;
            break;
        }
    }

    if (!result.converged && result.iterations > 0)
    {
        result.converged = std::isfinite(result.fitness_score);
    }
    return result;
}
} // namespace perception::lio