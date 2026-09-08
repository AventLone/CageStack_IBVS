#include "perception/LIO/SparsityAwareGICP.hpp"
#include <cub/block/block_reduce.cuh>
#include <cuco/static_map.cuh>
#include <cuda_runtime.h>
#include <math_constants.h>
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

namespace perception::lio
{
Eigen::Isometry3f sophusExpUpdate(const Eigen::Matrix<float, 6, 1>& delta);

namespace
{
constexpr int linear_system_block_size = 256;
constexpr int hessian_size = 21;
constexpr int valid_count_offset = hessian_size + 6;
constexpr int squared_error_offset = valid_count_offset + 1;
constexpr int linear_system_size = squared_error_offset + 1;

struct LinearSystemPartial
{
    float values[linear_system_size]{};

    __device__ LinearSystemPartial operator+(const LinearSystemPartial& other) const
    {
        LinearSystemPartial result;
        #pragma unroll
        for (int element = 0; element < linear_system_size; ++element)
        {
            result.values[element] = values[element] + other.values[element];
        }
        return result;
    }
};

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
    HostVoxelKey key{};
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
    int start{0};
    int count{0};
};

struct DeviceCorrespondence
{
    int target_index{-1};
    float transformed_x{0.0f};
    float transformed_y{0.0f};
    float transformed_z{0.0f};
};

struct TargetLayout
{
    std::vector<DevicePoint> points;
    std::vector<DeviceVoxelEntry> voxels;
    std::vector<std::int64_t> voxel_keys;
};

struct DeviceAlignmentState
{
    float fitness_score{std::numeric_limits<float>::infinity()};
    int num_correspondences{0};
    int iterations{0};
    int active{1};
    int converged{0};
};

using DeviceVoxelMap = decltype(cuco::static_map{std::size_t{2},
                                                   cuco::empty_key{std::numeric_limits<std::int64_t>::min()},
                                                   cuco::empty_value{-1},
                                                   cuda::std::equal_to<std::int64_t>{},
                                                   cuco::linear_probing<1, cuco::default_hash_function<std::int64_t>>{}});

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

        const bool too_close = std::any_of(voxel_points.cbegin(), voxel_points.cend(), [&](const std::size_t index)
            {
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

template<class VoxelMapRef>
__global__ void estimateCovariancesKernel(DevicePoint* points, const int num_points,
                                          const DeviceVoxelEntry* voxel_entries, VoxelMapRef voxels,
                                          const float voxel_size, const int voxel_radius,
                                          const int min_neighbors, const int max_neighbors,
                                          const float regularization)
{
    constexpr int neighbor_capacity = 64;
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= num_points)
    {
        return;
    }

    const DevicePoint point = points[index];
    const int base_x = static_cast<int>(floorf(point.x / voxel_size));
    const int base_y = static_cast<int>(floorf(point.y / voxel_size));
    const int base_z = static_cast<int>(floorf(point.z / voxel_size));
    float distances[neighbor_capacity];
    int neighbor_indices[neighbor_capacity];
    int neighbor_count = 0;

    for (int dx = -voxel_radius; dx <= voxel_radius; ++dx)
    {
        for (int dy = -voxel_radius; dy <= voxel_radius; ++dy)
        {
            for (int dz = -voxel_radius; dz <= voxel_radius; ++dz)
            {
                const auto found = voxels.find(packVoxelKey(base_x + dx, base_y + dy, base_z + dz));
                if (found == voxels.end())
                {
                    continue;
                }

                const DeviceVoxelEntry voxel = voxel_entries[found->second];
                for (int offset = 0; offset < voxel.count; ++offset)
                {
                    const int candidate_index = voxel.start + offset;
                    const DevicePoint candidate = points[candidate_index];
                    const float x_difference = candidate.x - point.x;
                    const float y_difference = candidate.y - point.y;
                    const float z_difference = candidate.z - point.z;
                    const float distance = x_difference * x_difference + y_difference * y_difference + z_difference * z_difference;
                    if (neighbor_count == max_neighbors && distance >= distances[max_neighbors - 1])
                    {
                        continue;
                    }

                    const int insertion_limit = min(neighbor_count, max_neighbors - 1);
                    int insertion_index = insertion_limit;

                    while (insertion_index > 0 && distance < distances[insertion_index - 1])
                    {
                        if (insertion_index < max_neighbors)
                        {
                            distances[insertion_index] = distances[insertion_index - 1];
                            neighbor_indices[insertion_index] = neighbor_indices[insertion_index - 1];
                        }
                        --insertion_index;
                    }

                    if (insertion_index < max_neighbors)
                    {
                        distances[insertion_index] = distance;
                        neighbor_indices[insertion_index] = candidate_index;
                    }
                    neighbor_count = min(neighbor_count + 1, max_neighbors);
                }
            }
        }
    }

    DevicePoint result = point;
    result.covariance_valid = 0;
    for (int element = 0; element < 9; ++element)
    {
        result.covariance[element] = element % 4 == 0 ? 1.0f : 0.0f;
    }
    if (neighbor_count < min_neighbors)
    {
        points[index] = result;
        return;
    }

    float mean_x = 0.0f;
    float mean_y = 0.0f;
    float mean_z = 0.0f;
    for (int neighbor = 0; neighbor < neighbor_count; ++neighbor)
    {
        const DevicePoint candidate = points[neighbor_indices[neighbor]];
        mean_x += candidate.x;
        mean_y += candidate.y;
        mean_z += candidate.z;
    }
    mean_x /= static_cast<float>(neighbor_count);
    mean_y /= static_cast<float>(neighbor_count);
    mean_z /= static_cast<float>(neighbor_count);

    float covariance[9]{};
    for (int neighbor = 0; neighbor < neighbor_count; ++neighbor)
    {
        const DevicePoint candidate = points[neighbor_indices[neighbor]];
        const float x = candidate.x - mean_x;
        const float y = candidate.y - mean_y;
        const float z = candidate.z - mean_z;
        covariance[0] += x * x;
        covariance[1] += x * y;
        covariance[2] += x * z;
        covariance[4] += y * y;
        covariance[5] += y * z;
        covariance[8] += z * z;
    }

    const float scale = 1.0f / static_cast<float>(neighbor_count - 1);
    covariance[0] = covariance[0] * scale + regularization;
    covariance[1] *= scale;
    covariance[2] *= scale;
    covariance[3] = covariance[1];
    covariance[4] = covariance[4] * scale + regularization;
    covariance[5] *= scale;
    covariance[6] = covariance[2];
    covariance[7] = covariance[5];
    covariance[8] = covariance[8] * scale + regularization;

    const float determinant = covariance[0] * (covariance[4] * covariance[8] - covariance[5] * covariance[7]) -
                              covariance[1] * (covariance[3] * covariance[8] - covariance[5] * covariance[6]) +
                              covariance[2] * (covariance[3] * covariance[7] - covariance[4] * covariance[6]);
    if (!isfinite(determinant) || fabsf(determinant) <= 1.0e-12f)
    {
        points[index] = result;
        return;
    }

    const float inverse_norm_squared =
        (covariance[4] * covariance[8] - covariance[5] * covariance[7]) * (covariance[4] * covariance[8] - covariance[5] * covariance[7]) +
        (covariance[2] * covariance[7] - covariance[1] * covariance[8]) * (covariance[2] * covariance[7] - covariance[1] * covariance[8]) +
        (covariance[1] * covariance[5] - covariance[2] * covariance[4]) * (covariance[1] * covariance[5] - covariance[2] * covariance[4]) +
        (covariance[5] * covariance[6] - covariance[3] * covariance[8]) * (covariance[5] * covariance[6] - covariance[3] * covariance[8]) +
        (covariance[0] * covariance[8] - covariance[2] * covariance[6]) * (covariance[0] * covariance[8] - covariance[2] * covariance[6]) +
        (covariance[2] * covariance[3] - covariance[0] * covariance[5]) * (covariance[2] * covariance[3] - covariance[0] * covariance[5]) +
        (covariance[3] * covariance[7] - covariance[4] * covariance[6]) * (covariance[3] * covariance[7] - covariance[4] * covariance[6]) +
        (covariance[1] * covariance[6] - covariance[0] * covariance[7]) * (covariance[1] * covariance[6] - covariance[0] * covariance[7]) +
        (covariance[0] * covariance[4] - covariance[1] * covariance[3]) * (covariance[0] * covariance[4] - covariance[1] * covariance[3]);
    const float inverse_norm = sqrtf(inverse_norm_squared) / fabsf(determinant);
    if (!isfinite(inverse_norm) || inverse_norm <= 0.0f)
    {
        points[index] = result;
        return;
    }

    for (int element = 0; element < 9; ++element)
    {
        result.covariance[element] = covariance[element] * inverse_norm;
    }
    result.covariance_valid = 1;
    points[index] = result;
}

thrust::device_vector<DevicePoint> estimateCovariancesCuda(const TargetLayout& layout, const SparsityAwareGICPConfig& config)
{
    const int min_neighbors = std::max(3, config.min_covariance_neighbors);
    const int max_neighbors = std::max(min_neighbors, config.max_covariance_neighbors);
    if (constexpr int neighbor_capacity = 64; max_neighbors > neighbor_capacity)
    {
        throw std::invalid_argument("max_covariance_neighbors must not exceed 64 for CUDA covariance estimation");
    }

    thrust::device_vector<DevicePoint> device_points(layout.points.begin(), layout.points.end());
    thrust::device_vector<DeviceVoxelEntry> device_voxels(layout.voxels.begin(), layout.voxels.end());
    std::vector<cuco::pair<std::int64_t, int>> host_pairs;
    host_pairs.reserve(layout.voxel_keys.size());
    for (int index = 0; index < static_cast<int>(layout.voxel_keys.size()); ++index)
    {
        host_pairs.emplace_back(layout.voxel_keys[index], index);
    }
    thrust::device_vector<cuco::pair<std::int64_t, int>> device_pairs(host_pairs.begin(), host_pairs.end());
    DeviceVoxelMap voxel_map(std::max<std::size_t>(2, layout.voxel_keys.size() * 2),
                             cuco::empty_key{std::numeric_limits<std::int64_t>::min()}, cuco::empty_value{-1},
                             cuda::std::equal_to<std::int64_t>{},
                             cuco::linear_probing<1, cuco::default_hash_function<std::int64_t>>{});
    voxel_map.insert(device_pairs.begin(), device_pairs.end());

    constexpr int block_size = 128;
    const int grid_size = std::max(1, static_cast<int>((device_points.size() + block_size - 1) / block_size));
    estimateCovariancesKernel<<<grid_size, block_size>>>(thrust::raw_pointer_cast(device_points.data()), static_cast<int>(device_points.size()),
                                                         thrust::raw_pointer_cast(device_voxels.data()), voxel_map.ref(cuco::find),
                                                         config.voxel_size, std::max(0, config.covariance_voxel_radius), min_neighbors,
                                                         max_neighbors, std::max(1.0e-6f, config.covariance_regularization));
    if (cudaDeviceSynchronize() != cudaSuccess)
    {
        throw std::runtime_error("CUDA GICP covariance estimation failed");
    }
    return device_points;
}

TargetLayout makeTargetLayout(const std::vector<SparsePoint>& points)
{
    TargetLayout layout;
    layout.points.reserve(points.size());
    std::vector<int> original_indices(points.size());
    std::iota(original_indices.begin(), original_indices.end(), 0);
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

template<class VoxelMapRef>
__global__ void findCorrespondencesKernel(const DevicePoint* source_points, const int num_source_points,
                                          const DevicePoint* target_points, const DeviceVoxelEntry* target_voxel_entries,
                                          VoxelMapRef target_voxels,
                                          DeviceCorrespondence* correspondences,
                                          const float voxel_size, const int adjacent_voxels,
                                          const float max_correspondence_distance2, const float* transform,
                                          const DeviceAlignmentState* state)
{
    if (state->active == 0)
    {
        return;
    }
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t stride = blockDim.x * gridDim.x;

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
                        if (const float distance2 = diff_x * diff_x + diff_y * diff_y + diff_z * diff_z;
                            distance2 < best_distance2)
                        {
                            best_distance2 = distance2;
                            best_target = target_index;
                        }
                    }
                }
            }
        }

        correspondences[index] = DeviceCorrespondence{best_target, transformed_x, transformed_y, transformed_z};
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
                                        const float* transform, const float cauchy_kernel_scale, float* partials,
                                        const DeviceAlignmentState* state)
{
    if (state->active == 0)
    {
        return;
    }
    using BlockReduce = cub::BlockReduce<LinearSystemPartial, linear_system_block_size, cub::BLOCK_REDUCE_WARP_REDUCTIONS>;
    __shared__ BlockReduce::TempStorage reduction_storage;
    LinearSystemPartial thread_sum;

    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t stride = blockDim.x * gridDim.x;
    while (index < num_correspondences)
    {
        if (const DeviceCorrespondence correspondence = correspondences[index]; correspondence.target_index >= 0)
        {
            const DevicePoint source = source_points[index];
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

            if (const Eigen::Matrix3f precision = covariance.inverse(); matrixAllFinite(precision))
            {
                const Eigen::Vector3f precision_residual = precision * residual;
                const float mahalanobis_error = residual.dot(precision_residual);
                const float kernel_scale2 = cauchy_kernel_scale * cauchy_kernel_scale;
                const float weight = cauchy_kernel_scale > 0.0f ? 1.0f / (1.0f + mahalanobis_error / kernel_scale2) : 1.0f;

                Eigen::Matrix<float, 3, 6> jacobian;
                jacobian.block<3, 3>(0, 0) = Eigen::Matrix3f::Identity();
                jacobian.block<3, 3>(0, 3) = -skewMatrix(transformed_source);

                const Eigen::Matrix<float, 6, 6> local_hessian = jacobian.transpose() * weight * precision * jacobian;
                const Eigen::Matrix<float, 6, 1> local_gradient = jacobian.transpose() * weight * precision_residual;
                int packed_index = 0;

                #pragma unroll
                for (int row = 0; row < 6; ++row)
                {
                    #pragma unroll
                    for (int col = row; col < 6; ++col)
                    {
                        thread_sum.values[packed_index++] += local_hessian(row, col);
                    }
                }

                #pragma unroll
                for (int element = 0; element < 6; ++element)
                {
                    thread_sum.values[hessian_size + element] += local_gradient(element);
                }
                thread_sum.values[valid_count_offset] += 1.0f;
                thread_sum.values[squared_error_offset] += residual.squaredNorm();
            }
        }
        index += stride;
    }

    const LinearSystemPartial block_sum = BlockReduce(reduction_storage).Sum(thread_sum);
    if (threadIdx.x == 0)
    {
        #pragma unroll
        for (int element = 0; element < linear_system_size; ++element)
        {
            partials[blockIdx.x * linear_system_size + element] = block_sum.values[element];
        }
    }
}

template<class VoxelMapRef>
void findCorrespondencesCuda(const thrust::device_vector<DevicePoint>& device_source,
                             const thrust::device_vector<DevicePoint>& device_target,
                             const thrust::device_vector<DeviceVoxelEntry>& device_voxels,
                             VoxelMapRef target_voxels,
                             thrust::device_vector<DeviceCorrespondence>& device_correspondences,
                             thrust::device_vector<float>& device_transform,
                             const SparsityAwareGICPConfig& config,
                             const thrust::device_vector<DeviceAlignmentState>& device_state)
{
    constexpr int block_size = linear_system_block_size;
    const int grid_size = std::max(1, std::min(1024, static_cast<int>((device_source.size() + block_size - 1) / block_size)));
    const float max_distance2 = config.max_correspondence_distance * config.max_correspondence_distance;
    findCorrespondencesKernel<<<grid_size, block_size>>>(thrust::raw_pointer_cast(device_source.data()), static_cast<int>(device_source.size()),
                                                         thrust::raw_pointer_cast(device_target.data()),
                                                         thrust::raw_pointer_cast(device_voxels.data()), target_voxels,
                                                         thrust::raw_pointer_cast(device_correspondences.data()), config.voxel_size,
                                                         config.adjacent_voxels, max_distance2,
                                                         thrust::raw_pointer_cast(device_transform.data()),
                                                         thrust::raw_pointer_cast(device_state.data()));
    if (cudaGetLastError() != cudaSuccess)
    {
        throw std::runtime_error("CUDA GICP correspondence-search launch failed");
    }
}

void buildLinearSystemCuda(const std::size_t num_source_points,
                           const thrust::device_vector<DevicePoint>& device_source,
                           const thrust::device_vector<DevicePoint>& device_target,
                           const thrust::device_vector<DeviceCorrespondence>& device_correspondences,
                           const thrust::device_vector<float>& device_transform,
                           const SparsityAwareGICPConfig& config,
                           thrust::device_vector<float>& device_partials,
                           const thrust::device_vector<DeviceAlignmentState>& device_state)
{
    constexpr int block_size = 256;
    const int grid_size = std::max(1, std::min(1024, static_cast<int>((num_source_points + block_size - 1) / block_size)));

    buildLinearSystemKernel<<<grid_size, block_size>>>(thrust::raw_pointer_cast(device_source.data()),
                                                       thrust::raw_pointer_cast(device_target.data()),
                                                       thrust::raw_pointer_cast(device_correspondences.data()),
                                                       static_cast<int>(num_source_points),
                                                       thrust::raw_pointer_cast(device_transform.data()),
                                                       config.cauchy_kernel_scale,
                                                       thrust::raw_pointer_cast(device_partials.data()),
                                                       thrust::raw_pointer_cast(device_state.data()));
    if (cudaGetLastError() != cudaSuccess)
    {
        throw std::runtime_error("CUDA GICP linear-system launch failed");
    }
}

__global__ void solveAndUpdateKernel(const float* partials, const int num_blocks, const float damping_factor,
                                     const float convergence_translation, const float convergence_rotation,
                                     float* transform, DeviceAlignmentState* state)
{
    if (threadIdx.x != 0 || state->active == 0)
    {
        return;
    }

    float augmented[6][7]{};
    float valid_count = 0.0f;
    float squared_error_sum = 0.0f;
    for (int block = 0; block < num_blocks; ++block)
    {
        const float* partial = partials + block * linear_system_size;
        int packed_index = 0;
        for (int row = 0; row < 6; ++row)
        {
            for (int col = row; col < 6; ++col)
            {
                const float value = partial[packed_index++];
                augmented[row][col] += value;
                if (row != col)
                {
                    augmented[col][row] += value;
                }
            }
            augmented[row][6] -= partial[hessian_size + row];
        }
        valid_count += partial[valid_count_offset];
        squared_error_sum += partial[squared_error_offset];
    }
    state->num_correspondences = static_cast<int>(valid_count);
    state->fitness_score = valid_count > 0.0f ? squared_error_sum / valid_count : CUDART_INF_F;
    if (valid_count <= 0.0f || !isfinite(squared_error_sum))
    {
        state->active = 0;
        return;
    }
    for (int diagonal = 0; diagonal < 6; ++diagonal)
    {
        augmented[diagonal][diagonal] += damping_factor;
    }
    for (int diagonal = 0; diagonal < 6; ++diagonal)
    {
        int pivot_row = diagonal;
        for (int row = diagonal + 1; row < 6; ++row)
        {
            if (fabsf(augmented[row][diagonal]) > fabsf(augmented[pivot_row][diagonal]))
            {
                pivot_row = row;
            }
        }

        if (!isfinite(augmented[pivot_row][diagonal]) || fabsf(augmented[pivot_row][diagonal]) <= 1.0e-12f)
        {
            state->active = 0;
            return;
        }

        for (int col = diagonal; col < 7; ++col)
        {
            const float temporary = augmented[diagonal][col];
            augmented[diagonal][col] = augmented[pivot_row][col];
            augmented[pivot_row][col] = temporary;
        }

        const float pivot = augmented[diagonal][diagonal];
        for (int col = diagonal; col < 7; ++col)
        {
            augmented[diagonal][col] /= pivot;
        }

        for (int row = 0; row < 6; ++row)
        {
            if (row == diagonal)
            {
                continue;
            }

            const float factor = augmented[row][diagonal];
            for (int col = diagonal; col < 7; ++col)
            {
                augmented[row][col] -= factor * augmented[diagonal][col];
            }
        }
    }

    float delta[6];
    for (int row = 0; row < 6; ++row)
    {
        delta[row] = augmented[row][6];
        if (!isfinite(delta[row]))
        {
            state->active = 0;
            return;
        }
    }
    const float theta = sqrtf(delta[3] * delta[3] + delta[4] * delta[4] + delta[5] * delta[5]);
    const float theta2 = theta * theta;
    const float sin_over_theta = theta > 1.0e-5f ? sinf(theta) / theta : 1.0f - theta2 / 6.0f;
    const float one_minus_cos_over_theta2 = theta > 1.0e-5f ? (1.0f - cosf(theta)) / theta2 : 0.5f - theta2 / 24.0f;
    const float theta_minus_sin_over_theta3 = theta > 1.0e-5f ? (theta - sinf(theta)) / (theta2 * theta) : 1.0f / 6.0f - theta2 / 120.0f;
    const float skew[9]{0.0f, -delta[5], delta[4], delta[5], 0.0f, -delta[3], -delta[4], delta[3], 0.0f};
    float skew_squared[9]{};
    for (int row = 0; row < 3; ++row)
    {
        for (int col = 0; col < 3; ++col)
        {
            for (int inner = 0; inner < 3; ++inner)
            {
                skew_squared[row * 3 + col] += skew[row * 3 + inner] * skew[inner * 3 + col];
            }
        }
    }

    float rotation[9]{};
    float translation[3]{};
    for (int row = 0; row < 3; ++row)
    {
        for (int col = 0; col < 3; ++col)
        {
            rotation[row * 3 + col] = (row == col ? 1.0f : 0.0f) + sin_over_theta * skew[row * 3 + col] + one_minus_cos_over_theta2 * skew_squared[row * 3 + col];
            translation[row] += ((row == col ? 1.0f : 0.0f) + one_minus_cos_over_theta2 * skew[row * 3 + col] +
                                 theta_minus_sin_over_theta3 * skew_squared[row * 3 + col]) * delta[col];
        }
    }

    float updated_transform[12]{};
    for (int row = 0; row < 3; ++row)
    {
        for (int col = 0; col < 3; ++col)
        {
            for (int inner = 0; inner < 3; ++inner)
            {
                updated_transform[row * 3 + col] += rotation[row * 3 + inner] * transform[inner * 3 + col];
            }
            updated_transform[9 + row] += rotation[row * 3 + col] * transform[9 + col];
        }
        updated_transform[9 + row] += translation[row];
    }

    for (int element = 0; element < 12; ++element)
    {
        transform[element] = updated_transform[element];
    }
    state->iterations += 1;
    state->num_correspondences = static_cast<int>(valid_count);
    state->fitness_score = squared_error_sum / valid_count;
    if (sqrtf(delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]) < convergence_translation &&
        theta < convergence_rotation)
    {
        state->converged = 1;
        state->active = 0;
    }
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

    TargetCache(const TargetLayout& layout, thrust::device_vector<DevicePoint> device_points, const std::size_t max_target_voxels)
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

SparsityAwareGICP::SparsityAwareGICP(const SparsityAwareGICPConfig& config) : mConfig(config)
{}

SparsityAwareGICP::~SparsityAwareGICP() = default;
SparsityAwareGICP::SparsityAwareGICP(SparsityAwareGICP&&) noexcept = default;
SparsityAwareGICP& SparsityAwareGICP::operator=(SparsityAwareGICP&&) noexcept = default;

const SparsityAwareGICPConfig& SparsityAwareGICP::config() const noexcept
{
    return mConfig;
}

void SparsityAwareGICP::setConfig(const SparsityAwareGICPConfig& config) noexcept
{
    mConfig = config;
    clearTarget();
}

void SparsityAwareGICP::initializeTarget(const pcl::PointCloud<pcl::PointXYZ>& target)
{
    const std::vector<SparsePoint> target_sparse = makeSparseCloud(target, mConfig);
    TargetLayout target_layout = makeTargetLayout(target_sparse);
    if (mConfig.max_target_voxels == 0 || target_layout.voxels.size() > mConfig.max_target_voxels)
    {
        throw std::invalid_argument("Initial target exceeds max_target_voxels or the voxel budget is zero");
    }
    thrust::device_vector<DevicePoint> device_points = estimateCovariancesCuda(target_layout, mConfig);
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

    thrust::device_vector<DevicePoint> device_points = estimateCovariancesCuda(new_layout, mConfig);
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

SparsityAwareGICPResult SparsityAwareGICP::align(const pcl::PointCloud<pcl::PointXYZ>& source,
                                                 const Eigen::Isometry3f& initial_guess) const
{
    SparsityAwareGICPResult result;
    result.transform = initial_guess;

    if (source.empty() || !hasTarget())
    {
        return result;
    }

    std::vector<SparsePoint> source_sparse = makeSparseCloud(source, mConfig);
    TargetLayout source_layout = makeTargetLayout(source_sparse);

    result.num_source_points = source_layout.points.size();
    result.num_target_points = mTarget->points.size();
    if (source_layout.points.empty() || mTarget->points.empty())
    {
        return result;
    }

    const thrust::device_vector<DevicePoint> device_source = estimateCovariancesCuda(source_layout, mConfig);
    thrust::device_vector<DeviceCorrespondence> device_correspondences(source_layout.points.size());
    const Eigen::Matrix3f initial_rotation = initial_guess.rotation();
    const Eigen::Vector3f initial_translation = initial_guess.translation();
    const std::array<float, 12> initial_transform{initial_rotation(0, 0), initial_rotation(0, 1), initial_rotation(0, 2),
                                                   initial_rotation(1, 0), initial_rotation(1, 1), initial_rotation(1, 2),
                                                   initial_rotation(2, 0), initial_rotation(2, 1), initial_rotation(2, 2),
                                                   initial_translation.x(), initial_translation.y(), initial_translation.z()};
    thrust::device_vector<float> device_transform(initial_transform.begin(), initial_transform.end());
    thrust::device_vector<DeviceAlignmentState> device_state(1, DeviceAlignmentState{});
    const auto target_voxel_ref = mTarget->voxel_map.ref(cuco::find);
    constexpr int block_size = linear_system_block_size;
    const int grid_size = std::max(1, std::min(1024, static_cast<int>((source_layout.points.size() + block_size - 1) / block_size)));
    thrust::device_vector<float> device_partials(static_cast<std::size_t>(grid_size * linear_system_size));

    for (int iteration = 0; iteration < mConfig.max_iterations; ++iteration)
    {
        findCorrespondencesCuda(device_source, mTarget->points, mTarget->voxels, target_voxel_ref,
                                                                device_correspondences, device_transform, mConfig, device_state);
        buildLinearSystemCuda(source_layout.points.size(), device_source, mTarget->points, device_correspondences,
                                                            device_transform, mConfig, device_partials, device_state);
        solveAndUpdateKernel<<<1, 1>>>(thrust::raw_pointer_cast(device_partials.data()), grid_size, mConfig.damping_factor,
                                       mConfig.convergence_translation, mConfig.convergence_rotation,
                                       thrust::raw_pointer_cast(device_transform.data()), thrust::raw_pointer_cast(device_state.data()));
        if (cudaGetLastError() != cudaSuccess)
        {
            throw std::runtime_error("CUDA GICP solve-and-update launch failed");
        }
    }

    thrust::host_vector<DeviceAlignmentState> host_state = device_state;
    thrust::host_vector<float> host_transform = device_transform;
    const DeviceAlignmentState& final_state = host_state.front();
    result.iterations = final_state.iterations;
    result.num_correspondences = static_cast<std::size_t>(std::max(0, final_state.num_correspondences));
    result.fitness_score = final_state.fitness_score;
    result.transform.matrix() << host_transform[0], host_transform[1], host_transform[2], host_transform[9],
                                 host_transform[3], host_transform[4], host_transform[5], host_transform[10],
                                 host_transform[6], host_transform[7], host_transform[8], host_transform[11],
                                 0.0f, 0.0f, 0.0f, 1.0f;
    result.converged = final_state.converged != 0;
    if (!result.converged && result.iterations > 0)
    {
        result.converged = std::isfinite(result.fitness_score);
    }
    return result;
}
} // namespace perception::lio