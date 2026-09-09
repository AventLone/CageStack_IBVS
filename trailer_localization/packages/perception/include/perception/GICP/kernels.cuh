#pragma once
#include "perception/GICP/SparsityAwareGICP.h"
#include <cub/block/block_reduce.cuh>
#include <cuco/static_map.cuh>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <limits>
#include <vector>

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

namespace cuda_func
{
__host__ __device__ inline std::int64_t packVoxelKey(const int x, const int y, const int z)
{
    constexpr std::int64_t coordinate_offset = 1 << 20;
    constexpr std::int64_t coordinate_mask = (1 << 21) - 1;
    const std::int64_t packed_x = (static_cast<std::int64_t>(x) + coordinate_offset) & coordinate_mask;
    const std::int64_t packed_y = (static_cast<std::int64_t>(y) + coordinate_offset) & coordinate_mask;
    const std::int64_t packed_z = (static_cast<std::int64_t>(z) + coordinate_offset) & coordinate_mask;
    return (packed_x << 42) | (packed_y << 21) | packed_z;
}

thrust::device_vector<DevicePoint> estimateCovariances(const TargetLayout& layout,
                                                           const SparsityAwareGICPConfig& config);

void findCorrespondences(const thrust::device_vector<DevicePoint>& device_source,
                             const thrust::device_vector<DevicePoint>& device_target,
                             const thrust::device_vector<DeviceVoxelEntry>& device_voxels,
                             const DeviceVoxelMap& target_voxel_map,
                             thrust::device_vector<DeviceCorrespondence>& device_correspondences,
                             thrust::device_vector<float>& device_transform,
                             const SparsityAwareGICPConfig& config,
                             const thrust::device_vector<DeviceAlignmentState>& device_state);

void buildLinearSystem(std::size_t num_source_points,
                           const thrust::device_vector<DevicePoint>& device_source,
                           const thrust::device_vector<DevicePoint>& device_target,
                           const thrust::device_vector<DeviceCorrespondence>& device_correspondences,
                           const thrust::device_vector<float>& device_transform,
                           const SparsityAwareGICPConfig& config,
                           thrust::device_vector<float>& device_partials,
                           const thrust::device_vector<DeviceAlignmentState>& device_state);

void solveAndUpdate(const thrust::device_vector<float>& device_partials,
                        int num_blocks,
                        float damping_factor,
                        float convergence_translation,
                        float convergence_rotation,
                        thrust::device_vector<float>& device_transform,
                        thrust::device_vector<DeviceAlignmentState>& device_state);
}
// } // namespace perception::lio::detail