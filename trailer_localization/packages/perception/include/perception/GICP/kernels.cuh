#pragma once
#include "perception/GICP/SparsityAwareGICP.h"
#include <cuco/static_map.cuh>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <limits>
#include <vector>

static constexpr int LINEAR_SYSTEM_BLOCK_SIZE = 256;
static constexpr int HESSIAN_SIZE = 21;
static constexpr int VALID_COUNT_OFFSET = HESSIAN_SIZE + 6;
static constexpr int SQUARED_ERROR_OFFSET = VALID_COUNT_OFFSET + 1;
static constexpr int LINEAR_SYSTEM_SIZE = SQUARED_ERROR_OFFSET + 1;

struct LinearSystemPartial
{
    float values[LINEAR_SYSTEM_SIZE]{};

    __device__ LinearSystemPartial operator+(const LinearSystemPartial& other) const
    {
        LinearSystemPartial result;
        #pragma unroll
        for (int element = 0; element < LINEAR_SYSTEM_SIZE; ++element)
        {
            result.values[element] = values[element] + other.values[element];
        }
        return result;
    }
};

// A GICP point stored on the device. The covariance describes the point's
// local surface neighborhood; voxels are used only to index nearby points.
struct DevicePoint
{
    float x{0.0f};
    float y{0.0f};
    float z{0.0f};
    float covariance[9]{};
    int covariance_valid{0};
};

// A voxel owns a contiguous [start, start + count) range in the point array.
struct DeviceVoxelEntry
{
    int start{0};
    int count{0};
};

// Correspondence search also caches the transformed source position so the
// linear-system kernel does not repeat the same rigid transformation.
struct DeviceCorrespondence
{
    int target_index{-1};      // 匹配到的目标点在 target_points 数组中的索引，-1 没有在距离阈值内找到有效目标点

    /* 源点经过当前位姿估计变换后的坐标 */
    float transformed_x{0.0f};
    float transformed_y{0.0f};
    float transformed_z{0.0f};
};

// Host-side, voxel-sorted representation used to create device arrays/maps.
struct TargetLayout
{
    std::vector<DevicePoint> points;
    std::vector<DeviceVoxelEntry> voxels;
    std::vector<std::int64_t> voxel_keys;
};

// Non-running states make later queued kernels return without synchronizing
// the host after every iteration.
enum class AlignmentStatus : int
{
    Running,
    Aborted,
    Converged
};

struct DeviceAlignmentState
{
    float fitness_score{std::numeric_limits<float>::infinity()};
    int num_correspondences{0};
    int iterations{0};
    AlignmentStatus status{AlignmentStatus::Running};
};

using DeviceVoxelMap = decltype(cuco::static_map{std::size_t{2},
                                                 cuco::empty_key{std::numeric_limits<std::int64_t>::min()},
                                                 cuco::empty_value{-1},
                                                 cuda::std::equal_to<std::int64_t>{},
                                                 cuco::linear_probing<1, cuco::default_hash_function<std::int64_t>>{}});

namespace cuda_func
{
// Pack three signed 21-bit voxel coordinates into one hash-map key.
__host__ __device__ inline std::int64_t packVoxelKey(const int x, const int y, const int z)
{
    constexpr std::int64_t coordinate_offset = 1 << 20;
    constexpr std::int64_t coordinate_mask = (1 << 21) - 1;
    const std::int64_t packed_x = (static_cast<std::int64_t>(x) + coordinate_offset) & coordinate_mask;
    const std::int64_t packed_y = (static_cast<std::int64_t>(y) + coordinate_offset) & coordinate_mask;
    const std::int64_t packed_z = (static_cast<std::int64_t>(z) + coordinate_offset) & coordinate_mask;
    return (packed_x << 42) | (packed_y << 21) | packed_z;
}

thrust::device_vector<DevicePoint> estimateCovariances(const TargetLayout& layout, const SparsityAwareGICP::Config& config);

void findCorrespondences(const thrust::device_vector<DevicePoint>& device_source,
                         const thrust::device_vector<DevicePoint>& device_target,
                         const thrust::device_vector<DeviceVoxelEntry>& device_voxels,
                         const DeviceVoxelMap& target_voxel_map,
                         thrust::device_vector<DeviceCorrespondence>& device_correspondences,
                         thrust::device_vector<float>& device_transform,
                         const SparsityAwareGICP::Config& config,
                         const thrust::device_vector<DeviceAlignmentState>& device_state);

void buildLinearSystem(std::size_t num_source_points,
                       const thrust::device_vector<DevicePoint>& device_source,
                       const thrust::device_vector<DevicePoint>& device_target,
                       const thrust::device_vector<DeviceCorrespondence>& device_correspondences,
                       const thrust::device_vector<float>& device_transform,
                       const SparsityAwareGICP::Config& config, thrust::device_vector<float>& device_partials,
                       const thrust::device_vector<DeviceAlignmentState>& device_state);

void solveAndUpdate(const thrust::device_vector<float>& device_partials, int num_blocks,
                    const SparsityAwareGICP::Config& config, thrust::device_vector<float>& device_transform,
                    thrust::device_vector<DeviceAlignmentState>& device_state);
}