#include "perception/GICP/kernels.cuh"
#include <cub/block/block_reduce.cuh>
#include <math_constants.h>
#include <sophus/se3.hpp>
#include <thrust/host_vector.h>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace
{
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
    const Eigen::Array3i base = (point.position / voxel_size).array().floor().cast<int>();
    float distances[neighbor_capacity];
    int neighbor_indices[neighbor_capacity];
    int neighbor_count = 0;

    // Search neighboring voxels while retaining only the nearest fixed-capacity
    // set. Squared distances preserve ordering without a square root.
    for (int dx = -voxel_radius; dx <= voxel_radius; ++dx)
    {
        for (int dy = -voxel_radius; dy <= voxel_radius; ++dy)
        {
            for (int dz = -voxel_radius; dz <= voxel_radius; ++dz)
            {
                const auto found = voxels.find(cuda_func::packVoxelKey(base.x() + dx, base.y() + dy, base.z() + dz));
                if (found == voxels.end())
                {
                    continue;
                }

                const DeviceVoxelEntry voxel = voxel_entries[found->second];
                for (int offset = 0; offset < voxel.count; ++offset)
                {
                    const int candidate_index = voxel.start + offset;
                    const DevicePoint candidate = points[candidate_index];
                    const float distance = (candidate.position - point.position).squaredNorm();
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
    result.covariance.setIdentity();
    if (neighbor_count < min_neighbors)
    {
        points[index] = result;
        return;
    }

    // Estimate one local sample covariance per point. This is GICP point
    // geometry, not one covariance shared by an entire voxel.
    Eigen::Vector3f mean = Eigen::Vector3f::Zero();
    for (int neighbor = 0; neighbor < neighbor_count; ++neighbor)
    {
        const DevicePoint candidate = points[neighbor_indices[neighbor]];
        mean += candidate.position;
    }
    mean /= static_cast<float>(neighbor_count);

    Eigen::Matrix3f covariance = Eigen::Matrix3f::Zero();
    for (int neighbor = 0; neighbor < neighbor_count; ++neighbor)
    {
        const DevicePoint candidate = points[neighbor_indices[neighbor]];
        const Eigen::Vector3f offset = candidate.position - mean;
        covariance.noalias() += offset * offset.transpose();
    }
    covariance /= static_cast<float>(neighbor_count - 1);
    covariance.diagonal().array() += regularization;

    if (const float determinant = covariance.determinant();
        !isfinite(determinant) || fabsf(determinant) <= 1.0e-12f)
    {
        points[index] = result;
        return;
    }

    // Normalize by the Frobenius norm of the inverse covariance. This limits
    // the scale of the precision matrix while retaining local anisotropy.
    const float inverse_norm = covariance.inverse().norm();
    if (!isfinite(inverse_norm) || inverse_norm <= 0.0f)
    {
        points[index] = result;
        return;
    }

    result.covariance = covariance * inverse_norm;
    result.covariance_valid = 1;
    points[index] = result;
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
    if (state->status != AlignmentStatus::Running)
    {
        return;
    }
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t stride = blockDim.x * gridDim.x;

    const Sophus::SE3f pose(Eigen::Map<const Eigen::Matrix<float, 3, 3, Eigen::RowMajor>>(transform),
                            Eigen::Map<const Eigen::Vector3f>(transform + 9));
    while (index < num_source_points)
    {
        const DevicePoint source = source_points[index];
        const Eigen::Vector3f transformed_position = pose * source.position;

        const Eigen::Array3i base = (transformed_position / voxel_size).array().floor().cast<int>();

        float best_distance2 = max_correspondence_distance2;
        int best_target = -1;
        for (int dx = -adjacent_voxels; dx <= adjacent_voxels; ++dx)
        {
            for (int dy = -adjacent_voxels; dy <= adjacent_voxels; ++dy)
            {
                for (int dz = -adjacent_voxels; dz <= adjacent_voxels; ++dz)
                {
                    const auto found = target_voxels.find(cuda_func::packVoxelKey(base.x() + dx, base.y() + dy, base.z() + dz));
                    if (found == target_voxels.end())
                    {
                        continue;
                    }
                    const DeviceVoxelEntry voxel = target_voxel_entries[found->second];
                    for (int i = 0; i < voxel.count; ++i)
                    {
                        const int target_index = voxel.start + i;
                        const DevicePoint target = target_points[target_index];
                        if (const float distance2 = (transformed_position - target.position).squaredNorm();
                            distance2 < best_distance2)
                        {
                            best_distance2 = distance2;
                            best_target = target_index;
                        }
                    }
                }
            }
        }

        correspondences[index] = DeviceCorrespondence{best_target, transformed_position};
        index += stride;
    }
}

__device__ Eigen::Matrix3f rotationMatrix(const float* transform)
{
    return Eigen::Map<const Eigen::Matrix<float, 3, 3, Eigen::RowMajor>>(transform);
}

__global__ void buildLinearSystemKernel(const DevicePoint* source_points, const DevicePoint* target_points,
                                        const DeviceCorrespondence* correspondences, const int num_correspondences,
                                        const float* transform, const float cauchy_kernel_scale, float* partials,
                                        const DeviceAlignmentState* state)
{
    if (state->status != AlignmentStatus::Running)
    {
        return;
    }

    using BlockReduce = cub::BlockReduce<LinearSystemPartial, LINEAR_SYSTEM_BLOCK_SIZE, cub::BLOCK_REDUCE_WARP_REDUCTIONS>;
    __shared__ BlockReduce::TempStorage reduction_storage;
    LinearSystemPartial thread_sum;

    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t stride = blockDim.x * gridDim.x;
    while (index < num_correspondences)
    {
        if (const auto [target_index, transformed_source] = correspondences[index]; target_index >= 0)
        {
            const DevicePoint source = source_points[index];
            const DevicePoint target = target_points[target_index];
            const Eigen::Vector3f residual = transformed_source - target.position;

            // GICP combines the target covariance with the rotated source
            // covariance, yielding the Mahalanobis metric for this match.
            Eigen::Matrix3f covariance = Eigen::Matrix3f::Identity();
            if (source.covariance_valid != 0)
            {
                const Eigen::Matrix3f rotation = rotationMatrix(transform);
                const Eigen::Matrix3f target_covariance = target.covariance_valid != 0 ? target.covariance : Eigen::Matrix3f::Identity();
                covariance = target_covariance + rotation * source.covariance * rotation.transpose();
            }

            if (const Eigen::Matrix3f precision = covariance.inverse(); precision.array().isFinite().all())
            {
                const Eigen::Vector3f precision_residual = precision * residual;
                const float mahalanobis_error = residual.dot(precision_residual);
                const float kernel_scale2 = cauchy_kernel_scale * cauchy_kernel_scale;
                const float weight = cauchy_kernel_scale > 0.0f ? 1.0f / (1.0f + mahalanobis_error / kernel_scale2) : 1.0f;

                Eigen::Matrix<float, 3, 6> jacobian;
                jacobian.leftCols<3>().setIdentity();
                jacobian.rightCols<3>() = -Sophus::SO3f::hat(transformed_source);

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
                    thread_sum.values[HESSIAN_SIZE + element] += local_gradient(element);
                }
                thread_sum.values[VALID_COUNT_OFFSET] += 1.0f;
                thread_sum.values[SQUARED_ERROR_OFFSET] += residual.squaredNorm();
            }
        }
        index += stride;
    }

    const LinearSystemPartial block_sum = BlockReduce(reduction_storage).Sum(thread_sum);
    if (threadIdx.x == 0)
    {
        #pragma unroll
        for (int element = 0; element < LINEAR_SYSTEM_SIZE; ++element)
        {
            partials[blockIdx.x * LINEAR_SYSTEM_SIZE + element] = block_sum.values[element];
        }
    }
}

__global__ void solveAndUpdateKernel(const float* partials, const int num_blocks, const SparsityAwareGICP::Config config,
                                     float* transform, DeviceAlignmentState* state)
{
    if (threadIdx.x != 0 || state->status != AlignmentStatus::Running)
    {
        return;
    }

    Eigen::Matrix<float, 6, 7, Eigen::RowMajor> augmented =
        Eigen::Matrix<float, 6, 7, Eigen::RowMajor>::Zero();
    float valid_count = 0.0f;
    float squared_error_sum = 0.0f;

    for (int block = 0; block < num_blocks; ++block)
    {
        const float* partial = partials + block * LINEAR_SYSTEM_SIZE;
        int packed_index = 0;
        for (int row = 0; row < 6; ++row)
        {
            for (int col = row; col < 6; ++col)
            {
                const float value = partial[packed_index++];
                augmented(row, col) += value;
                if (row != col)
                {
                    augmented(col, row) += value;
                }
            }
            augmented(row, 6) -= partial[HESSIAN_SIZE + row];
        }
        valid_count += partial[VALID_COUNT_OFFSET];
        squared_error_sum += partial[SQUARED_ERROR_OFFSET];
    }

    state->num_correspondences = static_cast<int>(valid_count);
    state->fitness_score = valid_count > 0.0f ? squared_error_sum / valid_count : CUDART_INF_F;
    if (valid_count <= 0.0f || !isfinite(squared_error_sum))
    {
        state->status = AlignmentStatus::Aborted;
        return;
    }

    augmented.leftCols<6>().diagonal().array() += config.damping_factor;

    for (int diagonal = 0; diagonal < 6; ++diagonal)
    {
        int pivot_row = diagonal;
        for (int row = diagonal + 1; row < 6; ++row)
        {
            if (fabsf(augmented(row, diagonal)) > fabsf(augmented(pivot_row, diagonal)))
            {
                pivot_row = row;
            }
        }

        if (!isfinite(augmented(pivot_row, diagonal)) || fabsf(augmented(pivot_row, diagonal)) <= 1.0e-12f)
        {
            state->status = AlignmentStatus::Aborted;
            return;
        }

        for (int col = diagonal; col < 7; ++col)
        {
            const float temporary = augmented(diagonal, col);
            augmented(diagonal, col) = augmented(pivot_row, col);
            augmented(pivot_row, col) = temporary;
        }

        const float pivot = augmented(diagonal, diagonal);
        for (int col = diagonal; col < 7; ++col)
        {
            augmented(diagonal, col) /= pivot;
        }

        for (int row = 0; row < 6; ++row)
        {
            if (row == diagonal)
            {
                continue;
            }

            const float factor = augmented(row, diagonal);
            for (int col = diagonal; col < 7; ++col)
            {
                augmented(row, col) -= factor * augmented(diagonal, col);
            }
        }
    }

    const Sophus::SE3f::Tangent delta = augmented.col(6);
    if (!delta.array().isFinite().all())
    {
        state->status = AlignmentStatus::Aborted;
        return;
    }

    // Left composition keeps the Jacobian convention above consistent:
    // T_next = Exp(delta) * T_current.
    const Sophus::SE3f pose(rotationMatrix(transform), Eigen::Map<const Eigen::Vector3f>(transform + 9));
    const Eigen::Vector3f omega = delta.tail<3>();
    const float theta_squared = omega.squaredNorm();
    const float theta = sqrtf(theta_squared);
    const float first_coefficient = theta > 1.0e-5f ? (1.0f - cosf(theta)) / theta_squared : 0.5f - theta_squared / 24.0f;
    const float second_coefficient = theta > 1.0e-5f ? (theta - sinf(theta)) / (theta_squared * theta) : 1.0f / 6.0f - theta_squared / 120.0f;
    const Eigen::Matrix3f omega_hat = Sophus::SO3f::hat(omega);
    const Eigen::Matrix3f left_jacobian = Eigen::Matrix3f::Identity() + first_coefficient * omega_hat +
                                        second_coefficient * omega_hat * omega_hat;
    const Sophus::SE3f update(Sophus::SO3f::exp(omega), left_jacobian * delta.head<3>());
    const Sophus::SE3f updated_pose = update * pose;
    Eigen::Map<Eigen::Matrix<float, 3, 3, Eigen::RowMajor>> updated_rotation(transform);
    Eigen::Map<Eigen::Vector3f> updated_translation(transform + 9);
    updated_rotation = updated_pose.rotationMatrix();
    updated_translation = updated_pose.translation();
    state->iterations += 1;
    state->num_correspondences = static_cast<int>(valid_count);
    state->fitness_score = squared_error_sum / valid_count;
    if (delta.head<3>().norm() < config.convergence_translation &&
        delta.tail<3>().norm() < config.convergence_rotation)
    {
        state->status = AlignmentStatus::Converged;
    }
}
} // namespace

namespace cuda_func
{
thrust::device_vector<DevicePoint> estimateCovariances(const VoxelPointLayout& layout, const SparsityAwareGICP::Config& config)
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

void findCorrespondences(const thrust::device_vector<DevicePoint>& device_source,
                             const thrust::device_vector<DevicePoint>& device_target,
                             const thrust::device_vector<DeviceVoxelEntry>& device_voxels,
                             const DeviceVoxelMap& target_voxel_map,
                             thrust::device_vector<DeviceCorrespondence>& device_correspondences,
                             thrust::device_vector<float>& device_transform,
                             const SparsityAwareGICP::Config& config,
                             const thrust::device_vector<DeviceAlignmentState>& device_state)
{
    constexpr int block_size = LINEAR_SYSTEM_BLOCK_SIZE;
    const int grid_size = std::max(1, std::min(1024, static_cast<int>((device_source.size() + block_size - 1) / block_size)));
    const float max_distance2 = config.max_correspondence_distance * config.max_correspondence_distance;
    findCorrespondencesKernel<<<grid_size, block_size>>>(thrust::raw_pointer_cast(device_source.data()), static_cast<int>(device_source.size()),
                                                         thrust::raw_pointer_cast(device_target.data()),
                                                         thrust::raw_pointer_cast(device_voxels.data()), target_voxel_map.ref(cuco::find),
                                                         thrust::raw_pointer_cast(device_correspondences.data()), config.voxel_size,
                                                         config.adjacent_voxels, max_distance2,
                                                         thrust::raw_pointer_cast(device_transform.data()),
                                                         thrust::raw_pointer_cast(device_state.data()));
    if (cudaGetLastError() != cudaSuccess)
    {
        throw std::runtime_error("CUDA GICP correspondence-search launch failed");
    }
}

void buildLinearSystem(const std::size_t num_source_points,
                           const thrust::device_vector<DevicePoint>& device_source,
                           const thrust::device_vector<DevicePoint>& device_target,
                           const thrust::device_vector<DeviceCorrespondence>& device_correspondences,
                           const thrust::device_vector<float>& device_transform,
                           const SparsityAwareGICP::Config& config,
                           thrust::device_vector<float>& device_partials,
                           const thrust::device_vector<DeviceAlignmentState>& device_state)
{
    constexpr int block_size = LINEAR_SYSTEM_BLOCK_SIZE;
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

void solveAndUpdate(const thrust::device_vector<float>& device_partials,
                    const int num_blocks,
                    const SparsityAwareGICP::Config& config,
                    thrust::device_vector<float>& device_transform,
                    thrust::device_vector<DeviceAlignmentState>& device_state)
{
    solveAndUpdateKernel<<<1, 1>>>(thrust::raw_pointer_cast(device_partials.data()), num_blocks, config,
                                   thrust::raw_pointer_cast(device_transform.data()),
                                   thrust::raw_pointer_cast(device_state.data()));
    if (cudaGetLastError() != cudaSuccess)
    {
        throw std::runtime_error("CUDA GICP solve-and-update launch failed");
    }
}
}
