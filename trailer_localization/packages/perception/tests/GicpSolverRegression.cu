#include "../source/LIO/SparsityAwareGICP.cu"
#include <iostream>
#include <string>

namespace
{
using namespace perception::lio;

void require(const bool condition, const char* message)
{
    if (!condition)
    {
        throw std::runtime_error(message);
    }
}

void testSolver(const bool zero_gradient, const bool singular)
{
    Eigen::Matrix<float, 6, 6> basis;
    basis << 1, 2, 0, 0, 0, 0,
             0, 1, 3, 0, 0, 0,
             0, 0, 1, 2, 0, 0,
             0, 0, 0, 1, 3, 0,
             0, 0, 0, 0, 1, 2,
             0, 0, 0, 0, 0, 1;
    Eigen::Matrix<float, 6, 6> hessian = basis.transpose() * basis;
    if (singular)
    {
        hessian.setZero();
    }
    Eigen::Matrix<float, 6, 1> gradient;
    gradient << 0.02f, -0.01f, 0.03f, 0.01f, -0.02f, 0.04f;
    if (zero_gradient)
    {
        gradient.setZero();
    }
    const float damping = singular ? 0.0f : 0.1f;
    std::vector<float> partials(2 * linear_system_size, 0.0f);
    for (int block = 0; block < 2; ++block)
    {
        int packed_index = 0;
        for (int row = 0; row < 6; ++row)
        {
            for (int col = row; col < 6; ++col)
            {
                partials[block * linear_system_size + packed_index++] = hessian(row, col) * 0.5f;
            }
            partials[block * linear_system_size + hessian_size + row] = gradient(row) * 0.5f;
        }
        partials[block * linear_system_size + valid_count_offset] = 10.0f;
        partials[block * linear_system_size + squared_error_offset] = 0.1f;
        partials[block * linear_system_size + raw_count_offset] = 12.0f;
    }
    Eigen::Isometry3f initial = Eigen::Isometry3f::Identity();
    initial.linear() = Eigen::AngleAxisf(0.3f, Eigen::Vector3f::UnitY()).toRotationMatrix();
    initial.translation() = Eigen::Vector3f(3.0f, -2.0f, 1.0f);
    std::array<float, 12> transform;
    for (int row = 0; row < 3; ++row)
    {
        for (int col = 0; col < 3; ++col)
        {
            transform[row * 3 + col] = initial.linear()(row, col);
        }
        transform[9 + row] = initial.translation()(row);
    }
    thrust::device_vector<float> device_partials(partials.begin(), partials.end());
    thrust::device_vector<float> device_transform(transform.begin(), transform.end());
    thrust::device_vector<DeviceAlignmentState> device_state(1, DeviceAlignmentState{});
    solveAndUpdateKernel<<<1, 1>>>(thrust::raw_pointer_cast(device_partials.data()), 2, damping,
                                   1.0e-5f, 1.0e-5f, thrust::raw_pointer_cast(device_transform.data()),
                                   thrust::raw_pointer_cast(device_state.data()));
    require(cudaDeviceSynchronize() == cudaSuccess, "Solver CUDA execution failed");
    const thrust::host_vector<DeviceAlignmentState> state = device_state;
    const thrust::host_vector<float> actual = device_transform;
    require(state[0].raw_correspondences == 24, "Raw-match reduction or failure diagnostics incorrect");
    require(state[0].num_correspondences == 20, "Valid-match reduction or failure diagnostics incorrect");
    require(std::abs(state[0].fitness_score - 0.01f) < 1.0e-6f, "Fitness reduction incorrect");
    if (singular)
    {
        require(state[0].iterations == 0 && state[0].active == 0, "Singular solve must stop without an update");
    }
    else
    {
        require(state[0].iterations == 1, "Valid system failed to solve");
    }
    Eigen::Isometry3f expected = initial;
    if (!singular)
    {
        hessian.diagonal().array() += damping;
        const Eigen::Matrix<float, 6, 1> delta = hessian.ldlt().solve(-gradient);
        expected = sophusExpUpdate(delta) * initial;
    }
    for (int row = 0; row < 3; ++row)
    {
        for (int col = 0; col < 3; ++col)
        {
            require(std::abs(actual[row * 3 + col] - expected.linear()(row, col)) < 2.0e-5f,
                    "GPU rotation differs from Eigen/Sophus reference");
        }
        require(std::abs(actual[9 + row] - expected.translation()(row)) < 2.0e-5f,
                "GPU translation differs from Eigen/Sophus reference");
    }
}

void testInactiveKernels()
{
    DeviceAlignmentState inactive;
    inactive.active = 0;
    thrust::device_vector<DeviceAlignmentState> state(1, inactive);
    thrust::device_vector<float> partials(linear_system_size, -123.0f);
    DeviceCorrespondence sentinel;
    sentinel.target_index = 123;
    thrust::device_vector<DeviceCorrespondence> correspondences(1, sentinel);
    DeviceVoxelMap voxel_map(2, cuco::empty_key{std::numeric_limits<std::int64_t>::min()},
                             cuco::empty_value{-1}, cuda::std::equal_to<std::int64_t>{},
                             cuco::linear_probing<1, cuco::default_hash_function<std::int64_t>>{});
    findCorrespondencesKernel<<<1, 256>>>(nullptr, 1, nullptr, nullptr, voxel_map.ref(cuco::find),
                                          thrust::raw_pointer_cast(correspondences.data()), 0.05f, 2, 0.04f,
                                          nullptr, thrust::raw_pointer_cast(state.data()));
    buildLinearSystemKernel<<<1, 256>>>(nullptr, nullptr, nullptr, 1, nullptr, 0.3f,
                                        thrust::raw_pointer_cast(partials.data()),
                                        thrust::raw_pointer_cast(state.data()));
    require(cudaDeviceSynchronize() == cudaSuccess, "Inactive kernels accessed their inputs");
    const thrust::host_vector<float> actual_partials = partials;
    const thrust::host_vector<DeviceCorrespondence> actual_correspondences = correspondences;
    require(std::all_of(actual_partials.begin(), actual_partials.end(), [](const float value) { return value == -123.0f; }),
            "Inactive linear-system kernel overwrote partials");
    require(actual_correspondences[0].target_index == 123, "Inactive search overwrote correspondences");
}

void testLinearSystem(const int num_points, const int num_blocks, const bool all_invalid)
{
    std::vector<DevicePoint> source(num_points);
    std::vector<DevicePoint> target(num_points);
    std::vector<DeviceCorrespondence> correspondences(num_points);
    const Eigen::Matrix3f rotation = Eigen::AngleAxisf(0.2f, Eigen::Vector3f(1.0f, 2.0f, 3.0f).normalized()).toRotationMatrix();
    const std::array<float, 12> transform{rotation(0, 0), rotation(0, 1), rotation(0, 2),
                                        rotation(1, 0), rotation(1, 1), rotation(1, 2),
                                        rotation(2, 0), rotation(2, 1), rotation(2, 2), 0.1f, -0.2f, 0.3f};
    Eigen::Matrix<double, 6, 6> expected_hessian = Eigen::Matrix<double, 6, 6>::Zero();
    Eigen::Matrix<double, 6, 1> expected_gradient = Eigen::Matrix<double, 6, 1>::Zero();
    int raw_count = 0;
    int valid_count = 0;
    double squared_error = 0.0;
    for (int index = 0; index < num_points; ++index)
    {
        const Eigen::Vector3f position(0.01f * (index % 31), 0.02f * (index % 17), 0.03f * (index % 13));
        const Eigen::Vector3f transformed = rotation * position + Eigen::Vector3f(0.1f, -0.2f, 0.3f);
        const Eigen::Vector3f residual(0.001f * (index % 9 - 4), 0.002f * (index % 7 - 3), 0.003f * (index % 5 - 2));
        const Eigen::Vector3f target_position = transformed - residual;
        source[index].x = position.x();
        source[index].y = position.y();
        source[index].z = position.z();
        target[index].x = target_position.x();
        target[index].y = target_position.y();
        target[index].z = target_position.z();
        source[index].covariance_valid = index % 3 != 0;
        target[index].covariance_valid = index % 4 != 0;
        Eigen::Matrix3f source_covariance;
        source_covariance << 0.2f, 0.01f, 0.02f, 0.01f, 0.3f, 0.01f, 0.02f, 0.01f, 0.4f;
        const Eigen::Matrix3f target_covariance = 2.0f * source_covariance;
        for (int row = 0; row < 3; ++row)
        {
            for (int col = 0; col < 3; ++col)
            {
                source[index].covariance[row * 3 + col] = source_covariance(row, col);
                target[index].covariance[row * 3 + col] = target_covariance(row, col);
            }
        }
        correspondences[index] = DeviceCorrespondence{!all_invalid && index % 7 != 0 ? index : -1,
                                   transformed.x(), transformed.y(), transformed.z()};
        if (correspondences[index].target_index < 0)
        {
            continue;
        }
        ++raw_count;
        Eigen::Matrix3f covariance = Eigen::Matrix3f::Identity();
        if (source[index].covariance_valid != 0)
        {
            covariance = target[index].covariance_valid != 0 ? target_covariance : Eigen::Matrix3f::Identity();
            covariance += rotation * source_covariance * rotation.transpose();
        }
        const Eigen::Matrix3f precision = covariance.inverse();
        const Eigen::Vector3f actual_residual = transformed - target_position;
        const float weight = 1.0f / (1.0f + actual_residual.dot(precision * actual_residual) / (0.3f * 0.3f));
        Eigen::Matrix<float, 3, 6> jacobian;
        jacobian.leftCols<3>().setIdentity();
        jacobian.rightCols<3>() << 0.0f, transformed.z(), -transformed.y(),
                                  -transformed.z(), 0.0f, transformed.x(),
                                  transformed.y(), -transformed.x(), 0.0f;
        expected_hessian += (jacobian.transpose() * weight * precision * jacobian).cast<double>();
        expected_gradient += (jacobian.transpose() * weight * precision * actual_residual).cast<double>();
        squared_error += actual_residual.squaredNorm();
        ++valid_count;
    }
    thrust::device_vector<DevicePoint> device_source(source.begin(), source.end());
    thrust::device_vector<DevicePoint> device_target(target.begin(), target.end());
    thrust::device_vector<DeviceCorrespondence> device_correspondences(correspondences.begin(), correspondences.end());
    thrust::device_vector<float> device_transform(transform.begin(), transform.end());
    thrust::device_vector<DeviceAlignmentState> state(1, DeviceAlignmentState{});
    thrust::device_vector<float> partials(num_blocks * linear_system_size, std::numeric_limits<float>::quiet_NaN());
    buildLinearSystemKernel<<<num_blocks, linear_system_block_size>>>(thrust::raw_pointer_cast(device_source.data()),
        thrust::raw_pointer_cast(device_target.data()), thrust::raw_pointer_cast(device_correspondences.data()), num_points,
        thrust::raw_pointer_cast(device_transform.data()), 0.3f, thrust::raw_pointer_cast(partials.data()),
        thrust::raw_pointer_cast(state.data()));
    require(cudaDeviceSynchronize() == cudaSuccess, "Linear-system kernel execution failed");
    const thrust::host_vector<float> actual = partials;
    std::array<double, linear_system_size> sum{};
    for (int block = 0; block < num_blocks; ++block)
    {
        for (int element = 0; element < linear_system_size; ++element)
        {
            sum[element] += actual[block * linear_system_size + element];
        }
    }
    int packed_index = 0;
    for (int row = 0; row < 6; ++row)
    {
        for (int col = row; col < 6; ++col)
        {
            require(std::abs(sum[packed_index++] - expected_hessian(row, col)) < 5.0e-5 * std::max(1.0, std::abs(expected_hessian(row, col))),
                    "Reduced Hessian differs from CPU reference");
        }
        require(std::abs(sum[hessian_size + row] - expected_gradient(row)) < 5.0e-5 * std::max(1.0, std::abs(expected_gradient(row))),
                "Reduced gradient differs from CPU reference");
    }
    require(sum[raw_count_offset] == raw_count && sum[valid_count_offset] == valid_count, "Reduced correspondence counts incorrect");
    require(std::abs(sum[squared_error_offset] - squared_error) < 1.0e-6, "Reduced fitness sum incorrect");
}

void testTargetCapacity()
{
    SparsityAwareGICPConfig config;
    config.max_target_voxels = 3;
    config.max_iterations = 1;
    SparsityAwareGICP gicp(config);
    pcl::PointCloud<pcl::PointXYZ> initial;
    initial.emplace_back(0.01f, 0.01f, 0.01f);
    initial.emplace_back(0.21f, 0.01f, 0.01f);
    gicp.initializeTarget(initial);
    pcl::PointCloud<pcl::PointXYZ> excess;
    excess.emplace_back(0.41f, 0.01f, 0.01f);
    excess.emplace_back(0.61f, 0.01f, 0.01f);
    gicp.insertTargetPoints(excess);
    require(gicp.align(initial, Eigen::Isometry3f::Identity()).num_target_points == 2, "Over-budget batch changed target");
    gicp.insertTargetPoints(initial);
    require(gicp.align(initial, Eigen::Isometry3f::Identity()).num_target_points == 2, "Duplicate voxels changed target");
    excess.resize(1);
    gicp.insertTargetPoints(excess);
    const auto inserted = gicp.align(excess, Eigen::Isometry3f::Identity());
    require(inserted.num_target_points == 3 && inserted.num_raw_correspondences == 1,
            "Incremental target layout or correspondence lookup incorrect");
    excess.front().x = 0.81f;
    gicp.insertTargetPoints(excess);
    require(gicp.align(initial, Eigen::Isometry3f::Identity()).num_target_points == 3, "Full target accepted new voxels");
    initial += excess;
    initial.emplace_back(1.01f, 0.01f, 0.01f);
    bool rejected = false;
    try
    {
        gicp.initializeTarget(initial);
    }
    catch (const std::invalid_argument&)
    {
        rejected = true;
    }
    require(rejected, "Oversized initial target was not rejected");
    require(gicp.align(excess, Eigen::Isometry3f::Identity()).num_target_points == 3,
            "Rejected initialization replaced the existing target");
}

void testAlignment()
{
    pcl::PointCloud<pcl::PointXYZ> target;
    pcl::PointCloud<pcl::PointXYZ> source;
    const Eigen::Vector3f translation(3.0f, -2.0f, 1.0f);
    for (int point_index = 0; point_index < 600; ++point_index)
    {
        const Eigen::Vector3f position(0.017f * (point_index % 13),
                                       0.023f * ((point_index / 13) % 11),
                                       0.031f * (point_index / 143));
        target.emplace_back(position.x(), position.y(), position.z());
        const Eigen::Vector3f shifted = position - translation;
        source.emplace_back(shifted.x(), shifted.y(), shifted.z());
    }
    SparsityAwareGICPConfig config;
    config.max_correspondence_distance = 0.2f;
    SparsityAwareGICP gicp(config);
    gicp.initializeTarget(target);
    Eigen::Isometry3f initial = Eigen::Isometry3f::Identity();
    initial.translation() = translation;
    const auto exact_result = gicp.align(source, initial);
    require(exact_result.converged && exact_result.num_raw_correspondences > 500,
            "Exact alignment failed or lost spatial matches");
    require((exact_result.transform.matrix() - initial.matrix()).norm() < 1.0e-4f,
            "Exact alignment changed a correct nonzero pose");
    initial.translation().x() += 0.004f;
    const auto perturbed_result = gicp.align(source, initial);
    require(perturbed_result.converged && perturbed_result.num_correspondences > 500,
            "Perturbed alignment failed");
    require((perturbed_result.transform.translation() - translation).norm() < 1.0e-3f,
            "Perturbed alignment did not recover the known translation");
    require(exact_result.iterations < config.max_iterations, "Exact alignment failed to stop early");
    std::cout << "Alignment iterations: exact=" << exact_result.iterations
              << ", perturbed=" << perturbed_result.iterations << '\n';
}
}

int main(const int argc, const char* const* argv)
{
    try
    {
        if (argc == 2 && std::string(argv[1]) == "--alignment-only")
        {
            testAlignment();
            return 0;
        }
        testSolver(true, false);
        testSolver(false, false);
        testSolver(false, true);
        testInactiveKernels();
        testLinearSystem(777, 2, false);
        testLinearSystem(17, 1, true);
        testLinearSystem(0, 2, false);
        testTargetCapacity();
        testAlignment();
        std::cout << "GPU solver and alignment regression tests passed\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}