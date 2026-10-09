#include "perception/LIO/IESKF.h"
#include <Eigen/Cholesky>
#include <algorithm>
#include <array>
#include <chrono>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace
{
void require(bool condition, const std::string& message)
{
    if (!condition)
    {
        throw std::runtime_error(message);
    }
}

void checkCovariance(const lio::IESKF& filter)
{
    const auto& covariance = filter.covariance();
    require(covariance.allFinite(), "Covariance must be finite");
    require(covariance.isApprox(covariance.transpose(), 1e-12), "Covariance must be symmetric");
    const Eigen::LLT<Eigen::Matrix<double, 15, 15>> factorization(covariance);
    require(factorization.info() == Eigen::Success, "Covariance must be positive definite");
}

std::vector<lio::ImuData> stationarySamples()
{
    std::vector<lio::ImuData> samples(201);
    for (std::size_t index = 0; index < samples.size(); ++index)
    {
        samples[index].timestamp = static_cast<double>(index) * 0.01;
        samples[index].accel = Eigen::Vector3d(0.0, 0.0, 9.81);
    }
    return samples;
}

using ErrorState = Eigen::Matrix<double, 15, 1>;
using StateCovariance = Eigen::Matrix<double, 15, 15>;

lio::NavState inject(lio::NavState state, const ErrorState& correction)
{
    state.p_wi += correction.segment<3>(0);
    state.R_wi *= Sophus::SO3d::exp(correction.segment<3>(3));
    state.v_wi += correction.segment<3>(6);
    state.accel_bias += correction.segment<3>(9);
    state.gyro_bias += correction.segment<3>(12);
    return state;
}

ErrorState difference(const lio::NavState& state, const lio::NavState& reference)
{
    ErrorState result;
    result.segment<3>(0) = state.p_wi - reference.p_wi;
    result.segment<3>(3) = (reference.R_wi.inverse() * state.R_wi).log();
    result.segment<3>(6) = state.v_wi - reference.v_wi;
    result.segment<3>(9) = state.accel_bias - reference.accel_bias;
    result.segment<3>(12) = state.gyro_bias - reference.gyro_bias;
    return result;
}

lio::NavState referenceStep(lio::NavState state, const lio::ImuData& sample)
{
    const double dt = sample.timestamp - state.timestamp;
    const Eigen::Vector3d phi = (sample.gyro - state.gyro_bias) * dt;
    const Eigen::Vector3d acceleration = (state.R_wi * Sophus::SO3d::exp(0.5 * phi)) *
        (sample.accel - state.accel_bias) + state.gravity;
    state.p_wi += dt * state.v_wi + 0.5 * dt * dt * acceleration;
    state.v_wi += dt * acceleration;
    state.R_wi *= Sophus::SO3d::exp(phi);
    state.timestamp = sample.timestamp;
    return state;
}

void checkPropagationJacobian()
{
    lio::IESKF filter;
    require(filter.initialize(stationarySamples()), "Initialization failed");
    lio::ImuData sample;
    sample.accel = Eigen::Vector3d(0.4, -0.3, 9.9);
    sample.gyro = Eigen::Vector3d(0.7, -0.2, 0.4);
    for (int index = 1; index <= 20; ++index)
    {
        sample.timestamp = 2.0 + 0.01 * index;
        filter.predict(std::vector<lio::ImuData>{sample});
    }
    const auto prior = filter.state();
    const auto prior_covariance = filter.covariance();
    sample.timestamp += 0.01;
    const auto nominal = referenceStep(prior, sample);
    constexpr double epsilon = 1e-6;
    StateCovariance transition;
    for (int column = 0; column < 15; ++column)
    {
        ErrorState perturbation = ErrorState::Zero();
        perturbation[column] = epsilon;
        transition.col(column) = (difference(referenceStep(inject(prior, perturbation), sample), nominal) -
            difference(referenceStep(inject(prior, -perturbation), sample), nominal)) / (2.0 * epsilon);
    }
    Eigen::Matrix<double, 15, 6> noise;
    for (int column = 0; column < 6; ++column)
    {
        auto positive = sample;
        auto negative = sample;
        if (column < 3)
        {
            positive.accel[column] += epsilon;
            negative.accel[column] -= epsilon;
        }
        else
        {
            positive.gyro[column - 3] += epsilon;
            negative.gyro[column - 3] -= epsilon;
        }
        noise.col(column) = (difference(referenceStep(prior, positive), nominal) -
            difference(referenceStep(prior, negative), nominal)) / (2.0 * epsilon);
    }
    Eigen::Matrix<double, 6, 6> noise_covariance = Eigen::Matrix<double, 6, 6>::Zero();
    noise_covariance.diagonal().head<3>().setConstant(0.02 * 0.02);
    noise_covariance.diagonal().tail<3>().setConstant(std::pow(0.01 * lio::DEG2RAD, 2));
    StateCovariance expected = transition * prior_covariance * transition.transpose() +
        noise * noise_covariance * noise.transpose();
    const double dt = sample.timestamp - prior.timestamp;
    expected.block<3, 3>(9, 9).diagonal().array() += dt * 1e-8;
    expected.block<3, 3>(12, 12).diagonal().array() += dt * std::pow(0.001 * lio::DEG2RAD, 2);
    filter.predict(std::vector<lio::ImuData>{sample});
    require(difference(filter.state(), nominal).norm() < 1e-12, "Midpoint nominal state differs");
    require((filter.covariance() - expected).norm() / expected.norm() < 1e-8,
        "Sparse propagation differs from finite-difference dense propagation");
    checkCovariance(filter);
}

void checkRotatingAcceleration()
{
    lio::IESKF filter;
    require(filter.initialize(stationarySamples()), "Initialization failed");
    std::vector<lio::ImuData> samples(100);
    constexpr double rate = 2.0;
    constexpr double acceleration = 2.0;
    for (std::size_t index = 0; index < samples.size(); ++index)
    {
        samples[index].timestamp = 2.0 + static_cast<double>(index + 1) * 0.01;
        samples[index].gyro.z() = rate;
        samples[index].accel = Eigen::Vector3d(acceleration, 0.0, 9.81);
    }
    filter.predict(samples);
    const Eigen::Vector3d expected_velocity(acceleration * std::sin(rate) / rate,
        acceleration * (1.0 - std::cos(rate)) / rate, 0.0);
    const Eigen::Vector3d expected_position(acceleration * (1.0 - std::cos(rate)) / (rate * rate),
        acceleration * (1.0 / rate - std::sin(rate) / (rate * rate)), 0.0);
    const double position_error = (filter.state().p_wi - expected_position).norm();
    const double velocity_error = (filter.state().v_wi - expected_velocity).norm();
    std::cout << "Rotating acceleration: position error=" << position_error
              << " velocity error=" << velocity_error << '\n';
    require(position_error < 1e-4, "Rotating acceleration position accuracy regressed");
    require(velocity_error < 1e-4, "Rotating acceleration velocity accuracy regressed");
    checkCovariance(filter);
}

void checkInvalidImu()
{
    lio::IESKF filter;
    require(filter.initialize(stationarySamples()), "Initialization failed");
    const auto prior = filter.state();
    const auto covariance = filter.covariance();
    lio::ImuData sample;
    sample.timestamp = 1.9;
    sample.accel = Eigen::Vector3d(0.0, 0.0, 9.81);
    filter.predict(std::vector<lio::ImuData>{sample});
    require(filter.state().timestamp == prior.timestamp, "Out-of-order IMU rewound timestamp");
    sample.timestamp = 2.01;
    sample.accel.x() = std::numeric_limits<double>::quiet_NaN();
    filter.predict(std::vector<lio::ImuData>{sample});
    require(filter.state().timestamp == prior.timestamp, "Invalid IMU changed timestamp");
    require(filter.state().p_wi.isApprox(prior.p_wi), "Invalid IMU changed position");
    require(filter.covariance().isApprox(covariance), "Invalid IMU changed covariance");
}

pcl::PointCloud<pcl::PointXYZ> planeMap()
{
    pcl::PointCloud<pcl::PointXYZ> cloud;
    for (int first = -15; first <= 15; ++first)
    {
        for (int second = -15; second <= 15; ++second)
        {
            const float first_coordinate = static_cast<float>(first) * 0.08f;
            const float second_coordinate = static_cast<float>(second) * 0.08f;
            cloud.emplace_back(3.0f, first_coordinate, second_coordinate);
            cloud.emplace_back(first_coordinate, 3.0f, second_coordinate);
            cloud.emplace_back(first_coordinate, second_coordinate, -3.0f);
        }
    }
    return cloud;
}

pcl::PointCloud<pcl::PointXYZ> makeScan(const pcl::PointCloud<pcl::PointXYZ>& map,
    const Sophus::SE3d& pose)
{
    pcl::PointCloud<pcl::PointXYZ> scan;
    for (std::size_t index = 0; index < map.size(); index += 2)
    {
        const Eigen::Vector3f point = (pose.inverse() * map[index].getVector3fMap().cast<double>()).cast<float>();
        scan.emplace_back(point.x(), point.y(), point.z());
    }
    return scan;
}

void checkObservation()
{
    const auto map = planeMap();
    const Sophus::SE3d expected_pose(Sophus::SO3d::exp(Eigen::Vector3d(0.01, -0.008, 0.006)),
        Eigen::Vector3d(0.04, -0.03, 0.02));
    const auto scan = makeScan(map, expected_pose);
    lio::IESKF filter;
    require(filter.initialize(stationarySamples()), "Initialization failed");
    filter.initialize(map);
    filter.observe(scan);
    const double position_error = (filter.state().p_wi - expected_pose.translation()).norm();
    const double rotation_error = (expected_pose.so3().inverse() * filter.state().R_wi).log().norm();
    std::cout << "Plane alignment: position error=" << position_error
              << " rotation error=" << rotation_error << '\n';
    require(position_error < 1e-4, "Plane alignment position accuracy regressed");
    require(rotation_error < 1e-4, "Plane alignment rotation accuracy regressed");
    checkCovariance(filter);
    const auto prior = filter.state();
    const auto covariance = filter.covariance();
    filter.observe(pcl::PointCloud<pcl::PointXYZ>{});
    pcl::PointCloud<pcl::PointXYZ> unmatched;
    unmatched.emplace_back(100.0f, 100.0f, 100.0f);
    filter.observe(unmatched);
    require(filter.state().p_wi.isApprox(prior.p_wi), "Rejected scan changed position");
    require(filter.covariance().isApprox(covariance), "Rejected scan changed covariance");
    auto invalid_scan = scan;
    invalid_scan.emplace_back(std::numeric_limits<float>::quiet_NaN(), 0.0f, 0.0f);
    filter.observe(invalid_scan);
    require(filter.state().p_wi.allFinite(), "NaN scan contaminated state");
    checkCovariance(filter);
}

void checkReducedObservation()
{
    const auto map = planeMap();
    lio::IESKF filter;
    require(filter.initialize(stationarySamples()), "Initialization failed");
    filter.initialize(map);
    lio::ImuData sample;
    sample.accel = Eigen::Vector3d(0.4, -0.3, 9.9);
    sample.gyro = Eigen::Vector3d(0.2, -0.1, 0.3);
    for (int index = 1; index <= 20; ++index)
    {
        sample.timestamp = 2.0 + 0.01 * index;
        filter.predict(std::vector<lio::ImuData>{sample});
    }
    const auto prior = filter.state();
    const auto prior_covariance = filter.covariance();
    const Sophus::SE3d pose(prior.R_wi * Sophus::SO3d::exp(Eigen::Vector3d(1e-5, -1e-5, 1e-5)),
        prior.p_wi + Eigen::Vector3d(2e-5, -2e-5, 2e-5));
    const auto scan = makeScan(map, pose);
    IVox reference_map;
    reference_map.initialize(map);
    StateCovariance information = prior_covariance.llt().solve(StateCovariance::Identity());
    ErrorState gradient = ErrorState::Zero();
    int effective_points = 0;
    for (const auto& point : scan)
    {
        const Eigen::Vector3d point_i = point.getVector3fMap().cast<double>();
        const Eigen::Vector3d point_w = prior.R_wi * point_i + prior.p_wi;
        const auto neighbors = reference_map.nearestNeighbors(point_w.cast<float>(), 5, 1.0f);
        const auto plane = estimatePlane(neighbors, 0.1f);
        if (!plane.valid)
        {
            continue;
        }
        const Eigen::Vector3d normal = plane.normal.cast<double>();
        const double residual = normal.dot(point_w - plane.center.cast<double>());
        require(std::abs(residual) < 0.05, "Reference case crossed robust weighting threshold");
        Eigen::Matrix<double, 6, 1> jacobian;
        jacobian.head<3>() = normal;
        jacobian.tail<3>() = point_i.cross(prior.R_wi.inverse() * normal);
        const double weight = 1.0 / (1e-6 + static_cast<double>(plane.normal_variance));
        information.topLeftCorner<6, 6>().noalias() += weight * jacobian * jacobian.transpose();
        gradient.head<6>() += weight * residual * jacobian;
        ++effective_points;
    }
    require(effective_points >= 10, "Insufficient reference correspondences");
    const Eigen::LLT<StateCovariance> solver(information);
    require(solver.info() == Eigen::Success, "Dense reference solve failed");
    const ErrorState correction = -solver.solve(gradient);
    require(correction.segment<3>(0).norm() < 1e-4 && correction.segment<3>(3).norm() < 1e-4,
        "Dense reference case should converge in one iteration");
    const auto expected_state = inject(prior, correction);
    StateCovariance reset = StateCovariance::Identity();
    const Eigen::Matrix3d rotation_hat = Sophus::SO3d::hat(correction.segment<3>(3));
    reset.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity() - 0.5 * rotation_hat + rotation_hat * rotation_hat / 6.0;
    const StateCovariance expected_covariance = reset * solver.solve(StateCovariance::Identity()) * reset.transpose();
    filter.observe(scan);
    require(difference(filter.state(), expected_state).norm() < 1e-9,
        "Reduced state update differs from dense 15-state solve");
    require((filter.covariance() - expected_covariance).norm() / expected_covariance.norm() < 1e-8,
        "Reduced covariance differs from dense 15-state solve");
    require((filter.state().accel_bias - prior.accel_bias).norm() > 1e-12,
        "Cross-covariance failed to update accelerometer bias");
    checkCovariance(filter);
}

void checkNeighborsAndPlanes()
{
    IVox map;
    pcl::PointCloud<pcl::PointXYZ> cloud;
    for (int first = -6; first <= 6; ++first)
    {
        for (int second = -6; second <= 6; ++second)
        {
            for (int third = -6; third <= 6; ++third)
            {
                cloud.emplace_back(0.2f * first, 0.2f * second, 0.2f * third);
            }
        }
    }
    map.initialize(cloud);
    const Eigen::Vector3f query(0.24f, 0.24f, 0.24f);
    std::vector<float> expected;
    for (const auto& point : cloud)
    {
        const float distance = (point.getVector3fMap() - query).squaredNorm();
        if (distance < 1.0f)
        {
            expected.push_back(distance);
        }
    }
    std::sort(expected.begin(), expected.end());
    for (const int count : {1, 5, 32, 100, 300})
    {
        const auto neighbors = map.nearestNeighbors(query, count, 1.0f);
        require(neighbors.size() == std::min(expected.size(), static_cast<std::size_t>(count)), "KNN count differs");
        for (std::size_t index = 0; index < neighbors.size(); ++index)
        {
            require(std::abs(neighbors[index].squared_distance - expected[index]) < 1e-6f,
                "KNN differs from brute force");
        }
    }
    const std::array<Eigen::Vector3f, 5> line{Eigen::Vector3f(-0.2f, 0.0f, 0.0f),
        Eigen::Vector3f(-0.1f, 0.0f, 0.0f), Eigen::Vector3f::Zero(),
        Eigen::Vector3f(0.1f, 0.0f, 0.0f), Eigen::Vector3f(0.2f, 0.0f, 0.0f)};
    std::array<IVox::Neighbor, 5> neighbors;
    for (std::size_t index = 0; index < line.size(); ++index)
    {
        neighbors[index].point = &line[index];
    }
    require(!estimatePlane(neighbors, 0.1f).valid, "Collinear points produced a plane");
}

void benchmark()
{
    const auto map = planeMap();
    const Sophus::SE3d expected_pose(Sophus::SO3d::exp(Eigen::Vector3d(0.01, -0.008, 0.006)),
        Eigen::Vector3d(0.04, -0.03, 0.02));
    const auto scan = makeScan(map, expected_pose);
    const auto initialization = stationarySamples();
    constexpr int repetitions = 12;
    double observe_ms = 0.0;
    for (int repetition = 0; repetition <= repetitions; ++repetition)
    {
        lio::IESKF filter;
        require(filter.initialize(initialization), "Initialization failed");
        filter.initialize(map);
        const auto start = std::chrono::steady_clock::now();
        filter.observe(scan);
        const auto end = std::chrono::steady_clock::now();
        if (repetition > 0)
        {
            observe_ms += std::chrono::duration<double, std::milli>(end - start).count();
        }
        checkCovariance(filter);
    }
    lio::IESKF filter;
    require(filter.initialize(initialization), "Initialization failed");
    std::vector<lio::ImuData> samples(10000);
    for (std::size_t index = 0; index < samples.size(); ++index)
    {
        samples[index].timestamp = 2.0 + static_cast<double>(index + 1) * 0.01;
        samples[index].accel = Eigen::Vector3d(2.0, 0.0, 9.81);
        samples[index].gyro = Eigen::Vector3d(0.2, -0.1, 0.5);
    }
    const auto start = std::chrono::steady_clock::now();
    filter.predict(samples);
    const auto end = std::chrono::steady_clock::now();
    checkCovariance(filter);
    std::cout << "observe_ms=" << observe_ms / repetitions
              << " scan_points=" << scan.size() << " predict_us="
              << std::chrono::duration<double, std::micro>(end - start).count() / samples.size() << '\n';
}
}

int main(int argc, char** argv)
{
    try
    {
        if (argc == 2 && std::string(argv[1]) == "--benchmark")
        {
            benchmark();
            return 0;
        }
        lio::IESKF filter;
        checkCovariance(filter);
        require(filter.initialize(stationarySamples()), "Stationary initialization failed");
        lio::ImuData sample;
        sample.accel = Eigen::Vector3d(0.0, 0.0, 9.81);
        for (int index = 1; index <= 100; ++index)
        {
            sample.timestamp = 2.0 + static_cast<double>(index) * 0.01;
            filter.predict(std::vector<lio::ImuData>{sample});
        }
        require(filter.state().p_wi.norm() < 1e-12, "Stationary IMU caused position drift");
        require(filter.state().v_wi.norm() < 1e-12, "Stationary IMU caused velocity drift");
        checkCovariance(filter);
        checkRotatingAcceleration();
        checkPropagationJacobian();
        checkInvalidImu();
        checkObservation();
        checkReducedObservation();
        checkNeighborsAndPlanes();
        std::cout << "IESKF regressions passed\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}