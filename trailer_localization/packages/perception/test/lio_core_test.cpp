#include "perception/LIO/ImuProcessor.h"
#include <Eigen/Eigenvalues>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace
{
void check(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}

std::vector<lio::ImuData> stationarySamples()
{
    std::vector<lio::ImuData> samples;
    for (int i = 0; i <= 100; ++i)
        samples.push_back({i * 0.005, Eigen::Vector3d::Zero(), {0.0, 0.0, 9.81}});
    return samples;
}

lio::ESKF initialized(const lio::ESKF::Config& config = {})
{
    lio::ESKF filter(config);
    filter.initialize(stationarySamples());
    return filter;
}

void checkCovariance(const lio::ESKF& filter)
{
    const auto& P = filter.covariance();
    check(P.allFinite(), "non-finite covariance");
    check((P - P.transpose()).norm() < 1e-12, "asymmetric covariance");
    check(Eigen::SelfAdjointEigenSolver<lio::ESKF::StateCov>(P).eigenvalues().minCoeff() > -1e-12,
          "covariance lost positive semidefiniteness");
}

void initializationAndMotion()
{
    auto samples = stationarySamples();
    const auto tilt = Sophus::SO3d::exp(Eigen::Vector3d(0.3, -0.2, 0.0));
    const Eigen::Vector3d bias(0.001, -0.002, 0.003);
    for (auto& sample : samples)
    {
        sample.accel = tilt.inverse() * Eigen::Vector3d(0.0, 0.0, 9.81);
        sample.gyro = bias;
    }
    lio::ESKF filter;
    filter.initialize(samples);
    check((filter.state().gyro_bias - bias).norm() < 1e-12, "gyro calibration");
    auto next = samples.back();
    for (int i = 1; i <= 1000; ++i)
    {
        next.timestamp = 0.5 + 0.005 * i;
        filter.predict(next);
    }
    check(filter.state().p_wi.norm() < 1e-10, "stationary position drift");
    check(filter.state().v_wi.norm() < 1e-10, "stationary velocity drift");
    checkCovariance(filter);

    filter = initialized();
    // Linearly increasing specific force a_x = t, midpoint integration gives exact velocity.
    for (int i = 1; i <= 1000; ++i)
    {
        const double t = i * 0.001;
        filter.predict({0.5 + t, Eigen::Vector3d::Zero(), {t, 0.0, 9.81}});
    }
    check(std::abs(filter.state().v_wi.x() - 0.5) < 1e-10, "accelerating velocity");
    check(std::abs(filter.state().p_wi.x() - 1.0 / 6.0) < 1e-6, "accelerating position");
    checkCovariance(filter);
}

void covarianceJacobian()
{
    lio::ESKF::Config config;
    config.accel_noise_density = config.gyro_noise_density = 0.0;
    auto filter = initialized(config);
    const auto P0 = filter.covariance();
    const double dt = 0.02;
    const lio::ImuData end{0.52, {0.7, -0.4, 1.2}, {1.1, 0.3, 9.6}};
    // Numerically differentiate the midpoint nominal dynamics in right-local coordinates.
    const auto motion = [&](const Eigen::Matrix<double, 9, 1>& dx)
    {
        const auto R = Sophus::SO3d::exp(dx.segment<3>(3));
        const auto dR = Sophus::SO3d::exp(0.5 * end.gyro * dt);
        const Eigen::Vector3d force = 0.5 * (end.accel + Eigen::Vector3d(0, 0, 9.81));
        const Eigen::Vector3d a = R * Sophus::SO3d::exp(0.25 * end.gyro * dt) * force - Eigen::Vector3d(0, 0, 9.81);
        lio::ImuState state;
        state.p_wi = dx.head<3>() + dx.tail<3>() * dt + a * (0.5 * dt * dt);
        state.v_wi = dx.tail<3>() + a * dt;
        state.R_wi = R * dR;
        return state;
    };
    const auto nominal = motion(Eigen::Matrix<double, 9, 1>::Zero());
    lio::ESKF::StateCov numerical_F;
    for (int i = 0; i < 9; ++i)
    {
        Eigen::Matrix<double, 9, 1> dx = Eigen::Matrix<double, 9, 1>::Zero();
        dx[i] = 1e-6;
        const auto plus = motion(dx);
        const auto minus = motion(-dx);
        numerical_F.block<3, 1>(0, i) = (plus.p_wi - minus.p_wi) / 2e-6;
        numerical_F.block<3, 1>(3, i) = ((nominal.R_wi.inverse() * plus.R_wi).log() -
                                       (nominal.R_wi.inverse() * minus.R_wi).log()) / 2e-6;
        numerical_F.block<3, 1>(6, i) = (plus.v_wi - minus.v_wi) / 2e-6;
    }
    filter.predict(end);
    check((filter.covariance() - numerical_F * P0 * numerical_F.transpose()).norm() < 1e-10,
          "midpoint covariance Jacobian does not match nominal propagation");
}

void noiseRateAndObservation()
{
    const auto propagated = [](double dt)
    {
        auto filter = initialized();
        const int count = static_cast<int>(1.0 / dt);
        for (int i = 1; i <= count; ++i)
            filter.predict({0.5 + i * dt, Eigen::Vector3d::Zero(), {0, 0, 9.81}});
        return filter;
    };
    const auto low_rate = propagated(0.01);
    auto high_rate = propagated(0.002);
    check((low_rate.covariance() - high_rate.covariance()).norm() < 1e-6,
          "continuous IMU noise depends on sample rate");
    const lio::ESKF::MeasurementCov covariance = lio::ESKF::MeasurementCov::Identity() * 1e-3;
    const auto before = high_rate;
    check(!high_rate.observe(Sophus::SE3d(Sophus::SO3d(), Eigen::Vector3d(100, 0, 0)), covariance),
          "outlier was accepted");
    check((high_rate.covariance() - before.covariance()).norm() == 0.0, "rejection changed covariance");
    check((high_rate.pose().matrix() - before.pose().matrix()).norm() == 0.0, "rejection changed pose");
    check(high_rate.observe(Sophus::SE3d(Sophus::SO3d::exp(Eigen::Vector3d(0, 0, 0.01)),
                                       Eigen::Vector3d(0.02, 0, 0)), covariance), "valid pose rejected");
    check(high_rate.state().p_wi.x() > 0.0 && high_rate.state().p_wi.x() < 0.02, "position correction");
    check(high_rate.state().v_wi.x() > 0.0, "pose observation did not correct velocity");
    checkCovariance(high_rate);
}

void translatingDeskew()
{
    auto filter = initialized();
    std::vector<lio::ImuData> imu{stationarySamples().back()};
    for (int i = 1; i <= 310; ++i)
        imu.push_back({0.5 + i * 0.001, Eigen::Vector3d::Zero(), {i * 0.001, 0, 9.81}});
    const Eigen::Vector3d world_point(4, 2, 1);
    StampedCloud cloud{0.6, 0.8, {}};
    for (const double time : {0.6, 0.65, 0.72, 0.8})
    {
        const double t = time - 0.5;
        const Eigen::Vector3d position(t * t * t / 6.0, 0, 0);
        cloud.points.push_back({(world_point - position).cast<float>(), 1.0f, time - cloud.begin_time});
    }
    lio::ImuProcessor processor{Sophus::SE3d()};
    processor.process(imu, cloud, filter);
    const Eigen::Vector3f expected = (world_point - Eigen::Vector3d(0.3 * 0.3 * 0.3 / 6.0, 0, 0)).cast<float>();
    for (const auto& point : cloud.points)
        check((point.position - expected).norm() < 2e-6, "accelerating deskew");
}

void deskewAndReplay()
{
    auto filter = initialized();
    const double omega = 0.8;
    const Sophus::SE3d T_il(Sophus::SO3d::exp(Eigen::Vector3d(0.1, -0.2, 0.3)), Eigen::Vector3d(0.8, 0.2, 0.1));
    const Eigen::Vector3d world_point(4.0, 2.0, 1.0);
    std::vector<lio::ImuData> imu{stationarySamples().back()};
    for (int i = 1; i <= 50; ++i)
        imu.push_back({0.5 + i * 0.01, {0, 0, omega}, {0, 0, 9.81}});
    const auto lidarPose = [&](double time)
    {
        return Sophus::SE3d(Sophus::SO3d::exp(Eigen::Vector3d(0, 0, omega * (time - 0.505))),
                            Eigen::Vector3d::Zero()) * T_il;
    };
    // Unsorted points, non-IMU-aligned scan boundaries, and a gap before scan start.
    StampedCloud cloud{0.625, 0.835, {}};
    for (const double time : {0.835, 0.7, 0.625, 0.777, 0.8})
        cloud.points.push_back({(lidarPose(time).inverse() * world_point).cast<float>(), 1.0f, time - cloud.begin_time});
    lio::ImuProcessor processor(T_il);
    processor.process(imu, cloud, filter);
    const Eigen::Vector3f expected = (lidarPose(cloud.end_time).inverse() * world_point).cast<float>();
    for (const auto& point : cloud.points)
        check((point.position - expected).norm() < 2e-6, "rotating lever-arm deskew");
    check(std::abs(filter.state().timestamp - cloud.end_time) < 1e-12, "scan-end timestamp");

    auto in_order = initialized();
    for (const auto& sample : imu)
        if (sample.timestamp > in_order.state().timestamp && sample.timestamp < cloud.end_time) in_order.predict(sample);
    in_order.predict(lio::ImuProcessor::imuAt(imu, cloud.end_time));
    const auto measured = Sophus::SE3d(filter.state().R_wi, Eigen::Vector3d(0.01, 0, 0));
    const lio::ESKF::MeasurementCov covariance = lio::ESKF::MeasurementCov::Identity() * 0.001;
    check(filter.observe(measured, covariance) && in_order.observe(measured, covariance), "replay observation rejected");
    auto replayed = filter;
    for (const auto& sample : imu)
    {
        if (sample.timestamp > replayed.state().timestamp) replayed.predict(sample);
        if (sample.timestamp > in_order.state().timestamp) in_order.predict(sample);
    }
    check((replayed.pose().matrix() - in_order.pose().matrix()).norm() < 1e-12, "delayed replay pose mismatch");
    check((replayed.covariance() - in_order.covariance()).norm() < 1e-12, "delayed replay covariance mismatch");
    check((replayed.state().v_wi - in_order.state().v_wi).norm() < 1e-12, "delayed replay velocity mismatch");
    checkCovariance(replayed);
}
} // namespace

int main()
{
    try
    {
        initializationAndMotion();
        covarianceJacobian();
        noiseRateAndObservation();
        translatingDeskew();
        deskewAndReplay();
        std::cout << "PASS: initialization, motion, Jacobian, noise rate, observation, deskew, replay\n";
    }
    catch (const std::exception& error)
    {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
