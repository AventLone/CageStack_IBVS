#include "perception/LIO/ESKF.h"
#include <Eigen/Cholesky>
#include <iostream>
#include <stdexcept>

namespace
{
using namespace lio;
void require(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}
template<class Function> void rejects(Function&& function)
{
    bool threw = false;
    try { function(); } catch (const std::exception&) { threw = true; }
    require(threw, "Invalid input was not rejected");
}

void priorJacobian()
{
    const Eigen::Vector3d rotation(0.3, -0.4, 0.2);
    const auto J = ESKF::rightJacobianInverse(rotation);
    for (int i = 0; i < 3; ++i)
    {
        const Eigen::Vector3d d = 1e-6 * Eigen::Vector3d::Unit(i);
        const Eigen::Vector3d a = (Sophus::SO3d::exp(rotation) * Sophus::SO3d::exp(d)).log();
        const Eigen::Vector3d b = (Sophus::SO3d::exp(rotation) * Sophus::SO3d::exp(-d)).log();
        require(((a - b) / 2e-6 - J.col(i)).norm() < 1e-8, "Prior tangent Jacobian error");
    }
}

void propagationAndBias()
{
    ESKF filter;
    ImuState initial;
    initial.gyro_bias = Eigen::Vector3d(0.01, -0.02, 0.03);
    initial.accel_bias = Eigen::Vector3d(0.1, -0.15, 0.2);
    filter.reset(initial);
    for (int i = 0; i <= 100; ++i)
        filter.predict({i * 0.01, initial.gyro_bias, Eigen::Vector3d(0.0, 0.0, 9.81) + initial.accel_bias});
    require(filter.state().matrix().isApprox(Eigen::Matrix4d::Identity(), 1e-10), "Stationary biased IMU drifted");
    require(filter.nominalState().v_WI.norm() < 1e-10, "Stationary velocity drifted");
    const auto& P = filter.covariance();
    require(P.isApprox(P.transpose(), 1e-12) && P.llt().info() == Eigen::Success, "Covariance lost symmetry/PSD");
    require(P.block<3, 3>(0, 6).norm() > 0.01 && P.block<3, 3>(3, 9).norm() > 1e-5 &&
            P.block<3, 3>(0, 12).norm() > 0.01, "Missing velocity/bias cross covariance");
    ImuState accelerating;
    ESKF accelerated;
    accelerated.reset(accelerating);
    for (int i = 0; i <= 100; ++i) accelerated.predict({0.01 * i, Eigen::Vector3d::Zero(), Eigen::Vector3d(2.0, 0.0, 9.81)});
    require(std::abs(accelerated.nominalState().p_WI.x() - 1.0) < 1e-10 &&
            std::abs(accelerated.nominalState().v_WI.x() - 2.0) < 1e-10, "Constant-acceleration integration incorrect");
}

void linearPosteriorAndRejection()
{
    ESKF filter;
    ImuState state;
    StateCovariance prior = StateCovariance::Identity() * 0.1;
    prior(0, 6) = prior(6, 0) = 0.02;
    prior(0, 12) = prior(12, 0) = -0.015;
    filter.reset(state, prior);
    const double target = 0.2;
    const auto model = [target](const ImuState& s)
    {
        ESKF::Measurement m;
        m.count = 30;
        m.information(0, 0) = 3000.0;
        m.gradient[0] = 3000.0 * (s.p_WI.x() - target);
        m.squared_error = 30.0 * std::pow(s.p_WI.x() - target, 2);
        return m;
    };
    const auto result = filter.update(model);
    require(result.accepted && result.iterations > 1, "Linear measurement was not accepted");
    const ErrorStateT expected = prior.col(0) * target / (prior(0, 0) + 1.0 / 3000.0);
    const StateCovariance posterior = prior - prior.col(0) * prior.row(0) / (prior(0, 0) + 1.0 / 3000.0);
    require((ESKF::boxMinus(filter.nominalState(), state) - expected).norm() < 1e-9, "Full-state Kalman update incorrect");
    require((filter.covariance() - posterior).norm() < 1e-10, "Prior/measurement counted more than once");
    require(std::abs(filter.nominalState().accel_bias.x()) > 1e-3, "LiDAR failed to update accel bias");

    ESKF::Config one_step_config;
    one_step_config.max_iterations = 1;
    one_step_config.max_fitness_score = 1.0;
    one_step_config.convergence_translation = 1e-12;
    one_step_config.convergence_rotation = 1e-12;
    ESKF one_step(one_step_config);
    one_step.reset(state, prior);
    const auto one_step_result = one_step.update(model);
    require(one_step_result.accepted && one_step_result.converged && one_step_result.iterations == 1,
            "Finite last GICP-style iteration was discarded at the iteration limit");

    const auto saved_state = filter.nominalState();
    const auto saved_covariance = filter.covariance();
    require(!filter.update([](const auto&) { return ESKF::Measurement{}; }).accepted, "Empty scan accepted");
    require(ESKF::boxMinus(filter.nominalState(), saved_state).norm() == 0.0 &&
            filter.covariance() == saved_covariance, "Rejected update changed prior");
    require(!filter.update([](const auto&) { ESKF::Measurement m; m.count = 30; m.squared_error = 30.0; return m; }).accepted,
            "Bad final GICP fitness accepted");
    require(filter.covariance() == saved_covariance, "Rejected fitness shrank covariance");
}

void deskewAndScanGap()
{
    const Sophus::SE3d T_IL(Sophus::SO3d::exp(Eigen::Vector3d(0.2, -0.1, 0.3)), Eigen::Vector3d(0.4, -0.2, 0.1));
    ESKF filter(ESKF::Config{}, T_IL);
    ImuState initial;
    initial.gravity.setZero();
    initial.v_WI = Eigen::Vector3d(1.0, -0.2, 0.0);
    filter.reset(initial);
    const auto pose = [&](double t)
    {
        return Sophus::SE3d(Sophus::SO3d::exp(Eigen::Vector3d(0.0, 0.0, 0.5 * t)), initial.v_WI * t) * T_IL;
    };
    std::vector<ImuData> imu;
    for (int i = 0; i <= 40; ++i) imu.push_back({i * 0.01, Eigen::Vector3d(0.0, 0.0, 0.5), Eigen::Vector3d::Zero()});
    const auto scan = [&](double begin, double end)
    {
        std::vector<PointXYZT> points;
        const Eigen::Vector3d fixed_world_point(3.0, 2.0, 0.5);
        for (int i = 0; i <= 10; ++i)
        {
            const double relative = (end - begin) * i / 10.0;
            const Eigen::Vector3d p = pose(begin + relative).inverse() * fixed_world_point;
            points.push_back({static_cast<float>(p.x()), static_cast<float>(p.y()), static_cast<float>(p.z()), relative});
        }
        filter.processScan(imu, points, begin, end);
        const Eigen::Vector3d expected = pose(end).inverse() * fixed_world_point;
        for (const auto& point : points)
            require((Eigen::Vector3d(point.x, point.y, point.z) - expected).norm() < 2e-6, "Deskew/extrinsic/scan-gap error");
        require(filter.lidarPose().matrix().isApprox(pose(end).matrix(), 1e-10), "Wrong scan-end pose");
    };
    scan(0.023, 0.117);
    scan(0.203, 0.337);  // Dropped scan / gap: propagate from previous posterior.
    const auto saved = filter.nominalState();
    std::vector<PointXYZT> invalid{{1.0f, 2.0f, 3.0f, 10.0}};
    rejects([&] { filter.processScan(imu, invalid, 0.34, 0.39); });
    require(ESKF::boxMinus(saved, filter.nominalState()).norm() == 0.0 && saved.timestamp == filter.nominalState().timestamp,
            "Invalid scan mutated state");
}

void initializationAndInputValidation()
{
    ESKF filter;
    std::vector<ImuData> imu;
    const auto R = Sophus::SO3d::exp(Eigen::Vector3d(0.3, -0.2, 0.0));
    const Eigen::Vector3d bias(0.01, -0.02, 0.015);
    for (int i = 0; i <= 100; ++i) imu.push_back({0.01 * i, bias, R.inverse() * Eigen::Vector3d(0.0, 0.0, 9.81)});
    require(filter.initialize(imu), "Stationary initialization failed");
    require((filter.nominalState().R_WI * imu.back().accel + filter.nominalState().gravity).norm() < 1e-10,
            "Gravity alignment sign incorrect");
    require((filter.nominalState().gyro_bias - bias).norm() < 1e-10, "Gyro bias initialization incorrect");
    for (auto& sample : imu) sample.gyro.x() += 0.5;
    require(!filter.initialize(imu), "Moving initialization accepted");
    rejects([&] { filter.predict({1.0, bias, Eigen::Vector3d::Zero()}); });
    rejects([&] { filter.predict({2.0, bias, Eigen::Vector3d::Zero()}); });
    imu[3].timestamp = imu[2].timestamp;
    rejects([&] { ImuProcessor::buildImuSequence(imu, 0.1, 0.2); });
    rejects([&] { filter.reset(ImuState{}, -StateCovariance::Identity()); });
    ESKF::Config invalid;
    invalid.accel_noise = -1;
    rejects([&] { ESKF bad(invalid); });
}

void propagationJacobian()
{
    ESKF::Config config;
    config.gyro_noise = config.accel_noise = config.gyro_bias_noise = config.accel_bias_noise = 1e-12;
    ESKF filter(config);
    ImuState initial;
    initial.R_WI = Sophus::SO3d::exp(Eigen::Vector3d(0.2, -0.3, 0.4));
    initial.gyro_bias = Eigen::Vector3d(0.01, 0.02, -0.01);
    initial.accel_bias = Eigen::Vector3d(0.1, 0.0, -0.2);
    const ImuData a{0.0, Eigen::Vector3d(0.2, -0.1, 0.3), Eigen::Vector3d(0.5, -0.2, 9.7)};
    const ImuData b{0.001, a.gyro, a.accel};
    const StateCovariance P = StateCovariance::Identity() * 0.01;
    filter.reset(initial, P);
    filter.predict(a); filter.predict(b);
    StateCovariance numerical;
    for (int i = 0; i < 15; ++i)
    {
        ErrorStateT delta = ErrorStateT::Zero(); delta[i] = 1e-6;
        auto plus = ESKF::boxPlus(initial, delta), minus = ESKF::boxPlus(initial, -delta);
        ImuProcessor::integrate(a, b, plus); ImuProcessor::integrate(a, b, minus);
        numerical.col(i) = (ESKF::boxMinus(plus, filter.nominalState()) - ESKF::boxMinus(minus, filter.nominalState())) / 2e-6;
    }
    require((filter.covariance() - numerical * P * numerical.transpose()).norm() < 1e-7,
            "Propagation covariance disagrees with numerical right-error dynamics");
}
}
int main()
{
    try
    {
        priorJacobian(); propagationAndBias(); linearPosteriorAndRejection();
        deskewAndScanGap(); initializationAndInputValidation(); propagationJacobian();
        std::cout << "PASS: prior Jacobian, propagation/cross covariance, analytic posterior/bias correction, rejection, deskew/gaps, initialization and invalid input\n";
        return 0;
    }
    catch (const std::exception& error) { std::cerr << "FAIL: " << error.what() << '\n'; return 1; }
}
