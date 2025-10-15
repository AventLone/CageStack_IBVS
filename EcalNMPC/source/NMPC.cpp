#include "colib/NMPC.h"

NMPC::NMPC(const uint16_t state_len, const uint16_t input_len, const Params& params) : mStateLen(state_len),
                                                                                       mInputLen(input_len),
                                                                                       mParams(params)
{
    const casadi::Dict opts = {
                {"ipopt.sb", "yes"},
                {"ipopt.print_level", 0},
                {"print_time", 0},
                {"ipopt.max_iter", 1000},
                {"ipopt.acceptable_tol", 1e-3},
                {"ipopt.acceptable_obj_change_tol", 1e-3}
            };
    mNLP.solver("ipopt", opts); // Choose IPOPT as solver

    mGoal = mNLP.parameter(3);
    mX0 = mNLP.parameter(mStateLen);

    mUs = mNLP.variable(mInputLen, mParams.horizon);
    mXs = mNLP.variable(mStateLen, mParams.horizon + 1);

    // mF = casadi::MX::eye(4) * std::vector<double>{3.2, 300.7, 30.6, 3.2};
    // mQ = casadi::MX::eye(4) * std::vector<double>{3.2, 300.7, 300.6, 3.2};
    // mR = casadi::MX::eye(2) * std::vector<double>{0.01, 0.005};

    // mF = casadi::MX::eye(3) * std::vector<double>{2.0 * 199.7, 2.0 * 199.7, 2.0 * 199.7};
    // mQ = casadi::MX::eye(3) * std::vector<double>{3.2, 199.7, 39.7};
    // mR = casadi::MX::eye(2) * std::vector<double>{0.1, 0.01};

    mF = casadi::MX::eye(4) * std::vector<double>{2.0, 2.0 * 199.7, 20.0 * 199.7, 2.0 * 199.7};
    mQ = casadi::MX::eye(4) * std::vector<double>{1.0, 879.7, 6.7, 0.1};
    mR = casadi::MX::eye(2) * std::vector<double>{1.0, 0.1};

    buildModel();
}

bool NMPC::solve(std::pair<Solution, Solution>& result)
{
    try
    {
        mNLP.solve();
        const casadi::DM result_u = mNLP.value(mUs);
        const casadi::DM result_x = mNLP.value(mXs);

        convertResult(result_u, result.first);
        convertResult(result_x, result.second);
        return true;
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
        return false;
    }
}

// void NMPC::buildModel()
// {
//     /* Formula of the control system */
//     static auto f = [this](const casadi::MX& vec_x, const casadi::MX& vec_u) -> casadi::MX
//         {
//             const auto x = vec_x(0);
//             const auto y = vec_x(1);
//             const auto th = vec_x(2);
//             const auto v = vec_x(3);
//             const auto del = vec_x(4);
//
//             const auto a = vec_u(0);
//             const auto w = vec_u(1);
//
//             return casadi::MX::vertcat({
//                     v * mParams.wheel_radius * casadi::MX::cos(th),
//                     v * mParams.wheel_radius * casadi::MX::sin(th),
//                     -v * mParams.wheel_radius / mParams.wheel_base * casadi::MX::tan(del),
//                     a, w
//                 });
//         };
//
//     static auto rk4 = [](const casadi::MX& x_k, const casadi::MX& u_k, const double dt) -> casadi::MX
//         {
//             const casadi::MX k1 = f(x_k, u_k);
//             const casadi::MX k2 = f(x_k + 0.5 * dt * k1, u_k);
//             const casadi::MX k3 = f(x_k + 0.5 * dt * k2, u_k);
//             const casadi::MX k4 = f(x_k + dt * k3, u_k);
//             return x_k + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0;
//         };
//
//     /* Control variables */
//     const auto a = mUs(0, casadi::Slice()); // Control variable: a [m/s]
//     const auto w = mUs(1, casadi::Slice()); // Control variable: delta [rad]
//
//     /* State variables */
//     // auto p_x = mXs(0, casadi::Slice()); // State variable: position.x [m]
//     // auto p_y = mXs(1, casadi::Slice()); // State variable: position.y [m]
//     // auto theta = mXs(2, casadi::Slice()); // State variable: theta [rad]
//
//     /* Constraints */
//     mNLP.subject_to(mXs(casadi::Slice(), 0) == mX0); // Initial condition
//     mNLP.subject_to(-mParams.max_acc <= a <= mParams.max_acc);
//     mNLP.subject_to(-mParams.max_steer_speed <= w <= mParams.max_steer_speed);
//
//     /* Objective function 目标函数, note: casadi::MX::mtimes represents matrix multiplication */
//     casadi::MX J = casadi::MX::zeros(1, 1);
//
//     // 累计阶段代价 + 动力学约束
//     for (int k = 0; k < mParams.horizon; ++k)
//     {
//         auto x_k = mXs(casadi::Slice(), k);
//         auto u_k = mUs(casadi::Slice(), k);
//
//         // 车体系误差（相对同一终点）
//         auto dx = mGoal(0) - x_k(0);
//         auto dy = mGoal(1) - x_k(1);
//         auto th = x_k(2);
//         auto dv = 0.5 * dx - x_k(3);
//         auto e_x = casadi::MX::cos(th) * dx + casadi::MX::sin(th) * dy;
//         auto e_y = -casadi::MX::sin(th) * dx + casadi::MX::cos(th) * dy;
//         auto e_th = casadi::MX::atan2(casadi::MX::sin(mGoal(2) - th), casadi::MX::cos(mGoal(2) - th));
//
//         casadi::MX e_k = casadi::MX::vertcat({e_x, e_y, e_th, dv}); // Error vector
//
//         // 阶段代价
//         J += casadi::MX::mtimes(casadi::MX::mtimes(e_k.T(), mQ), e_k) + casadi::MX::mtimes(
//             casadi::MX::mtimes(u_k.T(), mR), u_k);
//
//         // 动力学（RK4）
//         casadi::MX x_next = rk4(x_k, u_k, mParams.dt);
//         mNLP.subject_to(mXs(casadi::Slice(), k + 1) == x_next);
//     }
//
//     // 终端代价（到点 + 停车倾向）
//     {
//         auto x_n = mXs(casadi::Slice(), mParams.horizon);
//         casadi::MX dx = mGoal(0) - x_n(0);
//         casadi::MX dy = mGoal(1) - x_n(1);
//         auto th = x_n(2);
//         auto dv = 0.5 * dx - x_n(3);
//         casadi::MX e_x = casadi::MX::cos(th) * dx + casadi::MX::sin(th) * dy;
//         casadi::MX e_y = -casadi::MX::sin(th) * dx + casadi::MX::cos(th) * dy;
//         casadi::MX e_th = casadi::MX::atan2(casadi::MX::sin(mGoal(2) - th), casadi::MX::cos(mGoal(2) - th));
//
//         casadi::MX e_n = casadi::MX::vertcat({e_x, e_y, e_th, dv}); // Error vector
//
//         J += casadi::MX::mtimes(casadi::MX::mtimes(e_n.T(), mF), e_n);
//     }
//
//     mNLP.set_initial(mXs, casadi::DM::zeros(5, mParams.horizon + 1));
//     mNLP.set_initial(mUs, casadi::DM::zeros(2, mParams.horizon));
//     mNLP.minimize(J);
// }

void NMPC::buildModel()
{
    /* Control variables */
    const auto v = mUs(0, casadi::Slice()); // Control variable: a [m/s]
    const auto w = mUs(1, casadi::Slice()); // Control variable: delta [rad]

    const auto p_x = mXs(0, casadi::Slice()); // State variable: position.x [m]
    const auto steer_angle = mXs(3, casadi::Slice());

    /* Constraints */
    mNLP.subject_to(mXs(casadi::Slice(), 0) == mX0); // Initial condition
    mNLP.subject_to(-mParams.max_speed <= v <= mParams.max_speed);
    mNLP.subject_to(-mParams.max_steer_speed <= w <= mParams.max_steer_speed);
    mNLP.subject_to(-mParams.max_steer_angle <= steer_angle <= mParams.max_steer_angle);
    mNLP.subject_to(p_x <= mGoal(0));

    /* Objective function */
    casadi::MX J = casadi::MX::zeros(1, 1);

    // 累计阶段代价 + 动力学约束
    for (int k = 0; k < mParams.horizon; ++k)
    {
        auto x_k = mXs(casadi::Slice(), k);
        auto u_k = mUs(casadi::Slice(), k);

        // 车体系误差（相对同一终点）
        auto dx = mGoal(0) - x_k(0);
        auto dy = mGoal(1) - x_k(1);
        auto th = x_k(2);
        auto dv = 1.5 * dx - u_k(0);
        auto e_x = casadi::MX::cos(th) * dx + casadi::MX::sin(th) * dy;
        auto e_y = -casadi::MX::sin(th) * dx + casadi::MX::cos(th) * dy;
        // auto e_th = casadi::MX::atan2(casadi::MX::sin(mGoal(2) - th), casadi::MX::cos(mGoal(2) - th));
        auto e_th = mGoal(2) - th;
        // casadi::MX e_k = casadi::MX::vertcat({e_x, e_y, e_th, dv}); // Error vector
        casadi::MX e_k = casadi::MX::vertcat({e_x, e_y, e_th, x_k(3)}); // Error vector
        // casadi::MX e_k = casadi::MX::vertcat({e_x, e_y, e_th}); // Error vector

        // 阶段代价
        J += casadi::MX::mtimes(casadi::MX::mtimes(e_k.T(), mQ), e_k) + casadi::MX::mtimes(
            casadi::MX::mtimes(u_k.T(), mR), u_k);
        // J += casadi::MX::mtimes(casadi::MX::mtimes(e_k.T(), mQ), e_k);

        // 运动学
        casadi::MX x_next = sysFunc2(x_k, u_k);
        // casadi::MX x_next = rk4(x_k, u_k);
        mNLP.subject_to(mXs(casadi::Slice(), k + 1) == x_next);
    }

    // 终端代价（到点 + 停车倾向）
    {
        auto x_n = mXs(casadi::Slice(), mParams.horizon);
        auto u_n = mUs(casadi::Slice(), mParams.horizon - 1);
        casadi::MX dx = mGoal(0) - x_n(0);
        casadi::MX dy = mGoal(1) - x_n(1);
        auto th = x_n(2);
        auto dv = u_n(0);
        // auto dv = casadi::MX::pow(0.5 * dx, 2) - casadi::MX::pow(u_n(0), 2);
        casadi::MX e_x = casadi::MX::cos(th) * dx + casadi::MX::sin(th) * dy;
        casadi::MX e_y = -casadi::MX::sin(th) * dx + casadi::MX::cos(th) * dy;
        // casadi::MX e_th = casadi::MX::atan2(casadi::MX::sin(mGoal(2) - th), casadi::MX::cos(mGoal(2) - th));
        auto e_th = mGoal(2) - th;
        // casadi::MX e_n = casadi::MX::vertcat({e_x, e_y, e_th, dv}); // Error vector
        casadi::MX e_n = casadi::MX::vertcat({e_x, e_y, e_th, x_n(3)}); // Error vector
        // casadi::MX e_n = casadi::MX::vertcat({e_x, e_y, e_th}); // Error vector

        J += casadi::MX::mtimes(casadi::MX::mtimes(e_n.T(), mF), e_n);
    }

    mNLP.set_initial(mXs, casadi::DM::zeros(mStateLen, mParams.horizon + 1));
    mNLP.set_initial(mUs, casadi::DM::zeros(mInputLen, mParams.horizon));
    mNLP.minimize(J);
}

void NMPC::convertResult(const casadi::DM& input, std::vector<std::vector<double>>& output)
{
    std::vector<double> input_vec = input.nonzeros();

    output.reserve(input.columns());

    for (size_t i = 0; i < input_vec.size(); i += input.rows())
    {
        output.emplace_back(input_vec.begin() + i, input_vec.begin() + i + input.rows());
    }
}
