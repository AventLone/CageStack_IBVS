#include "ackermann_control//NMPC.h"

NMPC::NMPC()
{
    const casadi::Dict opts = {
        {"ipopt.sb", "yes"},
        {"ipopt.print_level", 0},
        {"print_time", 0},
        {"ipopt.max_iter", 80},
        {"ipopt.acceptable_tol", 1e-3},
        {"ipopt.acceptable_obj_change_tol", 1e-3}
    };
    mNLP.solver("ipopt", opts); // Choose IPOPT as solver

    mGoal = mNLP.parameter(3);
    mX0 = mNLP.parameter(3);

    mUs = mNLP.variable(2, N);
    mXs = mNLP.variable(3, N + 1);

    mF = casadi::MX::eye(3) * std::vector<double>{2.2, 2.6, 2.6};
    mQ = casadi::MX::eye(3) * std::vector<double>{1.2, 1.6, 1.6};
    mR = casadi::MX::eye(2) * std::vector<double>{0.2, 0.15};

    buildModel();
}

std::pair<Eigen::MatrixXd, Eigen::MatrixXd> NMPC::solve()
{
    try
    {
        mNLP.solve();
        casadi::DM result_u = mNLP.value(mUs);
        casadi::DM result_x = mNLP.value(mXs);
        return std::pair<Eigen::MatrixXd, Eigen::MatrixXd>{toEigen(result_u), toEigen(result_x)};
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
        return std::pair<Eigen::MatrixXd, Eigen::MatrixXd>{};
    }
}

void NMPC::buildModel()
{
    /* Formula of the control system */
    static auto f = [](const casadi::MX& vec_x, const casadi::MX& vec_u) -> casadi::MX
    {
        const auto x = vec_x(0);
        const auto y = vec_x(1);
        const auto th = vec_x(2);

        const auto v = vec_u(0);
        const auto del = vec_u(1);

        return casadi::MX::vertcat({
            v * casadi::MX::cos(th),
            v * casadi::MX::sin(th),
            v * WHEEL_BASE_INV * casadi::MX::tan(del)
        });
    };

    static auto rk4 = [](const casadi::MX& x_k, const casadi::MX& u_k, const double dt) -> casadi::MX
    {
        const casadi::MX k1 = f(x_k, u_k);
        const casadi::MX k2 = f(x_k + 0.5 * dt * k1, u_k);
        const casadi::MX k3 = f(x_k + 0.5 * dt * k2, u_k);
        const casadi::MX k4 = f(x_k + dt * k3, u_k);
        return x_k + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0;
    };

    /* Control variables */
    const auto velocity = mUs(0, casadi::Slice()); // Control variable: v [m/s]
    const auto delta = mUs(1, casadi::Slice()); // Control variable: delta [rad]

    /* State variables */
    auto p_x = mXs(0, casadi::Slice()); // State variable: position.x [m]
    auto p_y = mXs(1, casadi::Slice()); // State variable: position.y [m]
    auto theta = mXs(2, casadi::Slice()); // State variable: theta [rad]

    /* Constraints */
    mNLP.subject_to(mXs(casadi::Slice(), 0) == mX0); // Initial condition
    mNLP.subject_to(-MAX_VELOSITY <= velocity <= MAX_VELOSITY);
    mNLP.subject_to(-MAX_DELTA <= delta <= MAX_DELTA);

    /* Objective function 目标函数, note: casadi::MX::mtimes represents matrix multiplication */
    casadi::MX J = casadi::MX::zeros(1, 1);

    // 累计阶段代价 + 动力学约束
    for (int k = 0; k < N; ++k)
    {
        auto x_k = mXs(casadi::Slice(), k);
        auto u_k = mUs(casadi::Slice(), k);

        // 车体系误差（相对同一终点）
        auto dx = mGoal(0) - x_k(0);
        auto dy = mGoal(1) - x_k(1);
        auto th = x_k(2);
        auto e_x = casadi::MX::cos(th) * dx + casadi::MX::sin(th) * dy;
        auto e_y = -casadi::MX::sin(th) * dx + casadi::MX::cos(th) * dy;
        auto e_th = casadi::MX::atan2(casadi::MX::sin(mGoal(2) - th), casadi::MX::cos(mGoal(2) - th));

        casadi::MX e_k = casadi::MX::vertcat({e_x, e_y, e_th}); // Error vector

        // 阶段代价
        J += casadi::MX::mtimes(casadi::MX::mtimes(e_k.T(), mQ), e_k) + casadi::MX::mtimes(
            casadi::MX::mtimes(u_k.T(), mR), u_k);

        // 动力学（RK4）
        casadi::MX x_next = rk4(x_k, u_k, T);
        mNLP.subject_to(mXs(casadi::Slice(), k + 1) == x_next);
    }

    // 终端代价（到点 + 停车倾向）
    {
        auto x_n = mXs(casadi::Slice(), N);
        casadi::MX dx = mGoal(0) - x_n(0);
        casadi::MX dy = mGoal(1) - x_n(1);
        auto th = x_n(2);
        casadi::MX e_x = casadi::MX::cos(th) * dx + casadi::MX::sin(th) * dy;
        casadi::MX e_y = -casadi::MX::sin(th) * dx + casadi::MX::cos(th) * dy;
        casadi::MX e_th = casadi::MX::atan2(casadi::MX::sin(mGoal(2) - th), casadi::MX::cos(mGoal(2) - th));

        casadi::MX e_n = casadi::MX::vertcat({e_x, e_y, e_th}); // Error vector

        J += casadi::MX::mtimes(casadi::MX::mtimes(e_n.T(), mF), e_n);
    }

    mNLP.set_initial(mXs, casadi::DM::zeros(3, N + 1));
    mNLP.set_initial(mUs, casadi::DM::zeros(2, N));
    mNLP.minimize(J);
}
