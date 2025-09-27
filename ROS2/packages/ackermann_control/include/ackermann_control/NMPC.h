#pragma once
#include <casadi/casadi.hpp>
#include <cassert>
#include <Eigen/Core>

class NMPC final
{
    /* Hyperparameters */
    static constexpr int N{10}; // MPC Horizon, known as prediction horizon
    static constexpr double T{0.05}; // Sampling Interval [s]
    static constexpr double MAX_VELOSITY{2.0}, MAX_DELTA{M_PI / 2.0}; // Hard Constraints [m/s, rad/s]
    static constexpr double WHEEL_BASE{1.5};
    static constexpr double WHEEL_BASE_INV{1.0 / WHEEL_BASE};
    static constexpr float mSafeDistance{0.55f};

    using Solution = std::vector<std::vector<double>>;

public:
    NMPC();

    ~NMPC() = default;

    void setGoalAndState(const std::vector<double>& goal)
    {
        assert(goal.size() == 3);
        mNLP.set_value(mGoal, goal);
        mNLP.set_value(mX0, {0.0, 0.0, 0.0});
    }

    std::pair<Solution, Solution> solve();

private:
    casadi::Opti mNLP; // Construct NLP using CasADi
    casadi::MX mGoal, mX0;
    casadi::MX mF, mQ, mR; // Weighting matrices

    casadi::MX mUs; // Control Policy: a sequence of control vectors 控制序列
    casadi::MX mXs; // A sequence of state vectors 状态轨迹

    void buildModel();

    template<typename T>
    static T wrapAngle(T rad)
    {
        constexpr double two_pi = 2.0 * M_PI;
        double r = std::fmod(rad + M_PI, two_pi);
        if (r < 0.0)
        {
            r += two_pi;
        }
        return r - M_PI; // ∈ [-π, π)
    }

    // casadi::MX rk4(const casadi::MX& x_k, const casadi::MX& u_k, const double dt)
    // {
    //     const casadi::MX k1 = f(x_k, u_k);
    //     const casadi::MX k2 = f(x_k + 0.5 * dt * k1, u_k);
    //     const casadi::MX k3 = f(x_k + 0.5 * dt * k2, u_k);
    //     const casadi::MX k4 = f(x_k + dt * k3, u_k);
    //     return x_k + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0;
    // }


    static Eigen::MatrixXd toEigen(casadi::DM& input)
    {
        input.nonzeros();
        return Eigen::Map<Eigen::MatrixXd>(input.ptr(), input.rows(), input.columns());
    }

    static void convertResult(const casadi::DM& input, std::vector<std::vector<double>>& output);
};
