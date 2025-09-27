#pragma once
#include <casadi/casadi.hpp>
#include <cassert>
#include <Eigen/Core>

class NMPC final
{
    /* Hyperparameters */
    static constexpr int N{100}; // MPC Horizon, known as prediction horizon
    static constexpr double T{0.05}; // Sampling Interval [s]
    static constexpr double MAX_VELOSITY{3.0}, MAX_DELTA{0.6}; // Hard Constraints [m/s, rad/s]
    static constexpr double WHEEL_BASE{1.5};
    static constexpr double WHEEL_BASE_INV{1.0 / WHEEL_BASE};
    static constexpr float mSafeDistance{0.55f};

    using Solution = std::vector<std::vector<double>>;

public:
    NMPC();

    ~NMPC() = default;

    void setGoal(const std::vector<double>& goal)
    {
        assert(goal.size() == 3);
        mNLP.set_value(mGoal, goal);
        mNLP.set_value(mX0, {0.0, 0.0, 0.0});
    }

    std::pair<Eigen::MatrixXd, Eigen::MatrixXd> solve();

    std::pair<Solution, Solution> solve2();

private:
    casadi::Opti mNLP; // Construct NLP using CasADi
    casadi::MX mGoal, mX0;
    casadi::MX mF, mQ, mR; // Weighting matrices

    casadi::MX mUs; // Control Policy: a sequence of control vectors 控制序列
    casadi::MX mXs; // A sequence of state vectors 状态轨迹

    void buildModel();

    static Eigen::MatrixXd toEigen(casadi::DM& input)
    {
        return Eigen::Map<Eigen::MatrixXd>(input.ptr(), input.rows(), input.columns());
    }

    void convertResult(const casadi::DM& input, std::vector<std::vector<double>>& output);
};
