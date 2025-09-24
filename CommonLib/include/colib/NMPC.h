#pragma once
#include <casadi/casadi.hpp>
#include <cassert>

class NMPC final
{
    /* Hyperparameters */
    static constexpr int N{20}; // MPC Horizon, known as prediction horizon
    static constexpr double T{0.05}; // Sampling Interval [s]
    static constexpr double mMaxVelosity{0.5}, mMaxAngularVelocity{0.6}; // Hard Constraints [m/s, rad/s]
    static constexpr float mSafeDistance{0.55f};

    static Eigen::MatrixXd toEigen(casadi::DM& input)
    {
        return Eigen::Map<Eigen::MatrixXd>(input.ptr(), input.rows(), input.columns());
    }

    using Solution = std::vector<std::vector<double>>;

public:
    NMPC();

    ~NMPC() = default;

    void setGoalAndState(const std::vector<double>& goal, const std::vector<double>& state)
    {
        assert(goal.size() == 3 && state.size() == 2);
        mNLP.set_value(mGoal, goal);
        mNLP.set_value(mX0, {0.0, 0.0, 0.0, state[0], state[1]});
    }

    std::pair<Solution, Solution> solve();

private:
    casadi::Opti mNLP; // Construct NLP using CasADi

    casadi::MX mGoal, mX0;

    casadi::MX F, Q, R; // Weighting matrices

    casadi::MX mUs; // Control Policy: a sequence of control vectors
    casadi::MX mXs; // A sequence of state vectors

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
};
