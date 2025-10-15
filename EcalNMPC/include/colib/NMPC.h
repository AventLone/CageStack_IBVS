#pragma once
#include <casadi/casadi.hpp>
#include <cassert>
// #include <memory>

class NMPC final
{
    /* Hyperparameters */
    // static constexpr int N{200}; // MPC Horizon, known as prediction horizon
    // static constexpr double T{0.05}; // Sampling Interval [s]
    // static constexpr double MAX_ACC{M_PI}, MAX_STEER_VELOCITY{M_PI / 3.0}; // Hard Constraints [m/s, rad/s]
    // static constexpr double WHEEL_BASE{1.5};
    // static constexpr double WHEEL_RADIUS{0.3};
    // static constexpr double WHEEL_BASE_INV{1.0 / WHEEL_BASE};
    // static constexpr float mSafeDistance{0.55f};

public:
    using Solution = std::vector<std::vector<double>>;
    using Ptr = std::unique_ptr<NMPC>;

    struct Params
    {
        uint16_t horizon;
        double dt;
        double max_acc, max_speed;
        double max_steer_speed, max_steer_angle;
        double wheel_base, wheel_radius;
    };

    explicit NMPC(const uint16_t state_len, const uint16_t input_len, const Params& params);

    ~NMPC() = default;

    void setGoalAndState(const std::vector<double>& goal, const std::vector<double>& state)
    {
        assert(goal.size() == 3);
        mNLP.set_value(mGoal, goal);
        mNLP.set_value(mX0, state);
    }

    bool solve(std::pair<Solution, Solution>& result);

private:
    const uint16_t mStateLen, mInputLen;
    const Params mParams;

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

    casadi::MX sysFunc(const casadi::MX& vec_x, const casadi::MX& vec_u) const
    {
        const auto th = vec_x(2);
        const auto del = vec_x(3);

        const auto v = vec_u(0);
        const auto w = vec_u(1);

        return casadi::MX::vertcat({
                v * mParams.wheel_radius * casadi::MX::cos(th),
                v * mParams.wheel_radius * casadi::MX::sin(th),
                -v * mParams.wheel_radius / mParams.wheel_base * casadi::MX::tan(del),
                w
            });
    }

    casadi::MX sysFunc2(const casadi::MX& vec_x, const casadi::MX& vec_u) const
    {
        const auto th = vec_x(2);
        const auto del = vec_x(3);

        const auto v = vec_u(0);
        const auto w = vec_u(1);

        return vec_x + mParams.dt * casadi::MX::vertcat({
                       v * mParams.wheel_radius * casadi::MX::cos(th),
                       v * mParams.wheel_radius * casadi::MX::sin(th),
                       -v * mParams.wheel_radius / mParams.wheel_base * casadi::MX::tan(del),
                       w
                   });
    }

    casadi::MX rk4(const casadi::MX& vec_x, const casadi::MX& vec_u) const
    {
        const casadi::MX k1 = sysFunc(vec_x, vec_u);
        const casadi::MX k2 = sysFunc(vec_x + 0.5 * mParams.dt * k1, vec_u);
        const casadi::MX k3 = sysFunc(vec_x + 0.5 * mParams.dt * k2, vec_u);
        const casadi::MX k4 = sysFunc(vec_x + mParams.dt * k3, vec_u);
        return vec_x + mParams.dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0;
    }

    static void convertResult(const casadi::DM& input, std::vector<std::vector<double>>& output);
};
