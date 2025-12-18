#pragma once
#include <casadi/casadi.hpp>
#include <utility>
#include <std_msgs/msg/float32_multi_array.hpp>

namespace nmpc
{
struct Params
{
    uint16_t horizon;
    double dt;
    uint16_t input_len, state_len, output_len;

    /* Vehicle size */
    double wheel_base, wheel_radius;

    /* Constraits */
    double max_acc, max_speed;
    double max_steer_speed, max_steer_angle;

    std::vector<double> weight_Q, weight_F, weight_R;
};

inline std::ostream& operator<<(std::ostream& os, const Params& p)
{
    return os << "---------------------- NMPC Params ----------------------\n" <<
           "horizon: " << p.horizon << ", dt: " << p.dt
           << "\ninput_len: " << p.input_len << ", state_len: " << p.state_len << ", output_len: " << p.output_len
           << "\nwheel_base: " << p.wheel_base << ", wheel_radius: " << p.wheel_radius
           << "\nmax_acc: " << p.max_acc << "\nmax_speed: " << p.max_speed
           << "\nmax_steer_speed: " << p.max_steer_speed << ", max_steer_angle: " << p.max_steer_angle
           << "\nWeight_Q: " << p.weight_Q << "\nWeight_F: " << p.weight_F << "\nWeight_R: " << p.weight_R
           << "\n--------------------------------------------------------";
}

using Solution = std::vector<std::vector<double>>;

template<class Kinematics>
class Solver final
{
public:
    using Ptr = std::unique_ptr<Solver>;

    explicit Solver(Params params);

    ~Solver() = default;

    void setGoalAndState(const std::vector<double>& goal, const std::vector<double>& state)
    {
        mNLP.set_value(mGoal, goal);
        mNLP.set_value(mX0, state);
    }

    void setGoal(const std::vector<double>& goal)
    {
        mNLP.set_value(mGoal, goal);
        mNLP.set_value(mX0, {0.0, 0.0, 0.0, 0.0});
    }

    bool solve(std::pair<Solution, Solution>& result);

    void setWeightQ(const std::vector<double>& q)
    {
        mNLP.set_value(mQ, q);
    }

    void setWeightF(const std::vector<double>& f)
    {
        mNLP.set_value(mF, f);
    }

    void setWeightR(const std::vector<double>& r)
    {
        mNLP.set_value(mR, r);
    }

    void resetWeightQ()
    {
        setWeightQ(mParams.weight_Q);
    }

    void resetWeightF()
    {
        setWeightF(mParams.weight_F);
    }

    void resetWeightR()
    {
        setWeightR(mParams.weight_R);
    }

    void resetWeights()
    {
        resetWeightQ();
        resetWeightF();
        resetWeightR();
    }

private:
    const Params mParams;

    casadi::Opti mNLP;
    casadi::MX mGoal, mX0;
    casadi::MX mF, mQ, mR; // Weighting matrices

    casadi::MX mUs; // Control Policy: a sequence of control vectors 控制序列
    casadi::MX mXs; // A sequence of state vectors 状态轨迹

    const Kinematics kinematicFunc;

    void buildModel();

    static void convertResult(const casadi::DM& input, std::vector<std::vector<double>>& output);
};

template<class Kinematics>
Solver<Kinematics>::Solver(Params params) : mParams(std::move(params)), kinematicFunc(mParams)
{
    const casadi::Dict opts = {
                {"ipopt.sb", "yes"},
                {"ipopt.print_level", 0},
                {"print_time", 0},
                {"ipopt.max_iter", 600},
                {"ipopt.acceptable_tol", 1e-3},
                {"ipopt.acceptable_obj_change_tol", 1e-3}
            };
    mNLP.solver("ipopt", opts); // Choose IPOPT as solver

    mGoal = mNLP.parameter(3);
    mX0 = mNLP.parameter(mParams.state_len);

    mQ = mNLP.parameter(4);
    mF = mNLP.parameter(3);
    mR = mNLP.parameter(2);

    mUs = mNLP.variable(mParams.input_len, mParams.horizon);
    mXs = mNLP.variable(mParams.state_len, mParams.horizon + 1);

    setWeightQ(mParams.weight_Q);
    setWeightF(mParams.weight_F);
    setWeightR(mParams.weight_R);

    buildModel();
}

template<class Kinematics>
bool Solver<Kinematics>::solve(std::pair<Solution, Solution>& result)
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

template<class Kinematics>
void Solver<Kinematics>::buildModel()
{
    /* Control variables */
    const auto v = mUs(0, casadi::Slice()); // Control variable: a [m/s]
    const auto w = mUs(1, casadi::Slice()); // Control variable: delta [rad]

    const auto steer_angle = mXs(3, casadi::Slice());
    const auto p_x = mXs(0, casadi::Slice());

    /* Constraints */
    mNLP.subject_to(mXs(casadi::Slice(), 0) == mX0); // Initial condition
    mNLP.subject_to(-mParams.max_speed <= v <= mParams.max_speed);
    mNLP.subject_to(-mParams.max_steer_speed <= w <= mParams.max_steer_speed);
    mNLP.subject_to(-mParams.max_steer_angle <= steer_angle <= mParams.max_steer_angle);

    /* Weights */
    const auto weight_Q = casadi::MX::diag(mQ);
    const auto weight_F = casadi::MX::diag(mF);
    const auto weight_R = casadi::MX::diag(mR);

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
        auto dv = 10.0 * dx - u_k(0);
        auto e_x = casadi::MX::cos(th) * dx + casadi::MX::sin(th) * dy;
        auto e_y = -casadi::MX::sin(th) * dx + casadi::MX::cos(th) * dy;
        // auto e_th = casadi::MX::atan2(casadi::MX::sin(mGoal(2) - th), casadi::MX::cos(mGoal(2) - th));
        auto e_th = mGoal(2) - th;
        auto s_penalty = casadi::MX::fmax(0.0, -dx);
        auto mix = casadi::MX::fmax(0.0, u_k(0));

        // casadi::MX e_k = casadi::MX::vertcat({e_x, e_y, e_th, dv}); // Error vector
        casadi::MX e_k = casadi::MX::vertcat({e_x, e_y, e_th, mix * s_penalty}); // Error vector

        // 阶段代价
        J += casadi::MX::mtimes(casadi::MX::mtimes(e_k.T(), weight_Q), e_k) + casadi::MX::mtimes(
            casadi::MX::mtimes(u_k.T(), weight_R), u_k);

        // 运动学
        casadi::MX x_next = kinematicFunc(x_k, u_k);
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
        casadi::MX e_x = casadi::MX::cos(th) * dx + casadi::MX::sin(th) * dy;
        casadi::MX e_y = -casadi::MX::sin(th) * dx + casadi::MX::cos(th) * dy;
        // casadi::MX e_th = casadi::MX::atan2(casadi::MX::sin(mGoal(2) - th), casadi::MX::cos(mGoal(2) - th));
        auto e_th = mGoal(2) - th;
        casadi::MX e_n = casadi::MX::vertcat({e_x, e_y, e_th}); // Error vector

        J += casadi::MX::mtimes(casadi::MX::mtimes(e_n.T(), weight_F), e_n);
    }

    mNLP.set_initial(mXs, casadi::DM::zeros(mParams.state_len, mParams.horizon + 1));
    mNLP.set_initial(mUs, casadi::DM::zeros(mParams.input_len, mParams.horizon));
    mNLP.minimize(J);
}

template<class Kinematics>
void Solver<Kinematics>::convertResult(const casadi::DM& input, std::vector<std::vector<double>>& output)
{
    std::vector<double> input_vec = input.nonzeros();

    output.reserve(input.columns());

    for (long i = 0; i < input_vec.size(); i += input.rows())
    {
        output.emplace_back(input_vec.begin() + i, input_vec.begin() + i + input.rows());
    }
}
}
