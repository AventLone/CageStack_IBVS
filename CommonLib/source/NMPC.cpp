#include "NMPC.h"

namespace Eigen
{
casadi::DM matrix2DM(const Eigen::MatrixXd& input)
{
    const long rows = input.rows();
    const long cols = input.cols();
    casadi::DM result(rows, cols);
    for (long row = 0; row < rows; ++row)
    {
        for (long col = 0; col < cols; ++col)
        {
            result(row, col) = input(row, col);
        }
    }
    return result;
}

inline casadi::DM vector2DM(const Eigen::VectorXd& input)
{
    return casadi::DM(std::vector<double>(input.data(), input.data() + input.size()));
}
} // namespace Eigen

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
    mX0 = mNLP.parameter(5);

    mUs = mNLP.variable(N, 2);

    F = casadi::MX::eye(5) * std::vector<double>{1.2, 1.6, 1.6, 2.0, 1.1};
    Q = casadi::MX::eye(4) * std::vector<double>{1.2, 1.6, 1.6, 2.0};
    R = casadi::MX::eye(2) * std::vector<double>({0.2, 0.15});
}

// void NMPC::buildProblem(const Eigen::VectorXd& current_state, const Eigen::MatrixXd& goal_state,
//                         const std::vector<Eigen::Vector2d>& obstacle)
// {
//     casadi::DM goal = Eigen::matrix2DM(goal_state(Eigen::all, Eigen::seq(0, 3)));
//
//     mX0 = mNLP.parameter(3); // Initial condition of state vector
//     mUs = mNLP.variable(N, 2); // Control Policy: a sequence of control vectors
//     mVelocity = mUs(casadi::Slice(), 0); // Control variable: v [m/s]
//     mAngularVelocity = mUs(casadi::Slice(), 1); // Control variable: omega [rad/s]
//
//     /* State variables */
//     mXs = mNLP.variable(N + 1, 3); // A sequence of state vectors
//     mPosition_x = mXs(casadi::Slice(), 0); // State variable: position.x [m]
//     mPosition_y = mXs(casadi::Slice(), 1); // State variable: position.y [m]
//     mAngle = mXs(casadi::Slice(), 2); // State variable: theta [rad]
//
//     mNLP.subject_to(mXs(0, casadi::Slice()) == mX0.T()); // Initial condition
//
//     /* Hard constraints */
//     mNLP.subject_to(-mMaxVelosity <= mVelocity <= mMaxVelosity);
//     mNLP.subject_to(-mMaxAngularVelocity <= mAngularVelocity <= mMaxAngularVelocity);
//
//     /* Create funciont for F(x) */
//     double theta_x = current_state(4) * std::cos(current_state(2)) - current_state(3) * std::sin(current_state(2));
//     double theta_y = current_state(4) * std::cos(current_state(2)) + current_state(3) * std::sin(current_state(2));
//
//     /* Get the derivative of state vector */
//     auto f = [&](const casadi::MX& x, const casadi::MX& u) -> casadi::MX
//     {
//         return casadi::MX::vertcat(
//             {u(0) * casadi::MX::cos(x(2)) * std::cos(theta_x), u(0) * casadi::MX::sin(x(2)) * std::cos(theta_y), u(1)});
//     };
//
//     /* Control system model constraints */
//     for (int i = 0; i < N; i++)
//     {
//         /* x_{k+1} = x_k + T * (v * cos(theta), v * sin(theta), omega) */
//         casadi::MX x_next = mXs(i, casadi::Slice()) + T * f(mXs(i, casadi::Slice()), mUs(i, casadi::Slice())).T();
//         mNLP.subject_to(mXs(i + 1, casadi::Slice()) == x_next);
//     }
//
//     /* Avoidance of local obstacles: by remove the position which the obstacle is at from the state set */
//     for (const auto& point: obstacle)
//     {
//         for (int i = 0; i < N + 1; ++i)
//         {
//             mNLP.subject_to(mXs(i, 0) != point(0));
//             mNLP.subject_to(mXs(i, 1) != point(1));
//         }
//     }
//
//     /* Cost function, note: casadi::MX::mtimes represents matrix multiplication */
//     casadi::MX J = 0;
//     for (int i = 0; i < N; ++i)
//     {
//         J = J +
//             0.1 * casadi::MX::mtimes(casadi::MX::mtimes(mXs(i, casadi::Slice()) - goal(i), Q),
//                                      (mXs(i, casadi::Slice()) - goal(i)).T()) +
//             casadi::MX::mtimes(casadi::MX::mtimes(mUs(i, casadi::Slice()), R), mUs(i, casadi::Slice()).T());
//     }
//     J = J + 2 * casadi::MX::mtimes(casadi::MX::mtimes(mXs(N - 1, casadi::Slice()) - goal(N - 1), Q),
//                                    (mXs(N - 1, casadi::Slice()) - goal(N - 1)).T());
//
//     mNLP.minimize(J);
//     mNLP.set_value(mX0, Eigen::vector2DM(current_state(Eigen::seq(0, 2))));
// }

void NMPC::solve()
{
    try
    {
        mNLP.solve();
        casadi::DM uu = mNLP.value(mUs);
        casadi::DM xx = mNLP.value(mXs);
        // mSolution_u = toEigen(uu);
        // mSolution_x = toEigen(xx);
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << '\n';
        // mSolution_u = Eigen::MatrixXd::Zero(N, 2);
        // mSolution_x = Eigen::MatrixXd::Zero(N + 1, 5);
    }
}

void NMPC::buildModel()
{
    casadi::MX velocity = mUs(casadi::Slice(), 0); // Control variable: v [m/s]
    casadi::MX angular_velocity = mUs(casadi::Slice(), 1); // Control variable: omega [rad/s]

    /* State variables */
    // mXs = mNLP.variable(N + 1, 3); // A sequence of state vectors
    // mPosition_x = mXs(casadi::Slice(), 0); // State variable: position.x [m]
    // mPosition_y = mXs(casadi::Slice(), 1); // State variable: position.y [m]
    // mAngle = mXs(casadi::Slice(), 2); // State variable: theta [rad]

    mNLP.subject_to(mXs(0, casadi::Slice()) == mX0.T()); // Initial condition

    /* Hard constraints */
    mNLP.subject_to(-mMaxVelosity <= velocity <= mMaxVelosity);
    mNLP.subject_to(-mMaxAngularVelocity <= angular_velocity <= mMaxAngularVelocity);

    /* Create funciont for F(x) */
    // double theta_x = current_state(4) * std::cos(current_state(2)) - current_state(3) * std::sin(current_state(2));
    // double theta_y = current_state(4) * std::cos(current_state(2)) + current_state(3) * std::sin(current_state(2));

    /* Get the derivative of state vector */
    static auto f = [&](const casadi::MX& x, const casadi::MX& u) -> casadi::MX
    {
        return casadi::MX::vertcat(
            {u(0) * casadi::MX::cos(x(2)) * std::cos(theta_x), u(0) * casadi::MX::sin(x(2)) * std::cos(theta_y), u(1)});
    };

    /* Control system model constraints */
    for (int i = 0; i < N; i++)
    {
        /* x_{k+1} = x_k + T * (v * cos(theta), v * sin(theta), omega) */
        casadi::MX x_next = mXs(i, casadi::Slice()) + T * f(mXs(i, casadi::Slice()), mUs(i, casadi::Slice())).T();
        mNLP.subject_to(mXs(i + 1, casadi::Slice()) == x_next);
    }

    /* Avoidance of local obstacles: by remove the position which the obstacle is at from the state set */
    for (const auto& point : obstacle)
    {
        for (int i = 0; i < N + 1; ++i)
        {
            mNLP.subject_to(mXs(i, 0) != point(0));
            mNLP.subject_to(mXs(i, 1) != point(1));
        }
    }

    /* Cost function, note: casadi::MX::mtimes represents matrix multiplication */
    casadi::MX J = 0;
    for (int i = 0; i < N; ++i)
    {
        J = J +
            0.1 * casadi::MX::mtimes(casadi::MX::mtimes(mXs(i, casadi::Slice()) - goal(i), Q),
                                     (mXs(i, casadi::Slice()) - goal(i)).T()) +
            casadi::MX::mtimes(casadi::MX::mtimes(mUs(i, casadi::Slice()), R), mUs(i, casadi::Slice()).T());
    }
    J = J + 2 * casadi::MX::mtimes(casadi::MX::mtimes(mXs(N - 1, casadi::Slice()) - goal(N - 1), Q),
                                   (mXs(N - 1, casadi::Slice()) - goal(N - 1)).T());

    mNLP.minimize(J);
    mNLP.set_value(mX0, Eigen::vector2DM(current_state(Eigen::seq(0, 2))));
}
