#pragma once

#include "./nmpc_solver.hpp"

struct ForkliftE
{
    const nmpc::Params& mParams;

    explicit ForkliftE(const nmpc::Params& params) : mParams(params)
    {}

    casadi::MX operator()(const casadi::MX& vec_x, const casadi::MX& vec_u) const
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
};

struct ForkliftST
{
    const nmpc::Params& mParams;

    explicit ForkliftST(const nmpc::Params& params) : mParams(params)
    {}

    casadi::MX operator()(const casadi::MX& vec_x, const casadi::MX& vec_u) const
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
};

using ControllerST = nmpc::Solver<ForkliftST>;
using ControllerE = nmpc::Solver<ForkliftE>;
