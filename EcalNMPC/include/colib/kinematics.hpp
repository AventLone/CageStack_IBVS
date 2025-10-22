#pragma once
#include "colib/nmpc_solver.hpp"

struct Bicycle
{
    const nmpc::Params& mParams;

    explicit Bicycle(const nmpc::Params& params) : mParams(params)
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

using ControllerBicycle = nmpc::Solver<Bicycle>;
