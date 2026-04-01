#pragma once
#include "perception/types/common.hpp"

using StateT = Eigen::Vector3f; // [p_x, p_y, theta]'

struct SystemModel
{
    using ControlT = Eigen::Vector3f; // [v_x, v_y, w]'

    static constexpr float dt = 1.0 / 10.0f; // Sample interval 100 ms

    static const inline Eigen::DiagonalMatrix<float, 3>
    MatQ{std::pow(0.2f * dt, 2.0f), std::pow(0.2f * dt, 2.0f), std::pow(0.1f * dt, 2.0f)};

    StateT operator()(const StateT& vecX, const ControlT& vecU) const
    {
        StateT vecX_next(vecX);
        vecX_next[0] += dt * (vecU[0] * std::cos(vecX[2]) - vecU[1] * std::sin(vecX[2]));
        vecX_next[1] += dt * (vecU[0] * std::sin(vecX[2]) + vecU[1] * std::cos(vecX[2]));
        vecX_next[2] += dt * vecU[2];

        vecX_next[2] = std::clamp(vecX_next[2], -M_PI_2f, M_PI_2f);

        return vecX_next;
    }
};

struct MeasureModel
{
    using MeasurementT = Eigen::Vector3f; // [p_x, p_y, theta]'

    static const inline Eigen::DiagonalMatrix<float, 3>
    MatR{std::pow(0.03f, 2.0f), std::pow(0.03f, 2.0f), std::pow(0.03f, 2.0f)};

    MeasurementT operator()(const StateT& vecX) const
    {
        return vecX;
    }
};
