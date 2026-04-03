#pragma once
#include "perception/types/common.hpp"

using StateT = Eigen::Vector3f; // [p_x, p_y, theta]'

struct SystemModel
{
    using ControlT = Eigen::Vector2f; // [v, delta]'

    static inline float dt = 1.0 / 10.0f; // Sample interval 100 ms

    static constexpr float wheelbase = 1.20551f;
    static constexpr float wheelbase_inv = 1.0f / wheelbase;
    static constexpr float wheel_radius = 0.115f;

    static const inline Eigen::DiagonalMatrix<float, 3>
    MatQ{std::pow(0.006f, 2.0f), std::pow(0.006f, 2.0f), std::pow(0.05f, 2.0f)};
    // MatQ{std::pow(0.1f, 2.0f), std::pow(0.1f, 2.0f), std::pow(0.2f, 2.0f)};

    StateT operator()(const StateT& vecX, const ControlT& vecU) const
    {
        StateT vecX_next(vecX);
        const float v = vecU[0] * wheel_radius;
        vecX_next[0] += dt * v * std::cos(vecX[2]);
        vecX_next[1] += dt * v * std::sin(vecX[2]);
        vecX_next[2] += dt * v * wheelbase_inv * std::tan(vecU[1]);
        return vecX_next;
    }
};

struct MeasureModel
{
    using MeasurementT = Eigen::Vector3f; // [p_x, p_y, theta]'

    static const inline Eigen::DiagonalMatrix<float, 3>
    MatR{std::pow(0.01f, 2.0f), std::pow(0.01f, 2.0f), std::pow(0.1f, 2.0f)};
    // MatR{std::pow(0.001f, 2.0f), std::pow(0.001f, 2.0f), std::pow(0.05f, 2.0f)};

    MeasurementT operator()(const StateT& vecX) const
    {
        return vecX;
    }
};
