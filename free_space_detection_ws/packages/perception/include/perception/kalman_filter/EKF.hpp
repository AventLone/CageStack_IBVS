#pragma once
#include "./BaseEKF.hpp"

using StateT = Eigen::Vector3f; // [p_x, p_y, theta]'
using ControlT = Eigen::Vector2f; // [n_l, n_r]
using MeasurementT = Eigen::Vector2f; // [p_x, p_y]

constexpr double dt = 0.002; // Sample interval 2 ms
constexpr double wheelDiameter = 0.6; // Unit: m
constexpr double wheelCircumference = wheelDiameter * M_PI;
constexpr double tread = 0.8;

struct SystemModel
{
    StateT operator()(const StateT& vecX, const ControlT& vecU) const
    {
        StateT new_vecX = vecX;

        new_vecX[0] += dt * wheelCircumference * std::cos(vecX[2]) * (vecU[0] + vecU[1]) / 2.0;
        new_vecX[1] += dt * wheelCircumference * std::sin(vecX[2]) * (vecU[0] + vecU[1]) / 2.0;
        new_vecX[2] += dt * wheelCircumference * (vecU[1] - vecU[0]) / tread;

        if (new_vecX[2] > 2 * M_PI || new_vecX[2] < -2 * M_PI)
            new_vecX[2] = 0.0;

        return new_vecX;
    }
};

struct MeasureModel
{
    MeasurementT operator()(const StateT& vecX) const
    {
        MeasurementT estimate;
        estimate[0] = vecX[0];
        estimate[1] = vecX[1];
        return estimate;
    }
};

class EKF : public BaseEKF<StateT, ControlT, MeasurementT, SystemModel, MeasureModel>
{
public:
    explicit EKF(const CovarianceMatrix<StateT>& matQ, const CovarianceMatrix<MeasurementT>& matR,
                 const StateT& vecX0 = StateT::Zero(),
                 const CovarianceMatrix<StateT>& matP0 = CovarianceMatrix<StateT>::Identity())
        : BaseEKF(matQ, matR, vecX0, matP0)
    {}

private:
    void updateJacobians(const ControlT& vecU) override
    {
        mJacobianA(0, 2) = -dt * wheelCircumference * std::sin(mVecX[2]) * (vecU[0] + vecU[1]) / 2.0f;
        mJacobianA(1, 2) = dt * wheelCircumference * std::cos(mVecX[2]) * (vecU[0] + vecU[1]) / 2.0f;
    }
};
