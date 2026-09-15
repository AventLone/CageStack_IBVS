#pragma once
#include "perception/LIO/ESKF.h"

namespace lio
{
struct PointToPlane
{
    double residual;
    PoseGradient jacobian;

    // p_I includes the LiDAR-to-IMU lever arm. n_W is a UNIT plane normal.
    static PointToPlane linearize(const ImuState& state, const Eigen::Vector3d& p_I,
                                  const Eigen::Vector3d& n_W, const Eigen::Vector3d& plane_center)
    {
        PointToPlane result;
        result.residual = n_W.dot(state.R_WI * p_I + state.p_WI - plane_center);
        result.jacobian.head<3>() = n_W;
        result.jacobian.tail<3>() = Sophus::SO3d::hat(p_I) * (state.R_WI.inverse() * n_W);
        return result;
    }

    void accumulate(ESKF::Measurement& measurement, const double variance, const double huber_scale) const
    {
        const double weight = (std::abs(residual) > huber_scale ? huber_scale / std::abs(residual) : 1.0) / variance;
        measurement.information.noalias() += weight * jacobian * jacobian.transpose();
        measurement.gradient.noalias() += weight * jacobian * residual;
        measurement.squared_error += residual * residual;
        ++measurement.count;
    }
};
}  // namespace lio
