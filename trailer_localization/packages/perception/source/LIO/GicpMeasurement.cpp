#include "perception/LIO/GicpMeasurement.h"
#include <cmath>
#include <stdexcept>

namespace lio
{
GICP::Config GicpMeasurement::makeGicpConfig(const Config& config)
{
    GICP::Config result;
    result.voxel_size = config.voxel_size;
    result.max_correspondence_distance = config.max_correspondence_distance;
    result.cauchy_kernel_scale = config.cauchy_kernel_scale;
    return result;
}

GicpMeasurement::GicpMeasurement(const Config& config, const Sophus::SE3d& T_IL)
    : mConfig(config), mTIL(T_IL), mGicp(makeGicpConfig(config))
{
    if (!std::isfinite(config.voxel_size) || config.voxel_size <= 0.0f ||
        !std::isfinite(config.max_correspondence_distance) || config.max_correspondence_distance <= 0.0f ||
        !std::isfinite(config.cauchy_kernel_scale) || config.cauchy_kernel_scale < 0.0f ||
        !std::isfinite(config.lidar_noise) || config.lidar_noise <= 0.0 ||
        !T_IL.matrix().allFinite())
    {
        throw std::invalid_argument("Invalid tightly coupled GICP configuration.");
    }
}

std::unique_ptr<SparseVoxel> GicpMeasurement::prepareScan(
    const pcl::PointCloud<pcl::PointXYZ>& scan) const
{
    return mGicp.createSourceIndex(scan);
}

ESKF::Measurement GicpMeasurement::build(const ImuState& state,
    const SparseVoxel& scan, const SparseVoxel& map) const
{
    const Sophus::SE3d T_WL = Sophus::SE3d(state.R_WI, state.p_WI) * mTIL;
    const auto gicp = mGicp.linearize(scan, map, T_WL);

    ESKF::Measurement measurement;
    measurement.count = gicp.num_correspondences;
    measurement.squared_error = gicp.squared_error_sum;
    if (measurement.count == 0 || !gicp.hessian.allFinite() || !gicp.gradient.allFinite())
    {
        return measurement;
    }

    // GICP uses a full right perturbation of T_WL:
    //   T_WL' = T_WL * Exp(d_rho_L, d_theta_L).
    // ESKF uses additive dp_W and right dtheta_I on T_WI. The LiDAR-to-IMU
    // lever arm therefore contributes to the local translation increment.
    PoseInformation right_from_eskf = PoseInformation::Zero();
    const Eigen::Matrix3d R_WI = state.R_WI.matrix();
    const Eigen::Matrix3d R_IL = mTIL.rotationMatrix();
    const Eigen::Matrix3d R_WL = R_WI * R_IL;
    right_from_eskf.topLeftCorner<3, 3>() = R_WL.transpose();
    right_from_eskf.topRightCorner<3, 3>() =
        -R_WL.transpose() * R_WI * Sophus::SO3d::hat(mTIL.translation());
    right_from_eskf.bottomRightCorner<3, 3>() = R_IL.transpose();

    const double inverse_variance = 1.0 / (mConfig.lidar_noise * mConfig.lidar_noise);
    measurement.information = inverse_variance *
        right_from_eskf.transpose() * gicp.hessian * right_from_eskf;
    measurement.information = 0.5 *
        (measurement.information + measurement.information.transpose());
    measurement.gradient = inverse_variance * right_from_eskf.transpose() * gicp.gradient;
    return measurement;
}
}  // namespace lio
