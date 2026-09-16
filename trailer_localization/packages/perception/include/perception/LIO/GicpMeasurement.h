#pragma once
#include "perception/GICP/GICP.h"
#include "perception/LIO/ESKF.h"

namespace lio
{
// Direct GICP scan-to-map residuals for the ESKF. This is not a pose
// measurement: every source/target correspondence contributes its Hessian and
// gradient to the same normal equation as the propagated IMU prior.
class GicpMeasurement
{
public:
    struct Config
    {
        float voxel_size{0.5f};
        float max_correspondence_distance{0.5f};
        float cauchy_kernel_scale{0.3f};

        // SparseVoxel covariance is a dimensionless GICP shape matrix. This
        // scalar supplies the physical metre scale before it is combined with
        // the ESKF prior: Sigma = lidar_noise^2 * Sigma_gicp_shape.
        double lidar_noise{0.03};
    };

    GicpMeasurement(const Config& config, const Sophus::SE3d& T_IL);

    [[nodiscard]] std::unique_ptr<SparseVoxel> prepareScan(
        const pcl::PointCloud<pcl::PointXYZ>& scan) const;

    [[nodiscard]] ESKF::Measurement build(const ImuState& state,
                                          const SparseVoxel& scan,
                                          const SparseVoxel& map) const;

private:
    static GICP::Config makeGicpConfig(const Config& config);

    Config mConfig;
    Sophus::SE3d mTIL;
    GICP mGicp;
};
}  // namespace lio
