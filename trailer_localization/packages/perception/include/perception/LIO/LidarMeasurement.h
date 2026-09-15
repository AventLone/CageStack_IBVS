#pragma once
#include "perception/LIO/PointToPlane.hpp"
#include "perception/LIO/SparseVoxel.h"

namespace lio
{
// A scan-to-map measurement model, not an independent pose optimizer.
// The caller must keep the map fixed throughout ESKF::update().
class LidarMeasurement
{
public:
    struct Config
    {
        int neighbors{8};
        int voxel_radius{2};
        double max_neighbor_distance{1.0};
        double max_plane_distance{0.05};    // Plane-fit tolerance, m.
        double max_planarity_ratio{0.1};   // lambda_min / lambda_middle.
        double max_residual{0.3};          // Association gate, m.
        double lidar_noise{0.03};          // Point-to-plane standard deviation, m.
        double huber_scale{0.1};           // m.
    };

    LidarMeasurement(const Config& config, const Sophus::SE3d& T_IL);

    ESKF::Measurement build(const ImuState& state, const pcl::PointCloud<pcl::PointXYZ>& scan, const SparseVoxel& map) const;

private:
    Config mConfig;
    Sophus::SE3d mTi2l;
};
}  // namespace lio
