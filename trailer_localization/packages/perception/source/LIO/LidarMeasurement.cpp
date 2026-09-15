#include "perception/LIO/LidarMeasurement.h"
#include <Eigen/Eigenvalues>

namespace lio
{
LidarMeasurement::LidarMeasurement(const Config& config, const Sophus::SE3d& T_IL)
    : mConfig(config), mTi2l(T_IL)
{
    if (config.neighbors < 3 || config.voxel_radius < 1 || config.voxel_radius > 10)
        throw std::invalid_argument("Invalid LiDAR neighborhood configuration.");
    for (double value : {config.max_neighbor_distance, config.max_plane_distance,
                        config.max_planarity_ratio, config.max_residual, config.lidar_noise, config.huber_scale})
    {
        if (!std::isfinite(value) || value <= 0.0) throw std::invalid_argument("LiDAR thresholds must be positive and finite.");
    }
}

ESKF::Measurement LidarMeasurement::build(const ImuState& state,
    const pcl::PointCloud<pcl::PointXYZ>& scan, const SparseVoxel& map) const
{
    ESKF::Measurement measurement;
    for (const auto& point : scan)
    {
        const Eigen::Vector3d p_L(point.x, point.y, point.z);
        if (!p_L.allFinite()) continue;
        const Eigen::Vector3d p_I = mTi2l * p_L;
        const Eigen::Vector3d p_W = state.R_WI * p_I + state.p_WI;
        const auto neighbors = map.nearestNeighbors(p_W.cast<float>(), mConfig.neighbors, mConfig.voxel_radius);
        if (neighbors.size() < static_cast<std::size_t>(mConfig.neighbors)) continue;
        Eigen::Vector3d center = Eigen::Vector3d::Zero();
        bool valid = true;
        for (const auto& neighbor : neighbors)
        {
            if (neighbor.squared_distance > mConfig.max_neighbor_distance * mConfig.max_neighbor_distance) valid = false;
            center += neighbor.point->position.cast<double>();
        }
        if (!valid) continue;
        center /= static_cast<double>(neighbors.size());
        // Fit from raw neighbor positions. SparseVoxel's normalized GICP
        // covariances are not metric measurement-noise covariances.
        Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();
        for (const auto& neighbor : neighbors)
        {
            const Eigen::Vector3d offset = neighbor.point->position.cast<double>() - center;
            covariance.noalias() += offset * offset.transpose();
        }
        covariance /= static_cast<double>(neighbors.size());
        const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(covariance);
        if (solver.info() != Eigen::Success || solver.eigenvalues()[1] < 1e-5 ||
            solver.eigenvalues()[0] > mConfig.max_planarity_ratio * solver.eigenvalues()[1]) continue;
        const Eigen::Vector3d normal = solver.eigenvectors().col(0);
        for (const auto& neighbor : neighbors)
        {
            if (std::abs(normal.dot(neighbor.point->position.cast<double>() - center)) > mConfig.max_plane_distance) valid = false;
        }
        if (!valid) continue;
        const auto constraint = PointToPlane::linearize(state, p_I, normal, center);
        if (std::abs(constraint.residual) > mConfig.max_residual) continue;
        constraint.accumulate(measurement, mConfig.lidar_noise * mConfig.lidar_noise, mConfig.huber_scale);
    }
    return measurement;
}
}  // namespace lio
