#include "perception/LIO/SparseVoxel.h"
#include <algorithm>
#include <limits>
#include <stdexcept>

namespace
{
void require(const bool condition, const char* message)
{
    if (!condition)
    {
        throw std::runtime_error(message);
    }
}

pcl::PointCloud<pcl::PointXYZ> makeCloud(std::initializer_list<Eigen::Vector3f> positions)
{
    pcl::PointCloud<pcl::PointXYZ> cloud;
    cloud.reserve(positions.size());
    for (const Eigen::Vector3f& position : positions)
    {
        cloud.emplace_back(position.x(), position.y(), position.z());
    }
    return cloud;
}
}

int main()
{
    SparseVoxel target(1.0f, 26, 3);
    target.initialize(makeCloud({{0.1f, 0.0f, 0.0f}, {0.2f, 0.0f, 0.0f}, {2.1f, 0.0f, 0.0f}}));

    const SparseVoxel::Neighbor nearest = target.nearestNeighbor(Eigen::Vector3f(0.19f, 0.0f, 0.0f), 0.5f);
    require(nearest.point_index == 1, "nearest-neighbor query returned the wrong point");

    const std::vector<SparseVoxel::Neighbor> neighbors =
        target.nearestNeighbors(Eigen::Vector3f(0.15f, 0.0f, 0.0f), 1);
    require(neighbors.size() == 1, "K-neighbor query did not enforce its limit");
    require(target.nearestNeighbors(Eigen::Vector3f(0.15f, 0.0f, 0.0f),
                                    std::numeric_limits<int>::max()).size() == 2,
            "a large K-neighbor limit did not return all local candidates");

    const auto oversized_increment = makeCloud(
        {{0.3f, 0.0f, 0.0f}, {1.1f, 0.0f, 0.0f}, {3.1f, 0.0f, 0.0f}});
        require(!target.insert(oversized_increment, 3, 4, 1.0e-3f),
            "an increment larger than the remaining voxel budget was not rejected");
    require(target.voxelCount() == 2 && target.pointCount() == 3, "a rejected increment changed the target");

    const auto accepted_increment = makeCloud({{0.3f, 0.0f, 0.0f}, {1.1f, 0.0f, 0.0f}});
        require(target.insert(accepted_increment, 3, 4, 1.0e-3f), "a valid increment was rejected");
    require(target.voxelCount() == 3 && target.pointCount() == 4, "the valid increment was not appended");

    const SparseVoxel::Neighbor inserted = target.nearestNeighbor(Eigen::Vector3f(1.1f, 0.0f, 0.0f), 0.1f);
    require(inserted.point_index == 3, "the inserted point is not searchable");

    SparseVoxel covariance_voxel(1.0f);
    covariance_voxel.initialize(makeCloud(
        {{0.1f, 0.1f, 0.0f}, {0.3f, 0.1f, 0.0f}, {0.1f, 0.3f, 0.0f}, {0.3f, 0.3f, 0.0f}}));
    covariance_voxel.estimateCovariances(3, 4, 1.0e-3f);
    require(std::ranges::all_of(covariance_voxel.points(), [](const PointWithCovariance& point)
            {
                return point.covariance_valid && point.covariance.allFinite();
            }),
            "covariance estimation did not produce finite valid covariances");

    SparseVoxel insertion_covariance_voxel(1.0f);
    insertion_covariance_voxel.initialize(makeCloud({{0.1f, 0.1f, 0.0f}}));
    require(insertion_covariance_voxel.insert(makeCloud(
                {{1.1f, 0.1f, 0.0f}, {1.3f, 0.1f, 0.0f}, {1.1f, 0.3f, 0.0f}, {1.3f, 0.3f, 0.0f}}),
            3, 4, 1.0e-3f),
            "a covariance-bearing increment was rejected");
    require(std::ranges::all_of(insertion_covariance_voxel.points().begin() + 1,
                                insertion_covariance_voxel.points().end(),
                                [](const PointWithCovariance& point)
            {
                return point.covariance_valid && point.covariance.allFinite();
            }),
            "insert did not estimate covariances for the appended points");

    bool rejected_zero_capacity = false;
    try
    {
        SparseVoxel zero_capacity_target(1.0f, 26, 0);
        zero_capacity_target.initialize(makeCloud({{0.1f, 0.0f, 0.0f}}));
    }
    catch (const std::invalid_argument&)
    {
        rejected_zero_capacity = true;
    }
    require(rejected_zero_capacity, "a zero voxel capacity was accepted");

    bool rejected_invalid_voxel_size = false;
    try
    {
        const SparseVoxel invalid_target(0.0f);
        (void)invalid_target;
    }
    catch (const std::invalid_argument&)
    {
        rejected_invalid_voxel_size = true;
    }
    require(rejected_invalid_voxel_size, "an invalid voxel size was accepted");
    return 0;
}