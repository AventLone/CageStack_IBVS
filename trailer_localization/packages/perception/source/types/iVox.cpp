#include "perception/types/iVox.h"
#include <ranges>
#include <Eigen/Eigenvalues>

IVox::VoxelKey IVox::pointToVoxel(const Eigen::Vector3f& point)
{
    /*
     * Faster-LIO uses round(), rather than floor().
     *
     * The voxel represented by key (0,0,0) is therefore
     * centered around the origin.
     */
    const Eigen::Vector3f scaled = point * INVERSE_VOXEL_SIZE;

    return {static_cast<int>(std::round(scaled.x())),
            static_cast<int>(std::round(scaled.y())),
            static_cast<int>(std::round(scaled.z()))};
}


bool IVox::insert(const pcl::PointCloud<pcl::PointXYZ>& scan)
{
    if (scan.empty())
    {
        return false;
    }

    bool inserted_any = false;

    for (const auto& point : scan)
    {
        const Eigen::Vector3f eigen_point = point.getVector3fMap();
        const VoxelKey key = pointToVoxel(eigen_point);
        if (auto found = mVoxelMap.find(key); found == mVoxelMap.end())
        {
            /*
             * Same basic idea as Faster-LIO:
             *
             * create voxel
             *     ↓
             * put it at the front of LRU list
             *     ↓
             * insert into hash map
             */
            mVoxelCache.emplace_front(key, VoxelCell{});

            auto voxel_it = mVoxelCache.begin();
            voxel_it->second.add(eigen_point);

            mVoxelMap.emplace(key, voxel_it);

            ++mPointCount;
            inserted_any = true;

            /* Faster-LIO limits the number of occupied voxels */
            if (mVoxelMap.size() > VOXEL_CAPACITY)
            {
                const auto last = std::prev(mVoxelCache.end());
                mPointCount -= last->second.getPoints().size();
                mVoxelMap.erase(last->first);
                mVoxelCache.pop_back();
            }
        }
        else
        {
            auto voxel_it = found->second;
            voxel_it->second.add(eigen_point);

            ++mPointCount;
            inserted_any = true;

            /* Refresh this voxel in the LRU cache */
            mVoxelCache.splice(mVoxelCache.begin(), mVoxelCache, voxel_it);
            found->second = mVoxelCache.begin();
        }
    }

    return inserted_any;
}

std::vector<IVox::Neighbor> IVox::nearestNeighbors(const Eigen::Vector3f& query, const int max_neighbors, const float max_distance) const
{
    std::vector<Neighbor> neighbors;

    if (max_neighbors <= 0)
    {
        return neighbors;
    }

    const float max_distance_squared = max_distance * max_distance;
    neighbors.reserve(max_neighbors);

    const auto [i, j, k] = pointToVoxel(query);
    constexpr int voxel_radius = 1;
    for (int dx = -voxel_radius; dx <= voxel_radius; ++dx)
    {
        for (int dy = -voxel_radius; dy <= voxel_radius; ++dy)
        {
            for (int dz = -voxel_radius; dz <= voxel_radius; ++dz)
            {
                const auto found = mVoxelMap.find(VoxelKey{i + dx, j + dy, k + dz});
                if (found == mVoxelMap.end())
                {
                    continue;
                }

                for (const VoxelCell& voxel = found->second->second; const Eigen::Vector3f& point : voxel.getPoints())
                {
                    if (const float squared_distance = (point - query).squaredNorm();
                        squared_distance < max_distance_squared)
                    {
                        neighbors.push_back({&point, squared_distance});
                    }
                }
            }
        }
    }

    if (static_cast<int>(neighbors.size()) > max_neighbors)
    {
        std::ranges::nth_element(neighbors, neighbors.begin() + max_neighbors, {}, &Neighbor::squared_distance);
        neighbors.resize(max_neighbors);
    }

    return neighbors;
}

LocalPlane estimatePlane(const std::vector<IVox::Neighbor>& neighbors, const float distance_threshold)
{
    LocalPlane plane;

    /*
     * Faster-LIO:
     * NUM_MATCH_POINTS     = 5
     * MIN_NUM_MATCH_POINTS = 3
     */
    if (neighbors.size() < 3)
    {
        return plane;
    }

    Eigen::Vector3f center = Eigen::Vector3f::Zero();

    for (const auto& neighbor : neighbors)
    {
        center += *neighbor.point;
    }

    center /= static_cast<float>(neighbors.size());

    Eigen::Matrix3f covariance = Eigen::Matrix3f::Zero();
    for (const auto& neighbor : neighbors)
    {
        const Eigen::Vector3f delta = *neighbor.point - center;
        covariance.noalias() += delta * delta.transpose();
    }

    covariance /= static_cast<float>(neighbors.size());
    const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver(covariance);
    if (solver.info() != Eigen::Success)
    {
        return plane;
    }

    /*
     * Eigenvalues:
     * lambda0 <= lambda1 <= lambda2
     * lambda0 direction is the local
     * plane normal.
     */
    Eigen::Vector3f normal = solver.eigenvectors().col(0);
    normal.normalize();

    /*
     * Same spirit as Faster-LIO's esti_plane():
     *
     * Every KNN point must be sufficiently close
     * to the estimated plane.
     */
    for (const auto& neighbor : neighbors)
    {
        if (const float distance = std::abs(normal.dot(*neighbor.point - center));
            distance > distance_threshold)
        {
            return plane;
        }
    }

    plane.center = center;
    plane.normal = normal;
    plane.normal_variance = std::max(0.0f, solver.eigenvalues()[0]);
    plane.valid = true;

    return plane;
}