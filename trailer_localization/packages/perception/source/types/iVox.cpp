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
    if (max_neighbors <= 0 || empty())
    {
        return {};
    }
    std::vector<Neighbor> neighbors(static_cast<std::size_t>(max_neighbors));
    neighbors.resize(nearestNeighbors(query, std::span<Neighbor>(neighbors), max_distance));
    return neighbors;
}

std::size_t IVox::nearestNeighbors(const Eigen::Vector3f& query, const std::span<Neighbor> neighbors,
    const float max_distance) const
{
    if (neighbors.empty() || empty() || !query.allFinite() || !std::isfinite(max_distance) || max_distance <= 0.0f)
    {
        return 0;
    }

    std::size_t count = 0;
    float search_limit = max_distance * max_distance;
    const auto [i, j, k] = pointToVoxel(query);
    const Eigen::Vector3f query_center = Eigen::Vector3f(static_cast<float>(i), static_cast<float>(j),
        static_cast<float>(k)) * VOXEL_SIZE;
    const float query_offset = (query - query_center).cwiseAbs().maxCoeff();
    const auto visitVoxel = [&](int dx, int dy, int dz)
    {
        const Eigen::Vector3f center = Eigen::Vector3f(static_cast<float>(i + dx),
            static_cast<float>(j + dy), static_cast<float>(k + dz)) * VOXEL_SIZE;
        const Eigen::Vector3f nearest_delta = ((center - query).cwiseAbs().array() - 0.5f * VOXEL_SIZE).max(0.0f);
        if (nearest_delta.squaredNorm() >= search_limit)
        {
            return;
        }
        const auto found = mVoxelMap.find(VoxelKey{i + dx, j + dy, k + dz});
        if (found == mVoxelMap.end())
        {
            return;
        }
        for (const Eigen::Vector3f& point : found->second->second.getPoints())
        {
            const float squared_distance = (point - query).squaredNorm();
            if (squared_distance >= search_limit)
            {
                continue;
            }
            std::size_t position = count < neighbors.size() ? count++ : count - 1;
            while (position > 0 && squared_distance < neighbors[position - 1].squared_distance)
            {
                neighbors[position] = neighbors[position - 1];
                --position;
            }
            neighbors[position] = {&point, squared_distance};
            if (count == neighbors.size())
            {
                search_limit = neighbors[count - 1].squared_distance;
            }
        }
    };

    visitVoxel(0, 0, 0);
    const int voxel_radius = static_cast<int>(std::ceil(max_distance * INVERSE_VOXEL_SIZE));
    for (int radius = 1; radius <= voxel_radius; ++radius)
    {
        const float remaining_distance = (static_cast<float>(radius) - 0.5f) * VOXEL_SIZE - query_offset;
        if (remaining_distance >= 0.0f && remaining_distance * remaining_distance >= search_limit)
        {
            break;
        }
        for (int dx = -radius; dx <= radius; ++dx)
        {
            for (int dy = -radius; dy <= radius; ++dy)
            {
                for (int dz = -radius; dz <= radius; ++dz)
                {
                    if (std::max({std::abs(dx), std::abs(dy), std::abs(dz)}) == radius)
                    {
                        visitVoxel(dx, dy, dz);
                    }
                }
            }
        }
    }
    return count;
}

LocalPlane estimatePlane(const std::vector<IVox::Neighbor>& neighbors, const float distance_threshold)
{
    return estimatePlane(std::span<const IVox::Neighbor>(neighbors), distance_threshold);
}

LocalPlane estimatePlane(const std::span<const IVox::Neighbor> neighbors, const float distance_threshold)
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
    if (solver.info() != Eigen::Success || !solver.eigenvalues().allFinite() ||
        solver.eigenvalues()[1] <= 1e-6f || solver.eigenvalues()[1] <= 1e-3f * solver.eigenvalues()[2] ||
        solver.eigenvalues()[0] > 0.1f * solver.eigenvalues()[1])
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