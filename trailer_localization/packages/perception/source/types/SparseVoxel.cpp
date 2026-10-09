#include "perception/types/SparseVoxel.h"
#include <algorithm>
#include <cmath>
#include <execution>
#include <ranges>
#include <boost/unordered/unordered_flat_set.hpp>

bool SparseVoxel::insert(const pcl::PointCloud<pcl::PointXYZ>& cloud)
{
    if (cloud.empty())
    {
        return false;
    }

    boost::unordered_flat_set<VoxelKey> updated_voxels;
    for (const auto& point : cloud)
    {
        const Eigen::Vector3f eigen_point = point.getVector3fMap();
        const VoxelKey key = pointToVoxel(eigen_point);

        auto [it, inserted] = mOccupiedVoxels.try_emplace(key, mCells.size());
        if (inserted)
        {
            mCells.emplace_back();
        }

        if (mCells[it->second].add(eigen_point) != nullptr)
        {
            updated_voxels.insert(key);
            ++mPointCount;
        }
    }

    boost::unordered_flat_set<std::size_t> affected_cells;
    for (const auto& [i, j, k] : updated_voxels)
    {
        constexpr int voxel_radius = 1;
        for (int dx = -voxel_radius; dx <= voxel_radius; ++dx)
        {
            for (int dy = -voxel_radius; dy <= voxel_radius; ++dy)
            {
                for (int dz = -voxel_radius; dz <= voxel_radius; ++dz)
                {
                    if (const auto found = mOccupiedVoxels.find(VoxelKey{i + dx, j + dy, k + dz});
                        found != mOccupiedVoxels.end())
                    {
                        affected_cells.insert(found->second);
                    }
                }
            }
        }
    }

    std::vector<PointWithCovariance*> points;
    for (const std::size_t cell_index : affected_cells)
    {
        for (PointWithCovariance& point : mCells[cell_index].points)
        {
            points.push_back(&point);
        }
    }

    estimateCovariances(points);
	return true;
}

void SparseVoxel::estimateCovariances(const std::vector<PointWithCovariance*>& points) const
{
	std::for_each(std::execution::par, points.begin(), points.end(),
		[this](PointWithCovariance* point)
		{
			point->covariance = Eigen::Matrix3f::Identity();
			point->covariance_valid = false;

			const std::vector<Neighbor> neighbors = nearestNeighbors(point->position, MAX_COVARIANCE_NEIGHBORS);
			if (static_cast<int>(neighbors.size()) < MIN_COVARIANCE_NEIGHBORS)
			{
				return;
			}

			Eigen::Vector3f mean = Eigen::Vector3f::Zero();
			for (const auto& [neighbor, _] : neighbors)
			{
				mean += neighbor->position;
			}
			mean /= static_cast<float>(neighbors.size());

			Eigen::Matrix3f covariance = Eigen::Matrix3f::Zero();
			for (const auto& [neighbor, _] : neighbors)
			{
				const Eigen::Vector3f offset = neighbor->position - mean;
				covariance.noalias() += offset * offset.transpose();
			}
			covariance /= static_cast<float>(neighbors.size() - 1);

			point->covariance = covariance;
			point->covariance_valid = true;
		});
}

std::vector<const PointWithCovariance*> SparseVoxel::points() const
{
	std::vector<const PointWithCovariance*> points;
	points.reserve(mPointCount);
    for (const VoxelCell& cell : mCells)
	{
		for (const PointWithCovariance& point : cell.points)
		{
			points.push_back(&point);
		}
	}
	return points;
}

SparseVoxel::Neighbor SparseVoxel::nearestNeighbor(const Eigen::Vector3f& query, const float max_distance) const
{
	Neighbor nearest;
	nearest.squared_distance = max_distance * max_distance;
	const auto [i, j, k] = pointToVoxel(query);

    constexpr int voxel_radius = 1;
	for (int dx = -voxel_radius; dx <= voxel_radius; ++dx)
	{
		for (int dy = -voxel_radius; dy <= voxel_radius; ++dy)
		{
			for (int dz = -voxel_radius; dz <= voxel_radius; ++dz)
			{
				const auto found = mOccupiedVoxels.find(VoxelKey{i + dx, j + dy, k + dz});
				if (found == mOccupiedVoxels.end())
				{
					continue;
				}

                for (const PointWithCovariance& point : mCells[found->second].points)
				{
					if (const float squared_distance = (query - point.position).squaredNorm();
                        squared_distance < nearest.squared_distance)
					{
						nearest = Neighbor{&point, squared_distance};
					}
				}
			}
		}
	}
	return nearest;
}

std::vector<SparseVoxel::Neighbor> SparseVoxel::nearestNeighbors(const Eigen::Vector3f& query, const int max_neighbors) const
{
    if (max_neighbors <= 0)
    {
        return {};
    }

    std::vector<Neighbor> neighbors;
    neighbors.reserve(static_cast<std::size_t>(max_neighbors));

    const auto [i, j, k] = pointToVoxel(query);

    constexpr int voxel_radius = 1;

    // Max-heap:
    // neighbors.front() is always the farthest neighbor
    // among the current top-K nearest neighbors.
    const auto compare = [](const Neighbor& lhs, const Neighbor& rhs)
    {
        return lhs.squared_distance < rhs.squared_distance;
    };

    for (int dx = -voxel_radius; dx <= voxel_radius; ++dx)
    {
        for (int dy = -voxel_radius; dy <= voxel_radius; ++dy)
        {
            for (int dz = -voxel_radius; dz <= voxel_radius; ++dz)
            {
                const auto found = mOccupiedVoxels.find(VoxelKey{i + dx, j + dy, k + dz});

                if (found == mOccupiedVoxels.end())
                {
                    continue;
                }

                for (const PointWithCovariance& point : mCells[found->second].points)
                {
                    const float squared_distance = (query - point.position).squaredNorm();

                    if (neighbors.size() < static_cast<std::size_t>(max_neighbors))
                    {
                        neighbors.push_back(Neighbor{.point = &point, .squared_distance = squared_distance});
                        std::ranges::push_heap(neighbors, compare);
                        continue;
                    }

                    // Heap is full.
                    // front() is the farthest point currently retained.
                    if (squared_distance >= neighbors.front().squared_distance)
                    {
                        continue;
                    }

                    std::ranges::pop_heap(neighbors, compare);                                            // Remove current farthest.
                    neighbors.back() = Neighbor{.point = &point, .squared_distance = squared_distance};   // Replace it with the new closer point.
                    std::ranges::push_heap(neighbors, compare);                                           // Restore heap.
                }
            }
        }
    }

    return neighbors;
}