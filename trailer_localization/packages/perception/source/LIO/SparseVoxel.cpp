#include "perception/LIO/SparseVoxel.h"
#include <algorithm>
#include <cmath>
#include <execution>
#include <unordered_set>

void SparseVoxel::initialize(const pcl::PointCloud<pcl::PointXYZ>& cloud)
{
	if (mConfig.max_voxels_num == 0)
	{
		throw std::invalid_argument("Initial target exceeds max_target_voxels or the voxel budget is zero");
	}

	const std::vector<SparsePoint> sparse_points = makeSparseCloud(cloud);
	std::unordered_set<VoxelKey, VoxelKeyHash> voxel_keys;
	voxel_keys.reserve(sparse_points.size());
	for (const SparsePoint& point : sparse_points)
	{
		voxel_keys.insert(point.key);
	}

	if (voxel_keys.size() > mConfig.max_voxels_num)
	{
		throw std::invalid_argument("Initial target exceeds max_target_voxels or the voxel budget is zero");
	}

	clear();

	mOccupiedVoxels.reserve(voxel_keys.size());
	const std::size_t cell_capacity = static_cast<std::size_t>(std::max(1, mConfig.max_points_per_voxel));
	for (const SparsePoint& point : sparse_points)
	{
		auto [cell, inserted] = mOccupiedVoxels.try_emplace(point.key);
		if (inserted)
		{
			cell->second.reserve(cell_capacity);
		}
		cell->second.push_back(PointWithCovariance{.position = point.position});
		++mPointCount;
	}
	estimateCovariances();
}

bool SparseVoxel::insert(const pcl::PointCloud<pcl::PointXYZ>& cloud)
{
	if (cloud.empty())
	{
		return false;
	}

	const int max_points_per_voxel = std::max(1, mConfig.max_points_per_voxel);
	const float min_spacing_square = std::pow(std::max(0.001f, mConfig.voxel_size * 0.1f), 2.0f);
	std::unordered_set<VoxelKey, VoxelKeyHash> touched_voxels;
	touched_voxels.reserve(cloud.size());

	for (const auto& pcl_point : cloud)
	{
		if (!std::isfinite(pcl_point.x) || !std::isfinite(pcl_point.y) || !std::isfinite(pcl_point.z))
		{
			continue;
		}

		SparsePoint point;
		point.position = Eigen::Vector3f(pcl_point.x, pcl_point.y, pcl_point.z);
		point.key = pointToVoxel(point.position);

		auto found = mOccupiedVoxels.find(point.key);
		if (found == mOccupiedVoxels.end())
		{
			if (mOccupiedVoxels.size() >= mConfig.max_voxels_num)
			{
				continue;
			}

			found = mOccupiedVoxels.try_emplace(point.key).first;
			found->second.reserve(static_cast<std::size_t>(max_points_per_voxel));
		}

		auto& cell_points = found->second;
		if (static_cast<int>(cell_points.size()) >= max_points_per_voxel)
		{
			continue;
		}
		if (const bool too_close = std::ranges::any_of(cell_points, [&](const PointWithCovariance& existing_point)
			{
				return (existing_point.position - point.position).squaredNorm() < min_spacing_square;
			}); too_close)
		{
			continue;
		}

		cell_points.push_back(PointWithCovariance{.position = point.position});
		++mPointCount;
		touched_voxels.insert(point.key);
	}

	if (touched_voxels.empty())
	{
		return false;
	}

	std::unordered_set<VoxelKey, VoxelKeyHash> affected_voxels;
	affected_voxels.reserve(touched_voxels.size() * 27);
	for (const VoxelKey& key : touched_voxels)
	{
		for (int dx = -1; dx <= 1; ++dx)
		{
			for (int dy = -1; dy <= 1; ++dy)
			{
				for (int dz = -1; dz <= 1; ++dz)
				{
                    if (const VoxelKey affected_key{key.i + dx, key.j + dy, key.k + dz};
                        mOccupiedVoxels.contains(affected_key))
					{
						affected_voxels.insert(affected_key);
					}
				}
			}
		}
	}

	std::vector<PointWithCovariance*> affected_points;
	affected_points.reserve(affected_voxels.size() * static_cast<std::size_t>(max_points_per_voxel));
	for (const VoxelKey& key : affected_voxels)
	{
		for (PointWithCovariance& point : mOccupiedVoxels.at(key))
		{
			affected_points.push_back(&point);
		}
	}

	estimateCovariances(affected_points);
	return true;
}

void SparseVoxel::estimateCovariances()
{
	std::vector<PointWithCovariance*> points;
	points.reserve(mPointCount);
	for (auto& [_, cell_points] : mOccupiedVoxels)
	{
		for (PointWithCovariance& point : cell_points)
		{
			points.push_back(&point);
		}
	}
	estimateCovariances(points);
}

void SparseVoxel::estimateCovariances(const std::vector<PointWithCovariance*>& points) const
{
	const int required_neighbors = std::max(3, mConfig.min_covariance_neighbors);
	const int neighbor_limit = std::max(required_neighbors, mConfig.max_covariance_neighbors);
	const float diagonal_regularization = std::max(1.0e-6f, mConfig.covariance_regularization);

	std::for_each(std::execution::par, points.begin(), points.end(),
		[this, required_neighbors, neighbor_limit, diagonal_regularization](PointWithCovariance* point)
		{
			point->covariance = Eigen::Matrix3f::Identity();
			point->covariance_valid = false;

			constexpr int voxel_radius = 1;
			const std::vector<Neighbor> neighbors =
				nearestNeighbors(point->position, neighbor_limit, voxel_radius);
			if (static_cast<int>(neighbors.size()) < required_neighbors)
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
			covariance.diagonal().array() += diagonal_regularization;

			if (const float determinant = covariance.determinant();
				!std::isfinite(determinant) || std::abs(determinant) <= 1.0e-12f)
			{
				return;
			}

			const float inverse_norm = covariance.inverse().norm();
			if (!std::isfinite(inverse_norm) || inverse_norm <= 0.0f)
			{
				return;
			}
			point->covariance = covariance * inverse_norm;
			point->covariance_valid = true;
		});
}

std::vector<const PointWithCovariance*> SparseVoxel::points() const
{
	std::vector<const PointWithCovariance*> points;
	points.reserve(mPointCount);
	for (const auto& [_, cell_points] : mOccupiedVoxels)
	{
		for (const PointWithCovariance& point : cell_points)
		{
			points.push_back(&point);
		}
	}
	return points;
}

SparseVoxel::Neighbor SparseVoxel::nearestNeighbor(const Eigen::Vector3f& query, const float max_distance, const int voxel_radius) const
{
	Neighbor nearest;
	nearest.squared_distance = max_distance * max_distance;
	const auto [i, j, k] = pointToVoxel(query);

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

				for (const PointWithCovariance& point : found->second)
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

std::vector<SparseVoxel::Neighbor> SparseVoxel::nearestNeighbors(const Eigen::Vector3f& query,
                                                                 const int max_neighbors,
																 const int voxel_radius) const
{
	std::vector<Neighbor> neighbors;
	if (max_neighbors <= 0)
	{
		return neighbors;
	}

	neighbors.reserve(std::min(static_cast<std::size_t>(max_neighbors), mPointCount));
	const auto [i, j, k] = pointToVoxel(query);
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

				for (const PointWithCovariance& point : found->second)
				{
					neighbors.push_back(Neighbor{&point, (query - point.position).squaredNorm()});
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

std::vector<SparseVoxel::SparsePoint> SparseVoxel::makeSparseCloud(const pcl::PointCloud<pcl::PointXYZ>& cloud) const
{
    std::vector<SparsePoint> sparse_cloud;
    sparse_cloud.reserve(cloud.size());
	SparsePointIndices occupied_voxels;
    const int max_points_per_voxel = std::max(1, mConfig.max_points_per_voxel);
    const float min_spacing_square = std::pow(std::max(0.001f, mConfig.voxel_size * 0.1f), 2.0f);

    for (const auto& pcl_point : cloud)
    {
        if (!std::isfinite(pcl_point.x) || !std::isfinite(pcl_point.y) || !std::isfinite(pcl_point.z))
        {
            continue;
        }

        SparsePoint point;
        point.position = Eigen::Vector3f(pcl_point.x, pcl_point.y, pcl_point.z);
        point.key = pointToVoxel(point.position);

        auto& voxel_points = occupied_voxels[point.key];
        if (static_cast<int>(voxel_points.size()) >= max_points_per_voxel)
        {
            continue;
        }

        if (const bool too_close = std::ranges::any_of(std::as_const(voxel_points), [&](const std::size_t index)
            {
                return (sparse_cloud[index].position - point.position).squaredNorm() < min_spacing_square;
            }); too_close)
        {
            continue;
        }

        voxel_points.push_back(sparse_cloud.size());
        sparse_cloud.push_back(point);
    }
    return sparse_cloud;
}
