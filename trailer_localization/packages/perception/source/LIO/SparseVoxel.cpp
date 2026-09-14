#include "perception/LIO/SparseVoxel.h"
#include <algorithm>
#include <cmath>
#include <execution>
#include <numeric>
#include <unordered_set>
#include <utility>

bool SparseVoxel::insert(const pcl::PointCloud<pcl::PointXYZ>& cloud)
{
	if (cloud.empty() || mVoxels.size() >= mConfig.max_voxels_num)
	{
		return false;
	}

	SparseVoxel increment(mConfig);
	VoxelPointLayout layout = makeVoxelPointLayout(makeSparseCloud(cloud));

	VoxelPointLayout filtered;
	filtered.points.reserve(layout.points.size());
	filtered.voxels.reserve(layout.voxels.size());
	filtered.voxel_keys.reserve(layout.voxel_keys.size());

	for (std::size_t index = 0; index < layout.voxels.size(); ++index)
	{
		if (mVoxelMap.contains(layout.voxel_keys[index]))
		{
			continue;
		}

		const auto [start, count] = layout.voxels[index];
		const int filtered_start = static_cast<int>(filtered.points.size());
		filtered.points.insert(filtered.points.end(), layout.points.begin() + start,
							   layout.points.begin() + start + count);
		filtered.voxels.push_back(VoxelEntry{filtered_start, count});
		filtered.voxel_keys.push_back(layout.voxel_keys[index]);
	}

    if (const std::size_t available_voxels = mConfig.max_voxels_num - mVoxels.size();
        filtered.voxels.empty() || filtered.voxels.size() > available_voxels)
	{
		return false;
	}

	increment.initializeLayout(std::move(filtered));
	increment.estimateCovariances();

	const int point_offset = static_cast<int>(mPoints.size());
	const int voxel_offset = static_cast<int>(mVoxels.size());
	for (auto& voxel : increment.mVoxels)
	{
		voxel.start += point_offset;
	}

	mPoints.insert(mPoints.end(), std::make_move_iterator(increment.mPoints.begin()), std::make_move_iterator(increment.mPoints.end()));
	mVoxels.insert(mVoxels.end(), increment.mVoxels.begin(), increment.mVoxels.end());
	mVoxelKeys.insert(mVoxelKeys.end(), increment.mVoxelKeys.begin(), increment.mVoxelKeys.end());
	for (int index = 0; index < static_cast<int>(increment.mVoxelKeys.size()); ++index)
	{
		mVoxelMap.emplace(increment.mVoxelKeys[index], voxel_offset + index);
	}
	return true;
}

void SparseVoxel::estimateCovariances()
{
	const int required_neighbors = std::max(3, mConfig.min_covariance_neighbors);
	const int neighbor_limit = std::max(required_neighbors, mConfig.max_covariance_neighbors);
	const float diagonal_regularization = std::max(1.0e-6f, mConfig.covariance_regularization);

	std::for_each(std::execution::par, mPoints.begin(), mPoints.end(),
		[this, required_neighbors, neighbor_limit, diagonal_regularization](PointWithCovariance& point)
		{
			point.covariance = Eigen::Matrix3f::Identity();
			point.covariance_valid = false;

			constexpr int voxel_radius = 1;
			const std::vector<Neighbor> neighbors =
				nearestNeighbors(point.position, neighbor_limit, voxel_radius);
			if (static_cast<int>(neighbors.size()) < required_neighbors)
			{
				return;
			}

			Eigen::Vector3f mean = Eigen::Vector3f::Zero();
			for (const auto& [neighbor_index, _] : neighbors)
			{
				mean += mPoints[neighbor_index].position;
			}
			mean /= static_cast<float>(neighbors.size());

			Eigen::Matrix3f covariance = Eigen::Matrix3f::Zero();
			for (const auto& [neighbor_index, _] : neighbors)
			{
				const Eigen::Vector3f offset = mPoints[neighbor_index].position - mean;
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
			point.covariance = covariance * inverse_norm;
			point.covariance_valid = true;
		});
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
				const auto found = mVoxelMap.find(VoxelKey{i + dx, j + dy, k + dz});
				if (found == mVoxelMap.end())
				{
					continue;
				}

				const auto [start, count] = mVoxels[found->second];
				for (int offset = 0; offset < count; ++offset)
				{
					const int point_index = start + offset;
                    if (const float squared_distance = (query - mPoints[point_index].position).squaredNorm();
                        squared_distance < nearest.squared_distance)
					{
						nearest = Neighbor{point_index, squared_distance};
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

	neighbors.reserve(std::min(static_cast<std::size_t>(max_neighbors), mPoints.size()));
	const auto [i, j, k] = pointToVoxel(query);
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

				const auto [start, count] = mVoxels[found->second];
				for (int offset = 0; offset < count; ++offset)
				{
					const int point_index = start + offset;
					neighbors.push_back(Neighbor{point_index, (query - mPoints[point_index].position).squaredNorm()});
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
    OccupiedVoxels occupied_voxels;
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

SparseVoxel::VoxelPointLayout SparseVoxel::makeVoxelPointLayout(const std::vector<SparsePoint>& points)
{
    VoxelPointLayout layout;
    layout.points.reserve(points.size());
    std::vector<int> original_indices(points.size());
    std::iota(original_indices.begin(), original_indices.end(), 0);
    std::ranges::sort(original_indices, [&](const int first, const int second)
                          {
                              if (points[first].key == points[second].key)
                              {
                                  return first < second;
                              }
                              return points[first].key < points[second].key;
                          });

    VoxelKey current_key;
    bool have_current_key = false;
    for (int index = 0; index < static_cast<int>(original_indices.size()); ++index)
    {
        const auto& [position, key] = points[original_indices[index]];
        layout.points.push_back(PointWithCovariance{.position = position});
        if (!have_current_key || key != current_key)
        {
            layout.voxels.push_back(VoxelEntry{index, 1});
            layout.voxel_keys.push_back(key);
            current_key = key;
            have_current_key = true;
        }
        else
        {
            ++layout.voxels.back().count;
        }
    }
    return layout;
}