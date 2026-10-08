#pragma once
#include <cstddef>
#include <deque>
#include <limits>
#include <boost/container/static_vector.hpp>
#include <boost/unordered/unordered_flat_map.hpp>
#include <vector>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

struct PointWithCovariance
{
    Eigen::Vector3f position;
    Eigen::Matrix3f covariance;
    bool covariance_valid{false};
};

struct Correspondence
{
    const PointWithCovariance* target{nullptr};
    Eigen::Vector3d transformed_position{Eigen::Vector3d::Zero()};
};

class SparseVoxel
{
public:
    static constexpr float VOXEL_SIZE = 0.5;
    static constexpr int MIN_COVARIANCE_NEIGHBORS = 8;
    static constexpr int MAX_COVARIANCE_NEIGHBORS = 36;

	struct Neighbor
	{
        const PointWithCovariance* point{nullptr};
		float squared_distance{std::numeric_limits<float>::infinity()};
	};

    SparseVoxel() = default;

    bool insert(const pcl::PointCloud<pcl::PointXYZ>& cloud);

    void clear() noexcept
    {
        mOccupiedVoxels.clear();
        mCells.clear();
        mPointCount = 0;
    }

    bool empty() const noexcept
    {
        return mPointCount == 0;
    }

    std::size_t pointCount() const noexcept
    {
        return mPointCount;
    }

    std::size_t voxelCount() const noexcept
    {
        return mOccupiedVoxels.size();
    }

    std::vector<const PointWithCovariance*> points() const;

	Neighbor nearestNeighbor(const Eigen::Vector3f& query, float max_distance) const;

	std::vector<Neighbor> nearestNeighbors(const Eigen::Vector3f& query, int max_neighbors) const;

private:
    struct VoxelKey
    {
        int i{0};
        int j{0};
        int k{0};

        bool operator==(const VoxelKey& other) const noexcept
        {
            return i == other.i && j == other.j && k == other.k;
        }

        friend std::size_t hash_value(const VoxelKey& key) noexcept
        {
            std::size_t seed = 0;
            boost::hash_combine(seed, key.i);
            boost::hash_combine(seed, key.j);
            boost::hash_combine(seed, key.k);
            return seed;
        }
    };

    struct VoxelCell
    {
        static constexpr int CAPACITY = 36;
        static constexpr float MIN_DIST = 0.03f;
        static constexpr float MIN_DIST_SQUARE = MIN_DIST * MIN_DIST;

        using Points = boost::container::static_vector<PointWithCovariance, CAPACITY>;

        PointWithCovariance* add(const Eigen::Vector3f& position)
        {
            if (points.size() == CAPACITY)
            {
                return nullptr;
            }

            if (const bool too_close = std::ranges::any_of(points, [&](const PointWithCovariance& p)
            {
                return (p.position - position).squaredNorm() < MIN_DIST_SQUARE;
            }); too_close)
            {
                return nullptr;
            }

            points.emplace_back();
            points.back().position = position;

            return &points.back();
        }

        Points points;
    };

    using OccupiedVoxels = boost::unordered_flat_map<VoxelKey, std::size_t>;

    static VoxelKey pointToVoxel(const Eigen::Vector3f& point)
    {
        return VoxelKey{static_cast<int>(std::floor(point.x() / VOXEL_SIZE)),
                        static_cast<int>(std::floor(point.y() / VOXEL_SIZE)),
                        static_cast<int>(std::floor(point.z() / VOXEL_SIZE))};
    }

    OccupiedVoxels mOccupiedVoxels;
    std::deque<VoxelCell> mCells;
	std::size_t mPointCount{0};

    void estimateCovariances(const std::vector<PointWithCovariance*>& points) const;
};
