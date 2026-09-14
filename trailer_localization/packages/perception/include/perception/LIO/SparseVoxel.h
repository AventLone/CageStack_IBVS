#pragma once
#include <cstddef>
#include <limits>
#include <unordered_map>
#include <vector>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

struct PointWithCovariance
{
    Eigen::Vector3f position{Eigen::Vector3f::Zero()};
    Eigen::Matrix3f covariance{Eigen::Matrix3f::Identity()};
    bool covariance_valid{false};
};

struct Correspondence
{
    const PointWithCovariance* target{nullptr};
    Eigen::Vector3f transformed_position{Eigen::Vector3f::Zero()};
};

class SparseVoxel
{
public:
    struct Config
    {
        float voxel_size{0.1f};
        int max_points_per_voxel{26};

        int min_covariance_neighbors{8};
        int max_covariance_neighbors{36};
        float covariance_regularization{1.0e-3f};

        std::size_t max_voxels_num{999999};
    };

	struct Neighbor
	{
        const PointWithCovariance* point{nullptr};
		float squared_distance{std::numeric_limits<float>::infinity()};
	};

	explicit SparseVoxel(const Config& config) : mConfig(config)
    {
        if (!std::isfinite(config.voxel_size) || config.voxel_size <= 0.0f)
        {
            throw std::invalid_argument("voxel_size must be finite and positive");
        }
    }

    void initialize(const pcl::PointCloud<pcl::PointXYZ>& cloud);

    bool insert(const pcl::PointCloud<pcl::PointXYZ>& cloud);

    void clear() noexcept
    {
        mOccupiedVoxels.clear();
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

	Neighbor nearestNeighbor(const Eigen::Vector3f& query, float max_distance, int voxel_radius = 1) const;

	std::vector<Neighbor> nearestNeighbors(const Eigen::Vector3f& query, int max_neighbors, int voxel_radius = 1) const;

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

        bool operator<(const VoxelKey& other) const noexcept
        {
            if (i != other.i)
            {
                return i < other.i;
            }
            if (j != other.j)
            {
                return j < other.j;
            }
            return k < other.k;
        }
    };

    struct VoxelKeyHash
    {
        std::size_t operator()(const VoxelKey& key) const noexcept
        {
            const auto x = static_cast<std::size_t>(key.i);
            const auto y = static_cast<std::size_t>(key.j);
            const auto z = static_cast<std::size_t>(key.k);
            return (x * 73856093ULL) ^ (y * 19349663ULL) ^ (z * 83492791ULL);
        }
    };

    using OccupiedVoxels = std::unordered_map<VoxelKey, std::vector<PointWithCovariance>, VoxelKeyHash>;
    using SparsePointIndices = std::unordered_map<VoxelKey, std::vector<std::size_t>, VoxelKeyHash>;

    VoxelKey pointToVoxel(const Eigen::Vector3f& point) const
    {
        return VoxelKey{static_cast<int>(std::floor(point.x() / mConfig.voxel_size)),
                        static_cast<int>(std::floor(point.y() / mConfig.voxel_size)),
                        static_cast<int>(std::floor(point.z() / mConfig.voxel_size))};
    }

    struct SparsePoint
    {
        Eigen::Vector3f position{Eigen::Vector3f::Zero()};
        VoxelKey key{};
    };

    const Config mConfig;
    OccupiedVoxels mOccupiedVoxels;
	std::size_t mPointCount{0};

    void estimateCovariances();
    void estimateCovariances(const std::vector<PointWithCovariance*>& points) const;

    std::vector<SparsePoint> makeSparseCloud(const pcl::PointCloud<pcl::PointXYZ>& cloud) const;
};
