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
    int target_index{-1};
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
		int point_index{-1};
		float squared_distance{std::numeric_limits<float>::infinity()};
	};

	explicit SparseVoxel(const Config& config) : mConfig(config)
    {
        if (!std::isfinite(config.voxel_size) || config.voxel_size <= 0.0f)
        {
            throw std::invalid_argument("voxel_size must be finite and positive");
        }
    }

    void initialize(const pcl::PointCloud<pcl::PointXYZ>& cloud)
    {
        initializeLayout(makeVoxelPointLayout(makeSparseCloud(cloud)));
	    estimateCovariances();
    }

    bool insert(const pcl::PointCloud<pcl::PointXYZ>& cloud);

    void clear() noexcept
    {
        mPoints.clear();
        mVoxels.clear();
        mVoxelKeys.clear();
        mVoxelMap.clear();
    }

    bool empty() const noexcept
    {
        return mPoints.empty();
    }

    std::size_t pointCount() const noexcept
    {
        return mPoints.size();
    }

    std::size_t voxelCount() const noexcept
    {
        return mVoxels.size();
    }

    const std::vector<PointWithCovariance>& points() const noexcept
    {
        return mPoints;
    }



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

    struct VoxelEntry
    {
        int start{0};
        int count{0};
    };

    struct VoxelPointLayout
    {
        std::vector<PointWithCovariance> points;
        std::vector<VoxelEntry> voxels;
        std::vector<VoxelKey> voxel_keys;
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

    using OccupiedVoxels = std::unordered_map<VoxelKey, std::vector<std::size_t>, VoxelKeyHash>;

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

	// const float mVoxelSize, mMinSpace;
 //    const int mMaxPoints;
 //    const std::size_t mMaxVoxels;
    const Config mConfig;
	std::vector<PointWithCovariance> mPoints;
	std::vector<VoxelEntry> mVoxels;
	std::vector<VoxelKey> mVoxelKeys;
	std::unordered_map<VoxelKey, int, VoxelKeyHash> mVoxelMap;

    void initializeLayout(VoxelPointLayout layout)
    {
        if (mConfig.max_voxels_num == 0 || layout.voxels.size() > mConfig.max_voxels_num)
        {
            throw std::invalid_argument("Initial target exceeds max_target_voxels or the voxel budget is zero");
        }

        mPoints = std::move(layout.points);
        mVoxels = std::move(layout.voxels);
        mVoxelKeys = std::move(layout.voxel_keys);
        rebuildVoxelMap();
    }

    void rebuildVoxelMap()
    {
        mVoxelMap.clear();
        mVoxelMap.reserve(mVoxelKeys.size());
        for (int index = 0; index < static_cast<int>(mVoxelKeys.size()); ++index)
        {
            mVoxelMap.emplace(mVoxelKeys[index], index);
        }
    }

    void estimateCovariances();

    std::vector<SparsePoint> makeSparseCloud(const pcl::PointCloud<pcl::PointXYZ>& cloud) const;

    static VoxelPointLayout makeVoxelPointLayout(const std::vector<SparsePoint>& points);
};
