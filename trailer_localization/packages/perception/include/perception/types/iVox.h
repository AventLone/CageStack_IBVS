#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <list>
#include <memory>
#include <vector>
#include <Eigen/Core>
#include <boost/unordered/unordered_flat_map.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

class IVox
{
public:
    using Ptr = std::unique_ptr<IVox>;
    using Points = std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>;

    struct Config
    {
        float voxel_size{1.0f};

        // Faster-LIO limits the number of occupied grids, rather than the number of points in each voxel.
        std::size_t capacity{1000000};
    };

    struct Neighbor
    {
        const Eigen::Vector3f* point{nullptr};
        float squared_distance{std::numeric_limits<float>::infinity()};
    };

    explicit IVox(const Config& config) : mConfig(config), mInverseVoxelSize(1.0f / config.voxel_size)
    {
    }

    void initialize(const pcl::PointCloud<pcl::PointXYZ>& scan)
    {
        clear();
        insert(scan);
    }

    bool insert(const pcl::PointCloud<pcl::PointXYZ>& scan);

    void clear() noexcept
    {
        mVoxelMap.clear();
        mVoxelCache.clear();
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
        return mVoxelMap.size();
    }

    std::vector<Neighbor> nearestNeighbors(const Eigen::Vector3f& query, int max_neighbors, float max_distance) const;

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

    struct Voxel
    {
        Points points;
    };

    using VoxelCache = std::list<std::pair<VoxelKey, Voxel>>;
    using VoxelIterator = VoxelCache::iterator;
    using VoxelMap = boost::unordered_flat_map<VoxelKey, VoxelIterator>;

    Config mConfig;

    float mInverseVoxelSize{1.0f};

    VoxelMap mVoxelMap;
    VoxelCache mVoxelCache;

    std::size_t mPointCount{0};

    VoxelKey pointToVoxel(const Eigen::Vector3f& point) const;
};


struct LocalPlane
{
    Eigen::Vector3f center{Eigen::Vector3f::Zero()};
    Eigen::Vector3f normal{Eigen::Vector3f::Zero()};
    float normal_variance{0.0f};
    bool valid{false};
};


LocalPlane estimatePlane(const std::vector<IVox::Neighbor>& neighbors, float distance_threshold);