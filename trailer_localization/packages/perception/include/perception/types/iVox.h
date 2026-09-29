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
#include <boost/container/static_vector.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

class IVox
{
    static constexpr std::size_t VOXEL_CAPACITY = 1000000;
    static constexpr float VOXEL_SIZE = 0.5f;
    static constexpr float INVERSE_VOXEL_SIZE = 1.0f / VOXEL_SIZE;

public:
    using Ptr = std::unique_ptr<IVox>;

    // struct Config
    // {
    //     float voxel_size{0.5f};
    //
    //     // Faster-LIO limits the number of occupied grids, rather than the number of points in each voxel.
    //     std::size_t capacity{1000000};
    // };

    struct Neighbor
    {
        const Eigen::Vector3f* point{nullptr};
        float squared_distance{std::numeric_limits<float>::infinity()};
    };

    // explicit IVox(const Config& config) : mConfig(config), mInverseVoxelSize(1.0f / config.voxel_size)
    // {
    // }
    IVox() = default;

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

    struct VoxelCell
    {
        static constexpr int CAPACITY = 36;
        static constexpr float MIN_DIST = 0.03f;
        static constexpr float MIN_DIST_SQUARE = MIN_DIST * MIN_DIST;

        using Points = boost::container::static_vector<Eigen::Vector3f, CAPACITY>;

        void add(const Eigen::Vector3f& point)
        {
            if (const bool too_close = std::ranges::any_of(points, [&](const Eigen::Vector3f& p)
            {
                return (p - point).squaredNorm() < MIN_DIST_SQUARE;
            }); too_close)
            {
                return;
            }

            if (points.size() == CAPACITY)
            {
                return;
            }

            points.push_back(point);
        }

        [[nodiscard]] const Points& getPoints() const
        {
            return points;
        }

    private:
        Points points;
    };

    using VoxelCache = std::list<std::pair<VoxelKey, VoxelCell>>;
    using VoxelIterator = VoxelCache::iterator;
    using VoxelMap = boost::unordered_flat_map<VoxelKey, VoxelIterator>;

    VoxelMap mVoxelMap;
    VoxelCache mVoxelCache;
    std::size_t mPointCount{0};

    static VoxelKey pointToVoxel(const Eigen::Vector3f& point);
};


struct LocalPlane
{
    Eigen::Vector3f center{Eigen::Vector3f::Zero()};
    Eigen::Vector3f normal{Eigen::Vector3f::Zero()};
    float normal_variance{0.0f};
    bool valid{false};
};


LocalPlane estimatePlane(const std::vector<IVox::Neighbor>& neighbors, float distance_threshold);