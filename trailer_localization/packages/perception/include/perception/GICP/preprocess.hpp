#pragma once
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
// #include <algorithm>
#include <numeric>


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

struct PointWithCovariance
{
    Eigen::Vector3f position{Eigen::Vector3f::Zero()};
    Eigen::Matrix3f covariance{Eigen::Matrix3f::Identity()};
    bool covariance_valid{false};
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

static VoxelKey pointToVoxel(const Eigen::Vector3f& point, const float voxel_size)
{
    return VoxelKey{static_cast<int>(std::floor(point.x() / voxel_size)),
                    static_cast<int>(std::floor(point.y() / voxel_size)),
                    static_cast<int>(std::floor(point.z() / voxel_size))};
}

struct SparsePoint
{
    Eigen::Vector3f position{Eigen::Vector3f::Zero()};
    VoxelKey key{};
};

class Preprocesser
{
public:
    explicit Preprocesser(const float voxel_size, const int max_points_in_cell) :
        mMinSpace(0.1f * voxel_size), mMaxPoints(max_points_in_cell), mVoxelSize(voxel_size)
    {
    }

    void setConfig(const float voxel_size, const int max_points_in_cell)
    {
        mMinSpace = 0.1f * voxel_size;
        mMaxPoints = max_points_in_cell;
        mVoxelSize = voxel_size;
    }

    VoxelPointLayout process(const pcl::PointCloud<pcl::PointXYZ>& cloud) const
    {
        return makeVoxelPointLayout(makeSparseCloud(cloud));
    }

    std::vector<SparsePoint> makeSparseCloud(const pcl::PointCloud<pcl::PointXYZ>& cloud) const
    {
        std::vector<SparsePoint> sparse_cloud;
        sparse_cloud.reserve(cloud.size());
        OccupiedVoxels occupied_voxels;
        const int max_points_per_voxel = std::max(1, mMaxPoints);
        const float min_spacing_square = std::pow(std::max(0.001f, mMinSpace), 2.0f);

        for (const auto& pcl_point : cloud)
        {
            if (!std::isfinite(pcl_point.x) || !std::isfinite(pcl_point.y) || !std::isfinite(pcl_point.z))
            {
                continue;
            }

            SparsePoint point;
            point.position = Eigen::Vector3f(pcl_point.x, pcl_point.y, pcl_point.z);
            point.key = pointToVoxel(point.position, mVoxelSize);

            auto& voxel_points = occupied_voxels[point.key];
            if (static_cast<int>(voxel_points.size()) >= max_points_per_voxel)
            {
                continue;
            }

            if (const bool too_close = std::ranges::any_of(std::as_const(voxel_points),
                [&](const std::size_t index)
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

    static VoxelPointLayout makeVoxelPointLayout(const std::vector<SparsePoint>& points)
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

private:
    float mMinSpace;
    int mMaxPoints;
    float mVoxelSize;
};