#include "free_space_detection/tools/feature_detect_3d.h"
#include <boost/unordered/unordered_flat_set.hpp>
#include <memory>

namespace feature3d
{
float detectFloor(const pcl::PointCloud<pcl::PointXYZ>& src, pcl::PointCloud<pcl::PointXYZ>::Ptr ground_cloud,
                  pcl::PointCloud<pcl::PointXYZ>::Ptr non_ground_cloud)
{
    using CellSet = boost::unordered_flat_set<std::uint64_t>;

    constexpr float xy_cell_size = 0.15;
    constexpr float coarse_bin_size = 0.1f;
    constexpr float fine_bin_size = 0.01f;
    constexpr float refine_half_range = 0.15f;
    constexpr float candidate_band = 0.20f; // ±20 cm
    constexpr float distance_threshold = 0.03f; // 3 cm

    const auto xyCellKey = [](const float x, const float y, const float resolution) -> std::uint64_t
        {
            const auto ix = static_cast<std::int32_t>(std::floor(x / resolution));
            const auto iy = static_cast<std::int32_t>(std::floor(y / resolution));
            return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(ix)) << 32) | static_cast<std::uint32_t>(iy);
        };

    if (src.size() < 10)
    {
        return 0.0f;
    }

    // 1. Determine the valid Z range.
    float min_z = std::numeric_limits<float>::infinity();
    float max_z = -std::numeric_limits<float>::infinity();

    std::size_t finite_count = 0;

    for (const auto& p : src)
    {
        min_z = std::min(min_z, p.z);
        max_z = std::max(max_z, p.z);
        ++finite_count;
    }

    if (finite_count == 0)
    {
        return 0.0f;
    }

    // 2. Coarse histogram: count unique XY cells per Z bin.
    const std::size_t coarse_bin_count = static_cast<std::size_t>(std::floor((max_z - min_z) / coarse_bin_size)) + 1;
    if (coarse_bin_count == 0 || coarse_bin_count > 1'000'000) // Protect against unreasonable allocations.
    {
        return 0.0f;
    }

    std::vector<CellSet> coarse_cells(coarse_bin_count);
    for (const auto& p : src)
    {
        if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z) || p.z < min_z || p.z > max_z)
        {
            continue;
        }

        const auto iz = static_cast<std::size_t>(std::floor((p.z - min_z) / coarse_bin_size));
        if (iz >= coarse_bin_count)
        {
            continue;
        }

        coarse_cells[iz].insert(xyCellKey(p.x, p.y, xy_cell_size));
    }

    // Smooth the histogram using adjacent bins.
    std::size_t best_coarse_bin = 0;
    std::size_t best_coarse_score = 0;

    for (std::size_t i = 0; i < coarse_bin_count; ++i)
    {
        std::size_t score = coarse_cells[i].size();

        if (i > 0)
        {
            score += coarse_cells[i - 1].size();
        }

        if (i + 1 < coarse_bin_count)
        {
            score += coarse_cells[i + 1].size();
        }

        if (score > best_coarse_score)
        {
            best_coarse_score = score;
            best_coarse_bin = i;
        }
    }

    if (best_coarse_score == 0)
    {
        return 0.0f;
    }

    const float coarse_height = min_z + (static_cast<float>(best_coarse_bin) + 0.5f) * coarse_bin_size;

    // 3. Fine histogram around the coarse estimate.
    const float fine_z_min = coarse_height - refine_half_range;
    const float fine_z_max = coarse_height + refine_half_range;

    const auto fine_bin_count = std::max<std::size_t>(1, std::ceil((fine_z_max - fine_z_min) / fine_bin_size));

    std::vector<CellSet> fine_cells(fine_bin_count);

    for (const auto& p : src)
    {
        if (p.z < min_z || p.z > max_z || p.z < fine_z_min || p.z > fine_z_max)
        {
            continue;
        }

        auto iz = static_cast<std::size_t>(std::floor((p.z - fine_z_min) / fine_bin_size));

        if (iz >= fine_bin_count)
        {
            if (p.z <= fine_z_max)
            {
                iz = fine_bin_count - 1;
            }
            else
            {
                continue;
            }
        }
        fine_cells[iz].insert(xyCellKey(p.x, p.y, xy_cell_size));
    }

    std::size_t best_fine_bin = 0;
    std::size_t best_fine_score = 0;

    for (std::size_t i = 0; i < fine_bin_count; ++i)
    {
        if (fine_cells[i].size() > best_fine_score)
        {
            best_fine_score = fine_cells[i].size();
            best_fine_bin = i;
        }
    }

    if (best_fine_score == 0)
    {
        return 0.0f;
    }

    const float fine_height = fine_z_min + (static_cast<float>(best_fine_bin) + 0.5f) * fine_bin_size;

    // 4. Collect ground candidate points.
    auto candidates = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    candidates->points.reserve(src.size());

    for (const auto& p : src)
    {
        if (p.z < min_z || p.z > max_z)
        {
            continue;
        }

        if (std::abs(p.z - fine_height) <= candidate_band)
        {
            candidates->points.push_back(p);
        }
    }

    candidates->width = static_cast<std::uint32_t>(candidates->size());
    candidates->height = 1;
    candidates->is_dense = true;

    if (candidates->size() < 100)
    {
        return 0.0f;
    }

    // 5. Fit a plane using constrained RANSAC.
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PERPENDICULAR_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setMaxIterations(100);
    seg.setDistanceThreshold(0.03);
    seg.setAxis(Eigen::Vector3f(0.0f, 0.0f, 1.0f));
    seg.setEpsAngle(10.0 * M_PI / 180.0);
    seg.setInputCloud(candidates);

    pcl::PointIndices inliers;
    pcl::ModelCoefficients coefficients;
    seg.segment(inliers, coefficients);
    if (inliers.indices.size() < 100 || coefficients.values.size() < 4)
    {
        return 0.0f;
    }

    // Normalize plane: ax + by + cz + d = 0.
    Eigen::Vector4f plane(coefficients.values[0], coefficients.values[1], coefficients.values[2], coefficients.values[3]);
    const float norm = plane.head<3>().norm();
    if (!std::isfinite(norm) || norm < 1e-6f)
    {
        return 0.0f;
    }

    plane /= norm;

    if (plane.z() < 0.0f)
    {
        plane = -plane;
    }

    // 6. Verify spatial coverage of the fitted ground.
    CellSet ground_cells;
    ground_cells.reserve(inliers.indices.size());
    for (const int idx : inliers.indices)
    {
        if (idx < 0 || static_cast<std::size_t>(idx) >= candidates->size())
        {
            continue;
        }

        const auto& p = candidates->points[idx];

        ground_cells.insert(xyCellKey(p.x, p.y, xy_cell_size));
    }

    if (ground_cells.size() < 10)
    {
        return 0.0f;
    }

    // 7. Classify the original map against the fitted plane.
    ground_cloud->points.reserve(src.size());
    if (non_ground_cloud)
    {
        non_ground_cloud->points.reserve(src.size());
    }

    for (const auto& p : src)
    {
        if (const float distance = plane[0] * p.x + plane[1] * p.y + plane[2] * p.z + plane[3];
            std::abs(distance) <= distance_threshold)
        {
            ground_cloud->points.push_back(p);
        }
        else if (non_ground_cloud)
        {
            non_ground_cloud->points.push_back(p);
        }
    }

    return -plane[3] / plane.z();
}
}