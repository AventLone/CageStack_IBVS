#pragma once
#include <cstddef>
#include <limits>
#include <memory>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace perception::lio
{
struct SparsityAwareGICPConfig
{
    float voxel_size{0.20f};
    float min_point_spacing{0.04f};
    int max_points_per_voxel{24};
    int adjacent_voxels{2};
    int covariance_voxel_radius{2};
    int min_covariance_neighbors{8};
    int max_covariance_neighbors{20};
    float covariance_regularization{1.0e-3f};
    float max_correspondence_distance{0.45f};
    float cauchy_kernel_scale{0.30f};
    float max_fitness_score{0.10f};
    std::size_t min_correspondences{80};
    std::size_t max_target_voxels{20000};
    float damping_factor{1.0e-4f};
    int max_iterations{30};
    float convergence_translation{1.0e-4f};
    float convergence_rotation{1.0e-4f};
    bool constrain_to_se2{false};
};

struct SparsityAwareGICPResult
{
    bool converged{false};
    int iterations{0};
    std::size_t num_source_points{0};
    std::size_t num_target_points{0};
    std::size_t num_correspondences{0};
    float fitness_score{std::numeric_limits<float>::infinity()};
    Eigen::Isometry3f transform{Eigen::Isometry3f::Identity()};
};

class SparsityAwareGICP
{
public:
    explicit SparsityAwareGICP(const SparsityAwareGICPConfig& config = {});
    ~SparsityAwareGICP();

    SparsityAwareGICP(const SparsityAwareGICP&) = delete;
    SparsityAwareGICP& operator=(const SparsityAwareGICP&) = delete;
    SparsityAwareGICP(SparsityAwareGICP&&) noexcept;
    SparsityAwareGICP& operator=(SparsityAwareGICP&&) noexcept;

    [[nodiscard]] const SparsityAwareGICPConfig& config() const noexcept;
    void setConfig(const SparsityAwareGICPConfig& config) noexcept;
    void initializeTarget(const pcl::PointCloud<pcl::PointXYZ>& target);
    void insertTargetPoints(const pcl::PointCloud<pcl::PointXYZ>& points);
    void clearTarget() noexcept;
    [[nodiscard]] bool hasTarget() const noexcept;

    [[nodiscard]] SparsityAwareGICPResult align(const pcl::PointCloud<pcl::PointXYZ>& source,
                                                const Eigen::Isometry3f& initial_guess) const;

private:
    struct TargetCache;

    SparsityAwareGICPConfig mConfig;
    std::unique_ptr<TargetCache> mTarget;
};
} // namespace perception::lio