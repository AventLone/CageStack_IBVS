#pragma once
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include "perception/types/SparseVoxel.h"
#include <sophus/se3.hpp>

class GICP
{
public:
    struct Config
    {
        // 体素边长，单位 m，必须 > 0；同时用于稀疏化、协方差邻域和对应搜索。
        // 增大通常减少保留点、损失细节，但固定体素半径下的物理搜索范围变大；减小则相反。
        float voxel_size{0.05f};

        // 最近邻欧氏距离上限，单位 m，要求 > 0；只在 adjacent_voxels 覆盖的体素内查找。
        // 增大放宽匹配但增加误匹配风险；减小更严格、可能无对应点。单独增大不会扩大体素查询范围。
        float max_correspondence_distance{0.5f};

        std::size_t min_correspondences{1000};
        double max_fitness_score{0.01};

        // Cauchy 鲁棒核尺度 s，作用于马氏误差 e：权重= 1 / (1 + e/s^2)，不是直接的欧氏距离阈值。
        // 正值越小越抑制大残差，但也可能削弱有效约束；越大越接近普通 GICP；<= 0 禁用鲁棒降权。
        // double cauchy_kernel_scale{0.3};

        int max_iterations{60};
        double convergence_translation{1.0e-5};   // SE(3) 增量中平移分量的范数阈值，单位 m
        double convergence_rotation{1.0e-5};   // SE(3) 增量中旋转向量的范数阈值，单位 rad
    };

    struct Result
    {
        bool converged{false};
        int iterations{0};
        std::size_t num_source_points{0};
        std::size_t num_target_points{0};
        std::size_t num_correspondences{0};
        double fitness_score{std::numeric_limits<double>::infinity()};
        Eigen::Isometry3d transform{Eigen::Isometry3d::Identity()};
        Eigen::Matrix<double, 6, 6> measure_covariance{Eigen::Matrix<double, 6, 6>::Identity()};
    };

    GICP() : GICP(Config{})
    {
    }

    explicit GICP(const Config& config) : mConfig(config)
    {
        mSparseVoxelConfig.voxel_size = config.voxel_size;
    }

    GICP(GICP&&) noexcept = default;
    GICP& operator=(GICP&&) noexcept = default;

    GICP(const GICP&) = delete;
    GICP& operator=(const GICP&) = delete;

    ~GICP() = default;


    [[nodiscard]] const Config& config() const noexcept
    {
        return mConfig;
    }

    void setConfig(const Config& config) noexcept
    {
        mConfig = config;
        mSparseVoxelConfig.voxel_size = config.voxel_size;
        clearTarget();
    }

    void initializeTarget(const pcl::PointCloud<pcl::PointXYZ>& target);
    void insertTargetPoints(const pcl::PointCloud<pcl::PointXYZ>& points);


    void clearTarget() noexcept
    {
        mTarget.reset();
    }

    bool hasTarget() const noexcept
    {
        return mTarget != nullptr;
    }

    Result align(const pcl::PointCloud<pcl::PointXYZ>& source, const Eigen::Isometry3d& initial_guess) const;

private:
    Config mConfig;
    SparseVoxel::Config mSparseVoxelConfig{};
    SparseVoxel::Ptr mTarget;

    void findCorrespondences(const std::vector<const PointWithCovariance*>& source, const SparseVoxel& target,
                             const Sophus::SE3d& source_to_target, std::vector<Correspondence>& correspondences) const;


    std::optional<std::pair<std::size_t, double>> findCorrespondencesAndSolve(const std::vector<const PointWithCovariance*>& source,
                                                                              Sophus::SE3d& source_to_target,
                                                                              Sophus::SE3d::Tangent& left_increment,
                                                                              Eigen::Matrix<double, 6, 6>& hessian_) const;
};