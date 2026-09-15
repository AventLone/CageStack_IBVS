#pragma once
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include "perception/LIO/SparseVoxel.h"
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

        // Cauchy 鲁棒核尺度 s，作用于马氏误差 e：权重= 1 / (1 + e/s^2)，不是直接的欧氏距离阈值。
        // 正值越小越抑制大残差，但也可能削弱有效约束；越大越接近普通 GICP；<= 0 禁用鲁棒降权。
        float cauchy_kernel_scale{0.3f};

        // 调用方 TrailerLocalization 的结果验收上限，单位 m^2；不参与 GICP 求解或收敛判定。
        // fitness 是有效约束的未加权欧氏残差平方均值，不是 RMSE；0.01 对应 RMSE 0.1 m。
        // 增大放宽验收，减小更严格；当前统计使用最近一次已评估位姿的残差，并非更新后重新评估。
        float max_fitness_score{0.01f};

        // 调用方验收所需的最少有效约束数，不是原始最近邻命中数，也不控制求解器迭代。
        // 增大降低少量匹配被接受的风险，但更易拒绝稀疏扫描；减小更宽松，不保证几何约束充分。
        std::size_t min_correspondences{1000};

        // 求解 (H + lambda*I) * delta = -g 的固定阻尼，建议 > 0；不是自适应 LM 阻尼。
        // 增大通常使更新更保守、改善病态系统，但可能减慢收敛；减小更激进，也更易受退化和噪声影响。
        float damping_factor{1.0e-3f};

        // 最大迭代轮数，正常使用应 > 0；增大允许更多更新但增加计算预算，减小更快但可能未充分对齐。
        // CPU 最多执行这么多轮；达到收敛阈值或求解失败时，会提前停止。
        int max_iterations{60};

        // SE(3) 增量中平移分量的范数阈值，单位 m，要求 > 0；需与旋转阈值同时满足才停止更新。
        // 增大更早停止、精细程度降低；减小更严格、可能因浮点精度或噪声持续迭代；1e-6 m 为 1 微米。
        float convergence_translation{1.0e-5f};

        // SE(3) 增量中旋转向量的范数阈值，单位 rad，要求 > 0；增大更早停止，减小更严格。
        // 1e-6 rad 约为 0.000057 度。阈值仅判断更新量，不保证配准正确或残差足够小。
        float convergence_rotation{1.0e-5f};
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

    void initializeTarget(const pcl::PointCloud<pcl::PointXYZ>& target)
    {
        auto target_index = std::make_unique<SparseVoxel>(mSparseVoxelConfig);
        target_index->initialize(target);
        mTarget = std::move(target_index);
    }

    void insertTargetPoints(const pcl::PointCloud<pcl::PointXYZ>& points)
    {
        if (points.empty())
        {
            return;
        }
        if (!hasTarget())
        {
            initializeTarget(points);
            return;
        }
        mTarget->insert(points);
    }


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
    std::unique_ptr<SparseVoxel> mTarget;

    void findCorrespondences(const std::vector<const PointWithCovariance*>& source, const SparseVoxel& target,
                             const Sophus::SE3d& source_to_target, std::vector<Correspondence>& correspondences) const;

    bool buildAndSolve(const std::vector<const PointWithCovariance*>& source, const std::vector<Correspondence>& correspondences,
                       Sophus::SE3d& source_to_target, std::size_t& num_correspondences,
                       double& fitness_score, Sophus::SE3d::Tangent& left_increment) const;
};