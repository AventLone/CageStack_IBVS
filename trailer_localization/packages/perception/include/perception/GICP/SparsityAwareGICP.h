#pragma once
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include "perception/GICP/preprocess.hpp"
#include <unordered_set>
#include <sophus/se3.hpp>

class SparsityAwareGICP
{
public:
    struct Config
    {
        // 体素边长，单位 m，必须 > 0；同时用于稀疏化、协方差邻域和对应搜索。
        // 增大通常减少保留点、损失细节，但固定体素半径下的物理搜索范围变大；减小则相反。
        float voxel_size{0.05f};

        // 每体素最多保留的点数，按输入顺序筛选，实际至少为 1。
        // 增大保留更多局部几何、增加计算和内存开销；减小加快处理但可能使邻域过稀。
        int max_points_per_voxel{26};

        // 有效协方差所需的最少邻居数，包含点自身，实际至少为 3。
        // 增大对稀疏邻域更严格，更多点回退为无效协方差；减小更易获得协方差，但统计可靠性降低。
        int min_covariance_neighbors{8};

        // 计算协方差时最多使用的最近邻数 K；实际 K=max(3, min_covariance_neighbors, 本值)。
        // CPU 实现没有固定的 K 上限。增大通常更平滑、更慢；减小更局部、对噪声敏感。
        int max_covariance_neighbors{max_points_per_voxel * 3};

        // 原始样本协方差对角线正则项，单位 m^2，实际至少为 1e-6，之后还会做逆矩阵范数归一化。
        // 增大改善求逆稳定性，但弱化平面/边缘的方向性；减小保留方向性，但退化邻域更易数值不钱
        float covariance_regularization{1.0e-3f};

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

        // 目标地图体素容量预算，要求 > 0；不是点数上限。
        // 初始化体素数超过预算或预算为 0 时抛出 std::invalid_argument，已有目标保持不变。
        std::size_t max_target_voxels{999999};

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
        float fitness_score{std::numeric_limits<float>::infinity()};
        Eigen::Isometry3f transform{Eigen::Isometry3f::Identity()};
    };

    SparsityAwareGICP() : SparsityAwareGICP(Config{})
    {}

    explicit SparsityAwareGICP(const Config& config)
        : mConfig(config), mProcesser(mConfig.voxel_size, mConfig.max_points_per_voxel)
    {
    }

    SparsityAwareGICP(SparsityAwareGICP&&) noexcept = default;
    SparsityAwareGICP& operator=(SparsityAwareGICP&&) noexcept = default;

    SparsityAwareGICP(const SparsityAwareGICP&) = delete;
    SparsityAwareGICP& operator=(const SparsityAwareGICP&) = delete;

    ~SparsityAwareGICP() = default;


    [[nodiscard]] const Config& config() const noexcept
    {
        return mConfig;
    }

    void setConfig(const Config& config) noexcept
    {
        mConfig = config;
        mProcesser.setConfig(mConfig.voxel_size, mConfig.max_points_per_voxel);
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

    Result align(const pcl::PointCloud<pcl::PointXYZ>& source, const Eigen::Isometry3f& initial_guess) const;

private:
    struct TargetCache
    {
        std::unordered_set<VoxelKey, VoxelKeyHash> occupied_voxels;
        std::vector<PointWithCovariance> points;
        std::vector<VoxelEntry> voxels;
        std::unordered_map<VoxelKey, int, VoxelKeyHash> voxel_map;

        TargetCache(const VoxelPointLayout& layout, std::vector<PointWithCovariance> estimated_points)
            : points(std::move(estimated_points)), voxels(layout.voxels.begin(), layout.voxels.end())
        {
            occupied_voxels.insert(layout.voxel_keys.begin(), layout.voxel_keys.end());
            voxel_map.reserve(layout.voxel_keys.size());
            for (int index = 0; index < static_cast<int>(layout.voxel_keys.size()); ++index)
            {
                voxel_map.emplace(layout.voxel_keys[index], index);
            }
        }
    };

    struct Correspondence
    {
        int target_index{-1};
        Eigen::Vector3f transformed_position{Eigen::Vector3f::Zero()};
    };

    Config mConfig;
    Preprocesser mProcesser;
    std::unique_ptr<TargetCache> mTarget;

    std::vector<PointWithCovariance> estimateCovariances(VoxelPointLayout layout) const;

    void findCorrespondences(const std::vector<PointWithCovariance>& source, const std::vector<PointWithCovariance>& target,
                             const std::vector<VoxelEntry>& target_voxels,
                             const std::unordered_map<VoxelKey, int, VoxelKeyHash>& target_voxel_map,
                             const Sophus::SE3f& source_to_target,
                             std::vector<Correspondence>& correspondences) const;

    bool buildAndSolve(const std::vector<PointWithCovariance>& source, const std::vector<PointWithCovariance>& target,
                       const std::vector<Correspondence>& correspondences, Sophus::SE3f& source_to_target,
                       std::size_t& num_correspondences, float& fitness_score, Sophus::SE3f::Tangent& left_increment) const;
};