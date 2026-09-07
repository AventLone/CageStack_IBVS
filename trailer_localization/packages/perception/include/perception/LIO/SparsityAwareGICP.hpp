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
    // 体素边长，单位 m，必须 > 0；同时用于稀疏化、协方差邻域和对应搜索。
    // 增大通常减少保留点、损失细节，但固定体素半径下的物理搜索范围变大；减小则相反。
    float voxel_size{0.05f};

    // 同一体素内保留点的最小间距，单位 m；不检查跨体素点对。
    // 增大可减少点数和计算量，但可能丢失细节；减小保留更多点；<= 0 禁用间距筛选。
    float min_point_spacing{0.01f};

    // 每体素最多保留的点数，按输入顺序筛选，实际至少为 1。
    // 增大保留更多局部几何、增加计算和显存开销；减小加快处理但可能使邻域过稀。
    int max_points_per_voxel{26};

    // 对应搜索沿每个轴扩展的体素数 r，要求 >= 0；查询 (2*r+1)^3 个体素。
    // 增大可覆盖更大的初始位姿误差，但搜索开销和误匹配风险增加；减小更快但易漏匹配。
    int adjacent_voxels{2};

    // 协方差邻域沿每个轴扩展的体素数 r，实际至少为 0。
    // 增大可找到更多候选邻居，但查询量按 (2*r+1)^3 增长，并可能混入不同表面；减小更局部。
    int covariance_voxel_radius{2};

    // 有效协方差所需的最少邻居数，包含点自身，实际至少为 3。
    // 增大对稀疏邻域更严格，更多点回退为无效协方差；减小更易获得协方差，但统计可靠性降低。
    int min_covariance_neighbors{8};

    // 计算协方差时最多使用的最近邻数 K；实际 K=max(3, min_covariance_neighbors, 本值)。
    // 当前 CUDA 实现要求实际 K <= 64，否则抛异常。增大通常更平滑、更慢；减小更局部、对噪声敏感。
    int max_covariance_neighbors{20};

    // 原始样本协方差对角线正则项，单位 m^2，实际至少为 1e-6，之后还会做逆矩阵范数归一化。
    // 增大改善求逆稳定性，但弱化平面/边缘的方向性；减小保留方向性，但退化邻域更易数值不稳。
    float covariance_regularization{1.0e-3f};

    // 最近邻欧氏距离上限，单位 m，要求 > 0；只在 adjacent_voxels 覆盖的体素内查找。
    // 增大放宽匹配但增加误匹配风险；减小更严格、可能无对应点。单独增大不会扩大体素查询范围。
    float max_correspondence_distance{0.2f};

    // Cauchy 鲁棒核尺度 s，作用于马氏误差 e：权重=1/(1+e/s^2)，不是直接的欧氏距离阈值。
    // 正值越小越抑制大残差，但也可能削弱有效约束；越大越接近普通 GICP；<= 0 禁用鲁棒降权。
    float cauchy_kernel_scale{0.3f};

    // 调用方 TrailerLocalization 的结果验收上限，单位 m^2；不参与 GICP 求解或收敛判定。
    // fitness 是有效约束的未加权欧氏残差平方均值，不是 RMSE；0.01 对应 RMSE 0.1 m。
    // 增大放宽验收，减小更严格；当前统计使用最近一次已评估位姿的残差，并非更新后重新评估。
    float max_fitness_score{0.01f};

    // 调用方验收所需的最少有效约束数，不是原始最近邻命中数，也不控制求解器迭代。
    // 增大降低少量匹配被接受的风险，但更易拒绝稀疏扫描；减小更宽松，不保证几何约束充分。
    std::size_t min_correspondences{80};

    // 目标地图体素容量预算，要求 > 0；不是点数上限，也没有自动淘汰旧体素。
    // 增大允许地图增长，但占用更多显存；减小更早停止增量插入，新增体素超出剩余容量时整批拒绝。
    // 当前初始化不会按此值裁剪输入，需确保初始地图不超预算，以免超过固定哈希表容量。
    std::size_t max_target_voxels{20000};

    // 求解 (H + lambda*I)*delta = -g 的固定阻尼，建议 > 0；不是自适应 LM 阻尼。
    // 增大通常使更新更保守、改善病态系统，但可能减慢收敛；减小更激进，也更易受退化和噪声影响。
    float damping_factor{1.0e-4f};

    // 最大迭代轮数，正常使用应 > 0；增大允许更多更新但增加计算预算，减小更快但可能未充分对齐。
    // 当前 CPU 固定提交这么多轮，GPU 求解停止后对应搜索和 Hessian 构建仍会执行，因此仍有额外开销。
    int max_iterations{50};

    // SE(3) 增量中平移分量的范数阈值，单位 m，要求 > 0；需与旋转阈值同时满足才停止更新。
    // 增大更早停止、精细程度降低；减小更严格、可能因浮点精度或噪声持续迭代；1e-6 m 为 1 微米。
    float convergence_translation{1.0e-6f};

    // SE(3) 增量中旋转向量的范数阈值，单位 rad，要求 > 0；增大更早停止，减小更严格。
    // 1e-6 rad 约为 0.000057 度。阈值仅判断更新量，不保证配准正确或残差足够小。
    // 注意：当前 result.converged 还会接受“有成功迭代且 fitness 有限”，不等价于满足这两个阈值。
    float convergence_rotation{1.0e-6f};
};

struct SparsityAwareGICPResult
{
    bool converged{false};
    int iterations{0};
    std::size_t num_source_points{0};
    std::size_t num_target_points{0};
    std::size_t num_raw_correspondences{0};
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