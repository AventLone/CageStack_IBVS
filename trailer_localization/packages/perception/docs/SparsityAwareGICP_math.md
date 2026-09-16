# GICP Mathematical Principles

本文档解释 `GICP` 使用的稀疏 GICP 原理与数学约定。实现和数据结构细节请参考 [SparsityAwareGICP_algorithm.md](SparsityAwareGICP_algorithm.md)。

## 1 问题定义

给定源点云 $\mathcal{S}=\{p_i\}$ 与目标点云 $\mathcal{T}=\{q_j\}$，配准的目标是求刚体变换：

$$
T = (R, t) \in SE(3)
$$

使源点变换到目标坐标系后与目标表面一致：

$$
T p_i = R p_i + t
$$

当前实现中的 `source_to_target` 就是 $T$，因此结果 `transform` 也是从源点坐标系到目标地图坐标系的变换。

ICP 和 GICP 都需要重复执行两个步骤：

1. 在当前位姿 $T$ 下为每个源点建立目标对应；
2. 固定对应后，求使残差下降的位姿增量。

对应关系本身随 $T$ 离散变化，因此整体问题不是一个单次可解的光滑最小二乘问题，而是交替优化。



## 2 从点到点 ICP 到 GICP

### 2.1 点到点 ICP

普通点到点 ICP 把每个对应视为各方向不确定性相同，最小化欧氏残差：

$$
\min_T \sum_i \|T p_i - q_{c(i)}\|^2
$$

其中 $c(i)$ 是源点 $p_i$ 匹配到的目标点下标。

这种模型没有区分平面内方向和法向方向。例如，墙面上的点沿墙面滑动时应受到较弱约束，但普通 ICP 会将三个方向同等处理。

### 2.2 GICP 的局部高斯模型

GICP 为每个点估计局部协方差：

$$
C_{s,i}, \qquad C_{t,c(i)} \in \mathbb{R}^{3 \times 3}
$$

这里的“点协方差”不是由单个确定点计算出来的，也不表示该点自身是一组样本。对每个中心点 $p_i$，算法从其邻域 $\mathcal{N}(p_i)$ 收集一组附近点，以这组样本估计局部协方差，再将结果附着到中心点 $p_i$ 上。更准确地说，$C_{s,i}$ 是“附着于源点 $p_i$ 的局部邻域协方差”，$C_{t,c(i)}$ 也是同样的目标局部表面描述。

协方差描述该局部表面附近各方向的容许变化。对平面邻域而言，平面内特征值大、法向特征值小；因此精度矩阵会更强地惩罚偏离平面的残差，而对平面内滑动施加较弱约束。对线状邻域，一个主方向特征值大；对角点或体积散布的点，三个方向通常都具有约束。

源点协方差应随当前旋转带到目标坐标系：

$$
R C_{s,i} R^T
$$

若源、目标局部测量误差相互独立，则残差

$$
r_i = T p_i - q_{c(i)}
$$

的协方差为：

$$
C_i = C_{t,c(i)} + R C_{s,i} R^T
$$

其逆矩阵称为精度矩阵：

$$
W_i = C_i^{-1}
$$

于是每个对应的误差从欧氏距离变为马氏距离：

$$
e_i = r_i^T W_i r_i
$$

当 $C_i=I$ 时，$e_i=\|r_i\|^2$，GICP 就退化为点到点 ICP。



## 3 局部协方差如何得到

对每个中心稀疏点 $p_i$，在邻近体素中采集其局部邻域 $\mathcal{N}(p_i)$ 的最多 $N$ 个空间上最近的候选点，并计算这些邻居的均值：

$$
\mu = \frac{1}{N}\sum_{k=1}^{N} p_k
$$

由这些邻域样本得到的协方差随后附着到中心点 $p_i$。样本协方差为：

$$
\Sigma = \frac{1}{N-1}\sum_{k=1}^{N}(p_k-\mu)(p_k-\mu)^T + \lambda I
$$

其中 $\lambda > 0$ 是对角线正则项。

正则项有两个作用：

- 防止严格平面或重复点导致协方差奇异；
- 限制精度矩阵 $\Sigma^{-1}$ 的数值放大。

本实现会拒绝行列式不可靠或逆矩阵范数非有限的协方差。有效协方差还会按逆矩阵范数归一化：

$$
\Sigma \leftarrow \Sigma \cdot \|\Sigma^{-1}\|
$$

该缩放会保留协方差椭球的相对方向性，但改变其绝对尺度，使不同邻域的数值尺度更可比较。它是当前实现的权重设计，而不是 GICP 必需的通用步骤。



## 4 对应关系与稀疏化

每轮优化在当前位姿下计算变换后的源点：

$$
x_i = T p_i
$$

然后在目标地图中选择阈值内的最近点：

$$
c(i) = \arg\min_j \|x_i-q_j\|^2
$$

仅当：

$$
\|x_i-q_{c(i)}\|^2 < d_{\max}^2
$$

时该对应有效。这里 $d_{\max}$ 是 `max_correspondence_distance`。

体素索引只用于加速该最近邻搜索。当前实现只检查变换后源点所在体素及其每轴相邻一层的 27 个目标体素，因此搜索窗口还受 `voxel_size` 限制；单独增大 $d_{\max}$ 并不会扩大所查询的体素范围。

稀疏化减少重复或过密点对对总目标的主导，并限制 CPU 搜索成本。它改变的是用于优化的采样点集，不改变上述残差模型。



## 5 鲁棒 Cauchy 目标

错误对应、遮挡、动态物体和部分重叠会产生大残差。若所有对应都以相同二次代价累加，少量离群点就可能拉偏位姿。当前实现使用 Cauchy 鲁棒核。

令 $s>0$ 为 `cauchy_kernel_scale`，其鲁棒目标可写为：

$$
F(T) = \frac{1}{2}\sum_i s^2 \ln\left(1+\frac{e_i}{s^2}\right)
$$

其中 $e_i$ 是马氏误差。对该目标做迭代重加权最小二乘时，单个对应的权重为：

$$
w_i = \frac{1}{1+e_i/s^2}
$$

误差较小时 $w_i\approx1$，其行为接近普通 GICP；误差远大于 $s^2$ 时，权重接近零，离群对应对更新的影响被抑制。

当 `cauchy_kernel_scale <= 0` 时，代码令 $w_i=1$，即关闭鲁棒降权。



## 6 在 $SE(3)$ 上优化

### 6.1 为什么不能直接相加变换矩阵

刚体变换中的旋转必须保持在 $SO(3)$，直接对 $R$ 的矩阵元素做加法通常会破坏正交性：

$$
R^T R = I, \qquad \det R = 1
$$

因此优化变量使用李代数中的六维小量：

$$
\delta =
\begin{bmatrix}
\rho \\
\omega
\end{bmatrix}
=
\begin{bmatrix}
\rho_x & \rho_y & \rho_z & \omega_x & \omega_y & \omega_z
\end{bmatrix}^T
$$

其中 $\rho\in\mathbb{R}^3$ 是平移小量，$\omega\in\mathbb{R}^3$ 是旋转向量小量。Sophus 的 `SE3d::Tangent` 使用的正是这个排列。

### 6.2 左扰动：增量在目标坐标系表达

左扰动的定义是：

$$
T \leftarrow \exp(\delta)T
$$

也就是说，增量在目标坐标系中作用于当前已变换的点。令：

$$
x_i = T p_i
$$

对于足够小的 $\delta$，有：

$$
\exp(\delta)x_i
\approx x_i + \rho + \omega \times x_i
$$

定义叉乘矩阵：

$$
[x]_\times =
\begin{bmatrix}
0 & -x_z & x_y \\
x_z & 0 & -x_x \\
-x_y & x_x & 0
\end{bmatrix}
$$

因为：

$$
\omega \times x_i = -[x_i]_\times\omega
$$

残差的一阶近似为：

$$
r_i(\delta)
\approx r_i +
\begin{bmatrix}
I & -[x_i]_\times
\end{bmatrix}
\delta
$$

所以左扰动 Jacobian 是：

$$
J_i =
\begin{bmatrix}
I & -[T p_i]_\times
\end{bmatrix}
$$

对应实现形式为：

```cpp
jacobian.leftCols<3>().setIdentity();
jacobian.rightCols<3>() = -Sophus::SO3d::hat(transformed_position);
```

这里的左侧不是代码风格，而是群乘法中增量的位置。由于增量乘在 $T$ 左侧，$\rho$ 和 $\omega$ 都在目标坐标系中表达。这很**适合残差、地图或外部修正在目标/世界坐标系中表达的情况**。

### 6.3 右扰动：增量在源坐标系表达

右扰动把同样的位姿增量乘在当前变换右侧：

$$
T \leftarrow T\exp(\delta_r)
$$

其中 $\delta_r=[\rho_r,\omega_r]^T$ 在源点坐标系中表达。令 $T=(R,t)$，则对源点 $p_i$ 有：

$$
T\exp(\delta_r)p_i
\approx R(p_i+\rho_r+\omega_r\times p_i)+t
$$

因此残差的一阶近似为：

$$
r_i(\delta_r)
\approx r_i +
\begin{bmatrix}
R & -R[p_i]_\times
\end{bmatrix}
\delta_r
$$

右扰动 Jacobian 是：

$$
J_{r,i}=
\begin{bmatrix}
R & -R[p_i]_\times
\end{bmatrix}
$$

它依赖原始源点 $p_i$ 与当前旋转 $R$，而不是变换后的点 $Tp_i$。当前实现采用这一右扰动形式，并成对使用：

```cpp
jacobian.leftCols<3>() = source_to_target.rotationMatrix();
jacobian.rightCols<3>() = -source_to_target.rotationMatrix() *
                           Sophus::SO3d::hat(source_point.position.cast<double>());
source_to_target = source_to_target * Sophus::SE3d::exp(right_increment);
```

Jacobian 和更新乘法顺序必须保持这组配对关系。

### 6.4 两种增量如何表示同一个物理修正

左、右扰动不是两种不同的刚体运动，而是用不同坐标系表示同一个局部修正。若它们产生相同更新：

$$
\exp(\delta_l)T = T\exp(\delta_r)
$$

则其李代数小量通过当前位姿的 Adjoint 变换关联：

$$
\delta_l = \operatorname{Ad}_T\delta_r,
\qquad
\delta_r = \operatorname{Ad}_{T^{-1}}\delta_l
$$

对 $T=(R,t)$ 且切向量排列为 $[\rho,\omega]^T$，有：

$$
\operatorname{Ad}_T =
\begin{bmatrix}
R & [t]_\times R \\
0 & R
\end{bmatrix}
$$

因此，固定对应关系下，两种 Jacobian 也满足：

$$
J_r = J_l\operatorname{Ad}_T
$$

这说明两种线性化在正确变换增量坐标后描述相同的局部位姿变化。实际迭代中，阻尼、停止阈值、有限步长和每轮重新建立的对应关系会使两种参数化的数值轨迹不同；不存在脱离具体问题而总是更优的一种。

### 6.5 选择约定的实践准则

左、右扰动的选择应由状态误差和残差自然使用的参考系决定，最重要的是从 Jacobian、求解出的增量到位姿更新始终使用同一约定。

| 约定 | 更新 | 增量表达坐标系 | GICP 点残差 Jacobian | 常见适用情况 |
| --- | --- | --- | --- | --- |
| 左扰动 | $T\leftarrow\exp(\delta_l)T$ | 目标/世界坐标系 | $[I,-[Tp_i]_\times]$ | 扫描到地图配准；地图残差、全局位置修正或世界系先验直接进入优化 |
| 右扰动 | $T\leftarrow T\exp(\delta_r)$ | 源/机体系 | $[R,-R[p_i]_\times]$ | 连续时间轨迹、里程计或 IMU 预积分中以传感器/机体系表达的小运动和先验 |

对于纯扫描到地图 GICP，两种约定都可用。当前实现采用右扰动，使增量在源/LiDAR 系表达，并便于与使用右旋转误差的 IMU 状态结合。任何世界系位置先验仍需连同协方差和 Jacobian 一起映射，不能只交换位姿更新的乘法顺序。



## 7 Gauss-Newton 法方程

在一轮迭代中，先固定对应 $c(i)$、当前旋转导出的精度矩阵 $W_i$ 和鲁棒权重 $w_i$。将残差一阶线性化：

$$
r_i(\delta) \approx r_i + J_i\delta
$$

由加权最小二乘得到近似目标：

$$
\tilde{F}(\delta) =
\frac{1}{2}\sum_i w_i(r_i+J_i\delta)^T W_i(r_i+J_i\delta)
$$

令导数为零，得到正规方程：

$$
H\delta = -g
$$

其中：

$$
H = \sum_i J_i^T w_i W_i J_i
$$

$$
g = \sum_i J_i^T w_i W_i r_i
$$

$H\in\mathbb{R}^{6\times6}$ 是近似 Hessian，$g\in\mathbb{R}^6$ 是梯度。当前 CPU 实现直接逐对应累加完整稠密矩阵，不使用并行规约或压缩存储。

严格来说，$W_i$ 随旋转 $R$ 变化，鲁棒权重也随残差变化。当前实现按 Gauss-Newton/IRLS 的常见近似，在单轮线性化中将它们视为常量，不计算 $\partial W_i/\partial\delta$ 或 $\partial w_i/\partial\delta$。



## 8 阻尼、求解与更新

实际点云可能有退化几何，例如单个平面不能充分约束平面内运动和绕法向旋转。为降低病态系统的数值风险，代码使用固定对角线阻尼：

$$
(H+\lambda I)\delta=-g
$$

其中 $\lambda$ 是 `damping_factor`。较大的 $\lambda$ 通常会使更新更保守，但这不是根据代价自适应调整的完整 Levenberg-Marquardt 算法。

实现用 Eigen 的 `LDLT` 分解求解这个固定大小 $6\times6$ 系统。分解失败、增量含有 `NaN`/`Inf`，或没有有效约束时，本轮会终止。

求解成功后，Sophus 使用指数映射将李代数增量投影回刚体变换群：

$$
T_{k+1}=T_k\exp(\delta_k)
$$

对应实现为：

```cpp
source_to_target = source_to_target * Sophus::SE3d::exp(right_increment);
```



## 9 收敛、fitness 与可观测性

当前实现以增量大小判定严格收敛：

$$
\|\rho\| < \epsilon_t
\qquad\text{且}\qquad
\|\omega\| < \epsilon_r
$$

其中 $\epsilon_t$ 是 `convergence_translation`，单位为米；$\epsilon_r$ 是 `convergence_rotation`，单位为弧度。

同时，结果中还记录未加权欧氏残差平方平均值：

$$
fitness = \frac{1}{N}\sum_i\|r_i\|^2
$$

它不是 RMSE、马氏误差或 Cauchy 鲁棒代价；并且对应和残差在更新前计算，所以最后一次成功更新后不会自动重新评估。

当前 `converged` 还有宽松回退：只要至少完成一次成功更新且 `fitness_score` 有限，即使未达到上述增量阈值也会置为 `true`。调用方应联合检查：

$$
N \geq min\_correspondences
$$

以及：

$$
fitness \leq max\_fitness\_score
$$

低 fitness 并不保证所有六个自由度都被充分约束。点云若主要来自单平面、平行墙面或高度重复结构，$H$ 可能仍然病态或近似秩亏。更可靠的初始值、更多几何结构、运动先验或对退化方向的显式约束，能改善这种情况。



## 10 实现特有的协方差回退

理论 GICP 希望在源、目标协方差都可用时使用：

$$
C_i = C_{t,c(i)} + R C_{s,i} R^T
$$

当前实现的实际回退规则是：

$$
C_i =
\begin{cases}
C_t + R C_s R^T, & C_s \text{ 有效且 } C_t \text{ 有效}\\
I + R C_s R^T, & C_s \text{ 有效且 } C_t \text{ 无效}\\
I, & C_s \text{ 无效}
\end{cases}
$$

因此，当源点协方差无效时，即使目标协方差有效，当前约束仍使用单位阵。这是代码定义的回退策略，不是理论上的唯一选择；若希望任一可用协方差都能提供方向信息，需要相应调整该分支。



## 11 一轮迭代的概念伪代码

```text
for each source point p_i:
    x_i = T * p_i
    q_c = nearest target point to x_i within distance threshold
    if no q_c:
        continue

    r_i = x_i - q_c
    C_i = target covariance + rotated source covariance, with fallback
    W_i = inverse(C_i)
    if W_i is invalid:
        continue

    e_i = r_i^T * W_i * r_i
    w_i = CauchyWeight(e_i)
    J_i = [R, -R * hat(p_i)]

    H += J_i^T * w_i * W_i * J_i
    g += J_i^T * w_i * W_i * r_i

solve (H + lambda * I) * delta = -g
T = T * Exp(delta)
```

下一轮会用更新后的 $T$ 重新搜索对应、重新计算残差、协方差旋转和鲁棒权重，直到达到停止条件或耗尽 `max_iterations`。
