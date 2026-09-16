# IMU–LiDAR 紧耦合 ESKF

点云补偿到扫描结束时刻后，`GICP` 的逐点三维残差、源/目标协方差、Cauchy 权重和右扰动线性化直接进入 ESKF 正规方程。LIO 不先运行独立配准，也不把 GICP 输出的 6D 位姿当作观测，因此仍是紧耦合。

## 1 Scan 去畸变

**估计这一帧 LiDAR 扫描期间传感器的连续运动，然后把每个点“搬回”到同一个时刻的坐标系里。**

假设一帧 LiDAR scan 从 $t_0$ 扫到 $t_1$。机械式或非瞬时 LiDAR 的点不是同时采集的，所以第 $i$ 个点实际是在 $t_i$ 时刻测到的：
$$
t_0 \le t_i \le t_1
$$
如果车辆在这一帧期间发生了运动，那么直接把所有点当成在同一时刻采集，就会出现拉伸、弯曲、重影。

### 1.1 IMU 提供 scan 内部的运动

IMU 测量：
$$
\begin{align}
\boldsymbol{\omega}_m &= \boldsymbol{\omega} + \boldsymbol{b}_g + \boldsymbol{n}_g \\
\boldsymbol{a}_m &= \boldsymbol{R}^\top(\boldsymbol{a}-\boldsymbol{g})+\boldsymbol{b}_a+\boldsymbol{n}_a
\end{align}
$$
其中：

- $\boldsymbol{\omega}_m$：陀螺仪角速度
- $\boldsymbol{a}_m$：加速度计测量
- $\boldsymbol{b}_g,\boldsymbol{b}_a$：gyro / accel bias
- $\boldsymbol{R}$：IMU 姿态
- $\boldsymbol{g}$：重力

利用 IMU 在 LiDAR scan 的这几十毫秒内做积分，可以得到 $\boldsymbol{R}(t)$，$\boldsymbol{p}(t)$，$\boldsymbol{v}(t)$；也就是 scan 内任意时刻 IMU 的 pose。

最基本的传播形式类似：
$$
\begin{align}
\boldsymbol{R}_{k+1} 
&= \boldsymbol{R}_k\text{Exp} \big((\boldsymbol{\omega}_k-\boldsymbol{b}_g)\Delta t\big) \\
\boldsymbol{v}_{k+1} 
&= \boldsymbol{v}_k+ \big(\boldsymbol{R}_k(\boldsymbol{a}_k-\boldsymbol{b}_a)+\boldsymbol{g} \big)\Delta t \\
\boldsymbol{p}_{k+1} &= 
\boldsymbol{p}_k+ \boldsymbol{v}_k\Delta t+ \frac12 \left(\boldsymbol{R}_k(\boldsymbol{a}_k-\boldsymbol{b}_a)+\boldsymbol{g} \right)\Delta t^2
\end{align}
$$
实际 LIO 中通常就是 ESKF/IESKF propagation 的那套 IMU propagation。

### 1.2 每个 LiDAR 点都有自己的时间戳

例如一帧 scan 是 $t_0 = 10.000s$ 到 $t_1 = 10.1 \ \text{s}$，某个点 $\boldsymbol{p}_k$ 采集时间可能是 $t_k = 10.037 \ \text{s}$，IMU 积分可以得到这个时刻的位姿 $\boldsymbol{T}_{w\rightarrow i}(t_k)$ 以及你选定的参考时刻 $\boldsymbol{T}_{w\rightarrow i}(t_{\text{ref}})$。

通常 $t_{\text{ref}}$ 选：

- scan begin
- scan end
- scan midpoint

LIO 里很常见的是 **scan end**。

### 1.3 把每个点变换到统一参考时刻

先暂时假设 LiDAR 和 IMU 坐标系相同。

点 $\boldsymbol{p}_k$ 是在 $t_k$ 时刻的坐标 $\boldsymbol{p}_k^{I_k}$，变换到世界坐标 $\boldsymbol{p}_k^w = \boldsymbol{T}_{w\rightarrow i}(t_k) \cdot \boldsymbol{p}_k^{I_k}$，再变到 scan reference time：$\boldsymbol{p}_k^{I_{\text{ref}}} = \boldsymbol{T}_{w\rightarrow i}^{-1}(t_{\text{ref}})\cdot \boldsymbol{T}_{w\rightarrow i}(t_k)\cdot \boldsymbol{p}_k^{I_k}$

因此核心 deskew 公式就是：
$$
\boldsymbol{p}_k^{\text{ref}} = \boldsymbol{T}^{-1}(t_{\text{ref}})\cdot \boldsymbol{T}(t_k)\cdot \boldsymbol{p}_k
$$
这就是 IMU deskew 的本质。

实际系统中 LiDAR 和 IMU 显然不重合，因此还需要外参 $\boldsymbol{T}_{i\rightarrow l}$。

如果 LiDAR 点为 $\boldsymbol{p}_k^l$，先转到 IMU：
$$
\boldsymbol{p}_k^i = \boldsymbol{T}_{i\rightarrow l}\cdot \boldsymbol{p}_i^l
$$
进行运动补偿：
$$
\boldsymbol{p}_{k,\text{ref}}^{i} = \boldsymbol{T}_{w\rightarrow i}^{-1}(t_{\text{ref}})\cdot \boldsymbol{T}_{w\rightarrow i}(t_k)\cdot \boldsymbol{T}_{i\rightarrow l}\cdot\boldsymbol{p}_k^l
$$
最后再转回 LiDAR：
$$
\boxed{\boldsymbol{p}_{k,\text{ref}}^{l} = \boldsymbol{T}_{l\rightarrow i}\cdot \boldsymbol{T}_{w\rightarrow i}^{-1}(t_{\text{ref}})\cdot\boldsymbol{T}_{w\rightarrow i}(t_k)\cdot \boldsymbol{T}_{i\rightarrow l}\cdot\boldsymbol{p}_k^l}
$$
这基本就是完整的 deskew 变换。



## 2 状态与坐标系

- 名义状态：`R_wi, p_wi, v_wi, gyro_bias, accel_bias`。
- 15 维误差：`[dp_w, dtheta_i, dv_w, dbg_i, dba_i]`。
- 右扰动：`R_true = R_wi * Exp(dtheta_i)`；位置、速度误差在世界系，bias 在 IMU 系。
- `T_il`：LiDAR → IMU，`p_i = T_il * p_l`；`T_wl = T_wi * T_il`。
- `T_IB`：车体 → IMU；发布 `/base_pose` 的 `T_WB = T_WI * T_IB`。若使用默认单位外参，这个输出就是 IMU 位姿。
- 世界系由静止初始化定义，Z 轴向上。位置和 yaw 是局部坐标规范，不是绝对定位。

原先的 `Vector<double, 17>` 只是占位，没有对应的 S² 重力状态。本实现使用完整的 15 状态 ESKF，初始化后固定世界系重力，不估计重力方向的两个自由度，也不在线估计外参或时间偏移。状态、协方差、正规方程使用 `double`；PCL 点和体素存储沿用 `float`。



## 3 IMU 传播

传入未经去重力的加速度计 specific force（m/s²），以及角速度（rad/s）。先减 bias，再以区间中点姿态积分：

```text
omega = (gyro0 + gyro1)/2 - bg
force = (accel0 + accel1)/2 - ba
R_mid = R * Exp(omega * dt/2)
a_W = R_mid * force + g_W
p += v*dt + a_W*dt²/2
v += a_W*dt
R *= Exp(omega*dt)
```

误差动力学的非零块：

| 块 | 连续时间 F |
|---|---|
| p ← v | I |
| theta ← theta | -hat(omega) |
| theta ← bg | -I |
| v ← theta | -R_mid hat(force) |
| v ← ba | -R_mid |

噪声参数都是**连续时间标准差密度**，代码平方得到谱密度。状态转移采用二阶近似，过程噪声用 Simpson 积分；默认把积分步长细分到 0.01 s。协方差保留 p–v、theta–bg、v–ba 等交叉项。



## 4 GICP 紧耦合迭代更新

每帧仅建立一次源点 `SparseVoxel` 和局部协方差。每次迭代根据当前 `T_WL` 重新查找地图最近邻，并调用与 `GICP::align()` 完全相同的 `linearize()`：

```text
r_i = T_WL * p_Li - q_Wi
C_i = C_target_i + R_WL * C_source_i * R_WLᵀ
e_i = r_iᵀ C_i⁻¹ r_i
w_i = 1 / (1 + e_i / cauchy_scale²)
J_right = [R_WL, -R_WL * hat(p_Li)]
```

`GICP::linearize()` 同时供独立 GICP 和 LIO 使用，避免两套对应搜索、协方差或权重实现发生偏差。`SparseVoxel` 归一化协方差只提供各向异性形状；`lidar_noise²` 为它补充与 ESKF 先验比较所需的物理尺度：

```text
Omega_i = (lidar_noise² * C_i)⁻¹
```

这不是把无量纲归一化协方差直接冒充物理协方差。`lidar_noise` 是整体尺度，需要用真实残差或 NIS 标定。

GICP 使用 LiDAR 系右扰动 `T_WL' = T_WL * Exp(delta_right_L)`；ESKF 使用世界系加法位置误差和 IMU 系右旋转误差。实现通过解析变换把 GICP 的 6×6 Hessian 和梯度映射到 `[dp_W, dtheta_I]`。其中 `t_IL` 是 LiDAR 原点在 IMU 系的位置：

```text
delta_right_L = M * [dp_W, dtheta_I]
M = [R_WLᵀ, -R_WLᵀ R_WI hat(t_IL);
     0,                         R_ILᵀ]
H_eskf = Mᵀ H_gicp M
g_eskf = Mᵀ g_gicp
```

令 IMU 传播先验为 `x_bar, P_bar`，第 k 次迭代为 `x_k`，`e = x_k boxminus x_bar`。姿态部分的先验切空间雅可比为 SO(3) 右雅可比的逆 `Jr⁻¹(e_theta)`，其余块为单位阵，合成 A：

```text
Lambda = Aᵀ P_bar⁻¹ A + sum(Hᵀ R⁻¹ H)
b      = Aᵀ P_bar⁻¹ e + sum(Hᵀ R⁻¹ r)
delta  = -Lambda⁻¹ b
x_next = x_k boxplus delta
```

虽然 GICP 信息只有位姿块非零，15×15 的先验信息会使速度和 bias 同时获得修正。每轮重新查找对应，但源点协方差只计算一次；**整个迭代只使用同一份 P_bar**。求解时在位姿块加入与独立 GICP 相同的固定阻尼，最终后验协方差使用未加阻尼的信息矩阵。收敛只检查位姿增量；与 `GICP::align()` 一样，达到最大迭代次数后保留最后一次有限迭代。

验收条件包括有效对应数、最终 fitness、最大位姿修正和矩阵正定性。fitness 与独立 GICP 一致，是未加权欧氏残差平方均值，单位 `m²`。验收失败保留本帧 IMU 预测及其协方差，并跳过地图写入。地图只在首帧建立或成功更新后写入。退化方向仍主要受 IMU 先验约束。



## 5 时间同步与初始化

一个工作线程拥有滤波器和地图，ROS 回调只缓存数据。保留完整 IMU 历史及边界插值所需的两端样本，不再每次只留最后一帧 IMU。超过扫描队列容量时可以丢弃旧扫描，随后仍从上次修正状态连续积分 IMU。

| 逐点时间字段 | `auto` 解释 |
|---|---|
| `timestamp` | 绝对秒；扫描边界使用全点云最小/最大逐点时间 |
| `time` | 相对秒 |
| `t`、`offset_time` | 相对纳秒 |
| 自定义名称 | 必须配置 `lio.time_field`、`lio.time_scale`，并正确设置 `lio.time_mode` |

Hesai ROS 2 驱动的 `timestamp` 是 FLOAT64 字段，默认配置针对该格式。不同驱动/固件要核对实际消息。相对时间默认认为 header stamp 是扫描起点；若是扫描终点，设置 `lio.stamp_is_end:=true`。

- 先根据整个点云解析扫描时间，再做强度和空间过滤，最后去畸变、体素下采样。
- 只有 IMU 覆盖当前状态时间到扫描末端时才处理，扫描边界用插值，不外推。
- IMU 时间使用 `header.stamp + lio.imu_time_offset`；必须与点云处于同一时钟域。
- 默认拒绝没有逐点时间的点云；显式设置 `allow_untimed_cloud:=true` 时视作末端瞬时点云，无法补偿扫描内部运动。
- 默认使用扫描前 1 s 的静止数据初始化。估计 gyro bias，并使平均 specific force 对齐世界 +Z。单次静止姿态不能分离完整加速度 bias 与倾角，仅初始化沿重力的分量，其余依赖后续运动与激光观测。
- 静止检测只是均值/方差启发式，恒速运动无法仅靠 IMU 判定；请实际静止启动。
- 遇到超过 `max_imu_gap` 的缺测或历史缓存丢失时，清空局部地图和轨迹并等待重新静止初始化，日志明确报告坐标原点重置。
- 重复/乱序 IMU 被丢弃；若 rosbag 循环回放使时钟倒退，应重启节点。



## 使用

先填写真实的 `lio.lidar_to_imu` 和 `lio.base_to_imu` 外参，再核对时间字段、IMU 单位和 topic。参数示例在 `config/lio.yaml`，其中单位外参只是占位。

```bash
colcon build --base-paths trailer_localization/packages/perception --packages-select perception --cmake-args -DBUILD_TESTING=ON
source install/setup.bash
ros2 run perception perception --ros-args --params-file trailer_localization/packages/perception/config/lio.yaml
```

已有 system_manager launch 也会启动新的 LIO 入口，可把这些参数加入自己的参数文件。`/scan_vis`、`/voxel_map` 使用世界坐标，`/base_pose`、`/base_pose_path` 使用车体位姿。所有输出时间戳都对应扫描结束时刻。地图达到体素预算后停止创建新体素；当前版本不做滑动窗口剔除。

独立 C++ 调用顺序：

```cpp
lio::ESKF filter(config, T_IL);
filter.initialize(stationary_imu);  // 检查返回值；也可 reset(known_state, covariance)
filter.processScan(imu_covering_interval, timed_points, scan_begin, scan_end);
// 将 timed_points 转为扫描末端 LiDAR 系的 pcl cloud 并下采样。
auto source_index = gicp_measurement.prepareScan(scan);
auto result = filter.update([&](const lio::ImuState& state) {
    return gicp_measurement.build(state, *source_index, world_map);
});
// result.accepted 才把 scan 通过 filter.lidarPose() 变换后写入地图。
```

`processScan()` 已同时传播状态和协方差，不要对同一区间再调用 `predict()`。`predict()` 用于单独的 IMU 流式传播，reset 后首个样本必须锚定 state.timestamp。`ESKF.h` 不依赖 ROS 或 PCL；ROS 时间读取、体素地图和几何模型分别放在独立文件中。



## 验证与边界

`eskf_test` 检查有限差分几何/先验雅可比、带 bias 的静止传播、常加速度积分、协方差交叉项、解析卡尔曼后验、拒绝更新回滚、非单位外参下的旋转/平移去畸变、扫描间隙、初始化和非法输入。

`lio_measurement_test` 检查组织化点云/行 padding/字节序、绝对秒/相对纳秒、缺失与错误时间字段、共享 GICP 右扰动线性化、LiDAR 右扰动到 ESKF 误差状态的解析映射、三平面地图的位姿与 gyro bias 修正，以及对应距离门限。

```bash
ctest --test-dir build/perception --output-on-failure -R '^(eskf_test|lio_measurement_test)$'
```

这些是实现和合成数据回归测试，不代表真实传感器的定位精度已得到验证。上线前需回放自己的同步 LiDAR+IMU 数据，核对外参、时间偏移、运动激励、退化场景和噪声参数。扫描内去畸变轨迹在本帧更新期间固定，这是常见的一阶近似；极强运动或大 bias 修正时需要更精细的重新去畸变策略。
