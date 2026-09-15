# IMU–LiDAR 紧耦合 ESKF

`perception` 的运行入口现在启动 `Localization_LIO`。IMU 在扫描区间内传播状态与协方差，点云补偿到扫描结束时刻；每个点到地图平面的残差直接进入迭代 ESKF，更新位姿、速度和两类 IMU bias。LIO 不调用 `SparsityAwareGICP::align()`，也不把独立配准得到的 6D 位姿当作观测。原来的 `Localization_LO` / GICP 代码仍可单独使用。

## 状态与坐标系

- 名义状态：`R_WI, p_WI, v_WI, gyro_bias, accel_bias`。
- 15 维误差：`[dp_W, dtheta_I, dv_W, dbg_I, dba_I]`。
- 右扰动：`R_true = R_WI * Exp(dtheta_I)`；位置、速度误差在世界系，bias 在 IMU 系。
- `T_IL`：LiDAR → IMU，`p_I = T_IL * p_L`；`T_WL = T_WI * T_IL`。
- `T_IB`：车体 → IMU；发布 `/base_pose` 的 `T_WB = T_WI * T_IB`。若使用默认单位外参，这个输出就是 IMU 位姿。
- 世界系由静止初始化定义，Z 轴向上。位置和 yaw 是局部坐标规范，不是绝对定位。

原先的 `Vector<double, 17>` 只是占位，没有对应的 S² 重力状态。本实现使用完整的 15 状态 ESKF，初始化后固定世界系重力，不估计重力方向的两个自由度，也不在线估计外参或时间偏移。状态、协方差、正规方程使用 `double`；PCL 点和体素存储沿用 `float`。

## IMU 传播

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

## 点到平面迭代更新

对扫描末端 LiDAR 系中的点 `p_L`，在固定世界地图中查找邻居，从原始坐标拟合平面中心 `q_W` 和单位法向量 `n_W`：

```text
p_I = T_IL * p_L
r = n_Wᵀ (R_WI * p_I + p_WI - q_W)
H = [n_Wᵀ, -n_Wᵀ R_WI hat(p_I), 0, 0, 0]
```

不使用 SparseVoxel 中经过归一化的 GICP 协方差作为物理观测方差。平面拟合会拒绝邻居不足、距离过远、共线和明显非平面的邻域；残差采用距离门限和 Huber 权重。默认测量标准差为 0.03 m，需要根据真实点云噪声和下采样密度调整。

令 IMU 传播先验为 `x_bar, P_bar`，第 k 次迭代为 `x_k`，`e = x_k boxminus x_bar`。姿态部分的先验切空间雅可比为 SO(3) 右雅可比的逆 `Jr⁻¹(e_theta)`，其余块为单位阵，合成 A：

```text
Lambda = Aᵀ P_bar⁻¹ A + sum(Hᵀ R⁻¹ H)
b      = Aᵀ P_bar⁻¹ e + sum(Hᵀ R⁻¹ r)
delta  = -Lambda⁻¹ b
x_next = x_k boxplus delta
```

虽然 LiDAR 的 H 只有位姿列非零，15×15 的先验信息会使速度和 bias 同时获得修正。每轮重新查找对应和拟合平面；**整个迭代只使用同一份 P_bar**。收敛后在最终状态切空间重新线性化并求逆，得到已重置到新名义状态的后验协方差，不再额外乘一次 reset Jacobian。

验收条件包括有效平面数、收敛、最终残差 RMSE、最大位姿修正和矩阵正定性。验收失败保留本帧 IMU 预测及其协方差，并跳过地图写入。地图只在首帧建立或成功更新后写入。退化方向仍主要受 IMU 先验约束；长走廊中缺少端面时，沿走廊方向可能不可观，不能用更多相同方向平面消除这一几何退化。

## 时间同步与初始化

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
auto result = filter.update([&](const lio::ImuState& state) {
    return lidar_measurement.build(state, scan, world_map);
});
// result.accepted 才把 scan 通过 filter.lidarPose() 变换后写入地图。
```

`processScan()` 已同时传播状态和协方差，不要对同一区间再调用 `predict()`。`predict()` 用于单独的 IMU 流式传播，reset 后首个样本必须锚定 state.timestamp。`ESKF.h` 不依赖 ROS 或 PCL；ROS 时间读取、体素地图和几何模型分别放在独立文件中。

## 验证与边界

`eskf_test` 检查有限差分几何/先验雅可比、带 bias 的静止传播、常加速度积分、协方差交叉项、解析卡尔曼后验、拒绝更新回滚、非单位外参下的旋转/平移去畸变、扫描间隙、初始化和非法输入。

`lio_measurement_test` 检查组织化点云/行 padding/字节序、绝对秒/相对纳秒、缺失与错误时间字段、三平面地图的位姿/gyro bias 修正及共线邻域拒绝。

```bash
ctest --test-dir build/perception --output-on-failure -R '^(eskf_test|lio_measurement_test)$'
```

这些是实现和合成数据回归测试，不代表真实传感器的定位精度已得到验证。上线前需回放自己的同步 LiDAR+IMU 数据，核对外参、时间偏移、运动激励、退化场景和噪声参数。扫描内去畸变轨迹在本帧更新期间固定，这是常见的一阶近似；极强运动或大 bias 修正时需要更精细的重新去畸变策略。
