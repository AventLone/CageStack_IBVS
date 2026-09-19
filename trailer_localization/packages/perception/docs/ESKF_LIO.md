# 9 维松耦合 LIO

`Localization_LIO` 复用现有 CPU/TBB 稀疏 GICP、PCL、Eigen 和 Sophus。
参考 FAR-LIO 的“独立配准 + 滤波融合 + 高频预测/延迟补偿”结构；不是完整 FAR-LIO 移植，
不引入其 CUDA 后端，也不把逐点配准残差直接送入滤波器。

## 数据流

1. 启动时静止至少 `imu_init_duration` 秒、20 个样本：估计陀螺仪零偏、重力对齐姿态和加速度模长偏差。
2. IMU 回调持续保存样本并进行中点预测，按 IMU 频率发布 `/base_pose`。
3. LiDAR 线程等待 IMU 覆盖扫描结束，用同一个 ESKF 积分实现记录轨迹，将点去畸变到扫描结束时刻。
4. GICP 以预测的 LiDAR 位姿为初值，独立求解扫描到地图的位姿。
5. 检查收敛、对应数量、最终位姿的欧氏残差 MSE，再将 LiDAR 位姿和协方差转换到 IMU，做一次 ESKF 观测更新。
6. 在扫描结束时刻注入修正，复制完整滤波器（状态、协方差、边界 IMU），重放之后的 IMU。
   下一次 IMU 回调从修正后的最新时刻继续。不会重复融合同一帧 LiDAR 来人为提高观测频率。
7. 配准和创新门限都通过后，按位移/转角阈值插入关键帧；首帧在预测的世界位姿下建图。
   拒绝配准时继续惯性预测，不向地图插入失败帧。

仅保留一个待处理点云，计算跟不上时使用较新帧；IMU 历史不会随着点云丢帧而清空。
处理后的 IMU 队列保留扫描结束前最后一个样本，用于下一次边界插值。
GICP、去畸变与地图更新在锁外执行，重放和 IMU 回调共享一把锁。
`/base_pose` 只由 IMU 回调发布，不插入时间倒退的历史修正消息。

## 状态与数学约定

名义状态为位置 `p_wi`、SO(3) 姿态 `R_wi`、世界系速度 `v_wi`。
局部误差：

```
dx = [dp_world, dtheta_imu, dv_world]  (9 x 1)
R_true = R_wi * Exp(dtheta_imu)
```

重力固定为 `[0, 0, -9.81] m/s²`。陀螺仪和加速度零偏是固定标定参数，不扩展到状态中。
单一静止姿态无法同时辨识完整加速度零偏和倾角；初始化仅按重力方向对齐并去除模长误差。
世界原点在初始 IMU 位置，初始 yaw 没有绝对航向观测。

中点传播：先平均相邻原始 IMU，再用半步 SO(3) 旋转计算世界系加速度。
误差转移和噪声 Jacobian 对应同一个中点模型。
IMU 参数为连续噪声密度，离散协方差使用 `G * Qc/dt * G^T`，并补齐白加速度噪声的
`Q_pp = qa * dt³/3` 项，因此改变 IMU 频率不会把过程噪声人为缩小。

观测残差为 `[p_meas - p_pred, Log(R_pred^-1 * R_meas)]`，`H = [I6, 0]`。
使用 Eigen LDLT 求增益、Joseph 形式更新协方差，并用
`Sophus::SO3d::leftJacobian(-dtheta)` 做右扰动注入后的协方差重置。
通过交叉协方差更新速度，无需将相邻点云位移差分成第二个相关观测。

LiDAR 位姿测量协方差使用配置的平移/旋转标准差，假设局部小姿态误差；
它不是由 GICP Hessian 推导的完整退化估计。长直、缺少几何约束的车厢仍需实测调参。

## 外参与时间

- `T_il = lidar_to_imu`：`p_i = T_il * p_l`。
- `T_ib = base_to_imu`：`p_i = T_ib * p_b`；发布 `T_wb = T_wi * T_ib`。
- 配准初值 `T_wl = T_wi * T_il`；观测 `T_wi_meas = T_wl_meas * T_il^-1`。
- 位姿观测协方差由 `[dp_world, dtheta_lidar]` 转为 `[dp_world, dtheta_imu]`，
  包含非零杆臂造成的平移/旋转交叉项。
- 点云按照现有 Hesai 格式读取：`x/y/z/intensity` 为 FLOAT32，`timestamp` 为 FLOAT64 绝对秒。
  取点时间最小/最大值为扫描起止；点不需要按时间排序。空点云与不匹配的字段格式会跳过。
- IMU 为 rad/s、m/s²，包含重力的 specific force，时间戳必须与点云同源、递增。
  不额外检测 IMU 数值、采样间隙或静止性。
- 初始化结束时若落在扫描内部，丢弃这帧，从下一帧完整扫描开始。
  单 LiDAR 扫描窗口应不重叠。点云/路径使用扫描结束时间，高频位姿使用 IMU 时间。
- 历史样本由点云处理进度回收；该实现面向持续流入的 LiDAR/IMU，不提供长时间 LiDAR
  中断后的历史限长、自动重定位或时钟重置恢复。

## 构建与运行

在 `trailer_localization` 下执行（需要 ROS2、PCL、Sophus、TBB、Eigen，
以及现有 `SparseVoxel` 所需的 Boost `unordered_flat_map`，Boost >= 1.81）：

```bash
colcon build --packages-select perception --cmake-args -DBUILD_TESTING=ON
source install/setup.bash
colcon test --packages-select perception --ctest-args -R 'lio_.*test' --output-on-failure
colcon test-result --verbose
ros2 run perception perception --ros-args \
  --params-file "$(ros2 pkg prefix --share perception)/config/lio.yaml"
```

先将 `config/lio.yaml` 的两组单位外参替换为标定值，再运行；不要根据变量名猜外参方向。
仿真或 rosbag `--clock` 时设置 `use_sim_time:=true`。直接使用原 `system_manager` launch 时，
它仍加载自己的 `system.yaml`，需要将 LIO 参数放入该文件或显式传入新的参数文件。

主要参数及单位见 `config/lio.yaml`。`gicp.max_fitness_score` 是 MSE，`0.01` 对应 10 cm RMSE。
`eskf.innovation_gate` 是 6 维残差平方 Mahalanobis 门限，默认 22.458；不是米或弧度阈值。
当前 IMU 与 LiDAR 噪声参数是初始值，需要用实际传感器和 rosbag 调整。

输出：

| Topic | 内容 | 时间 |
|---|---|---|
| `/base_pose` | 配置的 base 位姿，世界系 | IMU 时刻 |
| `/base_pose_path` | 扫描时刻融合轨迹，最多 `path_max_poses` | 扫描结束 |
| `/scan_vis` | 去畸变、滤波后的世界系点云 | 扫描结束 |
| `/voxel_map` | 关键帧体素地图 | 扫描结束 |

不会额外发布动态 TF；原项目的下游可继续订阅上述 topics。

## 验证范围

`lio_core_test` 不依赖 ROS，验证静止标定、加速传播、数值差分 Jacobian、噪声随采样率的一致性、
观测拒绝不修改状态、速度修正、协方差对称/半正定、带旋转外参和杆臂的去畸变、
非采样点扫描边界及延迟重放一致性。测试通过不会证明真实数据上的定位精度。

本次环境验证仅覆盖独立 C++ 核心测试；没有 ROS2/PCL 运行环境及实测 rosbag，
整包编译、回调压力测试和实际定位精度仍需在目标机器上验证。

参考：[FAR-LIO](https://github.com/TUMFTM/FAR-LIO)、
[论文](https://arxiv.org/abs/2606.26010)。
