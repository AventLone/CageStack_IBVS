
import numpy as np
import omni
import os

from isaacsim.sensors.rtx import LidarRtx
from isaacsim.sensors.physx import _range_sensor
from pxr import UsdGeom, UsdPhysics, UsdLux, Sdf, Gf,Usd
from omni.isaac import RangeSensorSchema
from dataclasses import dataclass
from typing import Dict, Optional

import omni.syntheticdata as sd
from omni.replicator.core import annotators
import omni.replicator.core as rep
from collections import deque

from utils import SysTimer

import math




@dataclass
class PointCloudWithMeta:
    points: np.ndarray          # (N,3) 坐标数组（x,y,z）
    timestamps: Optional[np.ndarray] = None  # (N,) 每个点的时间戳（可选）
    intensities: Optional[np.ndarray] = None # (N,) 反射强度（可选）
    labels: Optional[np.ndarray] = None      # (N,) 分类标签（如障碍物/地面，可选）

class VN_Mid360:
    def __init__(
            self, prim_path: str, config: Optional[dict], vehicle_xform_path: Optional[str] = None,parent_xform_path: Optional[str] = None):
        """
        MID360激光雷达初始化（非重复流式扫描）
        :param prim_path: 雷达在USD场景中的路径（如"/World/MID360_Lidar"）
        :param npy_path: MID360角度表.npy文件路径（默认从环境变量读取）
        :param sampling_hz: 采样频率（默认24000Hz）
        :param min_range: 最小探测距离（默认0.1m）
        :param max_range: 最大探测距离（默认60m）
        :param batch_size: 每次投喂的射线批次大小（平衡性能）
        :param draw_points: 是否可视化点云（默认True）
        :param draw_lines: 是否可视化射线（默认False）
        """
        # 1. 基础参数初始化
        self.prim_path = prim_path
        self.config_lidar(config)
        self.parent_xform_path = parent_xform_path  # 保存父Xform路径
        self.vehicle_xform_path = vehicle_xform_path


        # 2. 核心状态变量
        self._range_sensor_if = None  # 激光雷达接口
        self._timeline = None  # 仿真时间线
        self._angle_pattern = None  # (2, N) 角度矩阵（方位角, 仰角，弧度）
        self._ray_offsets = None  # (N, 3) 射线原点偏移（默认0）
        self._total_rays = 0  # 总射线数
        self._head = 0  # 当前射线索引（用于批次投喂）
        self._update_sub = None  # 帧更新事件订阅
        self._is_running = False  # 雷达运行状态

        # 3. 初始化核心组件
        self._init_timeline()  # 初始化时间线
        self._load_angle_pattern()  # 加载角度图样（真实/模拟）
        self._create_lidar_sensor()  # 创建MID360雷达传感器

        if self.parent_xform_path:
            self.link_to_xform(self.parent_xform_path)
        self._subscribe_update_event()  # 订阅帧更新事件（用于批次投喂射线）

        #时间戳
        self._emit_ts_queue = deque()  # 存每条射线的发射时刻（秒，仿真时间）
        self._dt_ray = 1000 / float(self.sampling_hz)    #单位ms
        self._c_light = 299792458.0  # m/s

    def config_lidar(self, config: Optional[dict]):
        if not config:
            return

        if "npy_path" in config:
            self.npy_path = config["npy_path"]

        if "sampling_hz" in config:
            self.sampling_hz = config["sampling_hz"]


        if "min_range" in config:
            self.min_range = config["min_range"]

        if "max_range" in config:
            self.max_range = config["max_range"]

        if "batch_size" in config:
            self.batch_size = config["batch_size"]

        if "draw_points" in config:
            self.draw_points = config["draw_points"]

        if "draw_lines" in config:
            self.draw_lines = config["draw_lines"]

        if "position" in config:
            self.position = config["position"]

        if "rotation" in config:
            self.rotation = config["rotation"]

    import math

    def _make_delta_matrix_parent_RH_from_LH(self,delta_t_lh, delta_r_deg_lh):
        """
        父级同为左手系

        """
        tx, ty, tz = delta_t_lh if delta_t_lh is not None else (0.0, 0.0, 0.0)
        rx, ry, rz = delta_r_deg_lh if delta_r_deg_lh is not None else (0.0, 0.0, 0.0)

        # 用 Gf.Transform 组装 LH 下的 Δ 矩阵（注意角度单位是“度”）
        xf = Gf.Transform()
        xf.SetTranslation(Gf.Vec3d(tx, ty, tz))
        rot = (Gf.Rotation(Gf.Vec3d(1, 0, 0), rx) *
               Gf.Rotation(Gf.Vec3d(0, 1, 0), ry) *
               Gf.Rotation(Gf.Vec3d(0, 0, 1), rz))
        xf.SetRotation(rot)
        # 如需缩放，可额外 SetScale，这里默认 1
        return xf.GetMatrix()  # -> Gf.Matrix4d

    def relative_pose_A_in_B(self,pos_A_in_B, eul_A_in_B_deg,
                                                      pos_B_in_W, eul_B_in_W_deg):
        """已知 A@B 与 B@W，求 A@W：T_A^W = T_B^W * T_A^B"""

        def _R_from_euler_xyz_deg(rx, ry, rz):
            """R = Rz(yaw) * Ry(pitch) * Rx(roll)  —— 与下方提取函数一致"""
            return (Gf.Rotation(Gf.Vec3d(0, 0, 1), rz) *
                    Gf.Rotation(Gf.Vec3d(0, 1, 0), ry) *
                    Gf.Rotation(Gf.Vec3d(1, 0, 0), rx))

        def _T_from_pos_euler(pos_xyz, rot_deg_xyz):
            """由 位置 + 欧拉角(度,XYZ) 生成 4x4 变换矩阵"""
            tx, ty, tz = pos_xyz
            rx, ry, rz = rot_deg_xyz  # roll(X), pitch(Y), yaw(Z)
            xf = Gf.Transform()
            xf.SetTranslation(Gf.Vec3d(tx, ty, tz))
            xf.SetRotation(_R_from_euler_xyz_deg(rx, ry, rz))
            return xf.GetMatrix()  # Gf.Matrix4d

        def _euler_xyz_from_matrix(M):
            # 让 Decompose 返回满足 R ≈ Rz(yaw)*Ry(pitch)*Rx(roll) 的角度
            yaw, pitch, roll = rot.Decompose(
                Gf.Vec3d(0, 0, 1),  # Z (yaw)
                Gf.Vec3d(0, 1, 0),  # Y (pitch)
                Gf.Vec3d(1, 0, 0),  # X (roll)
            )
            return (roll, pitch, yaw)  # 按 XYZ 顺序返回


        T_A_B = _T_from_pos_euler(pos_A_in_B, eul_A_in_B_deg)
        T_B_W = _T_from_pos_euler(pos_B_in_W, eul_B_in_W_deg)
        T_A_W =  T_A_B*T_B_W

        t = T_A_W.ExtractTranslation()
        rot = T_A_W.ExtractRotation()  # Gf.Rotation（角度单位=度）
        eul_deg = _euler_xyz_from_matrix(rot)
        return (float(t[0]), float(t[1]), float(t[2])), eul_deg

    def link_to_xform(self,
                      parent_xform_path: str):  # (rx,ry,rz) 增量欧拉角(度)，左手系，XYZ 顺序
        """
        把雷达挂靠到 parent_xform_path 下，“保持世界位姿不变”，
        然后在“父级坐标系”里叠加一段“左手系”增量位姿（可选）。
        最终本地矩阵：L_new = Δ_parent(RH) * ( inv(P_world) * W_before )

        用法：
          link_to_xform("/World/Vehicle/BaseLink",
                        delta_translation_lh=(1.2, 0.35, 0.0),   # 前1.2m、右0.35m、上0
                        delta_rotation_deg_lh=(0.0, 0.0, 5.0))   # 绕Z旋 5°（LH）
        """
        stage = omni.usd.get_context().get_stage()
        if not stage:
            raise RuntimeError("未获取到USD舞台，无法建立关联")
        car_prim = stage.GetPrimAtPath(self.vehicle_xform_path)
        car_xf =  UsdGeom.Xformable(car_prim)
        M_car = car_xf.ComputeLocalToWorldTransform(Usd.TimeCode.Default()).RemoveScaleShear()
        t = M_car.ExtractTranslation()
        pos_car = [float(t[0]), float(t[1]), float(t[2])]
        rotation_car = M_car.ExtractRotation().Decompose(
            Gf.Vec3d(1, 0, 0),  # X轴（roll）
            Gf.Vec3d(0, 1, 0),  # Y轴（pitch）
            Gf.Vec3d(0, 0, 1)  # Z轴（yaw）  #这里添加一个负号？？？
        )

        position_fally,rotation_fally=self.relative_pose_A_in_B(self.position,self.rotation,pos_car,rotation_car)

        delta_translation_lh = position_fally  # (dx,dy,dz) 增量平移，父级坐标，左手系：X前/Y右/Z上
        delta_rotation_deg_lh =rotation_fally

        # 0) 拿 Stage/父 prim
        parent_prim = stage.GetPrimAtPath(parent_xform_path)
        if not parent_prim.IsValid():
            raise ValueError(f"目标Xform不存在：{parent_xform_path}")

        # 1) 拿雷达 prim（缓存优先，兜底按路径）
        lidar_prim = getattr(self, "_lidar_prim_pxr", None)
        if lidar_prim is None or not lidar_prim.IsValid():
            lidar_prim = stage.GetPrimAtPath(self.prim_path)
        if not lidar_prim.IsValid():
            raise ValueError(f"雷达Prim无效：{self.prim_path}（请先创建雷达）")

        # 2) 暂停运行，避免并发批次发送
        was_running = getattr(self, "_is_running", False)
        self._is_running = False

        # 3) 记录移动前“世界矩阵” W_before
        lidar_xf = UsdGeom.Xformable(lidar_prim)
        W_before: Gf.Matrix4d = lidar_xf.ComputeLocalToWorldTransform(Usd.TimeCode.Default())

        # 4) 生成目标路径（处理重名）
        name = lidar_prim.GetName()
        dst_path = Sdf.Path(parent_xform_path).AppendChild(name)
        if stage.GetPrimAtPath(str(dst_path)).IsValid():
            i = 1
            while stage.GetPrimAtPath(str(dst_path)).IsValid():
                dst_path = Sdf.Path(parent_xform_path).AppendChild(f"{name}_{i}")
                i += 1

        # 5) MovePrim 改父子关系
        omni.kit.commands.execute(
            "MovePrim",
            path_from=str(lidar_prim.GetPath()),
            path_to=str(dst_path),
        )

        # 6) 计算“保持世界位姿”的本地矩阵 L = inv(P) * W_before
        moved_prim = stage.GetPrimAtPath(str(dst_path))
        P_world: Gf.Matrix4d = UsdGeom.Xformable(parent_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        L: Gf.Matrix4d = P_world.GetInverse() * W_before

        # 7) 如果给了“父级增量(LH)”，换算为 RH 并左乘到 L（注意‘左乘’）
        if (delta_translation_lh is not None) or (delta_rotation_deg_lh is not None):
            Delta_parent_RH: Gf.Matrix4d = self._make_delta_matrix_parent_RH_from_LH(
                delta_translation_lh, delta_rotation_deg_lh
            )
            L = Delta_parent_RH * L  # 左乘：先在父系里施加增量，再接原本地矩阵

        # 8) 写回本地矩阵（单一 matrix4d op；不与 TRS 混用）
        moved_xf = UsdGeom.Xformable(moved_prim)
        moved_xf.ClearXformOpOrder()
        op = moved_xf.AddTransformOp()  # xformOp:transform (matrix4d)
        op.Set(L)

        # 9) 刷新内部句柄
        self._lidar_prim_pxr = moved_prim
        self._lidar_prim = RangeSensorSchema.Generic(moved_prim)
        self.prim_path = str(dst_path)
        self.parent_xform_path = parent_xform_path

        # 10) 恢复运行
        self._is_running = was_running

        print(f"[VN_Mid360] 保持世界位姿挂靠完成 -> {dst_path}"
              + (f" | 叠加父级增量(LH)：T={delta_translation_lh}, Rdeg={delta_rotation_deg_lh}"
                 if (delta_translation_lh is not None or delta_rotation_deg_lh is not None) else ""))


    def _init_timeline(self):
        """初始化Isaac Sim仿真时间线"""
        self._timeline = omni.timeline.get_timeline_interface()
        if not self._timeline:
            raise RuntimeError("获取仿真时间线失败，请确保Isaac Sim环境正常")

    def _load_angle_pattern(self):
        """加载MID360角度图样（优先真实.npy，其次模拟玫瑰线）"""
        # 情况1：加载真实角度表.npy
        if os.path.isfile(self.npy_path):
            try:
                data = np.load(self.npy_path)
                assert data.ndim == 2 and data.shape[1] == 2, f".npy格式错误，需为(N,2)，实际为{data.shape}"
                az, el = data[:, 0].astype(np.float32), data[:, 1].astype(np.float32)
                # 计算并打印az和el的最大值
                max_az = np.max(az)
                max_el = np.max(el)
                print("max_az ===========",max_az)
                print("max_el ===========", max_el)
                self._angle_pattern = np.stack([az, el], axis=0)
                self._ray_offsets = np.zeros((az.shape[0], 3), dtype=np.float32)
                self._total_rays = az.shape[0]
                print(f"[VN_Mid360] 加载真实角度表: {self.npy_path}，总射线数: {self._total_rays}")

            except Exception as e:
                raise RuntimeError(f"加载.npy角度表失败: {str(e)}")

        # #没有npy的情况下生成模拟玫瑰线（非重复扫描）
        else:
            n_total = 200000  # 模拟总射线数
            t = np.linspace(0, 1.0, n_total, endpoint=False)
            f_az, f_el = 23.0, 31.0  # 互质频率，避免扫描重复
            az = (t * 2 * np.pi * f_az) % (2 * np.pi) - np.pi  # 方位角: [-π, π)
            el = 0.35 * np.sin(2 * np.pi * f_el * t + 0.7)  # 仰角: ~±20°



            self._angle_pattern = np.stack([az, el], axis=0)
            self._ray_offsets = np.zeros((n_total, 3), dtype=np.float32)
            self._total_rays = n_total
            print(f"[VN_Mid360] 未找到.npy : {self.npy_path}，生成模拟玫瑰线图样，总射线数: {self._total_rays}")

    def _make_timestamps(self, N: int, points_world: np.ndarray, mode: str = "return") -> np.ndarray:
        # 1) 消费发射时刻（可能队列里比 N 多/少）
        # emit = np.empty(N, dtype=np.float64)
        # k = min(N, len(self._emit_ts_queue))
        # for i in range(k):
        #     emit[i] = self._emit_ts_queue.popleft()
        # if k < N:
        #     # 兜底：用当前仿真时间往前均匀补齐
        #     t_end = self._get_current_timestamp()
        #     missing = N - k
        #     start = t_end - missing * self._dt_ray
        #     emit[k:] = start + self._dt_ray * np.arange(missing, dtype=np.float64)
        #
        #
        # return emit
        emit = np.empty(N, dtype=np.float64)

        # 只保留最新的时间戳，丢弃历史时间戳
        if len(self._emit_ts_queue) > N:
            for _ in range(len(self._emit_ts_queue) - N):
                self._emit_ts_queue.popleft()  # 丢弃旧的时间戳

        # 获取当前仿真时间
        t_end = self._get_current_timestamp()

        # 补齐时间戳队列
        missing = N - len(self._emit_ts_queue)
        start = t_end - missing * self._dt_ray
        emit[:missing] = start + self._dt_ray * np.arange(missing, dtype=np.float64)

        # 填充剩余的时间戳
        for i in range(missing, N):
            emit[i] = self._emit_ts_queue[i - missing]

        # 更新队列，存储当前帧的时间戳
        self._emit_ts_queue.extend(emit.tolist())

        return emit


    def _create_lidar_sensor(self):
        """创建MID360激光雷达传感器（基于Generic Range Sensor）"""
        # 获取当前USD舞台
        stage = omni.usd.get_context().get_stage()
        if not stage:
            raise RuntimeError("未获取到USD舞台，请先初始化场景")

        # 1. 定义雷达传感器Prim
        if not stage.GetPrimAtPath(self.prim_path):
            self._lidar_prim = RangeSensorSchema.Generic.Define(stage, Sdf.Path(self.prim_path))
        else:
            self._lidar_prim = RangeSensorSchema.Generic(stage.GetPrimAtPath(self.prim_path))
        self._lidar_prim_pxr = self._lidar_prim.GetPrim()  # 保存USD原生Prim对象（用于后续设置父节点）

        # 2. 设置雷达核心参数
        self._lidar_prim.CreateStreamingAttr().Set(True)  # 启用流式扫描
        self._lidar_prim.CreateSamplingRateAttr().Set(self.sampling_hz)  # 采样频率
        self._lidar_prim.CreateMinRangeAttr().Set(self.min_range)  # 最小探测距离
        self._lidar_prim.CreateMaxRangeAttr().Set(self.max_range)  # 最大探测距离
        self._lidar_prim.CreateDrawPointsAttr().Set(self.draw_points)  # 点云可视化
        self._lidar_prim.CreateDrawLinesAttr().Set(self.draw_lines)  # 射线可视化

        # # 设置lidar的位置
        # self._lidar_prim.GetPrim().GetAttribute("xformOp:translate").Set(Gf.Vec3d(self.position))
        #
        # # 设置lidar的旋转
        # self._lidar_prim.GetPrim().GetAttribute("xformOp:rotateXYZ").Set(Gf.Vec3d(self.rotation))

        # 3. 设置雷达的「局部变换」（相对于父Xform，非世界坐标）
        # 若已挂靠父Xform：position是雷达相对于父Xform的偏移（如父Xform在车辆中心，雷达在车顶则设(0,0,1)）
        # self._lidar_prim_pxr.GetAttribute("xformOp:rotateXYZ").Set(Gf.Vec3d(self.rotation))
        # self._lidar_prim_pxr.GetAttribute("xformOp:translate").Set(Gf.Vec3d(self.position))



        # 3. 获取雷达控制接口
        self._range_sensor_if = _range_sensor.acquire_generic_sensor_interface()
        if not self._range_sensor_if:
            raise RuntimeError("获取激光雷达接口失败，请检查Isaac Sim传感器插件")


    def _subscribe_update_event(self):
        """订阅帧更新事件，每帧投喂射线批次"""

        def _on_frame_update(_):
            if not self._is_running or not self._range_sensor_if:
                return

            # 传感器需要下一批射线时，投喂数据
            if self._range_sensor_if.send_next_batch(self.prim_path):
                rays, offsets = self._get_next_ray_batch()

                n = rays.shape[1]  # 本批射线数（你的 rays 是形状 (2, n) 的 [az, el]）

                # 记录本批每条射线的“发射时刻”（系统时间）
                t0 = self._get_current_timestamp()  # 仿真时间，单位ms
                # 假设等间隔扫描（sampling_hz）
                t_batch = t0 + self._dt_ray * np.arange(n, dtype=np.float64)
                self._emit_ts_queue.extend(t_batch.tolist())

                self._range_sensor_if.set_next_batch_rays(self.prim_path, rays)
                self._range_sensor_if.set_next_batch_offsets(self.prim_path, offsets)

        # 订阅Isaac Sim全局更新事件流
        self._update_sub = omni.kit.app.get_app().get_update_event_stream().create_subscription_to_pop(
            _on_frame_update, name=f"VN_Mid360_Update_{self.prim_path}"
        )

    def _get_next_ray_batch(self):
        """获取下一批射线角度和偏移（循环流式投喂）"""
        if self._total_rays == 0:
            return np.empty((2, 0)), np.empty((0, 3))

        # 批次截取逻辑（到末尾时环回）
        if self._head + self.batch_size <= self._total_rays:
            batch_rays = self._angle_pattern[:, self._head: self._head + self.batch_size]
            batch_offsets = self._ray_offsets[self._head: self._head + self.batch_size]
            self._head += self.batch_size
        else:
            # 拼接末尾剩余射线 + 开头射线（实现循环）
            remaining = self._total_rays - self._head
            batch_rays = np.concatenate([
                self._angle_pattern[:, self._head:],
                self._angle_pattern[:, : self.batch_size - remaining]
            ], axis=1)
            batch_offsets = np.concatenate([
                self._ray_offsets[self._head:],
                self._ray_offsets[: self.batch_size - remaining]
            ], axis=0)
            self._head = self.batch_size - remaining

        return batch_rays, batch_offsets

    def enable_visualization(self, draw_points: bool = True, draw_lines: bool = False):
        """启用雷达可视化（点云/射线）"""
        self.draw_points = draw_points
        self.draw_lines = draw_lines
        self._lidar_prim.GetDrawPointsAttr().Set(draw_points)
        self._lidar_prim.GetDrawLinesAttr().Set(draw_lines)
        print(f"[VN_Mid360] 可视化已启用：点云={draw_points}，射线={draw_lines}")

    def disable_visualization(self):
        """禁用雷达可视化"""
        self.enable_visualization(draw_points=False, draw_lines=False)

    def start(self):
        """启动雷达（开始扫描和点云生成）"""
        if not self._timeline.is_playing():
            self._timeline.play()
        self._is_running = True
        print(f"[VN_Mid360] 雷达已启动：{self.prim_path}")

    def stop(self):
        """停止雷达"""
        self._is_running = False
        if self._timeline.is_playing():
            self._timeline.stop()
        print(f"[VN_Mid360] 雷达已停止：{self.prim_path}")

    def _get_current_timestamp(self) -> float:
        # return SysTimer.get_timestamp()
        return SysTimer.get_timestamp_plus8()

    def get_pointcloud(self) -> PointCloudWithMeta:
        """
        获取当前帧点云数据
        :return: 点云数组，形状为(N, 3)，每个元素为(x, y, z)坐标（单位：米）
        """
        if not self._range_sensor_if or not self._is_running:
            print("[VN_Mid360] 雷达未运行或接口无效，无法获取点云")
            return np.empty((0, 3))

        # 从雷达接口获取点云（返回numpy数组）
        pointcloud = self._range_sensor_if.get_point_cloud_data(self.prim_path)
        N = len(pointcloud)

        # 赋值时间戳和强度（逻辑同示例1）
        timestamps = self._make_timestamps(N, pointcloud, mode="return")
        print("timestamps:", timestamps[N-1])
        print(f"devices the sys timestamps  is {SysTimer.get_timestamp_plus8()}")
        #timestamps = base_timestamp + np.arange(N) * (0.1 / N)
        intensities = np.random.randint(0, 256, size=N, dtype=np.uint8)  #强度（Generic 没有原生强度后续用相机可补充

        # 返回自定义类实例
        return PointCloudWithMeta(
            points=pointcloud,
            timestamps=timestamps,
            intensities=intensities
        )




class VN_LidarRtx:
    def __init__(self, prim_path : str):
        """
        lidar init
        :param prim_path:
        """
        self.prim_path = prim_path
        self.lidar = LidarRtx(prim_path=self.prim_path)
        self.lidar.enable_visualization()

    def get_pointcloud(self):
        """

        :return:
        """
        current_frame = self.lidar.get_current_frame()
        print(current_frame)