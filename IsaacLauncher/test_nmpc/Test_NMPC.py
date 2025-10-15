from utils import ConfigLoader
from isaacsim import SimulationApp
import asyncio
from protos import VehicleControl_pb2
import ecal.core.core as ecal_core
from ecal.core.subscriber import ProtoSubscriber
from ecal.core.publisher import ProtoPublisher
import math, sys, queue
import numpy as np
from collections import deque
from nmpc import NMPC
from utils import SimTimer
# from ecal.core._ecal_cffi import lib as ecal_c   # 暴露 C API, 包含 etime_set_nanoseconds

def quat2rpy(quat):
    try:
        qw = quat.GetReal()
        qx, qy, qz = quat.GetImaginary()
    except:
        qw, qx, qy, qz = quat

    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (qw * qy - qz * qx)
    sinp = max(-1.0, min(1.0, sinp))  # 数值钳制
    pitch = math.asin(sinp)

    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw  # (rx, ry, rz)

def wrap_to_pi(a: float) -> float:
    # 映射到 [-pi, pi)
    return (a + math.pi) % (2 * math.pi) - math.pi



class TestPre:
    def __init__(self) -> None:
        if not ecal_core.is_initialized():
            ecal_core.initialize(sys.argv, "Test NMPC Pre")

        config = ConfigLoader("/home/vn/Documents/Projects/IsaacSimLauncher/configs/e_test.yaml").load()
        simulate_config = config["simulation_app"]
        self.app = SimulationApp(simulate_config["config"])

        from isaacsim.core.utils.stage import open_stage
        # open_stage(simulate_config["stage_file_path"])
        open_stage("/home/vn/Documents/IsaacSimAssets/Assets/Isaac/5.0/Isaac/VN/Stages/WarehouseFokrliftE_pallet.usd")

        from isaacsim.core.api.world import World
        self.world = World(physics_dt=simulate_config["physics_dt"],       # Physics time step
                           rendering_dt=simulate_config["rendering_dt"],   # Render timestep
                           stage_units_in_meters=1.0)
        self.require_reset = False
        self.world.reset()

        self.simu_timer = SimTimer(self.world)

        from devices import Vehicle2
        vehicle_cfg = config["vehicle"]
        self.forklift = Vehicle2(world=self.world, cfg=vehicle_cfg)


        self.cmds_queue = deque(maxlen=1)

        # Ecal publisher and subscriber
        self.state_publisher = ProtoPublisher(vehicle_cfg["vehicle_state_topic"], VehicleControl_pb2.State)
        self.goal_publisher = ProtoPublisher("goal", VehicleControl_pb2.Pose)
        self.cmd_subscriber = ProtoSubscriber("nmpc_cmd", VehicleControl_pb2.State)
        self.cmd_subscriber.set_callback(self._cmdHandler)

        # import carb
        # settings = carb.settings.get_settings()
        # settings.set("/app/runLoops/main/rateLimitEnabled", True)   # 开启限速
        # settings.set("/app/runLoops/main/rateLimitFrequency", 100)   # 例如 30Hz ⇒ RTF = 30 * physics_dt = 0.5×

        # Start sub tasks
        asyncio.get_event_loop().create_task(self._statePub())
        asyncio.get_event_loop().create_task(self._step())
        asyncio.get_event_loop().create_task(self._printError())
        # asyncio.get_event_loop().create_task(self._printTimeGap())

        from omni.usd import get_context
        stage = get_context().get_stage()

        self.goal_prim  = stage.GetPrimAtPath("/World/SM_PaletteA_02")
        self.forklift_prim = stage.GetPrimAtPath("/World/forklift_E/Links/front_left_wheel/mid")
        if not self.goal_prim .IsValid():
            raise RuntimeError("Prim not found: /World/SM_PaletteA_02")

        # relative_goal_position, yaw = self.self_and_goal
        # print(f"Start goal error: x {relative_goal_position[0]:.3f}, y {relative_goal_position[1]:.3f}, yaw {yaw:.3f}")


        # --- 用法示例 ---
        # poses = [(0.0, 0.0, 0.0),
        #          (1.0, 0.2, 0.1),
        #          (2.0, 0.6, 0.2),
        #          (3.0, 1.1, 0.25),
        #          (4.0, 1.6, 0.30)]
        # visualize_xytheta_path_with_arrows("PlannedPath",
        #                                    poses,
        #                                    draw_points=True,
        #                                    arrow_every=2,
        #                                    arrow_len=0.4)
        # from visualize_path import visualize_xytheta_path_with_arrows
        # poses = [(0.0, 0.0, 0.0),
        #          (1.0, 0.2, 0.1),
        #          (2.0, 0.6, 0.2),
        #          (3.0, 1.1, 0.25),
        #          (4.0, 1.6, 0.30)]
        # visualize_xytheta_path_with_arrows("PlannedPath",
        #                                    poses,
        #                                    draw_points=True,
        #                                    arrow_every=2,
        #                                    arrow_len=0.4)


    def run(self):
        while self.app.is_running():
            self.world.step(render=True)

            if self.world.is_stopped() and not self.require_reset:   # 播放/暂停与重置逻辑
                self.require_reset = True

            if self.world.is_playing():
                if self.require_reset:
                    self.world.reset()
                    self.require_reset = False

        self.app.close()

        if ecal_core.is_initialized():
            ecal_core.finalize()


    def _cmdHandler(self, topic_name, msg: VehicleControl_pb2.State, msg_time):
        self.cmds_queue.append(msg)

    async def _step(self):
        dt = 0.2
        ddt = 0.02
        steps = round(dt / ddt)
        while ecal_core.ok():
            if not self.world.is_playing():
                await asyncio.sleep(0.05)
                continue

            if len(self.cmds_queue) == 0:
                self.forklift.move(0.0)
                self.forklift.setSteerVelocity(0.0)
                await asyncio.sleep(0.1)
                continue

            cmd:VehicleControl_pb2.State = self.cmds_queue.popleft()

            self.forklift.move(cmd.drive_velocity)

            steer_ang = self.forklift.steer_angle
            for _ in range(steps):
                await self.simu_timer.sleep(ddt)
                steer_ang += cmd.steer_velocity * ddt
                self.forklift.steer(steer_ang)

            # print(f"Start: x {self.forklift.get_world_pose()[0][0]:.3f}, y {self.forklift.get_world_pose()[0][1]:.3f}")
            # self.forklift.steer(math.pi / 6.0)
            # self.forklift.move(1.0)
            # await self.simu_timer.sleep(10.0)
            # self.forklift.move(-1.0)
            # await self.simu_timer.sleep(10.0)
            # print(f"End: x {self.forklift.get_world_pose()[0][0]:.3f}, y {self.forklift.get_world_pose()[0][1]:.3f}")
            # break

    async def _printError(self):
        while ecal_core.ok():
            if not self.world.is_playing() or self.require_reset:
                await asyncio.sleep(0.05)
                continue
            # position, yaw = self.self_and_goal
            self_pose, goal_pose = self.self_and_goal
            print(f"error: x {goal_pose.x - self_pose.x:.3f}, y {goal_pose.y - self_pose.y:.3f}, yaw {goal_pose.yaw - self_pose.yaw:.3f}")
            # print(self_pose)
            await asyncio.sleep(2.0)


    async def _statePub(self):
        while ecal_core.ok():
            if not self.world.is_playing() or self.require_reset:
                await asyncio.sleep(0.05)
                continue

            state = VehicleControl_pb2.State()
            self_pose, goal_pose = self.self_and_goal

            state.drive_velocity = self.forklift.drive_velocity[0]
            state.steer_angle = self.forklift.steer_angle
            state.pose.x = self_pose.x
            state.pose.y = self_pose.y
            state.pose.yaw = self_pose.yaw
            self.state_publisher.send(state)

            self.goal_publisher.send(goal_pose)

            await asyncio.sleep(0.006)

    async def _printTimeGap(self):
        import time
        while ecal_core.ok():
            begin = time.perf_counter()
            await self.simu_timer.sleep(1.0)
            end = time.perf_counter()
            print(f"Real time{(end - begin) * 1000}")

    @property
    def self_and_goal(self):
        # from omni.usd import get_local_transform_matrix
        from omni.usd import get_world_transform_matrix

        position, orientation = self.forklift.get_world_pose()
        # forklift_pose = get_world_transform_matrix(self.forklift_prim)
        # forklift_position = forklift_pose.ExtractTranslation()
        # _, _, forklift_yaw = quat2rpy(forklift_pose.ExtractRotationQuat().GetNormalized())
        _, _, forklift_yaw = quat2rpy(orientation)

        self_pose = VehicleControl_pb2.Pose()
        self_pose.x = position[0] + 0.2687796
        self_pose.y = position[1]
        self_pose.yaw = forklift_yaw

        goal_pose = get_world_transform_matrix(self.goal_prim)
        goal_position = goal_pose.ExtractTranslation()
        _, _, yaw_a = quat2rpy(goal_pose.ExtractRotationQuat().GetNormalized())

        goal = VehicleControl_pb2.Pose()
        goal.x = goal_position[0] - 1.2
        goal.y = goal_position[1]
        goal.yaw = yaw_a

        return self_pose, goal

    @staticmethod
    def get_pose_relative_to(child_path: str, parent_path: str):
        from omni.usd import get_context, get_local_transform_matrix
        from pxr import UsdGeom, Gf, Usd

        stage = get_context().get_stage()

        child_prim  = stage.GetPrimAtPath(child_path)
        parent_prim = stage.GetPrimAtPath(parent_path)

        if not child_prim.IsValid():
            raise RuntimeError(f"Prim not found: {child_path}")

        if not parent_prim.IsValid():
            raise RuntimeError(f"Prim not found: {parent_path}")

        pose_a = get_local_transform_matrix(child_prim)
        pose_b = get_local_transform_matrix(parent_prim)

        position_a = pose_a.ExtractTranslation()
        _, _, yaw_a = quat2rpy(pose_a.ExtractRotationQuat().GetNormalized())
        yaw_a += math.pi / 2.0

        position_b = pose_b.ExtractTranslation()
        _, _, yaw_b = quat2rpy(pose_b.ExtractRotationQuat().GetNormalized())

        # pose_a2b = pose_b.GetInverse() * pose_a

        # position = pose_a2b.ExtractTranslation()
        # rotation = pose_a2b.ExtractRotationQuat().GetNormalized()
        # _, _, yaw = quat2rpy(rotation)
        return position_a - position_b, yaw_a - yaw_b

    @staticmethod
    def get_world_pose(prim_path: str):
        from omni.usd import get_context, get_local_transform_matrix
        from pxr import UsdGeom, Gf, Usd
        stage = get_context().get_stage()
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            raise RuntimeError(f"Prim not found: {prim_path}")

        pose = get_local_transform_matrix(prim)

        position = pose.ExtractTranslation()
        rotation = pose.ExtractRotationQuat().GetNormalized()
        _, _, yaw = quat2rpy(rotation)

        return position, yaw


if __name__ == "__main__":
    test = TestPre()
    test.run()
