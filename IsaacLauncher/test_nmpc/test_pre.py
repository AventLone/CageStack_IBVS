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

# def get_pose_relative_to(child_path: str, parent_path: str):
#     stage = get_context().get_stage()
#     cache = UsdGeom.XformCache(Usd.TimeCode.Default())
#     A = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(child_path))
#     B = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(parent_path))
#     M = B.GetInverse() * A                     # parent系下的 child 变换
#     tf = Gf.Transform(); tf.SetMatrix(M)
#     return tf.GetTranslation(), tf.GetRotation().GetQuat(), tf.GetScale()


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
        open_stage(simulate_config["stage_file_path"])

        from isaacsim.core.api.world import World
        self.world = World(physics_dt=simulate_config["physics_dt"],       # Physics time step
                           rendering_dt=simulate_config["rendering_dt"],   # Render timestep
                           stage_units_in_meters=1.0)
        self.require_reset = False
        self.world.reset()

        from devices import Vehicle2
        vehicle_cfg = config["vehicle"]
        self.forklift = Vehicle2(world=self.world, cfg=vehicle_cfg)
        # self.world.add_physics_callback("cmd_handler", self._on_cmd)
        # self.world.scene.add(self.forklift)
        # self.world.reset()

        # from isaacsim.robot.wheeled_robots.controllers.ackermann_controller import AckermannController
        # self.controller = AckermannController(
        #     "ackermann_controller", wheel_base=1.5, track_width=1.0, front_wheel_radius=0.3
        # )

        # self.cmd_queue = queue.Queue(maxsize=16)
        self.cmd_queue = deque(maxlen=16)

        # Ecal publisher and subscriber
        self.state_publisher = ProtoPublisher(vehicle_cfg["vehicle_state_topic"], VehicleControl_pb2.State)
        self.goal_publisher = ProtoPublisher("goal", VehicleControl_pb2.Pose)
        self.cmd_subscriber = ProtoSubscriber("nmpc_cmd", VehicleControl_pb2.State)
        self.cmd_subscriber.set_callback(self._cmdHandler)

        # Start sub tasks
        asyncio.get_event_loop().create_task(self._statePub())
        asyncio.get_event_loop().create_task(self._step())

        from omni.usd import get_context
        # from pxr import UsdGeom, Gf, Usd

        stage = get_context().get_stage()
        
        self.goal_prim  = stage.GetPrimAtPath("/World/SM_PaletteA_02")
        if not self.goal_prim .IsValid():
            raise RuntimeError("Prim not found: /World/SM_PaletteA_02")
        # parent_prim = stage.GetPrimAtPath(parent_path)

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
        self.cmd_queue.append(msg)


    async def _step(self):
        while ecal_core.ok():
            if not self.world.is_playing():
                await asyncio.sleep(0.05)
                continue

            if len(self.cmd_queue) == 0:
                self.forklift.move(0.0)
                self.forklift.setSteerVelocity(0.0)
                await asyncio.sleep(0.01)
                continue

            cmd:VehicleControl_pb2.State = self.cmd_queue.popleft()

            self.forklift.move(cmd.drive_velocity)
            self.forklift.setSteerVelocity(cmd.steer_velocity)
            await asyncio.sleep(0.1)


    async def _statePub(self):
        while ecal_core.ok():
            if not self.world.is_playing() or self.require_reset:
                await asyncio.sleep(0.05)
                continue

            state = VehicleControl_pb2.State()
            goal = VehicleControl_pb2.Pose()

            state.drive_velocity = self.forklift.drive_velocity[0]
            state.steer_angle = self.forklift.steer_angle
            self.state_publisher.send(state)

            # position, yaw = self.get_pose_relative_to(child_path="/World/SM_PaletteA_02",
            #                                           parent_path="/World/forklift_E/Links/body")
            position, yaw = self.relative_goal

            goal.x = position[0]
            goal.y = position[1]
            goal.yaw = wrap_to_pi(yaw)

            self.goal_publisher.send(goal)
            
            await asyncio.sleep(0.02)

    @property
    def relative_goal(self):
        from omni.usd import get_local_transform_matrix
        goal_pose = get_local_transform_matrix(self.goal_prim)

        goal_position = goal_pose.ExtractTranslation()
        _, _, yaw_a = quat2rpy(goal_pose.ExtractRotationQuat().GetNormalized())
        yaw_a += math.pi / 2.0

        # position_b = pose_b.ExtractTranslation()
        position, orientation = self.forklift.get_world_pose()
        _, _, yaw_b = quat2rpy(orientation)

        # pose_a2b = pose_b.GetInverse() * pose_a

        # position = pose_a2b.ExtractTranslation()
        # rotation = pose_a2b.ExtractRotationQuat().GetNormalized()
        # _, _, yaw = quat2rpy(rotation)
        return goal_position - position, yaw_a - yaw_b

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
