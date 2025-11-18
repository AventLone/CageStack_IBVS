import time
import numpy as np
from isaacsim.core.api.robots.robot import Robot
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.api.world import World
from isaacsim.core.simulation_manager import SimulationManager
import omni.timeline

timeline = omni.timeline.get_timeline_interface()

from dataclasses import dataclass

@dataclass
class ForkPose:
    z: float = 0.0
    y: float = 0.0
    x: float = 0.0
    roll: float = 0.0
    pitch: float = 0.0
    cl: float = 0.0
    cr: float = 0.0


class Vehicle(Robot):
    def __init__(self, world: World, cfg: dict) -> None:
        super().__init__(prim_path=cfg["prim_path"], name=cfg["name"])
        world.scene.add(self)
        world.reset()
        # SimulationManager.enable_on_stop_callback(False)
        # self._world = world

        self.set_joint_velocities(np.zeros(len(self.dof_names), dtype=np.float32))
        self._fork_joints = [
            self.get_dof_index(joint) for joint in cfg["fork_joint_names"]
        ]
        self._fork_joints_1 = self._fork_joints[0]
        # self._door_joints = [
        #     self.get_dof_index(joint) for joint in cfg["door_joint_names"]
        # ]
        self._drive_wheels = [
            self.get_dof_index(joint) for joint in cfg["drive_wheel_joint_names"]
        ]
        self._steer_wheels = [
            self.get_dof_index(joint) for joint in cfg["steer_wheel_joint_names"]
        ]

        self._drive_wheel_radius: float = cfg["drive_wheel_radius"]

        self._drive_position_cumu = [0, 0]
        self._drive_position_cumu2 = [0, 0]
        self._drive_position_prev = None

        self._drive_velocity_w: float = 0.0
        self._drive_velocity_v: float = 0.0
        self._time_last_step = 0.0

        world.add_physics_callback(cfg["name"], self._on_step)

    def _on_step(self, dt: float):   # Callback in World
        if not timeline.is_playing():
            return
        current_position = self.drive_position
        if self._drive_position_prev is None:
            self._drive_position_prev = current_position
            return
        try:
            dp = (current_position - self._drive_position_prev + np.pi) % (2.0 * np.pi) - np.pi
        except Exception:
            return
        self._drive_position_cumu += dp
        self._drive_position_prev = current_position

        time_now = time.time()
        if self._time_last_step != 0:
            dt = time_now - self._time_last_step
        self._drive_velocity_w = dp[0] / dt
        self._drive_velocity_v = self._drive_velocity_w * self._drive_wheel_radius
        self._time_last_step = time_now


    # def moveFork(self, pose: np.ndarray):   # Position Control
    #     action = ArticulationAction(joint_positions=pose[:len(self._fork_joints)],
    #                                 joint_indices=self._fork_joints)
    #     self._robot.apply_action(action)

    def moveFork(self, velocities: np.ndarray | list):   # Velocity Control
        action = ArticulationAction(joint_velocities=velocities[:len(self._fork_joints)],
                                    joint_indices=self._fork_joints)
        self.apply_action(action)

    def moveFork2(self, pose: np.ndarray):
        target_pose = pose.copy()

        if target_pose[0] > 1.1:
            door_joint = self._door_joints[0]
            self.apply_action(
                ArticulationAction(
                    joint_positions=np.array([float(self._fork_pose)]),
                    joint_indices=[self._fork_joints_1],
                )
            )

            target_pose[0] -= 1.1
            fork_joints = [door_joint] + self._fork_joints[1:]
        else:
            self._fork_pose = float(target_pose[0])
            fork_joints = self._fork_joints

        self.apply_action(
            ArticulationAction(
                joint_positions=target_pose[: len(fork_joints)],
                joint_indices=fork_joints,
            )
        )

    def move(self, velocity: float):
        action = ArticulationAction(joint_velocities=np.full(len(self._drive_wheels), velocity, np.float32),
                                   joint_indices=self._drive_wheels)
        self.apply_action(action)

    def steer(self, angle: float):   # Position Control
        action = ArticulationAction(joint_positions=np.full(len(self._drive_wheels), angle, np.float32),
                                    joint_indices=self._steer_wheels)
        self.apply_action(action)

    def setCmd(self, cmd: np.ndarray):
        self.move(cmd[0])
        self.steer(cmd[1])
        self.moveFork(cmd[2:])
        # self.moveFork2(cmd[2:])

    @property
    def pose(self):
        """
        Pose of the vehicle
        """
        return self.get_world_pose()

    @property
    def fork_pose(self) -> ForkPose:
        return ForkPose(*self.get_joint_positions(self._fork_joints))

    @property
    def linear_velocity(self) -> float:
        return self.get_joint_velocities(self._drive_wheels)[0] * self._drive_wheel_radius

    @property
    def angular_velocity(self) -> float:
        return self.get_joint_velocities(self._drive_wheels)[0]

    @property
    def rpm(self) -> float:
        """
        Revolutions per minute
        """
        return self.get_joint_velocities(self._drive_wheels)[0] / (2.0 * np.pi) * 60.0


    @property
    def drive_velocity_rad(self) -> np.ndarray:
        """
        Velocity of the drive wheel, unit: rad/s
        """
        return self.get_joint_velocities(self._drive_wheels)

    @property
    def drive_velocity_meter(self) -> np.ndarray:
        """
        Velocity of the drive wheel, unit: meter/s
        """
        return self.get_joint_velocities(self._drive_wheels) * self._drive_wheel_radius

    @property
    def drive_velocity_w(self) -> float:
        """
        Velocity of the drive wheel, unit: rad/s
        """
        return float(self._drive_velocity_w)

    @property
    def drive_velocity_v(self) -> float:
        """
        Velocity of the drive wheel, unit: meter/s
        """
        return self._drive_velocity_v

    @property
    def drive_rpm(self) -> np.ndarray:
        """
        Velocity of the drive wheel, unit: r/min
        """
        return self.get_joint_velocities(self._drive_wheels) / (2.0 * np.pi) * 60.0

    @property
    def drive_position(self) -> np.ndarray:
        """
        Incremental Encoder: return angle of the drive wheel, unit: radian
        """
        return self.get_joint_positions(self._drive_wheels)

    @property
    def drive_cumulative_position(self):
        """
        Absolute Encoder: Cumulative rotation of the drive wheel, unit: radian
        """
        return self._drive_position_cumu

    @property
    def drive_cumulative_distance(self):
        """
        Total distance that the drive wheel has gone, unit: meter
        """
        return np.array(self._drive_position_cumu) * self._drive_wheel_radius
        # return self._drive_position_cumu * self._drive_wheel_radius, self._drive_position_cumu2

    @property
    def steer_angle(self) -> float:
        return self.get_joint_positions(self._steer_wheels)[0]

class AckermannVehicle(Vehicle):
    def __init__(self, world: World, cfg: dict):
        super().__init__(world=world, cfg=cfg)
        self._drive_wheels = [self.get_dof_index(joint)
                              for joint in cfg["drive_wheel_joint_names"]]
        self._steer_wheels = [self.get_dof_index(joint)
                              for joint in cfg["steer_wheel_joint_names"]]
        self._drive_wheel_radius = cfg["drive_wheel_radius"]

        self._drive_position_cumu = [0, 0]
        self._drive_position_prev = None

    def _on_step(self, dt: float):   # Callback in World
        current_position = self.drive_position
        if self._drive_position_prev is None:
            self._drive_position_prev = current_position
            return
        dp = (current_position - self._drive_position_prev + np.pi) % (2 * np.pi) - np.pi
        self._drive_position_cumu += dp
        self._drive_position_prev = current_position


    def move(self, velocity: float):
        action = ArticulationAction(
            joint_velocities=[velocity, velocity], joint_indices=self._drive_wheels
        )
        self.apply_action(action)

    def steer(self, angle: float):  # Position Control
        action = ArticulationAction(
            joint_positions=np.array([angle, angle], dtype=np.float32),
            joint_indices=self._steer_wheels,
        )
        self.apply_action(action)

    def setCmd(self, cmd: np.ndarray):
        self.move(cmd[0])
        self.steer(cmd[1])
        self.moveFork(cmd[2:5])

    @property
    def drive_velocity_rad(self) -> np.ndarray:
        """
        Velocity of the drive wheel, unit: rad/s
        """
        return self.get_joint_velocities(self._drive_wheels)

    @property
    def drive_velocity_meter(self) -> np.ndarray:
        """
        Velocity of the drive wheel, unit: meter/s
        """
        return self.get_joint_velocities(self._drive_wheels) * self._drive_wheel_radius

    @property
    def drive_position(self) -> np.ndarray:
        """
        Incremental Encoder: return angle of the drive wheel, unit: radian
        """
        return self.get_joint_positions(self._drive_wheels)

    @property
    def drive_cumulative_position(self):
        """
        Absolute Encoder: Cumulative rotation of the drive wheel, unit: radian
        """
        return self._drive_position_cumu

    @property
    def drive_cumulative_distance(self):
        """
        Total distance that the drive wheel has gone, unit: meter
        """
        return np.array(self._drive_position_cumu) * self._drive_wheel_radius

    @property
    def steer_angle(self) -> float:
        return self.get_joint_positions(self._steer_wheels)[0]
