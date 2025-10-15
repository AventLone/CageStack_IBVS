import numpy as np
from typing import List
from isaacsim.core.api.robots.robot import Robot
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.api.world import World
# from transforms3d.euler import quat2euler
import math
from dataclasses import dataclass

@dataclass
class ForkPose:
    z: float = 0.0
    y: float = 0.0
    x: float = 0.0
    roll: float = 0.0
    pitch: float = 0.0
    lc: float = 0.0
    rc: float = 0.0


class Vehicle():
    def __init__(self, world: World, cfg: dict) -> None:
        super().__init__()
        self._robot = Robot(prim_path=cfg["prim_path"], name=cfg["name"])
        world.scene.add(self._robot)
        world.reset()
        world.add_physics_callback(cfg["name"], self._on_step)

        self._robot.set_joint_velocities(np.zeros(len(self._robot.dof_names), dtype=np.float32))

        self._fork_joints = [self._robot.get_dof_index(joint) for joint in cfg["fork_joint_names"]]

        self._drive_wheels = [self._robot.get_dof_index(joint)
                              for joint in cfg["drive_wheel_joint_names"]]
        self._steer_wheels = [self._robot.get_dof_index(joint)
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

    def moveFork(self, pose: np.ndarray):   # Position Control
        action = ArticulationAction(joint_positions=pose[:len(self._fork_joints)],
                                    joint_indices=self._fork_joints)
        self._robot.apply_action(action)

    def move(self, velocity: float):
        action = ArticulationAction(joint_velocities=np.full(len(self._drive_wheels), velocity, np.float32),
                                   joint_indices=self._drive_wheels)
        self._robot.apply_action(action)

    def steer(self, angle: float):   # Position Control
        action = ArticulationAction(joint_positions=np.full(len(self._drive_wheels), angle, np.float32),
                                    joint_indices=self._steer_wheels)
        self._robot.apply_action(action)

    def setSteerVelocity(self, velocity: float):
        action = ArticulationAction(joint_velocities=np.full(len(self._drive_wheels), velocity, np.float32),
                                    joint_indices=self._steer_wheels)
        self._robot.apply_action(action)

    def setCmd(self, cmd: np.ndarray):
        self.move(cmd[0])
        self.steer(cmd[1])
        self.moveFork(cmd[2:])

    @property
    def pose(self):
        """
        Pose of the vehicle
        """
        return self._robot.get_world_pose()

    @property
    def fork_pose(self) -> ForkPose:
        return ForkPose(*self._robot.get_joint_positions(self._fork_joints))

    @property
    def drive_velocity(self) -> List[float]:
        """
        Velocity of the drive wheel, unit: rad/s
        """
        return self._robot.get_joint_velocities(self._drive_wheels)

    @property
    def drive_position(self) -> List[float]:
        """
        Incremental Encoder: return angle of the drive wheel, unit: radian
        """
        return self._robot.get_joint_positions(self._drive_wheels)

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
        return self._drive_position_cumu * self._drive_wheel_radius

    @property
    def steer_angle(self) -> float:
        return self._robot.get_joint_positions(self._steer_wheels)[0]

class AckermannVehicle(Vehicle):
    def __init__(self, world: World, cfg: dict):
        super().__init__(world=world, cfg=cfg)
        self._drive_wheels = [self._robot.get_dof_index(joint)
                              for joint in cfg["drive_wheel_joint_names"]]
        self._steer_wheels = [self._robot.get_dof_index(joint)
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
        action = ArticulationAction(joint_velocities=[velocity, velocity],
                                   joint_indices=self._drive_wheels)
        self._robot.apply_action(action)

    def steer(self, angle: float):   # Position Control
        action = ArticulationAction(joint_positions=np.array([angle, angle], dtype=np.float32),
                                    joint_indices=self._steer_wheels)
        self._robot.apply_action(action)

    def setCmd(self, cmd: np.ndarray):
        self.move(cmd[0])
        self.steer(cmd[1])
        self.moveFork(cmd[2:5])

    @property
    def drive_velocity(self) -> List[float]:
        """
        Velocity of the drive wheel, unit: rad/s
        """
        return self._robot.get_joint_velocities(self._drive_wheels)

    @property
    def drive_position(self) -> List[float]:
        """
        Incremental Encoder: return angle of the drive wheel, unit: radian
        """
        return self._robot.get_joint_positions(self._drive_wheels)

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
        return self._drive_position_cumu * self._drive_wheel_radius

    @property
    def steer_angle(self) -> float:
        return self._robot.get_joint_positions(self._steer_wheels)[0]


# class VehicleST(AckermannVehicle):
#     def __init__(self, world: World, cfg: dict):
#         super().__init__(world, cfg)

#     def setCmd(self, cmd: np.ndarray):
#         self.move(cmd[0])
#         self.steer(cmd[1])
#         self.moveFork(cmd[2])

@dataclass
class Pose2D:
    x: float
    y: float
    th: float

    def __format__(self, spec: str):
        return (f"Pose2D("
                f"x={format(self.x, spec)}, "
                f"y={format(self.y, spec)}, "
                f"th={format(self.th, spec)})")

class Vehicle2(Robot):
    def __init__(self, world: World, cfg: dict) -> None:
        super().__init__(prim_path=cfg["prim_path"], name=cfg["name"])
        world.scene.add(self)
        world.reset()
        # world.add_physics_callback(cfg["name"], self._on_step)
        # self.initialize()

        self.set_joint_velocities(np.zeros(len(self.dof_names), dtype=np.float32))

        self._fork_joints = [self.get_dof_index(joint) for joint in cfg["fork_joint_names"]]

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

    def moveFork(self, pose: np.ndarray):   # Position Control
        action = ArticulationAction(joint_positions=pose[:len(self._fork_joints)],joint_indices=self._fork_joints)
        self.apply_action(action)

    def move(self, velocity: float):
        action = ArticulationAction(joint_velocities=np.full(len(self._drive_wheels), velocity, np.float32),
                                   joint_indices=self._drive_wheels)
        self.apply_action(action)

    def steer(self, angle: float):   # Position Control
        action = ArticulationAction(joint_positions=np.full(len(self._drive_wheels), angle, np.float32),
                                    joint_indices=self._steer_wheels)
        self.apply_action(action)

    def setSteerVelocity(self, velocity: float):
        action = ArticulationAction(joint_velocities=np.full(len(self._drive_wheels), velocity, np.float32),
                                    joint_indices=self._steer_wheels)
        self.apply_action(action)

    def setCmd(self, cmd: np.ndarray):
        self.move(cmd[0])
        self.steer(cmd[1])
        self.moveFork(cmd[2:])

    @property
    def pose(self) -> Pose2D:
        position, orientation = self.get_world_pose()

        position.astype(float)
        orientation.astype(float)

        qw, qx, qy, qz = orientation
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        return Pose2D(float(position[0]), float(position[1]), float(yaw))

    @property
    def fork_pose(self) -> ForkPose:
        return ForkPose(*self.get_joint_positions(self._fork_joints))

    @property
    def drive_velocity(self) -> np.ndarray:
        """
        Velocity of the drive wheel, unit: rad/s
        """
        return self.get_joint_velocities(self._drive_wheels)

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
        return self._drive_position_cumu * self._drive_wheel_radius

    @property
    def steer_angle(self) -> float:
        return self.get_joint_positions(self._steer_wheels)[0]
