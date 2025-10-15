import numpy as np
import threading, time, sys
import ecal.core.core as ecal_core
from ecal.core.publisher import ProtoPublisher
from ecal.core.subscriber import ProtoSubscriber
from devices import AckermannVehicle, ForkPose, PrimPose, VN_IMU
from protos import vehicle_state_msg_pb2
from utils import LoggerUtil, SysTimer
import asyncio
from dataclasses import dataclass


logger = LoggerUtil.get_logger("vehicle_state")


@dataclass
class VehicleState_Forklift:
    steer_angle = 0
    drive_velocity = 0
    wheel_l, wheel_r = 0, 0
    fork_pose = ForkPose()
    robot_pose = [[0, 0, 0], [1, 0, 0, 0]]



class VehicleStateServer(threading.Thread):
    def __init__(self, vehicle : AckermannVehicle, imu : VN_IMU, config : dict, time_period : float = 0.01) -> None:
        super().__init__()
        self.pub = ProtoPublisher(config["vehicle_state_topic"], vehicle_state_msg_pb2.VehicleStateMsg)
        self.run_thread = True
        self.vehicle = vehicle
        self.imu = imu
        self.config = config
        self.time_period = time_period
        self.vehicle_pose = None
        if "pose_prim_path" in config:
            self.vehicle_pose = PrimPose(config["pose_prim_path"])
        
        # 初始化IMU设备
        self.imu_device = None
        if "imu_prim_path" in config:
            try:
                imu_config = config.get("imu_config", {})
                self.imu_device = VN_IMU(config["imu_prim_path"], imu_config)
                logger.info(f"IMU设备初始化成功: {config['imu_prim_path']}")
            except Exception as e:
                logger.error(f"IMU设备初始化失败: {e}")
        
        # 订阅键盘控制主题
        self.keyboard_sub = ProtoSubscriber(config["keyboard_topic"], vehicle_state_msg_pb2.VehicleStateMsg)
        self.keyboard_sub.set_callback(self.keyboardCallback)
        
        # 订阅调整控制主题
        self.adjustctrl_sub = ProtoSubscriber(config["adjust_ctrl_topic"], vehicle_state_msg_pb2.VehicleStateMsg)
        self.adjustctrl_sub.set_callback(self.adjustCtrlCallback)
        
        # 新增：订阅vehicle/cmd主题以接收来自UI的直接控制指令
        self.ui_cmd_sub = ProtoSubscriber(config["ui_cmd_topic"], vehicle_state_msg_pb2.VehicleStateMsg)
        self.ui_cmd_sub.set_callback(self.uiCmdCallback)

        self.cmd_msg = None
        self.cmd_msg_lock = threading.Lock()
        self.vehicle_state = VehicleState_Forklift()
        self.vehicle_state_lock = threading.Lock()
        self.control_mode = 1  # 默认UI控制模式(1)
        self.pallet_pose = PrimPose("/World/SM_PaletteA_01")
        
        print("VehicleStateServer initialized with control mode switching support")

    def step(self):

        if self.vehicle is None:
            logger.error("vehicle is None")
            return

        # control
        cmd_msg = self.get_cmd_msg()
        if cmd_msg is not None:
            self.vehicle.setCmd(cmd_msg)

        # get state
        self.vehicle_state_lock.acquire()
        self.vehicle_state.steer_angle = self.vehicle.steer_angle
        self.vehicle_state.drive_velocity = np.sum(self.vehicle.drive_velocity) / 2
        self.vehicle_state.wheel_l, self.vehicle_state.wheel_r = self.vehicle.drive_cumulative_position
        self.vehicle_state.fork_pose = self.vehicle.fork_pose
        self.vehicle_state_lock.release()

    def get_vehicle_state(self):
        self.vehicle_state_lock.acquire()
        vehicle_state = self.vehicle_state
        self.vehicle_state_lock.release()
        return vehicle_state


    def run(self):
        """线程启动时执行的方法"""
        next_t = time.perf_counter()
        while self.run_thread:

            if self.vehicle is None:
                logger.error("vehicle is None")
                continue

            state = vehicle_state_msg_pb2.VehicleStateMsg()
            state.timestamp = SysTimer.get_timestamp()  # 使用毫秒时间戳
            state.seq_num = SysTimer.get_time_seqnum()
            vehicle_state = self.get_vehicle_state()
            state.steer_angle = vehicle_state.steer_angle
            state.drive_velocity = np.sum(vehicle_state.drive_velocity) / 2
            state.wheel_l, state.wheel_r = vehicle_state.wheel_l, vehicle_state.wheel_r
            state.fork_x = vehicle_state.fork_pose.x
            state.fork_y = vehicle_state.fork_pose.y
            state.fork_z = vehicle_state.fork_pose.z
            state.fork_pitch = vehicle_state.fork_pose.pitch
            state.fork_cl = vehicle_state.fork_pose.lc
            state.fork_cr = vehicle_state.fork_pose.rc
            state.fork_roll = vehicle_state.fork_pose.roll
            state.robot_pose.position.x = vehicle_state.robot_pose[0][0]
            state.robot_pose.position.y = vehicle_state.robot_pose[0][1]
            state.robot_pose.position.z = vehicle_state.robot_pose[0][2]
            state.robot_pose.orientation.x = vehicle_state.robot_pose[1][0]
            state.robot_pose.orientation.y = vehicle_state.robot_pose[1][1]
            state.robot_pose.orientation.z = vehicle_state.robot_pose[1][2]
            state.robot_pose.orientation.w = vehicle_state.robot_pose[1][3]
            
            # 设置默认值，确保所有字段都有值
            state.adjust_control = self.control_mode
            state.data_record = False
            state.dataidx_upload = 0
            state.ClampPressure = 0.0
            state.HSwitchL = False
            state.HSwitchR = False
            state.VSwitchL = False
            state.VSwitchR = False
            state.torquestatus = False

            if self.control_mode == 3:
                state.adjust_control = True
                state.target_pose.position.x = -1
                state.target_pose.position.y = 0
                state.target_pose.position.z = 0.22
                state.target_pose.orientation.x = 1
                state.target_pose.orientation.y = 0
                state.target_pose.orientation.z = 0
                state.target_pose.orientation.w = 0

            if self.imu is not None:
                lin_acc, ang_vel, orientation = self.imu.get_frame()
                state.imu.linear_acceleration_x = lin_acc[0]
                state.imu.linear_acceleration_y = lin_acc[1]
                state.imu.linear_acceleration_z = lin_acc[2]
                state.imu.anguldar_velocity_x = ang_vel[0]
                state.imu.anguldar_velocity_y = ang_vel[1]
                state.imu.anguldar_velocity_z = ang_vel[2]
                state.imu.orientation.x = orientation[0]
                state.imu.orientation.y = orientation[1]
                state.imu.orientation.z = orientation[2]
                state.imu.orientation.w = orientation[3]

            # logger.info(state)
            self.pub.send(state)

            # 固定周期调度，避免频率漂移
            next_t += self.time_period
            dt = next_t - time.perf_counter()
            if dt > 0:
                time.sleep(dt)
            else:
                # 掉帧时重置起点，避免一直为负
                next_t = time.perf_counter()


    def stop(self) -> None:
        self.run_thread = False

    def get_cmd_msg(self):
        self.cmd_msg_lock.acquire()
        cmd_msg = self.cmd_msg
        self.cmd_msg_lock.release()
        return cmd_msg

    def set_cmd_msg(self, cmd_msg:np.ndarray):
        self.cmd_msg_lock.acquire()
        self.cmd_msg = cmd_msg
        self.cmd_msg_lock.release()

    def keyboardCallback(self, topic_name, msg, msg_time) -> None:
        # 更新控制模式
        if hasattr(msg, 'adjust_control'):
            self.control_mode = msg.adjust_control
        
        # 只有在键盘控制模式(2)下才处理键盘指令
        if self.control_mode == 2:
            drive_velocity = msg.drive_velocity
            steer_angle = msg.steer_angle
            fork_x = msg.fork_x
            fork_y = msg.fork_y
            fork_z = msg.fork_z
            fork_pitch = msg.fork_pitch
            data_record = msg.data_record
            cmd_msg = np.array([drive_velocity, steer_angle, fork_z, fork_y, fork_pitch])
            # print("topic_name : ", topic_name, "cmd_msg : ", cmd_msg, msg_time)
            self.set_cmd_msg(cmd_msg)

    def adjustCtrlCallback(self, topic_name, msg, msg_time) -> None:
        # 更新控制模式
        if hasattr(msg, 'adjust_control'):
            self.control_mode = msg.adjust_control
        
        # 只有在自适应控制模式(3)下才处理调整控制指令
        if self.control_mode == 3:
            print(msg)
            drive_velocity = msg.drive_velocity
            steer_angle = msg.steer_angle
            fork_x = msg.fork_x
            fork_y = msg.fork_y
            fork_z = msg.fork_z
            fork_pitch = msg.fork_pitch
            cmd_msg = np.array([drive_velocity, steer_angle, fork_z, fork_y, fork_pitch])
            self.set_cmd_msg(cmd_msg)

    def uiCmdCallback(self, topic_name, msg, msg_time) -> None:
        """处理来自ui/cmd主题的UI控制指令"""
        # 更新控制模式
        if hasattr(msg, 'adjust_control'):
            self.control_mode = msg.adjust_control
        
        # 只有在UI控制模式(1)下才处理UI指令
        if self.control_mode == 1:
            drive_velocity = msg.drive_velocity
            steer_angle = msg.steer_angle
            fork_z = msg.fork_z
            fork_y = msg.fork_y 
            fork_pitch = msg.fork_pitch
            print(f"收到UI指令: 速度={msg.drive_velocity:.2f}, 转向={msg.steer_angle:.2f}, "
                  f"货叉Z={msg.fork_z:.2f}, 货叉Y={msg.fork_y:.2f}, "
                  f"货叉P={msg.fork_pitch:.2f}, 货叉C={msg.fork_cl:.2f}, "
                  f"时间戳={msg.timestamp}")
            # 创建控制指令数组
            cmd_msg = np.array([drive_velocity, steer_angle, fork_z, fork_y, fork_pitch])
            self.set_cmd_msg(cmd_msg)