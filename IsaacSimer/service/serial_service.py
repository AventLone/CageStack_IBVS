from typing import Dict, List, Optional, Callable, Any
import threading
import time
from dataclasses import dataclass
from utils.parser import InputDecoder, OutputEncoder, Package
from protos import vehicle_state_msg_pb2
from ecal.core.publisher import ProtoPublisher, StringPublisher
from ecal.core.subscriber import ProtoSubscriber, StringSubscriber
from service.base import BytesSubscriber, BytesPublisher
import ecal.core.core as ecal_core
import sys
from utils import PoseTrans, SimulationTimer, LoggerUtil
import numpy as np

logger = LoggerUtil.get_logger("agv_interface")

class AGVMsgTrans:
    index = 0

    @staticmethod
    def simState2AGVSensor(encoder, msg, data_index_return) -> Optional[Package]:
        raise NotImplementedError("Subclasses must implement the simState2AGVSensor method")

    @staticmethod
    def AGVActuator2SimCmd(decoder, msg):
        raise NotImplementedError("Subclasses must implement the AGVActuator2SimCmd method")

class ST_MsgTrans(AGVMsgTrans):
    @staticmethod
    def simState2AGVSensor(encoder, msg, data_index_return) -> Optional[Package]:
        battery = 66
        velocity = msg.drive_velocity
        drive_dist = msg.drive_dist
        steering_angle = msg.steer_angle
        fork_z = msg.fork_z
        linear_acceleration_x = msg.imu.linear_acceleration_x
        linear_acceleration_y = msg.imu.linear_acceleration_y
        linear_acceleration_z = msg.imu.linear_acceleration_z
        anguldar_velocity_x = msg.imu.anguldar_velocity_x
        anguldar_velocity_y = msg.imu.anguldar_velocity_y
        anguldar_velocity_z = msg.imu.anguldar_velocity_z

        yaw, pitch, roll = PoseTrans.quaternion_to_euler(msg.imu.orientation.x, msg.imu.orientation.y, \
                                                         msg.imu.orientation.z, msg.imu.orientation.w)

        wheel_coder_l = msg.wheel_l
        wheel_coder_r = msg.wheel_r
        data_idx = msg.seq_num
        # logger.info(f"timestamp = {msg.timestamp}, seq_num = {data_idx}, steering_angle = {steering_angle}, " + \
        #             f"velocity = {velocity}, drive_dist={drive_dist}, " + \
        #             f"roll = {roll}, pitch = {pitch}, yaw = {yaw}")
        if data_idx != ST_MsgTrans.index + 1:
            logger.warning(f"index jump err!!! ST_MsgTrans.index = {ST_MsgTrans.index}, seq_num = {data_idx}")
        ST_MsgTrans.index = data_idx

        encoder.update_value2("DataIndex", data_idx, 4)
        encoder.update_value2("DataIndexReturn", data_index_return, 4)
        encoder.update_value("RPMSensor", 2, "M", velocity * 0.1)
        encoder.update_value("HolzerCoder", 2, "M", drive_dist)
        encoder.update_value("IncrementalSteeringCoder", 2, "", steering_angle)
        encoder.update_value2("BatterySencer", battery, 2)
        # imu sensor msg
        encoder.update_value("Gyroscope", 4, "X", roll)
        encoder.update_value("Gyroscope", 4, "Y", pitch)
        encoder.update_value("Gyroscope", 4, "Z", -yaw)
        encoder.update_value("AngularVelocitySensor", 2, "X", anguldar_velocity_x)
        encoder.update_value("AngularVelocitySensor", 2, "Y", anguldar_velocity_y)
        encoder.update_value("AngularVelocitySensor", 2, "Z", anguldar_velocity_z)
        encoder.update_value("Accelerometer", 2, "X", linear_acceleration_x)
        encoder.update_value("Accelerometer", 2, "Y", linear_acceleration_y)
        encoder.update_value("Accelerometer", 2, "Z", linear_acceleration_z)
        # SwitchSencer
        encoder.update_switch_value("SwitchSencer", 50, msg.HSwitchL, "")
        encoder.update_switch_value("SwitchSencer", 51, msg.HSwitchR, "")
        if fork_z >= 0.19:
            encoder.update_switch_value("SwitchSencer", 32, True, "")
            encoder.update_switch_value("SwitchSencer", 33, False, "")
        elif fork_z <= 0.01:
            encoder.update_switch_value("SwitchSencer", 32, False, "")
            encoder.update_switch_value("SwitchSencer", 33, True, "")
        else:
            encoder.update_switch_value("SwitchSencer", 32, False, "")
            encoder.update_switch_value("SwitchSencer", 33, False, "")

        # get pack
        pack = encoder.encode_package()
        return pack

    @staticmethod
    def AGVActuator2SimCmd(decoder, msg):
        pack = Package(msg, len(msg))
        decoder.decode_package(pack)
        velocity = []
        steering = []
        fork_z, switch = [], []
        decoder.get_value("MoveDevice", velocity, "")
        decoder.get_value("SteeringDevice", steering, "")
        decoder.get_value("ForkDevice", fork_z, "Z")
        decoder.get_value("SwitchActuator", switch)
        index = bytearray()
        decoder.get_value2("DataIndex", index, 4)
        index = int.from_bytes(index, byteorder='big', signed=False)
        data_index_return = index
        # print("velocity : ", velocity, "steering : ", steering, "fork_z : ", fork_z, "switch : ", switch)
        vehicle_state = vehicle_state_msg_pb2.VehicleStateMsg()
        vehicle_state.drive_velocity = velocity[0] if len(velocity) > 0 else 0
        vehicle_state.steer_angle = steering[0] if len(steering) > 0 else 0
        vehicle_state.fork_z = fork_z[0] if len(fork_z) > 0 else 0
        return data_index_return, vehicle_state

class E_MsgTrans(AGVMsgTrans):
    @staticmethod
    def simState2AGVSensor(encoder, msg, data_index_return) -> Optional[Package]:
        battery = 88
        velocity = msg.drive_velocity
        steering_angle = msg.steer_angle
        fork_z = msg.fork_z
        fork_y = msg.fork_y
        fork_c = msg.fork_cl
        fork_p = msg.fork_pitch
        linear_acceleration_x = msg.imu.linear_acceleration_x
        linear_acceleration_y = msg.imu.linear_acceleration_y
        linear_acceleration_z = msg.imu.linear_acceleration_z
        anguldar_velocity_x = msg.imu.anguldar_velocity_x
        anguldar_velocity_y = msg.imu.anguldar_velocity_y
        anguldar_velocity_z = msg.imu.anguldar_velocity_z
        if msg.HasField("robot_pose"):
            roll, pitch, yaw = PoseTrans.quaternion_to_euler(msg.robot_pose.orientation.x, msg.robot_pose.orientation.y, \
                                                             msg.robot_pose.orientation.z, msg.robot_pose.orientation.w)
        else:
            roll, pitch, yaw = 0, 0, 0
        wheel_coder_l = msg.wheel_l
        wheel_coder_r = msg.wheel_r
        # print("wheel_coder_l : ", wheel_coder_l, " wheel_coder_r : ", wheel_coder_r)
        # print("fork_z : ", fork_z, " fork_y : ", fork_y,  " fork_p : ", fork_p)

        data_idx = msg.seq_num
        encoder.update_value2("DataIndex", data_idx, 4)
        encoder.update_value2("DataIndexReturn", data_idx, 4)
        encoder.update_value("WheelCoder", 8, "", wheel_coder_l, wheel_coder_r)
        encoder.update_value("IncrementalSteeringCoder", 4, "LF", steering_angle)
        encoder.update_value("IncrementalSteeringCoder", 4, "RF", steering_angle)
        encoder.update_value2("BatterySencer", battery, 2)
        # fork pose
        encoder.update_value("HeightCoder", 4, "", fork_z)
        encoder.update_value("ForkDisplacementSencer", 4, "Y", fork_y)
        encoder.update_value("ForkDisplacementSencer", 4, "C", fork_c)
        encoder.update_value("ForkDisplacementSencer", 2, "P", fork_p)
        # imu sensor msg
        encoder.update_value("Gyroscope", 4, "X", roll)
        encoder.update_value("Gyroscope", 4, "Y", pitch)
        encoder.update_value("Gyroscope", 4, "Z", yaw)
        encoder.update_value("AngularVelocitySensor", 2, "X", anguldar_velocity_x)
        encoder.update_value("AngularVelocitySensor", 2, "Y", anguldar_velocity_y)
        encoder.update_value("AngularVelocitySensor", 2, "Z", anguldar_velocity_z)
        encoder.update_value("Accelerometer", 2, "X", linear_acceleration_x)
        encoder.update_value("Accelerometer", 2, "Y", linear_acceleration_y)
        encoder.update_value("Accelerometer", 2, "Z", linear_acceleration_z)

        pack = encoder.encode_package()
        return pack

    @staticmethod
    def AGVActuator2SimCmd(self, decoder, msg):
        pack = Package(msg, len(msg))
        decoder.decode_package(pack)
        velocity_rr, velocity_lr = [], []
        steering_lf, steering_rf = [], []
        fork_z, fork_y, fork_p, fork_c, switch = [], [], [], [], []
        decoder.get_value("MoveDevice", velocity_rr, "RR")
        decoder.get_value("MoveDevice", velocity_lr, "LR")
        decoder.get_value("SteeringDevice", steering_lf, "LF")
        decoder.get_value("SteeringDevice", steering_rf, "RF")
        decoder.get_value("ForkDevice", fork_z, "Z")
        decoder.get_value("ForkDevice", fork_y, "Y")
        decoder.get_value("ForkDevice", fork_p, "P")
        decoder.get_value("ForkDevice", fork_c, "C")
        decoder.get_value("SwitchActuator", switch)
        index = bytearray()
        decoder.get_value2("DataIndex", index, 4)
        index = int.from_bytes(index, byteorder='big', signed=False)
        data_index_return = index
        print("velocity : ", velocity_rr, "steering : ", steering_lf, "fork_z : ", fork_z, \
              "fork_y : ", fork_y, "fork_p : ", fork_p, "fork_c : ", fork_c, "switch : ", switch)
        vehicle_state = vehicle_state_msg_pb2.VehicleStateMsg()
        vehicle_state.drive_velocity = steering_lf[0] if len(steering_lf) > 0 else 0
        vehicle_state.steer_angle = steering_lf[0] if len(steering_lf) > 0 else 0
        vehicle_state.fork_z = fork_z[0] if len(fork_z) > 0 else 0
        vehicle_state.fork_y = fork_y[0] if len(fork_y) > 0 else 0
        vehicle_state.fork_pitch = fork_p[0] if len(fork_p) > 0 else 0
        vehicle_state.fork_cl = fork_c[0] if len(fork_c) > 0 else 0
        # self.sim_pub.send(vehicle_state)
        return data_index_return, vehicle_state


class SerialController:
    """服务控制器，处理消息订阅、发布和回调"""

    def __init__(self,
                 sub_sim : str = "vehicle/status",
                 pub2sim : str = "agv/cmd",
                 sub_agv : str = "Actuator/write",
                 pub2agv : str = "Sensor/read",
                 test : bool = False) -> None:
        '''

        :param sub_sim:
        :param pub2sim:
        :param sub_agv:
        :param pub2agv:
        :param sub_ui: UI控制指令订阅主题
        :param pub2ui: UI状态发布主题
        '''
        self.decoder = InputDecoder()
        self.encoder = OutputEncoder()
        self.sim_pub = ProtoPublisher(pub2sim, vehicle_state_msg_pb2.VehicleStateMsg)
        self.agv_pub = BytesPublisher(pub2agv)
        self.run_thread = True
        self.sim_sub = ProtoSubscriber(sub_sim, vehicle_state_msg_pb2.VehicleStateMsg)
        self.sim_sub.set_callback(self.simCallback)
        self.agv_sub = BytesSubscriber(sub_agv)
        self.agv_sub.set_callback(self.serialCallback)
        self.data_index_return = 0
        
        self.test = test

    def testActuatorPub(self):
        while True:
            wheel_coder_l = wheel_coder_r = 1.567
            fork_z = 1.2
            fork_y = fork_c = 0.1
            fork_p = 0.02
            battery = 99
            linear_acceleration_x = linear_acceleration_y = linear_acceleration_z = 0.0
            anguldar_velocity_x = anguldar_velocity_y = anguldar_velocity_z = 0.0
            roll = pitch = yaw = 0.0
            steering_angle = 30 /180 * 3.1415926
            # print("wheel_coder_l : ", wheel_coder_l, " wheel_coder_r : ", wheel_coder_r)
            # print("fork_z : ", fork_z, " fork_y : ", fork_y,  " fork_p : ", fork_p)
            data_idx = SimulationTimer.get_time_seqnum()
            self.encoder.update_value2("DataIndex", data_idx, 4)
            self.encoder.update_value2("DataIndexReturn", data_idx, 4)
            # encoder.update_value("WheelCoder", 8, "", wheel_coder_l, wheel_coder_r)
            self.encoder.update_value("IncrementalSteeringCoder", 4, "LF", steering_angle)
            self.encoder.update_value("IncrementalSteeringCoder", 4, "RF", steering_angle)
            self.encoder.update_value2("BatterySencer", battery, 2)
            # fork pose
            self.encoder.update_value("HeightCoder", 4, "", fork_z)
            self.encoder.update_value("ForkDisplacementSencer", 4, "Y", fork_y)
            self.encoder.update_value("ForkDisplacementSencer", 4, "C", fork_c)
            self.encoder.update_value("ForkDisplacementSencer", 2, "P", fork_p)
            # imu sensor msg
            self.encoder.update_value("Gyroscope", 4, "X", roll)
            self.encoder.update_value("Gyroscope", 4, "Y", pitch)
            self.encoder.update_value("Gyroscope", 4, "Z", yaw)
            self.encoder.update_value("AngularVelocitySensor", 2, "X", anguldar_velocity_x)
            self.encoder.update_value("AngularVelocitySensor", 2, "Y", anguldar_velocity_y)
            self.encoder.update_value("AngularVelocitySensor", 2, "Z", anguldar_velocity_z)
            self.encoder.update_value("Accelerometer", 2, "X", linear_acceleration_x)
            self.encoder.update_value("Accelerometer", 2, "Y", linear_acceleration_y)
            self.encoder.update_value("Accelerometer", 2, "Z", linear_acceleration_z)


            pack = self.encoder.encode_package()
            # print(len(pack.buf), pack.buf)
            self.agv_pub.send(bytes(pack.buf))
            time.sleep(0.01)
    
    
    def initialize(self, actuator_config: str, sensor_config: str) -> bool:
        """
        初始化服务

        Args:
            actuator_config: 执行器配置文件路径
            sensor_config: 传感器配置文件路径

        Returns:
            是否初始化成功
        """
        # 加载配置
        if not self.decoder.load_config(actuator_config):
            return False

        if not self.encoder.load_config(sensor_config):
            return False
        
        if self.test is True:
            self.thread_test = threading.Thread(target=self.testActuatorPub)
            self.thread_test.start()

        return True

    def simCallback(self, topic_name, msg, msg_time) -> None:
        # print(topic_name, len(msg), msg, msg_time)
        pack = ST_MsgTrans.simState2AGVSensor(self.encoder, msg, self.data_index_return)
        # print(len(pack.buf), pack.buf)
        self.agv_pub.send(bytes(pack.buf))

    def serialCallback(self, topic_name, msg, msg_time) -> None:
        # print(topic_name, len(msg), msg, msg_time)
        self.data_index_return, vehicle_state = ST_MsgTrans.AGVActuator2SimCmd(self.decoder, msg)
        self.sim_pub.send(vehicle_state)




if __name__ == "__main__":

    # 初始化 eCAL
    ecal_core.initialize(sys.argv, "Serial service")
    # publish
    service = SerialController()
    service.initialize("configs/st/Actuators.config",
                       "configs/st/Sencers.config")

    while ecal_core.ok():
        time.sleep(0.001)

    # 关闭 eCAL
    ecal_core.finalize()
