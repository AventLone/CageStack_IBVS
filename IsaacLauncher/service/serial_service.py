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
from utils import PoseTrans, SysTimer

@dataclass
class ServiceMessage:
    """服务消息容器"""
    topic: str
    data: Any
    timestamp: float = 0.0


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
            data_idx = SysTimer.get_time_seqnum()
            self.encoder.update_value2("DataIndex", data_idx, 4)
            self.encoder.update_value2("DataIndexReturn", data_idx, 4)
            # self.encoder.update_value("WheelCoder", 8, "", wheel_coder_l, wheel_coder_r)
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
        # print(topic_name, msg, msg_time)
        pack = self.st_sensor(msg)
        # print(len(pack.buf), pack.buf)
        self.agv_pub.send(bytes(pack.buf))

    def st_sensor(self, msg) -> Optional[Package]:
        battery = 66
        velocity = msg.drive_velocity
        steering_angle = msg.steer_angle
        fork_z = msg.fork_z
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
        # print("wheel_coder_l : ", wheel_coder_l, " wheel_coder_r : ", wheel_coder_r, " steering_angle : ", steering_angle)

        data_idx = msg.seq_num
        self.encoder.update_value2("DataIndex", data_idx, 4)
        self.encoder.update_value2("DataIndexReturn", self.data_index_return, 4)
        self.encoder.update_value("RPMSensor", 2, "M", wheel_coder_l)
        self.encoder.update_value("IncrementalSteeringCoder", 2, "", steering_angle)
        self.encoder.update_value2("BatterySencer", battery, 2)
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
        # SwitchSencer
        self.encoder.update_switch_value("SwitchSencer", 50, msg.HSwitchL, "")
        self.encoder.update_switch_value("SwitchSencer", 51, msg.HSwitchR, "")
        if fork_z >= 0.29:
            self.encoder.update_switch_value("SwitchSencer", 32, True, "")
            self.encoder.update_switch_value("SwitchSencer", 33, False, "")
        elif fork_z <= 0.01:
            self.encoder.update_switch_value("SwitchSencer", 32, False, "")
            self.encoder.update_switch_value("SwitchSencer", 33, True, "")
        else:
            self.encoder.update_switch_value("SwitchSencer", 32, False, "")
            self.encoder.update_switch_value("SwitchSencer", 33, False, "")

        # get pack
        pack = self.encoder.encode_package()
        return pack

    def e_sensor(self, msg) -> Optional[Package]:
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
        self.encoder.update_value2("DataIndex", data_idx, 4)
        self.encoder.update_value2("DataIndexReturn", data_idx, 4)
        self.encoder.update_value("WheelCoder", 8, "", wheel_coder_l, wheel_coder_r)
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
        return pack


    def serialCallback(self, topic_name, msg, msg_time) -> None:
        # print(topic_name, len(msg), msg, msg_time)
        pack = Package(msg, len(msg))
        self.decoder.decode_package(pack)
        velocity, velocity_rr, velocity_lr = [], [], []
        steering, steering_lf, steering_rf = [], [], []
        fork_z, fork_y, fork_p, fork_c, switch = [], [], [], [], []
        self.decoder.get_value("MoveDevice", velocity, "")
        self.decoder.get_value("MoveDevice", velocity_rr, "RR")
        self.decoder.get_value("MoveDevice", velocity_lr, "LR")
        self.decoder.get_value("SteeringDevice", steering, "")
        self.decoder.get_value("SteeringDevice", steering_lf, "LF")
        self.decoder.get_value("SteeringDevice", steering_rf, "RF")
        self.decoder.get_value("ForkDevice", fork_z,"Z")
        self.decoder.get_value("ForkDevice", fork_y,"Y")
        self.decoder.get_value("ForkDevice", fork_p,"P")
        self.decoder.get_value("ForkDevice", fork_c,"C")
        self.decoder.get_value("SwitchActuator", switch)
        index = bytearray()
        self.decoder.get_value2("DataIndex", index, 4)
        index = int.from_bytes(index, byteorder='big', signed=False)
        self.data_index_return = index
        # print("velocity : ", velocity, "steering : ", steering, "fork_z : ", fork_z, \
        #       "fork_y : ", fork_y, "fork_p : ", fork_p, "fork_c : ", fork_c, "switch : ", switch)
        vehicle_state = vehicle_state_msg_pb2.VehicleStateMsg()
        vehicle_state.drive_velocity = velocity[0] if len(velocity) > 0 else 0
        vehicle_state.steer_angle = steering[0]if len(steering) > 0 else 0
        vehicle_state.fork_z = fork_z[0] if len(fork_z) > 0 else 0
        vehicle_state.fork_y = fork_y[0] if len(fork_y) > 0 else 0
        vehicle_state.fork_pitch = fork_p[0] if len(fork_p) > 0 else 0
        vehicle_state.fork_cl = fork_c[0] if len(fork_c) > 0 else 0
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
