#!/usr/bin/env python3
"""
eCAL UI Service for Vehicle Control

This module provides eCAL communication services for the vehicle control UI,
handling command publishing and status subscription using protobuf protocol.
"""

import ecal.core.core as ecal_core
from ecal.core.publisher import ProtoPublisher, StringPublisher
from ecal.core.subscriber import ProtoSubscriber, StringSubscriber
import json
import threading
import time
from typing import Dict, Any, Callable, Optional
from utils import SysTimer

# 导入protobuf消息
from protos import vehicle_state_msg_pb2

# 导入基础服务
from .ecal_base_service import EcalServiceBase, VehicleControlCommand, KeyboardState, EcalMessageSerializer, CommandType


class EcalUIService(EcalServiceBase):
    """UI专用的eCAL服务"""
    
    def __init__(self, node_name: str = "VehicleControlUI", config: Dict[str, Any] = None):
        super().__init__(node_name)
        self._latest_status: Dict[str, Any] = {}
        self._status_callback: Optional[Callable] = None
        
        # Protobuf发布器和订阅器
        self._proto_publishers: Dict[str, ProtoPublisher] = {}
        self._proto_subscribers: Dict[str, ProtoSubscriber] = {}
        
        # 连接状态检测
        self._last_status_time: float = 0.0
        self._connection_timeout: float = 2.0  # 2秒超时
        
        # 主题配置
        self._config = config or {}
        self._ui_cmd_topic = self._config.get("ui_cmd_topic", "ui/cmd")
        self._ui_status_topic = self._config.get("vehicle_state_topic", "vehicle/status")
    
    def set_status_callback(self, callback: Callable):
        """设置状态回调函数"""
        self._status_callback = callback
    
    def check_connection_status(self) -> bool:
        """检查连接状态"""
        current_time = time.time()
        time_since_last_status = current_time - self._last_status_time
        
        # 如果从未收到过状态，或者超时，则连接断开
        if self._last_status_time == 0.0 or time_since_last_status > self._connection_timeout:
            return False
        return True
    
    def initialize_ui_service(self) -> bool:
        """初始化UI服务"""
        if not self.initialize():
            return False
        
        # 创建protobuf命令发布器 (使用配置中的主题)
        if not self.create_proto_publisher(self._ui_cmd_topic, vehicle_state_msg_pb2.VehicleStateMsg):
            return False
        
        # 创建protobuf状态订阅器 (使用配置中的主题)
        if not self.create_proto_subscriber(self._ui_status_topic, vehicle_state_msg_pb2.VehicleStateMsg, self._on_ui_status_received):
            return False
        
        print(f"UI服务初始化完成 - 发布主题: {self._ui_cmd_topic}, 订阅主题: {self._ui_status_topic}")
        return True
    
    def create_proto_publisher(self, topic_name: str, message_type) -> bool:
        """创建protobuf发布器"""
        try:
            self._proto_publishers[topic_name] = ProtoPublisher(topic_name, message_type)
            return True
        except Exception as e:
            print(f"Failed to create proto publisher for {topic_name}: {e}")
            return False
    
    def create_proto_subscriber(self, topic_name: str, message_type, callback: Callable) -> bool:
        """创建protobuf订阅器"""
        try:
            subscriber = ProtoSubscriber(topic_name, message_type)
            subscriber.set_callback(callback)
            self._proto_subscribers[topic_name] = subscriber
            return True
        except Exception as e:
            print(f"Failed to create proto subscriber for {topic_name}: {e}")
            return False
    
    def publish_vehicle_command(self, command: VehicleControlCommand) -> bool:
        """发布车辆控制命令到配置的主题"""
        try:
            # 创建VehicleStateMsg消息 - 扩展支持6自由度控制
            msg = vehicle_state_msg_pb2.VehicleStateMsg()
            msg.drive_velocity = command.drive_speed
            msg.steer_angle = command.steering_angle
            msg.fork_x = command.fork_x
            msg.fork_y = command.fork_y
            msg.fork_z = command.fork_z  # Z轴位置（货叉高度）
            msg.fork_roll = command.fork_roll
            msg.fork_pitch = command.fork_pitch
            msg.fork_cl = command.fork_clamp_left
            msg.fork_cr = command.fork_clamp_right
            msg.control_mode = command.control_mode  # 设置控制模式
            msg.timestamp = SysTimer.get_timestamp()  # 毫秒时间戳
            
            # 发布到配置的主题
            if self._ui_cmd_topic in self._proto_publishers:
                self._proto_publishers[self._ui_cmd_topic].send(msg)
                return True
            else:
                print(f"Proto publisher for {self._ui_cmd_topic} not found")
                return False
                
        except Exception as e:
            print(f"Error publishing vehicle command: {e}")
            return False
    
    def publish_keyboard_state(self, state: KeyboardState) -> bool:
        """发布键盘状态到ui/cmd"""
        # 键盘状态暂时使用JSON格式发布到备用主题
        message = EcalMessageSerializer.serialize_command(
            CommandType.KEYBOARD_STATE, state
        )
        return self.publish("ui/keyboard", message)
    
    def publish_reset_command(self) -> bool:
        """发布重置命令到配置的主题"""
        try:
            msg = vehicle_state_msg_pb2.VehicleStateMsg()
            msg.drive_velocity = 0.0
            msg.steer_angle = 0.0
            msg.fork_z = 0.0
            msg.fork_x = 0.0  # 重置X轴位置
            msg.fork_y = 0.0  # 重置Y轴位置
            msg.fork_roll = 0.0  # 重置横滚角
            msg.fork_pitch = 0.0  # 重置俯仰角
            msg.fork_cl = 0.0  # 重置左夹钳压力
            msg.fork_cr = 0.0  # 重置右夹钳压力
            msg.timestamp = SysTimer.get_timestamp()
            
            if self._ui_cmd_topic in self._proto_publishers:
                self._proto_publishers[self._ui_cmd_topic].send(msg)
                return True
            else:
                print(f"Proto publisher for {self._ui_cmd_topic} not found")
                return False
                
        except Exception as e:
            print(f"Error publishing reset command: {e}")
            return False
    
    def publish_shutdown_command(self) -> bool:
        """发布关闭命令"""
        message = EcalMessageSerializer.serialize_command(
            CommandType.SHUTDOWN, {}
        )
        return self.publish("ui/shutdown", message)
        
    def publish_control_mode(self, control_mode: int) -> bool:
        """发布独立的控制模式切换命令
        
        Args:
            control_mode: 控制模式值 (1=UI控制, 2=键盘控制, 3=自适应控制)
            
        Returns:
            bool: 发布是否成功
        """
        try:
            # 创建仅包含control_mode字段的VehicleStateMsg消息
            msg = vehicle_state_msg_pb2.VehicleStateMsg()
            msg.control_mode = control_mode
            msg.timestamp = int(time.time() * 1000)  # 毫秒时间戳
            
            # 发布到配置的主题
            if self._ui_cmd_topic in self._proto_publishers:
                self._proto_publishers[self._ui_cmd_topic].send(msg)
                return True
            else:
                print(f"Proto publisher for {self._ui_cmd_topic} not found")
                return False
                
        except Exception as e:
            print(f"Error publishing control mode: {e}")
            return False
    
    def _on_ui_status_received(self, topic_name: str, msg, time_stamp: int):
        """处理来自ui/status的状态消息 - 完整VehicleStateMsg字段映射"""
        try:
            # 更新最后状态接收时间
            self._last_status_time = time.time()
            
            # 将protobuf消息转换为完整的字典格式
            status_data = self._map_vehicle_state_msg(msg)
            
            # 添加连接状态信息
            status_data['connection_status'] = True
            status_data['last_status_time'] = self._last_status_time
            
            with self._lock:
                self._latest_status = status_data
                
            # 调用状态回调函数
            if self._status_callback:
                self._status_callback(status_data)
                
        except Exception as e:
            print(f"Error processing UI status message: {e}")
    
    def _map_vehicle_state_msg(self, msg) -> dict:
        """将VehicleStateMsg映射为完整的状态字典"""
        return {
            # 基础运动状态
            'basic_status': {
                'drive_velocity': float(msg.drive_velocity),
                'drive_dist': float(msg.drive_dist),
                'steer_angle': float(msg.steer_angle),
                'timestamp': int(msg.timestamp)
            },
            
            # 货叉完整状态
            'fork_status': {
                'fork_x': float(msg.fork_x),
                'fork_y': float(msg.fork_y), 
                'fork_z': float(msg.fork_z),
                'fork_cl': float(msg.fork_cl),
                'fork_cr': float(msg.fork_cr),
                'fork_roll': float(msg.fork_roll),
                'fork_pitch': float(msg.fork_pitch),
                'clamp_pressure': float(msg.ClampPressure)
            },
            
            # 位姿状态
            'pose_status': {
                'robot_pose': self._extract_pose(msg.robot_pose),
                'target_pose': self._extract_pose(msg.target_pose)
            },
            
            # 传感器状态
            'sensor_status': {
                'imu_data': self._extract_imu(msg.imu),
                'hswitch_left': bool(msg.HSwitchL),
                'hswitch_right': bool(msg.HSwitchR),
                'vswitch_left': bool(msg.VSwitchL),
                'vswitch_right': bool(msg.VSwitchR),
                'torque_status': bool(msg.torquestatus)
            },
            
            # 系统状态
            'system_status': {
                'control_mode': int(msg.control_mode),
                'data_record': bool(msg.data_record),
                'data_index': int(msg.dataidx_upload)
            }
        }
    
    def _extract_pose(self, pose) -> dict:
        """提取位姿信息"""
        if pose and pose.HasField('position') and pose.HasField('orientation'):
            return {
                'position': {
                    'x': float(pose.position.x),
                    'y': float(pose.position.y),
                    'z': float(pose.position.z)
                },
                'orientation': {
                    'x': float(pose.orientation.x),
                    'y': float(pose.orientation.y),
                    'z': float(pose.orientation.z),
                    'w': float(pose.orientation.w)
                }
            }
        else:
            return {
                'position': {'x': 0.0, 'y': 0.0, 'z': 0.0},
                'orientation': {'x': 0.0, 'y': 0.0, 'z': 0.0, 'w': 1.0}
            }
    
    def _extract_imu(self, imu) -> dict:
        """提取IMU数据"""
        if imu:
            return {
                'linear_acceleration': {
                    'x': float(imu.linear_acceleration_x),
                    'y': float(imu.linear_acceleration_y),
                    'z': float(imu.linear_acceleration_z)
                },
                'angular_velocity': {
                    'x': float(imu.anguldar_velocity_x),
                    'y': float(imu.anguldar_velocity_y),
                    'z': float(imu.anguldar_velocity_z)
                }
            }
        else:
            return {
                'linear_acceleration': {'x': 0.0, 'y': 0.0, 'z': 0.0},
                'angular_velocity': {'x': 0.0, 'y': 0.0, 'z': 0.0}
            }
    
    def _on_status_received(self, topic_name: str, msg: str, time_stamp: int):
        """状态消息回调 (JSON格式)"""
        try:
            status_data = EcalMessageSerializer.deserialize_message(msg)
            with self._lock:
                self._latest_status = status_data
                
            # 调用状态回调函数
            if self._status_callback:
                self._status_callback(status_data)
                
        except Exception as e:
            print(f"Error processing status message: {e}")
    
    def get_latest_status(self) -> Dict[str, Any]:
        """获取最新状态"""
        with self._lock:
            return self._latest_status.copy()


class EcalUIServiceSingleton:
    """Singleton wrapper for EcalUIService."""
    
    _instance: Optional[EcalUIService] = None
    _lock = threading.Lock()
    
    @classmethod
    def get_instance(cls, node_name: str = "VehicleControlUI", config: Dict[str, Any] = None) -> EcalUIService:
        """Get singleton instance of EcalUIService."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = EcalUIService(node_name, config)
        return cls._instance
    
    @classmethod
    def shutdown_instance(cls):
        """Shutdown singleton instance."""
        if cls._instance is not None:
            with cls._lock:
                if cls._instance is not None:
                    cls._instance.shutdown()
                    cls._instance = None


# Convenience functions
def get_ecal_service(node_name: str = "VehicleControlUI", config: Dict[str, Any] = None) -> EcalUIService:
    """Get eCAL UI service instance."""
    return EcalUIServiceSingleton.get_instance(node_name, config)


def shutdown_ecal_service():
    """Shutdown eCAL UI service."""
    EcalUIServiceSingleton.shutdown_instance()


if __name__ == "__main__":
    # Test the eCAL service
    import time
    
    def test_status_callback(status: Dict[str, Any]):
        print(f"Received status: {status}")
    
    # 测试配置
    test_config = {
        "ui_cmd_topic": "ui/cmd",
        "vehicle_state_topic": "vehicle/status"
    }
    
    # Create and initialize service with test config
    service = EcalUIService("TestVehicleUI", test_config)
    
    if service.initialize():
        print("eCAL service initialized successfully")
        
        # Set status callback
        service.set_status_callback(test_status_callback)
        
        # Test publishing commands
        for i in range(5):
            success = service.publish_vehicle_command(
                drive_speed=1.0 + i * 0.1,
                steering_angle=0.1 * i,
                fork_height=0.5 + i * 0.05
            )
            print(f"Command {i+1} published: {success}")
            time.sleep(1)
        
        # Test keyboard state
        service.publish_keyboard_state({
            'w': True,
            'a': False,
            's': False,
            'd': False
        })
        
        print("Test completed, shutting down...")
        service.shutdown()
    else:
        print("Failed to initialize eCAL service")