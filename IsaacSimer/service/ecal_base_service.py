#!/usr/bin/env python3
"""
eCAL Base Service - 提供统一的eCAL通信基础功能
"""

import ecal.core.core as ecal_core
from ecal.core.publisher import StringPublisher
from ecal.core.subscriber import StringSubscriber
import json
import threading
import time
from typing import Dict, Any, Callable, Optional, List
from dataclasses import dataclass, asdict
from enum import Enum


class CommandType(Enum):
    """eCAL命令类型枚举"""
    VEHICLE_CONTROL = "vehicle_control"
    KEYBOARD_STATE = "keyboard_state"
    RESET = "reset"
    SHUTDOWN = "shutdown"


@dataclass
class VehicleControlCommand:
    """车辆控制命令数据结构"""
    drive_speed: float = 0.0
    steering_angle: float = 0.0
    fork_x: float = 0.0  # X轴位置
    fork_y: float = 0.0  # Y轴位置
    fork_z: float = 0.0  # Z轴位置（货叉高度）
    fork_roll: float = 0.0  # 横滚角
    fork_pitch: float = 0.0  # 俯仰角
    fork_clamp_left: float = 0.0  # 左夹钳压力
    fork_clamp_right: float = 0.0  # 右夹钳压力
    control_mode: int = 1  # 控制模式: 1=UI控制, 2=键盘控制, 3=自适应控制


@dataclass  
class KeyboardState:
    """键盘状态数据结构"""
    w: bool = False
    a: bool = False
    s: bool = False
    d: bool = False
    up: bool = False
    down: bool = False
    space: bool = False
    r: bool = False


class EcalMessageSerializer:
    """eCAL消息序列化器"""
    
    @staticmethod
    def serialize_command(command_type: CommandType, data: Any) -> str:
        """序列化命令消息"""
        message = {
            'type': command_type.value,
            'timestamp': time.time(),
            'data': asdict(data) if hasattr(data, '__dataclass_fields__') else data
        }
        return json.dumps(message)
    
    @staticmethod
    def deserialize_message(message: str) -> Dict[str, Any]:
        """反序列化消息"""
        try:
            return json.loads(message)
        except json.JSONDecodeError:
            return {}


class EcalServiceBase:
    """eCAL服务基类"""
    
    def __init__(self, node_name: str):
        self.node_name = node_name
        self._initialized = False
        self._running = False
        self._publishers: Dict[str, StringPublisher] = {}
        self._subscribers: Dict[str, StringSubscriber] = {}
        self._callbacks: Dict[str, Callable] = {}
        self._lock = threading.Lock()
    
    def initialize(self) -> bool:
        """初始化eCAL"""
        try:
            ecal_core.initialize([], self.node_name)
            self._initialized = True
            self._running = True
            print(f"eCAL Service '{self.node_name}' initialized")
            return True
        except Exception as e:
            print(f"Failed to initialize eCAL: {e}")
            return False
    
    def create_publisher(self, topic_name: str) -> bool:
        """创建发布器"""
        try:
            self._publishers[topic_name] = StringPublisher(topic_name)
            return True
        except Exception as e:
            print(f"Failed to create publisher for {topic_name}: {e}")
            return False
    
    def create_subscriber(self, topic_name: str, callback: Callable) -> bool:
        """创建订阅器"""
        try:
            subscriber = StringSubscriber(topic_name)
            subscriber.set_callback(callback)
            self._subscribers[topic_name] = subscriber
            self._callbacks[topic_name] = callback
            return True
        except Exception as e:
            print(f"Failed to create subscriber for {topic_name}: {e}")
            return False
    
    def publish(self, topic_name: str, message: str) -> bool:
        """发布消息"""
        if topic_name not in self._publishers:
            print(f"Publisher for {topic_name} not created")
            return False
        
        try:
            self._publishers[topic_name].send(message)
            return True
        except Exception as e:
            print(f"Failed to publish to {topic_name}: {e}")
            return False
    
    def shutdown(self):
        """关闭服务"""
        self._running = False
        if self._initialized:
            try:
                ecal_core.finalize()
                print(f"eCAL Service '{self.node_name}' shutdown")
            except Exception as e:
                print(f"Error during eCAL shutdown: {e}")
        self._initialized = False
    
    def is_initialized(self) -> bool:
        """检查是否已初始化"""
        return self._initialized
    
    def is_running(self) -> bool:
        """检查是否在运行"""
        return self._running