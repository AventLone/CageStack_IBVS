#!/usr/bin/env python3
"""
UI Client - 独立运行PyQt5界面的客户端进程
通过eCAL与Isaac Sim服务器通信，使用protobuf协议
"""

import sys
import os
import time
import json
from pathlib import Path
from typing import Dict, Any  

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# PyQt5 imports
try:
    from PyQt5.QtWidgets import QApplication
    from PyQt5.QtCore import QTimer
    PYQT5_AVAILABLE = True
except ImportError as e:
    print(f"Warning: PyQt5 not available: {e}")
    PYQT5_AVAILABLE = False

# eCAL imports
try:
    import ecal.core.core as ecal_core
    from ecal.core.publisher import StringPublisher, ProtoPublisher
    from ecal.core.subscriber import StringSubscriber, ProtoSubscriber
    ECAL_AVAILABLE = True
except ImportError as e:
    print(f"Warning: eCAL not available: {e}")
    ECAL_AVAILABLE = False

# Project imports
from ui.vehicle_control_ui import VehicleControlUI
from service.ecal_ui_service import EcalUIService, VehicleControlCommand, KeyboardState


class UIClient:
    """UI客户端"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.app = None
        self.ui = None
        self.ecal_service = None
        self.running = False
        self.status_timer = None
        self.ecal_initialized = False
        self.config = config or {}
        
        print("UI Client initialized with protobuf protocol")
    
    def init_ecal(self) -> bool:
        """初始化eCAL通信"""
        if not ECAL_AVAILABLE:
            print("eCAL not available, running without communication")
            self.ecal_initialized = False
            return False
        
        try:
            # 使用新的eCAL服务，传递配置参数
            self.ecal_service = EcalUIService("ui_client", self.config)
            success = self.ecal_service.initialize_ui_service()
            self.ecal_initialized = success
            
            # 设置状态回调函数
            if success:
                self.ecal_service.set_status_callback(self.on_isaac_status)
            
            return success
            
        except Exception as e:
            print(f"Failed to initialize eCAL: {e}")
            self.ecal_initialized = False
            return False
    
    def on_isaac_status(self, status_data):
        """处理来自Isaac Sim的状态更新"""
        try:
            # 使用线程安全的UI更新方式 - 通过信号发送状态数据
            if self.ui and hasattr(self.ui, 'status_updated'):
                self.ui.status_updated.emit(status_data)
            
        except Exception as e:
            print(f"Error processing Isaac Sim status: {e}")
    
    def send_command(self, command_type, **kwargs):
        """发送命令到Isaac Sim服务器"""
        if not self.ecal_service:
            return
        
        try:
            # 使用新的命令格式
            if command_type == 'vehicle_control':
                command = VehicleControlCommand(
                    drive_speed=kwargs.get('drive_speed', 0.0),
                    steering_angle=kwargs.get('steering_angle', 0.0),
                    fork_x=kwargs.get('fork_x', 0.0),  # X轴位置
                    fork_y=kwargs.get('fork_y', 0.0),  # Y轴位置
                    fork_z=kwargs.get('fork_z', 0.0),  # Z轴位置（货叉高度）
                    fork_roll=kwargs.get('fork_roll', 0.0),  # 横滚角
                    fork_pitch=kwargs.get('fork_pitch', 0.0),  # 俯仰角
                    fork_clamp_left=kwargs.get('fork_clamp_left', 0.0),  # 左夹钳压力
                    fork_clamp_right=kwargs.get('fork_clamp_right', 0.0),  # 右夹钳压力
                    control_mode=kwargs.get('control_mode', 1)  # 默认UI控制模式
                )
                self.ecal_service.publish_vehicle_command(command)
            elif command_type == 'keyboard_state':
                state = KeyboardState(**kwargs)
                self.ecal_service.publish_keyboard_state(state)
            elif command_type == 'reset':
                self.ecal_service.publish_reset_command()
            elif command_type == 'shutdown':
                self.ecal_service.publish_shutdown_command()
                
        except Exception as e:
            print(f"Error sending command: {e}")
    
    def setup_ui(self) -> bool:
        """设置UI界面"""
        if not PYQT5_AVAILABLE:
            print("PyQt5 not available")
            return False
        
        try:
            # Create QApplication
            self.app = QApplication(sys.argv)
            
            # Create UI
            self.ui = VehicleControlUI()
            
            # Connect UI signals to eCAL commands
            self.setup_ui_connections()
            
            # Setup connection status check timer
            self.connection_timer = QTimer()
            self.connection_timer.timeout.connect(self.check_connection_status)
            self.connection_timer.start(500)  # 每500ms检查一次连接状态
            
            # Show UI
            self.ui.show()
            
            print("UI setup complete")
            return True
            
        except Exception as e:
            print(f"Failed to setup UI: {e}")
            return False
    
    def setup_ui_connections(self):
        """设置UI信号连接"""
        if not self.ui:
            return
        
        # Connect UI command signal to eCAL sender
        self.ui.command_sent.connect(self.on_ui_command_sent)
        
        # Connect reset button
        if hasattr(self.ui, 'btn_reset'):
            self.ui.btn_reset.clicked.connect(lambda: self.send_command('reset'))
    
    def on_ui_command_sent(self, command_type, command_data):
        """处理来自UI的命令信号"""
        if command_type == "vehicle_control":
            # 发送车辆控制命令 - 扩展支持6自由度控制
            self.send_command('vehicle_control', 
                            drive_speed=command_data.get('steer_speed', 0.0),
                            steering_angle=command_data.get('steer_yaw', 0.0),
                            fork_x=command_data.get('fork_x', 0.0),
                            fork_y=command_data.get('fork_y', 0.0),
                            fork_z=command_data.get('fork_z', 0.0),  # 添加Z轴位置
                            fork_roll=command_data.get('fork_roll', 0.0),
                            fork_pitch=command_data.get('fork_pitch', 0.0),
                            fork_clamp_left=command_data.get('fork_clamp_left', 0.0),  # 左夹钳压力
                            fork_clamp_right=command_data.get('fork_clamp_right', 0.0),  # 右夹钳压力
                            control_mode=command_data.get('control_mode', 1))  # 发送控制模式
    
    # Vehicle and fork command methods removed - now handled via UI signals
    
    def check_connection_status(self):
        """检查并更新连接状态"""
        if not self.ui or not self.ecal_service:
            return
        
        try:
            # 检查eCAL服务连接状态
            connected = self.ecal_service.check_connection_status()
            
            # 更新UI连接状态显示
            if hasattr(self.ui, 'update_connection_status'):
                self.ui.update_connection_status(connected)
            
        except Exception as e:
            print(f"Error checking connection status: {e}")
    
    def update_ui_status(self):
        """更新UI状态显示 - 这个方法现在不需要了，因为状态通过信号实时更新"""
        if not self.ui:
            return
        
        # Update connection status - 连接状态现在由Isaac Sim服务器提供
        # connected = self.ecal_initialized
        # self.ui.update_connection_status(connected)
        
        # Update timestamp - 时间戳现在由Isaac Sim服务器提供
        # current_time = time.strftime("%H:%M:%S")
        # self.ui.update_timestamp(current_time)
    
    def run(self):
        """运行UI客户端"""
        print("Starting UI Client...")
        print("=====================")
        print(f"PyQt5 Available: {PYQT5_AVAILABLE}")
        print(f"eCAL Available: {ECAL_AVAILABLE}")
        print("Protocol: Protobuf")
        print()
        
        # Initialize components
        ecal_ok = self.init_ecal()
        ui_ok = self.setup_ui()
        
        if not ui_ok:
            print("Failed to setup UI, exiting")
            return
        
        self.running = True
        
        try:
            # Run Qt event loop in main thread
            print("Running UI in main thread...")
            exit_code = self.app.exec_()
            
        except KeyboardInterrupt:
            print("\nReceived interrupt signal")
            exit_code = 0
        
        # Cleanup
        self.shutdown()
        return exit_code
    
    def shutdown(self):
        """关闭客户端"""
        print("Shutting down UI Client...")
        
        self.running = False
        
        # Send shutdown command to Isaac Sim server
        if self.ecal_service:
            self.send_command('shutdown')
            time.sleep(0.1)  # Give time for message to send
        
        # Stop connection status timer
        if hasattr(self, 'connection_timer') and self.connection_timer:
            self.connection_timer.stop()
        
        # Cleanup UI
        if self.ui:
            try:
                self.ui.close()
            except Exception as e:
                print(f"Error closing UI: {e}")
        
        if self.app:
            try:
                self.app.quit()
            except Exception as e:
                print(f"Error quitting app: {e}")
        
        # Cleanup eCAL
        if self.ecal_initialized:
            try:
                ecal_core.finalize()
                print("eCAL finalized")
            except Exception as e:
                print(f"Error finalizing eCAL: {e}")
        
        print("UI Client shutdown complete")


def main():
    """主函数"""
    # 解析命令行参数
    import argparse
    parser = argparse.ArgumentParser(description='UI客户端')
    parser.add_argument('--config', default='configs/e_test.yaml', 
                       help='配置文件路径 (默认: configs/e_test.yaml)')
    
    args = parser.parse_args()
    
    # 加载配置
    config = {}
    try:
        # 尝试从配置文件加载配置
        from utils.config_loader import ConfigLoader
        config_loader = ConfigLoader()
        config = config_loader.load_config(args.config)
        print(f"Loaded config from {args.config}")
    except Exception as e:
        print(f"Failed to load config: {e}, using default config")
    
    client = UIClient(config)
    
    try:
        exit_code = client.run()
        sys.exit(exit_code)
    except Exception as e:
        print(f"Unexpected error: {e}")
        client.shutdown()
        sys.exit(1)


if __name__ == "__main__":
    main()