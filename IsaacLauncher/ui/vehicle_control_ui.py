#!/usr/bin/env python3
"""
Vehicle Control UI for Isaac Sim Launcher

This module provides a Qt-based UI for manual vehicle control including:
- Vehicle movement control (SpeedUp, Stop, SpeedDown, TurnLeft, TurnRight)
- Fork Z-axis control (LiftZUp, LiftZStop, LiftZDown)
- Real-time status display (wheel speeds, steering angles, fork height)
- eCAL communication integration

Author: Ported from webots_ctrl_dataset_with_timestamp
Date: 2024
"""

import sys
import time
import numpy as np
from typing import Dict, Any, Optional, Callable
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QPushButton, QLabel, QGroupBox, QSpinBox, QDoubleSpinBox,
    QCheckBox, QFrame, QSizePolicy, QTextEdit, QRadioButton, QButtonGroup,
    QTabWidget
)
from PyQt5.QtCore import QTimer, Qt, pyqtSignal, QThread
from PyQt5.QtGui import QFont, QKeyEvent

# eCAL imports removed - communication handled by ui_client.py


class VehicleControlUI(QMainWindow):
    """
    Main UI window for vehicle control.
    
    Provides manual control interface for AGV vehicles with:
    - Movement controls (forward/backward, left/right turn)
    - Fork controls (up/down/stop)
    - Real-time status display
    - Keyboard shortcuts
    """
    
    # Signals for communication
    status_updated = pyqtSignal(dict)
    command_sent = pyqtSignal(str, dict)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Control state
        self.manual_state = {
            'steer_speed': 0.0,      # Vehicle forward/backward speed
            'steer_yaw': 0.0,        # Vehicle steering angle
            'fork_speed': 0.0,       # Fork Z-axis speed
            'is_manual': True,       # Manual control mode
            'real_speed': 0.0,       # Actual vehicle speed
            'drivenL_speed': 0.0,    # Left wheel speed
            'drivenR_speed': 0.0,    # Right wheel speed
            'position': [0.0, 0.0, 0.0],  # Vehicle position [x, y, z]
            'calculated_speed': 0.0,  # Calculated speed from velocity
            
            # 扩展叉臂多轴控制状态
            'fork_x': 0.0,           # X轴位置（前后）
            'fork_y': 0.0,           # Y轴位置（左右）
            'fork_z': 0.0,           # Z轴位置（上下，货叉高度）
            'fork_roll': 0.0,        # 翻滚角
            'fork_pitch': 0.0,       # 俯仰角
            'fork_cl': 0.0,          # 左夹紧器
            'fork_cr': 0.0,          # 右夹紧器
            'clamp_pressure': 0.0    # 夹紧压力
        }
        
        # Connection status (managed by ui_client.py)
        self.connection_status = "Disconnected"
        
        # UI update timer - 移除这个定时器，因为状态更新通过信号处理
        # self.update_timer = QTimer()
        # self.update_timer.timeout.connect(self.update_display)
        
        # Keyboard state
        self.pressed_keys = set()
        self.key_timer = QTimer()
        self.key_timer.timeout.connect(self.handle_keyboard_movement)
        
        # Connect signals for thread-safe UI updates
        self.status_updated.connect(self.on_status_updated)
        
        self.init_ui()
        
        # Start timers - 只保留键盘处理定时器
        # self.update_timer.start(200)  # 移除状态更新定时器
        self.key_timer.start(20)     # 50Hz keyboard handling
        
    def _has_state_changed(self, old_state: dict) -> bool:
        """检查状态是否发生变化"""
        return (self.manual_state['steer_speed'] != old_state.get('steer_speed', self.manual_state['steer_speed']) or
            self.manual_state['steer_yaw'] != old_state.get('steer_yaw', self.manual_state['steer_yaw']) or
            self.manual_state['fork_speed'] != old_state.get('fork_speed', self.manual_state['fork_speed']) or
            self.manual_state['fork_x'] != old_state.get('fork_x', self.manual_state['fork_x']) or
            self.manual_state['fork_y'] != old_state.get('fork_y', self.manual_state['fork_y']) or
            self.manual_state['fork_z'] != old_state.get('fork_z', self.manual_state['fork_z']) or
            self.manual_state['fork_roll'] != old_state.get('fork_roll', self.manual_state['fork_roll']) or
            self.manual_state['fork_pitch'] != old_state.get('fork_pitch', self.manual_state['fork_pitch']) or
            self.manual_state['fork_cl'] != old_state.get('fork_cl', self.manual_state['fork_cl']) or
            self.manual_state['fork_cr'] != old_state.get('fork_cr', self.manual_state['fork_cr']) or
            self.manual_state['clamp_pressure'] != old_state.get('clamp_pressure', self.manual_state['clamp_pressure'])
        )
    
    def _get_current_state(self) -> dict:
        """获取当前控制状态"""
        return {
            'steer_speed': self.manual_state['steer_speed'],
            'steer_yaw': self.manual_state['steer_yaw'],
            'fork_speed': self.manual_state['fork_speed'],
            'fork_x': self.manual_state['fork_x'],
            'fork_y': self.manual_state['fork_y'],
            'fork_z': self.manual_state['fork_z'],
            'fork_roll': self.manual_state['fork_roll'],
            'fork_pitch': self.manual_state['fork_pitch'],
            'fork_clamp_left': self.manual_state['fork_cl'],  # 左夹钳压力
            'fork_clamp_right': self.manual_state['fork_cr'],  # 右夹钳压力
            'clamp_pressure': self.manual_state['clamp_pressure']
        }
        
    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("Vehicle Control Panel")
        self.setGeometry(100, 100, 1200, 800)  # 增加窗口尺寸以容纳更多内容
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout - 使用垂直布局，顶部为状态栏，中间为内容区域，底部为连接状态
        main_layout = QVBoxLayout(central_widget)
        
        # 顶部状态栏
        status_bar = QWidget()
        status_layout = QHBoxLayout(status_bar)
        
        # 连接状态
        self.connection_label = QLabel("Status: Disconnected")
        self.connection_label.setStyleSheet("color: red; font-weight: bold; padding: 5px; background-color: #f8f9fa; border: 1px solid #dee2e6; border-radius: 4px;")
        status_layout.addWidget(self.connection_label)
        
        # 时间戳显示
        self.timestamp_label = QLabel("Last Update: --:--:--.---")
        self.timestamp_label.setStyleSheet("color: blue; font-weight: bold; padding: 5px; background-color: #e3f2fd; border: 1px solid #bbdefb; border-radius: 4px;")
        status_layout.addWidget(self.timestamp_label)
        
        # 控制模式显示
        self.mode_label = QLabel("Current Mode: Manual")
        self.mode_label.setStyleSheet("color: #1976d2; font-weight: bold; padding: 5px; background-color: #e8f5e8; border: 1px solid #c8e6c9; border-radius: 4px;")
        status_layout.addWidget(self.mode_label)
        
        status_layout.addStretch()
        main_layout.addWidget(status_bar)
        
        # 主要内容区域 - 使用水平分割
        content_widget = QWidget()
        content_layout = QHBoxLayout(content_widget)
        
        # 左侧控制面板 (40%宽度)
        control_panel = self.create_control_panel()
        content_layout.addWidget(control_panel, 4)
        
        # 右侧状态面板 (60%宽度)
        status_panel = self.create_status_panel()
        content_layout.addWidget(status_panel, 6)
        
        main_layout.addWidget(content_widget, 1)
        
        # 底部信息栏
        info_bar = QWidget()
        info_layout = QHBoxLayout(info_bar)
        
        # 版本信息
        version_label = QLabel("Vehicle Control System v1.0")
        version_label.setStyleSheet("color: #666; font-size: 10px;")
        info_layout.addWidget(version_label)
        
        info_layout.addStretch()
        
        # 快捷键提示
        shortcut_label = QLabel("Press 'H' for help")
        shortcut_label.setStyleSheet("color: #666; font-size: 10px;")
        info_layout.addWidget(shortcut_label)
        
        main_layout.addWidget(info_bar)
        
        # Set focus policy for keyboard events
        self.setFocusPolicy(Qt.StrongFocus)
    
    def add_command_log(self, message: str):
        """Add a message to the command log with timestamp."""
        timestamp = time.strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        
        # 添加到log文本框
        self.log_text.append(log_entry)
        
        # 自动滚动到底部
        self.log_text.verticalScrollBar().setValue(
            self.log_text.verticalScrollBar().maximum()
        )
        
        # 同时在控制台输出（用于调试）
        print(log_entry)

    def on_status_updated(self, status_data: dict):
        """Handle status updates in main thread (slot for status_updated signal)."""
        # 连接状态现在由定时器统一管理，这里不再单独处理
        
        # 更新时间戳显示（使用状态数据中的时间戳）
        if 'basic_status' in status_data and 'timestamp' in status_data['basic_status']:
            timestamp = status_data['basic_status']['timestamp']
            self.update_timestamp(timestamp)
        
        # 更新车辆状态显示
        self.update_display_with_status(status_data)

    def update_display_with_status(self, status_data: dict):
        """使用从Isaac Sim接收到的状态数据更新显示"""
        try:
            # 提取分层状态数据
            basic_status = status_data.get('basic_status', {})
            fork_status = status_data.get('fork_status', {})
            sensor_status = status_data.get('sensor_status', {})
            system_status = status_data.get('system_status', {})
            pose_status = status_data.get('pose_status', {})
            
            # 提取IMU数据
            imu_data = sensor_status.get('imu_data', {})
            linear_accel = imu_data.get('linear_acceleration', {})
            angular_vel = imu_data.get('angular_velocity', {})
            
            # 提取位姿数据
            robot_pose = pose_status.get('robot_pose', {})
            position = robot_pose.get('position', {})
            
            # 更新速度显示
            drive_velocity = basic_status.get('drive_velocity', 0.0)
            self.lbl_current_speed.setText(f"{drive_velocity:.2f} m/s")
            
            # 更新转向角度
            steer_angle = basic_status.get('steer_angle', 0.0)
            self.lbl_steering_angle.setText(f"{steer_angle:.3f} rad")
            
            # 更新轮速（使用默认值，因为原始数据中没有这些字段）
            self.lbl_left_wheel.setText("0.00 rad/s")
            self.lbl_right_wheel.setText("0.00 rad/s")
            
            # 更新叉臂多轴控制状态
            fork_x = fork_status.get('fork_x', 0.0)
            fork_y = fork_status.get('fork_y', 0.0)
            fork_z = fork_status.get('fork_z', 0.0)
            fork_roll = fork_status.get('fork_roll', 0.0)
            fork_pitch = fork_status.get('fork_pitch', 0.0)
            fork_cl = fork_status.get('fork_cl', 0.0)
            fork_cr = fork_status.get('fork_cr', 0.0)
            clamp_pressure = fork_status.get('clamp_pressure', 0.0)
            
            self.lbl_fork_x.setText(f"{fork_x:.3f} m")
            self.lbl_fork_y.setText(f"{fork_y:.3f} m")
            self.lbl_fork_z.setText(f"{fork_z:.3f} m")
            self.lbl_fork_roll.setText(f"{fork_roll:.3f} rad")
            self.lbl_fork_pitch.setText(f"{fork_pitch:.3f} rad")
            self.lbl_fork_cl.setText(f"{fork_cl:.3f}")
            self.lbl_fork_cr.setText(f"{fork_cr:.3f}")
            self.lbl_clamp_pressure.setText(f"{clamp_pressure:.1f} kPa")
            
            # 更新位置信息
            pos_x = position.get('x', 0.0)
            pos_y = position.get('y', 0.0)
            pos_z = position.get('z', 0.0)
            self.lbl_position_x.setText(f"{pos_x:.3f} m")
            self.lbl_position_y.setText(f"{pos_y:.3f} m")
            self.lbl_position_z.setText(f"{pos_z:.3f} m")
            
            # 更新IMU数据
            imu_accel_x = linear_accel.get('x', 0.0)
            imu_accel_y = linear_accel.get('y', 0.0)
            imu_accel_z = linear_accel.get('z', 0.0)
            imu_gyro_x = angular_vel.get('x', 0.0)
            imu_gyro_y = angular_vel.get('y', 0.0)
            imu_gyro_z = angular_vel.get('z', 0.0)
            
            self.lbl_imu_accel_x.setText(f"{imu_accel_x:.3f} m/s²")
            self.lbl_imu_accel_y.setText(f"{imu_accel_y:.3f} m/s²")
            self.lbl_imu_accel_z.setText(f"{imu_accel_z:.3f} m/s²")
            self.lbl_imu_gyro_x.setText(f"{imu_gyro_x:.3f} rad/s")
            self.lbl_imu_gyro_y.setText(f"{imu_gyro_y:.3f} rad/s")
            self.lbl_imu_gyro_z.setText(f"{imu_gyro_z:.3f} rad/s")
            
            # 更新开关状态
            h_switch_l = sensor_status.get('hswitch_left', False)
            h_switch_r = sensor_status.get('hswitch_right', False)
            v_switch_l = sensor_status.get('vswitch_left', False)
            v_switch_r = sensor_status.get('vswitch_right', False)
            
            self.lbl_h_switch_l.setText("ON" if h_switch_l else "OFF")
            self.lbl_h_switch_l.setStyleSheet("color: green;" if h_switch_l else "color: red;")
            self.lbl_h_switch_r.setText("ON" if h_switch_r else "OFF")
            self.lbl_h_switch_r.setStyleSheet("color: green;" if h_switch_r else "color: red;")
            self.lbl_v_switch_l.setText("ON" if v_switch_l else "OFF")
            self.lbl_v_switch_l.setStyleSheet("color: green;" if v_switch_l else "color: red;")
            self.lbl_v_switch_r.setText("ON" if v_switch_r else "OFF")
            self.lbl_v_switch_r.setStyleSheet("color: green;" if v_switch_r else "color: red;")
            
            # 更新其他状态
            torque_status = sensor_status.get('torque_status', False)
            adjust_control = system_status.get('adjust_control', False)
            data_record = system_status.get('data_record', False)
            data_index = system_status.get('data_index', 0)
            
            self.lbl_torque_status.setText("OVERLOAD" if torque_status else "NORMAL")
            self.lbl_torque_status.setStyleSheet("color: red;" if torque_status else "color: green;")
            self.lbl_adjust_control.setText("ENABLED" if adjust_control else "DISABLED")
            self.lbl_adjust_control.setStyleSheet("color: green;" if adjust_control else "color: red;")
            self.lbl_data_record.setText("ACTIVE" if data_record else "INACTIVE")
            self.lbl_data_record.setStyleSheet("color: green;" if data_record else "color: red;")
            self.lbl_data_index.setText(f"{data_index}")
            

            
            # 自动选择适当的标签页
            self._auto_select_tab(status_data)
            
        except Exception as e:
            print(f"Error updating display with status: {e}")
    

    
    def _auto_select_tab(self, status_data: dict):
        """根据状态变化自动选择最相关的标签页"""
        # 禁用自动标签页切换功能，保持用户当前选择的标签页
        # 用户手动选择的标签页不会被状态变化所改变
        pass

    def update_display(self):
        """Update the status display."""
        # 这个方法现在只用于手动模式下的显示更新
        # 实时状态更新由update_display_with_status处理
        pass

    def update_connection_status(self, connected: bool):
        """Update connection status display."""
        if connected:
            self.connection_status = "Connected"
            self.connection_label.setText("Status: Connected")
            self.connection_label.setStyleSheet("color: green; font-weight: bold;")
        else:
            self.connection_status = "Disconnected"
            self.connection_label.setText("Status: Disconnected")
            self.connection_label.setStyleSheet("color: red; font-weight: bold;")
    
    def update_timestamp(self, timestamp):
        """Update timestamp display."""
        try:
            # Convert timestamp to readable format
            import datetime
            import time
            
            # Handle different timestamp types
            if isinstance(timestamp, str):
                # If it's already a formatted string, use current time
                current_timestamp = time.time()
            elif isinstance(timestamp, (int, float)):
                # Convert to seconds (timestamp might be in milliseconds)
                current_timestamp = float(timestamp)
                # If timestamp is too large (likely milliseconds), convert to seconds
                if current_timestamp > 1e10:  # If timestamp > 2001-09-09 (in seconds)
                    current_timestamp = current_timestamp / 1000.0  # Convert milliseconds to seconds
            else:
                current_timestamp = time.time()
            
            dt = datetime.datetime.fromtimestamp(current_timestamp)
            time_str = dt.strftime("%H:%M:%S.%f")[:-3]  # Show milliseconds
            
            # Update timestamp label if it exists
            if hasattr(self, 'timestamp_label'):
                self.timestamp_label.setText(f"Last Update: {time_str}")
                
        except Exception as e:
            # Fallback to current time if conversion fails
            import datetime
            import time
            dt = datetime.datetime.fromtimestamp(time.time())
            time_str = dt.strftime("%H:%M:%S.%f")[:-3]
            if hasattr(self, 'timestamp_label'):
                self.timestamp_label.setText(f"Last Update: {time_str} (Error: {str(e)})")
        
    def create_control_panel(self) -> QWidget:
        """Create the control panel with buttons."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Vehicle movement controls
        vehicle_group = QGroupBox("Vehicle Movement")
        vehicle_layout = QGridLayout(vehicle_group)
        
        # Speed controls
        self.btn_speed_up = QPushButton("Speed Up")
        self.btn_speed_up.clicked.connect(self.on_speed_up)
        vehicle_layout.addWidget(self.btn_speed_up, 0, 1)
        
        self.btn_turn_left = QPushButton("Turn Left")
        self.btn_turn_left.clicked.connect(self.on_turn_left)
        vehicle_layout.addWidget(self.btn_turn_left, 1, 0)
        
        self.btn_stop = QPushButton("STOP")
        self.btn_stop.clicked.connect(self.on_stop)
        self.btn_stop.setStyleSheet("QPushButton { background-color: red; color: white; font-weight: bold; }")
        vehicle_layout.addWidget(self.btn_stop, 1, 1)
        
        self.btn_turn_right = QPushButton("Turn Right")
        self.btn_turn_right.clicked.connect(self.on_turn_right)
        vehicle_layout.addWidget(self.btn_turn_right, 1, 2)
        
        self.btn_speed_down = QPushButton("Speed Down")
        self.btn_speed_down.clicked.connect(self.on_speed_down)
        vehicle_layout.addWidget(self.btn_speed_down, 2, 1)
        
        layout.addWidget(vehicle_group)
        
        # Fork controls - 替换为多轴控制面板
        fork_group = self.create_fork_multi_axis_panel()
        layout.addWidget(fork_group)
        
        # Control mode selection (Manual, Keyboard, Auto)
        mode_group = QGroupBox("Control Mode")
        mode_layout = QVBoxLayout(mode_group)
        
        self.mode_group = QButtonGroup()
        
        self.manual_radio = QRadioButton("Manual Control")
        self.keyboard_radio = QRadioButton("Keyboard Control") 
        self.auto_radio = QRadioButton("Auto Control")
        
        self.mode_group.addButton(self.manual_radio, 0)
        self.mode_group.addButton(self.keyboard_radio, 1)
        self.mode_group.addButton(self.auto_radio, 2)
        
        mode_layout.addWidget(self.manual_radio)
        mode_layout.addWidget(self.keyboard_radio)
        mode_layout.addWidget(self.auto_radio)
        
        # Set manual mode as default
        self.manual_radio.setChecked(True)
        self.mode_group.buttonClicked.connect(self.on_mode_changed)
        
        layout.addWidget(mode_group)
        
        # Manual mode checkbox
        self.cb_manual = QCheckBox("Manual Control Mode")
        self.cb_manual.setChecked(True)
        self.cb_manual.stateChanged.connect(self.on_manual_mode_changed)
        layout.addWidget(self.cb_manual)
        
        # Reset button
        self.btn_reset = QPushButton("Reset All")
        self.btn_reset.clicked.connect(self.on_reset)
        layout.addWidget(self.btn_reset)
        
        layout.addStretch()
        return panel
        
    def create_status_panel(self) -> QWidget:
        """Create the status display panel with tabbed layout."""
        panel = QWidget()
        main_layout = QVBoxLayout(panel)
        
        # Create tab widget for organized status display
        self.tab_widget = QTabWidget()
        
        # Basic Status Tab
        basic_tab = QWidget()
        basic_layout = QGridLayout(basic_tab)
        
        row = 0
        # Motion status
        basic_layout.addWidget(QLabel("Current Speed:"), row, 0)
        self.lbl_current_speed = QLabel("0.0 m/s")
        basic_layout.addWidget(self.lbl_current_speed, row, 1)
        
        row += 1
        basic_layout.addWidget(QLabel("Steering Angle:"), row, 0)
        self.lbl_steering_angle = QLabel("0.0 rad")
        basic_layout.addWidget(self.lbl_steering_angle, row, 1)
        
        row += 1
        basic_layout.addWidget(QLabel("Left Wheel Speed:"), row, 0)
        self.lbl_left_wheel = QLabel("0.00 rad/s")
        basic_layout.addWidget(self.lbl_left_wheel, row, 1)
        
        row += 1
        basic_layout.addWidget(QLabel("Right Wheel Speed:"), row, 0)
        self.lbl_right_wheel = QLabel("0.00 rad/s")
        basic_layout.addWidget(self.lbl_right_wheel, row, 1)
        
        row += 1
        basic_layout.addWidget(QLabel("Position X:"), row, 0)
        self.lbl_position_x = QLabel("0.000 m")
        basic_layout.addWidget(self.lbl_position_x, row, 1)
        
        row += 1
        basic_layout.addWidget(QLabel("Position Y:"), row, 0)
        self.lbl_position_y = QLabel("0.000 m")
        basic_layout.addWidget(self.lbl_position_y, row, 1)
        
        row += 1
        basic_layout.addWidget(QLabel("Position Z:"), row, 0)
        self.lbl_position_z = QLabel("0.000 m")
        basic_layout.addWidget(self.lbl_position_z, row, 1)
        
        # System status
        row += 1
        basic_layout.addWidget(QLabel("Torque Status:"), row, 0)
        self.lbl_torque_status = QLabel("NORMAL")
        self.lbl_torque_status.setStyleSheet("color: green;")
        basic_layout.addWidget(self.lbl_torque_status, row, 1)
        
        row += 1
        basic_layout.addWidget(QLabel("Adjust Control:"), row, 0)
        self.lbl_adjust_control = QLabel("DISABLED")
        self.lbl_adjust_control.setStyleSheet("color: red;")
        basic_layout.addWidget(self.lbl_adjust_control, row, 1)
        
        row += 1
        basic_layout.addWidget(QLabel("Data Recording:"), row, 0)
        self.lbl_data_record = QLabel("INACTIVE")
        self.lbl_data_record.setStyleSheet("color: red;")
        basic_layout.addWidget(self.lbl_data_record, row, 1)
        
        row += 1
        basic_layout.addWidget(QLabel("Data Index:"), row, 0)
        self.lbl_data_index = QLabel("0")
        basic_layout.addWidget(self.lbl_data_index, row, 1)
        
        self.tab_widget.addTab(basic_tab, "Basic Status")
        
        # Fork Status Tab
        fork_tab = QWidget()
        fork_layout = QGridLayout(fork_tab)
        
        row = 0
        fork_layout.addWidget(QLabel("Fork X:"), row, 0)
        self.lbl_fork_x = QLabel("0.000 m")
        fork_layout.addWidget(self.lbl_fork_x, row, 1)
        
        row += 1
        fork_layout.addWidget(QLabel("Fork Y:"), row, 0)
        self.lbl_fork_y = QLabel("0.000 m")
        fork_layout.addWidget(self.lbl_fork_y, row, 1)
        
        row += 1
        fork_layout.addWidget(QLabel("Fork Z:"), row, 0)
        self.lbl_fork_z = QLabel("0.000 m")
        fork_layout.addWidget(self.lbl_fork_z, row, 1)
        
        row += 1
        fork_layout.addWidget(QLabel("Fork Roll:"), row, 0)
        self.lbl_fork_roll = QLabel("0.000 rad")
        fork_layout.addWidget(self.lbl_fork_roll, row, 1)
        
        row += 1
        fork_layout.addWidget(QLabel("Fork Pitch:"), row, 0)
        self.lbl_fork_pitch = QLabel("0.000 rad")
        fork_layout.addWidget(self.lbl_fork_pitch, row, 1)
        
        row += 1
        fork_layout.addWidget(QLabel("Left Clamp:"), row, 0)
        self.lbl_fork_cl = QLabel("0.000")
        fork_layout.addWidget(self.lbl_fork_cl, row, 1)
        
        row += 1
        fork_layout.addWidget(QLabel("Right Clamp:"), row, 0)
        self.lbl_fork_cr = QLabel("0.000")
        fork_layout.addWidget(self.lbl_fork_cr, row, 1)
        
        row += 1
        fork_layout.addWidget(QLabel("Clamp Pressure:"), row, 0)
        self.lbl_clamp_pressure = QLabel("0.0 kPa")
        fork_layout.addWidget(self.lbl_clamp_pressure, row, 1)
        
        self.tab_widget.addTab(fork_tab, "Fork Status")
        
        # Sensor Status Tab
        sensor_tab = QWidget()
        sensor_layout = QGridLayout(sensor_tab)
        
        row = 0
        sensor_layout.addWidget(QLabel("IMU Accel X:"), row, 0)
        self.lbl_imu_accel_x = QLabel("0.000 m/s²")
        sensor_layout.addWidget(self.lbl_imu_accel_x, row, 1)
        
        row += 1
        sensor_layout.addWidget(QLabel("IMU Accel Y:"), row, 0)
        self.lbl_imu_accel_y = QLabel("0.000 m/s²")
        sensor_layout.addWidget(self.lbl_imu_accel_y, row, 1)
        
        row += 1
        sensor_layout.addWidget(QLabel("IMU Accel Z:"), row, 0)
        self.lbl_imu_accel_z = QLabel("0.000 m/s²")
        sensor_layout.addWidget(self.lbl_imu_accel_z, row, 1)
        
        row += 1
        sensor_layout.addWidget(QLabel("IMU Gyro X:"), row, 0)
        self.lbl_imu_gyro_x = QLabel("0.000 rad/s")
        sensor_layout.addWidget(self.lbl_imu_gyro_x, row, 1)
        
        row += 1
        sensor_layout.addWidget(QLabel("IMU Gyro Y:"), row, 0)
        self.lbl_imu_gyro_y = QLabel("0.000 rad/s")
        sensor_layout.addWidget(self.lbl_imu_gyro_y, row, 1)
        
        row += 1
        sensor_layout.addWidget(QLabel("IMU Gyro Z:"), row, 0)
        self.lbl_imu_gyro_z = QLabel("0.000 rad/s")
        sensor_layout.addWidget(self.lbl_imu_gyro_z, row, 1)
        
        self.tab_widget.addTab(sensor_tab, "Sensor Status")
        
        # Switch Status Tab
        switch_tab = QWidget()
        switch_layout = QGridLayout(switch_tab)
        
        row = 0
        switch_layout.addWidget(QLabel("H Switch L:"), row, 0)
        self.lbl_h_switch_l = QLabel("OFF")
        self.lbl_h_switch_l.setStyleSheet("color: red;")
        switch_layout.addWidget(self.lbl_h_switch_l, row, 1)
        
        row += 1
        switch_layout.addWidget(QLabel("H Switch R:"), row, 0)
        self.lbl_h_switch_r = QLabel("OFF")
        self.lbl_h_switch_r.setStyleSheet("color: red;")
        switch_layout.addWidget(self.lbl_h_switch_r, row, 1)
        
        row += 1
        switch_layout.addWidget(QLabel("V Switch L:"), row, 0)
        self.lbl_v_switch_l = QLabel("OFF")
        self.lbl_v_switch_l.setStyleSheet("color: red;")
        switch_layout.addWidget(self.lbl_v_switch_l, row, 1)
        
        row += 1
        switch_layout.addWidget(QLabel("V Switch R:"), row, 0)
        self.lbl_v_switch_r = QLabel("OFF")
        self.lbl_v_switch_r.setStyleSheet("color: red;")
        switch_layout.addWidget(self.lbl_v_switch_r, row, 1)
        
        self.tab_widget.addTab(switch_tab, "Switch Status")
        
        main_layout.addWidget(self.tab_widget)
        
        # Control values group
        control_group = QGroupBox("Control Values")
        control_layout = QGridLayout(control_group)
        
        # Speed control spinbox
        control_layout.addWidget(QLabel("Target Speed:"), 0, 0)
        self.sb_target_speed = QDoubleSpinBox()
        self.sb_target_speed.setRange(-4.0, 4.0)
        self.sb_target_speed.setSingleStep(0.2)
        self.sb_target_speed.setValue(0.0)
        self.sb_target_speed.valueChanged.connect(self.on_target_speed_changed)
        control_layout.addWidget(self.sb_target_speed, 0, 1)
        
        # Steering control spinbox
        control_layout.addWidget(QLabel("Target Steering:"), 1, 0)
        self.sb_target_steering = QDoubleSpinBox()
        self.sb_target_steering.setRange(-1.57, 1.57)
        self.sb_target_steering.setSingleStep(0.16)
        self.sb_target_steering.setValue(0.0)
        self.sb_target_steering.valueChanged.connect(self.on_target_steering_changed)
        control_layout.addWidget(self.sb_target_steering, 1, 1)
        
        # Fork Z position control spinbox
        control_layout.addWidget(QLabel("Target Fork Z:"), 2, 0)
        self.sb_target_fork = QDoubleSpinBox()
        self.sb_target_fork.setRange(0.0, 2.0)
        self.sb_target_fork.setSingleStep(0.1)
        self.sb_target_fork.setValue(0.0)
        self.sb_target_fork.valueChanged.connect(self.on_target_fork_changed)
        control_layout.addWidget(self.sb_target_fork, 2, 1)
        
        main_layout.addWidget(control_group)
        
        # Keyboard help button
        help_group = QGroupBox("帮助")
        help_layout = QVBoxLayout(help_group)
        
        self.btn_help = QPushButton("键盘控制帮助")
        self.btn_help.clicked.connect(self.show_keyboard_help)
        self.btn_help.setStyleSheet("QPushButton { padding: 8px; font-weight: bold; }")
        help_layout.addWidget(self.btn_help)
        
        main_layout.addWidget(help_group)
        
        # Command log
        log_group = QGroupBox("Command Log")
        log_layout = QVBoxLayout(log_group)
        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(150)
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)
        main_layout.addWidget(log_group)
        
        main_layout.addStretch()
        return panel
        
    # eCAL initialization removed - handled by ui_client.py
    
    def send_command(self):
        """Send current control command via signal to ui_client."""
        # 创建命令摘要用于日志
        command_summary = []
        
        # 检查并添加有变化的参数到日志
        if self.manual_state['steer_speed'] != 0.0:
            command_summary.append(f"Speed: {self.manual_state['steer_speed']:.2f} m/s")
        if self.manual_state['steer_yaw'] != 0.0:
            command_summary.append(f"Steering: {self.manual_state['steer_yaw']:.2f} rad")
        if self.manual_state['fork_z'] != 0.0:
            command_summary.append(f"Fork Z: {self.manual_state['fork_z']:.2f} m")
        if self.manual_state['fork_x'] != 0.0:
            command_summary.append(f"Fork X: {self.manual_state['fork_x']:.2f} m")
        if self.manual_state['fork_y'] != 0.0:
            command_summary.append(f"Fork Y: {self.manual_state['fork_y']:.2f} m")
        if self.manual_state['fork_roll'] != 0.0:
            command_summary.append(f"Fork Roll: {self.manual_state['fork_roll']:.2f} rad")
        if self.manual_state['fork_pitch'] != 0.0:
            command_summary.append(f"Fork Pitch: {self.manual_state['fork_pitch']:.2f} rad")
        if self.manual_state['fork_cl'] != 0.0:
            command_summary.append(f"Clamp Left: {self.manual_state['fork_cl']:.2f}")
        if self.manual_state['fork_cr'] != 0.0:
            command_summary.append(f"Clamp Right: {self.manual_state['fork_cr']:.2f}")
        
        # 如果没有变化的参数，记录为"No movement"
        if not command_summary:
            command_summary.append("No movement")
        
        # 添加到command log
        self.add_command_log(f"Sending command: {', '.join(command_summary)}")
        
        # Emit signal for ui_client to handle eCAL communication
        self.command_sent.emit("vehicle_control", self.manual_state.copy())
    
    # Status receiving removed - handled by ui_client.py
    
    # Vehicle control button handlers
    def on_speed_up(self):
        """Increase vehicle speed."""
        current_speed = self.manual_state['steer_speed']

        if current_speed < 4.0:
            new_speed = min(4.0, current_speed + 0.2)
            if new_speed != current_speed:
                self.manual_state['steer_speed'] = new_speed
                self.sb_target_speed.setValue(new_speed)
                self.send_command()
    
    def on_speed_down(self):
        """Decrease vehicle speed."""
        current_speed = self.manual_state['steer_speed']
        if current_speed > -4.0:
            new_speed = max(-4.0, current_speed - 0.2)
            if new_speed != current_speed:
                self.manual_state['steer_speed'] = new_speed
                self.sb_target_speed.setValue(new_speed)
                self.send_command()
    
    def on_stop(self):
        """Stop vehicle movement."""
        if self.manual_state['steer_speed'] != 0.0:
            self.manual_state['steer_speed'] = 0.0
            self.sb_target_speed.setValue(0.0)
            self.send_command()
    
    def on_turn_left(self):
        """Turn vehicle left."""
        current_yaw = self.manual_state['steer_yaw']
        if current_yaw < 1.57:
            new_yaw = min(1.57, current_yaw + 0.16)
            if new_yaw != current_yaw:
                self.manual_state['steer_yaw'] = new_yaw
                self.sb_target_steering.setValue(new_yaw)
                self.send_command()
    
    def on_turn_right(self):
        """Turn vehicle right."""
        current_yaw = self.manual_state['steer_yaw']
        if current_yaw > -1.57:
            new_yaw = max(-1.57, current_yaw - 0.16)
            if new_yaw != current_yaw:
                self.manual_state['steer_yaw'] = new_yaw
                self.sb_target_steering.setValue(new_yaw)
                self.send_command()
    
    # Fork control button handlers
    def on_lift_up(self):
        """Lift fork up."""
        current_height = self.manual_state['fork_z']
        if current_height < 2.0:
            new_height = min(2.0, current_height + 0.01)
            if new_height != current_height:
                self.manual_state['fork_z'] = new_height
                self.manual_state['fork_speed'] = 0.1  # Positive speed for up
                self.sb_target_fork.setValue(new_height)
                self.send_command()
    
    def on_lift_down(self):
        """Lift fork down."""
        current_height = self.manual_state['fork_z']
        if current_height > 0.0:
            new_height = max(0.0, current_height - 0.01)
            if new_height != current_height:
                self.manual_state['fork_z'] = new_height
                self.manual_state['fork_speed'] = -0.1  # Negative speed for down
                self.sb_target_fork.setValue(new_height)
                self.send_command()
    
    def on_lift_stop(self):
        """Stop fork movement."""
        if self.manual_state['fork_speed'] != 0.0:
            self.manual_state['fork_speed'] = 0.0
            self.send_command()
    
    # Other control handlers
    def on_manual_mode_changed(self, state):
        """Handle manual mode checkbox change."""
        self.manual_state['is_manual'] = state == Qt.Checked
        # Enable/disable controls based on manual mode
        controls_enabled = self.manual_state['is_manual']
        
        for btn in [self.btn_speed_up, self.btn_speed_down, self.btn_stop,
                   self.btn_turn_left, self.btn_turn_right,
                   self.btn_lift_up, self.btn_lift_down, self.btn_lift_stop,
                   # 叉臂多轴控制按钮
                   self.btn_fork_x_minus, self.btn_fork_x_stop, self.btn_fork_x_plus,
                   self.btn_fork_y_minus, self.btn_fork_y_stop, self.btn_fork_y_plus,
                   self.btn_fork_z_minus, self.btn_fork_z_stop, self.btn_fork_z_plus,
                   self.btn_fork_roll_minus, self.btn_fork_roll_stop, self.btn_fork_roll_plus,
                   self.btn_fork_pitch_minus, self.btn_fork_pitch_stop, self.btn_fork_pitch_plus,
                   self.btn_fork_cl_minus, self.btn_fork_cl_stop, self.btn_fork_cl_plus,
                   self.btn_fork_cr_minus, self.btn_fork_cr_stop, self.btn_fork_cr_plus,
                   self.btn_pressure_minus, self.btn_pressure_stop, self.btn_pressure_plus,
                   self.btn_fork_all_stop]:
            btn.setEnabled(controls_enabled)
    
    def on_reset(self):
        """Reset all control values."""
        # 保存旧状态用于比较
        old_state = self._get_current_state()
        
        self.manual_state.update({
            'steer_speed': 0.0,
            'steer_yaw': 0.0,
            'fork_speed': 0.0,
        })
        
        self.sb_target_speed.setValue(0.0)
        self.sb_target_steering.setValue(0.0)
        self.sb_target_fork.setValue(0.0)
        
        # 添加到command log
        self.add_command_log("Resetting all control values to zero")
        
        # 检查状态是否发生变化
        if self._has_state_changed(old_state):
            self.send_command()
    
    def show_keyboard_help(self):
        """显示键盘帮助对话框"""
        from PyQt5.QtWidgets import QDialog, QVBoxLayout, QTextEdit, QPushButton
        
        help_dialog = QDialog(self)
        help_dialog.setWindowTitle("键盘控制帮助")
        help_dialog.setMinimumSize(500, 600)
        
        layout = QVBoxLayout(help_dialog)
        
        # 创建文本编辑框显示帮助信息
        help_text = QTextEdit()
        help_text.setReadOnly(True)
        help_text.setPlainText(
            "=== 车辆控制 ===\n"
            "W/S: 前进/后退\n"
            "A/D: 左转/右转\n"
            "空格键: 停止车辆\n"
            "R: 重置所有控制\n\n"
            "=== 货叉多轴控制 ===\n"
            "I/K: 货叉 X轴 +/-\n"
            "J/L: 货叉 Y轴 +/-\n"
            "U/O: 货叉 Z轴 +/-\n"
            "8/2: 货叉俯仰角 +/-\n"
            "4/6: 货叉横滚角 +/-\n"
            "1/2: 左侧夹具 +/-\n"
            "3/4: 右侧夹具 +/-\n"
            "5/6: 夹具压力 +/-\n"
            "F: 停止货叉运动\n\n"
            "=== 控制模式 ===\n"
            "(通过界面按钮切换，无键盘快捷键)"
        )
        help_text.setStyleSheet("font-family: monospace; font-size: 12px;")
        
        # 关闭按钮
        close_button = QPushButton("关闭")
        close_button.clicked.connect(help_dialog.accept)
        
        layout.addWidget(help_text)
        layout.addWidget(close_button)
        
        help_dialog.exec()
    
    def on_mode_changed(self, button):
        """Handle control mode change."""
        mode_id = self.mode_group.id(button)
        
        if mode_id == 0:  # Manual mode
            self.current_mode = "manual"
            self.update_ui_for_mode("manual")
            self.mode_label.setText("Current Mode: Manual")
            self.mode_label.setStyleSheet("color: #1976d2; font-weight: bold; padding: 5px; background-color: #e8f5e8; border: 1px solid #c8e6c9; border-radius: 4px;")
        elif mode_id == 1:  # Keyboard mode
            self.current_mode = "keyboard"
            self.update_ui_for_mode("keyboard")
            self.mode_label.setText("Current Mode: Keyboard")
            self.mode_label.setStyleSheet("color: #f57c00; font-weight: bold; padding: 5px; background-color: #fff3e0; border: 1px solid #ffcc80; border-radius: 4px;")
        elif mode_id == 2:  # Auto mode
            self.current_mode = "auto"
            self.update_ui_for_mode("auto")
            self.mode_label.setText("Current Mode: Auto")
            self.mode_label.setStyleSheet("color: #7b1fa2; font-weight: bold; padding: 5px; background-color: #f3e5f5; border: 1px solid #ce93d8; border-radius: 4px;")
        
        # Send control mode switch command
        self.send_control_mode_switch(mode_id)
        
    def update_ui_for_mode(self, mode: int):
        """Update UI controls based on selected control mode."""
        if mode == 0:  # Manual mode
            self.cb_manual.setEnabled(True)
            self.cb_manual.setChecked(True)
            self.on_manual_mode_changed(2)  # Enable manual controls
        elif mode == 1:  # Keyboard mode
            self.cb_manual.setEnabled(False)
            self.cb_manual.setChecked(False)
            self.on_manual_mode_changed(0)  # Disable manual controls
        elif mode == 2:  # Auto mode
            self.cb_manual.setEnabled(False)
            self.cb_manual.setChecked(False)
            self.on_manual_mode_changed(0)  # Disable manual controls
            
    def send_control_mode_switch(self, mode: int):
        """Send control mode switch command."""
        # This will be handled by the communication layer
        print(f"Sending control mode switch: {mode}")
        # The actual eCAL message sending is handled by ui_client.py
    
    # Spinbox value change handlers
    def on_target_speed_changed(self, value):
        """Handle target speed spinbox change."""
        # 保存旧状态用于比较
        old_speed = self.manual_state['steer_speed']
        
        self.manual_state['steer_speed'] = value
        
        # 检查状态是否发生变化
        if self.manual_state['steer_speed'] != old_speed:
            self.send_command()
    
    def on_target_steering_changed(self, value):
        """Handle target steering spinbox change."""
        # 保存旧状态用于比较
        old_yaw = self.manual_state['steer_yaw']
        
        self.manual_state['steer_yaw'] = value
        
        # 检查状态是否发生变化
        if self.manual_state['steer_yaw'] != old_yaw:
            self.send_command()
    
    def on_target_fork_changed(self, value):
        """Handle target fork Z position spinbox change."""
        # 保存旧状态用于比较
        old_height = self.manual_state['fork_z']
        
        self.manual_state['fork_z'] = value
        
        # 检查状态是否发生变化
        if self.manual_state['fork_z'] != old_height:
            self.send_command()
    
    # 键盘事件处理方法
    def keyPressEvent(self, event: QKeyEvent):
        """处理按键按下事件"""
        key = event.key()
        
        # 添加按键到已按下集合
        self.pressed_keys.add(key)
        
        # 处理立即响应的按键
        if key == Qt.Key_Space:
            # 空格键 - 停止车辆
            self.on_stop()
        elif key == Qt.Key_R:
            # R键 - 重置所有控制
            self.on_reset()
        elif key == Qt.Key_F:
            # F键 - 停止货叉运动
            self.on_fork_stop()
        elif key == Qt.Key_H:
            # H键 - 显示帮助对话框
            self.show_help_dialog()
        
        # 调用父类处理其他按键
        super().keyPressEvent(event)
    
    def keyReleaseEvent(self, event: QKeyEvent):
        """处理按键释放事件"""
        key = event.key()
        
        # 从已按下集合中移除按键
        if key in self.pressed_keys:
            self.pressed_keys.remove(key)
        
        # 调用父类处理其他按键
        super().keyReleaseEvent(event)
    
    def handle_keyboard_movement(self):
        """处理持续键盘移动控制"""
        if not self.manual_state['is_manual']:
            return
            
        # 保存旧状态用于比较
        old_state = self._get_current_state()
        
        # 检查是否有移动按键被按下
        movement_keys_pressed = (Qt.Key_W in self.pressed_keys or Qt.Key_Up in self.pressed_keys or
                               Qt.Key_S in self.pressed_keys or Qt.Key_Down in self.pressed_keys)
        
        # 车辆移动控制 - 简化逻辑，避免方向切换时的速度重置
        current_speed = self.manual_state['steer_speed']
        
        if Qt.Key_W in self.pressed_keys or Qt.Key_Up in self.pressed_keys:
            # W键或上箭头 - 加速/前进
            new_speed = min(4.0, current_speed + 0.1)
            self.manual_state['steer_speed'] = new_speed
            
        elif Qt.Key_S in self.pressed_keys or Qt.Key_Down in self.pressed_keys:
            # S键或下箭头 - 减速/后退
            new_speed = max(-4.0, current_speed - 0.1)
            self.manual_state['steer_speed'] = new_speed
        
        # 车辆转向控制
        if Qt.Key_A in self.pressed_keys or Qt.Key_Left in self.pressed_keys:
            # A键或左箭头 - 左转
            self.manual_state['steer_yaw'] = min(1.57, self.manual_state['steer_yaw'] + 0.05)
        elif Qt.Key_D in self.pressed_keys or Qt.Key_Right in self.pressed_keys:
            # D键或右箭头 - 右转
            self.manual_state['steer_yaw'] = max(-1.57, self.manual_state['steer_yaw'] - 0.05)
        
        # 叉臂多轴键盘控制 - 避免键位冲突
        if Qt.Key_I in self.pressed_keys:
            # I键 - 叉臂向前移动(X+)
            self.manual_state['fork_x'] = min(1.0, self.manual_state['fork_x'] + 0.01)
        elif Qt.Key_K in self.pressed_keys:
            # K键 - 叉臂向后移动(X-)
            self.manual_state['fork_x'] = max(-1.0, self.manual_state['fork_x'] - 0.01)
            
        if Qt.Key_J in self.pressed_keys:
            # J键 - 叉臂向左移动(Y-)
            self.manual_state['fork_y'] = max(-0.5, self.manual_state['fork_y'] - 0.01)
        elif Qt.Key_L in self.pressed_keys:
            # L键 - 叉臂向右移动(Y+)
            self.manual_state['fork_y'] = min(0.5, self.manual_state['fork_y'] + 0.01)
            
        if Qt.Key_U in self.pressed_keys:
            # U键 - 叉臂上升(Z+)
            self.manual_state['fork_z'] = min(2.0, self.manual_state['fork_z'] + 0.01)
        elif Qt.Key_O in self.pressed_keys:
            # O键 - 叉臂下降(Z-)
            self.manual_state['fork_z'] = max(0.0, self.manual_state['fork_z'] - 0.01)
            
        if Qt.Key_8 in self.pressed_keys:
            # 8键 - 增加俯仰角
            self.manual_state['fork_pitch'] = min(0.5, self.manual_state['fork_pitch'] + 0.01)
        elif Qt.Key_2 in self.pressed_keys:
            # 2键 - 减少俯仰角
            self.manual_state['fork_pitch'] = max(-0.5, self.manual_state['fork_pitch'] - 0.01)
            
        if Qt.Key_4 in self.pressed_keys:
            # 4键 - 增加翻滚角
            self.manual_state['fork_roll'] = min(0.5, self.manual_state['fork_roll'] + 0.01)
        elif Qt.Key_6 in self.pressed_keys:
            # 6键 - 减少翻滚角
            self.manual_state['fork_roll'] = max(-0.5, self.manual_state['fork_roll'] - 0.01)
            
        if Qt.Key_1 in self.pressed_keys:
            # 1键 - 左夹紧器收紧
            self.manual_state['fork_cl'] = min(1.0, self.manual_state['fork_cl'] + 0.01)
        elif Qt.Key_2 in self.pressed_keys:
            # 2键 - 左夹紧器放松
            self.manual_state['fork_cl'] = max(0.0, self.manual_state['fork_cl'] - 0.01)
            
        if Qt.Key_3 in self.pressed_keys:
            # 3键 - 右夹紧器收紧
            self.manual_state['fork_cr'] = min(1.0, self.manual_state['fork_cr'] + 0.01)
        elif Qt.Key_4 in self.pressed_keys:
            # 4键 - 右夹紧器放松
            self.manual_state['fork_cr'] = max(0.0, self.manual_state['fork_cr'] - 0.01)
            
        if Qt.Key_5 in self.pressed_keys:
            # 5键 - 增加夹紧压力
            self.manual_state['clamp_pressure'] = min(100.0, self.manual_state['clamp_pressure'] + 1.0)
        elif Qt.Key_6 in self.pressed_keys:
            # 6键 - 减少夹紧压力
            self.manual_state['clamp_pressure'] = max(0.0, self.manual_state['clamp_pressure'] - 1.0)
        
        # 检查状态是否发生变化，或者有移动按键被按下时发送命令
        # 这样可以确保即使速度达到极限值，按键按下时仍然发送命令
        if self._has_state_changed(old_state) or movement_keys_pressed:
            self.send_command()

    # 叉臂多轴控制事件处理方法
    def on_fork_x_change(self, delta: float):
        """X轴位置变化"""
        old_x = self.manual_state['fork_x']
        new_x = max(-1.0, min(1.0, old_x + delta))
        if new_x != old_x:
            self.manual_state['fork_x'] = new_x
            self.send_command()
    
    def on_fork_y_change(self, delta: float):
        """Y轴位置变化"""
        old_y = self.manual_state['fork_y']
        new_y = max(-0.5, min(0.5, old_y + delta))
        if new_y != old_y:
            self.manual_state['fork_y'] = new_y
            self.send_command()
    
    def on_fork_z_change(self, delta: float):
        """Z轴位置变化"""
        old_z = self.manual_state['fork_z']
        new_z = max(0.0, min(2.0, old_z + delta))
        if new_z != old_z:
            self.manual_state['fork_z'] = new_z
            self.send_command()
    
    def on_fork_pitch_change(self, delta: float):
        """俯仰角变化"""
        old_pitch = self.manual_state['fork_pitch']
        new_pitch = max(-0.5, min(0.5, old_pitch + delta))
        if new_pitch != old_pitch:
            self.manual_state['fork_pitch'] = new_pitch
            self.send_command()
    
    def on_fork_roll_change(self, delta: float):
        """翻滚角变化"""
        old_roll = self.manual_state['fork_roll']
        new_roll = max(-0.5, min(0.5, old_roll + delta))
        if new_roll != old_roll:
            self.manual_state['fork_roll'] = new_roll
            self.send_command()
    
    def on_fork_cl_change(self, delta: float):
        """左夹紧器变化"""
        old_cl = self.manual_state['fork_cl']
        new_cl = max(0.0, min(1.0, old_cl + delta))
        if new_cl != old_cl:
            self.manual_state['fork_cl'] = new_cl
            self.send_command()
    
    def on_fork_cr_change(self, delta: float):
        """右夹紧器变化"""
        old_cr = self.manual_state['fork_cr']
        new_cr = max(0.0, min(1.0, old_cr + delta))
        if new_cr != old_cr:
            self.manual_state['fork_cr'] = new_cr
            self.send_command()
    
    def on_pressure_change(self, delta: float):
        """夹紧压力变化"""
        old_pressure = self.manual_state['clamp_pressure']
        new_pressure = max(0.0, min(100.0, old_pressure + delta))
        if new_pressure != old_pressure:
            self.manual_state['clamp_pressure'] = new_pressure
            self.send_command()
    
    def on_fork_stop(self):
        """停止所有叉臂运动"""
        old_state = self._get_current_state()
        
        # 重置所有叉臂控制状态
        self.manual_state.update({
            'fork_x': 0.0,
            'fork_y': 0.0,
            'fork_z': 0.0,
            'fork_roll': 0.0,
            'fork_pitch': 0.0,
            'fork_cl': 0.0,
            'fork_cr': 0.0,
            'clamp_pressure': 0.0
        })
        
        # 检查状态是否发生变化
        if self._has_state_changed(old_state):
            self.send_command()

    def create_fork_multi_axis_panel(self) -> QWidget:
        """创建叉臂多轴控制面板"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # 叉臂多轴控制组
        fork_group = QGroupBox("Fork Multi-Axis Control")
        fork_layout = QGridLayout(fork_group)
        
        # X轴控制
        row = 0
        fork_layout.addWidget(QLabel("X Axis:"), row, 0)
        
        self.btn_fork_x_minus = QPushButton("← X-")
        self.btn_fork_x_minus.clicked.connect(lambda: self.on_fork_x_change(-0.1))
        fork_layout.addWidget(self.btn_fork_x_minus, row, 1)
        
        self.btn_fork_x_stop = QPushButton("X Stop")
        self.btn_fork_x_stop.clicked.connect(lambda: self.on_fork_x_change(0.0))
        fork_layout.addWidget(self.btn_fork_x_stop, row, 2)
        
        self.btn_fork_x_plus = QPushButton("X+ →")
        self.btn_fork_x_plus.clicked.connect(lambda: self.on_fork_x_change(0.1))
        fork_layout.addWidget(self.btn_fork_x_plus, row, 3)
        
        # Y轴控制
        row += 1
        fork_layout.addWidget(QLabel("Y Axis:"), row, 0)
        
        self.btn_fork_y_minus = QPushButton("← Y-")
        self.btn_fork_y_minus.clicked.connect(lambda: self.on_fork_y_change(-0.05))
        fork_layout.addWidget(self.btn_fork_y_minus, row, 1)
        
        self.btn_fork_y_stop = QPushButton("Y Stop")
        self.btn_fork_y_stop.clicked.connect(lambda: self.on_fork_y_change(0.0))
        fork_layout.addWidget(self.btn_fork_y_stop, row, 2)
        
        self.btn_fork_y_plus = QPushButton("Y+ →")
        self.btn_fork_y_plus.clicked.connect(lambda: self.on_fork_y_change(0.05))
        fork_layout.addWidget(self.btn_fork_y_plus, row, 3)
        
        # Z轴控制
        row += 1
        fork_layout.addWidget(QLabel("Z Axis:"), row, 0)
        
        self.btn_fork_z_minus = QPushButton("Z- ↓")
        self.btn_fork_z_minus.clicked.connect(lambda: self.on_fork_z_change(-0.1))
        fork_layout.addWidget(self.btn_fork_z_minus, row, 1)
        
        self.btn_fork_z_stop = QPushButton("Z Stop")
        self.btn_fork_z_stop.clicked.connect(lambda: self.on_fork_z_change(0.0))
        fork_layout.addWidget(self.btn_fork_z_stop, row, 2)
        
        self.btn_fork_z_plus = QPushButton("Z+ ↑")
        self.btn_fork_z_plus.clicked.connect(lambda: self.on_fork_z_change(0.1))
        fork_layout.addWidget(self.btn_fork_z_plus, row, 3)
        
        # 姿态控制
        row += 1
        fork_layout.addWidget(QLabel("Roll:"), row, 0)
        
        self.btn_fork_roll_minus = QPushButton("Roll-")
        self.btn_fork_roll_minus.clicked.connect(lambda: self.on_fork_roll_change(-0.05))
        fork_layout.addWidget(self.btn_fork_roll_minus, row, 1)
        
        self.btn_fork_roll_stop = QPushButton("Roll 0")
        self.btn_fork_roll_stop.clicked.connect(lambda: self.on_fork_roll_change(0.0))
        fork_layout.addWidget(self.btn_fork_roll_stop, row, 2)
        
        self.btn_fork_roll_plus = QPushButton("Roll+")
        self.btn_fork_roll_plus.clicked.connect(lambda: self.on_fork_roll_change(0.05))
        fork_layout.addWidget(self.btn_fork_roll_plus, row, 3)
        
        row += 1
        fork_layout.addWidget(QLabel("Pitch:"), row, 0)
        
        self.btn_fork_pitch_minus = QPushButton("Pitch-")
        self.btn_fork_pitch_minus.clicked.connect(lambda: self.on_fork_pitch_change(-0.05))
        fork_layout.addWidget(self.btn_fork_pitch_minus, row, 1)
        
        self.btn_fork_pitch_stop = QPushButton("Pitch 0")
        self.btn_fork_pitch_stop.clicked.connect(lambda: self.on_fork_pitch_change(0.0))
        fork_layout.addWidget(self.btn_fork_pitch_stop, row, 2)
        
        self.btn_fork_pitch_plus = QPushButton("Pitch+")
        self.btn_fork_pitch_plus.clicked.connect(lambda: self.on_fork_pitch_change(0.05))
        fork_layout.addWidget(self.btn_fork_pitch_plus, row, 3)
        
        # 夹紧器控制
        row += 1
        fork_layout.addWidget(QLabel("Left Clamp:"), row, 0)
        
        self.btn_fork_cl_minus = QPushButton("Cl- ←")
        self.btn_fork_cl_minus.clicked.connect(lambda: self.on_fork_cl_change(-0.1))
        fork_layout.addWidget(self.btn_fork_cl_minus, row, 1)
        
        self.btn_fork_cl_stop = QPushButton("Cl 0")
        self.btn_fork_cl_stop.clicked.connect(lambda: self.on_fork_cl_change(0.0))
        fork_layout.addWidget(self.btn_fork_cl_stop, row, 2)
        
        self.btn_fork_cl_plus = QPushButton("Cl+ →")
        self.btn_fork_cl_plus.clicked.connect(lambda: self.on_fork_cl_change(0.1))
        fork_layout.addWidget(self.btn_fork_cl_plus, row, 3)
        
        row += 1
        fork_layout.addWidget(QLabel("Right Clamp:"), row, 0)
        
        self.btn_fork_cr_minus = QPushButton("Cr- ←")
        self.btn_fork_cr_minus.clicked.connect(lambda: self.on_fork_cr_change(-0.1))
        fork_layout.addWidget(self.btn_fork_cr_minus, row, 1)
        
        self.btn_fork_cr_stop = QPushButton("Cr 0")
        self.btn_fork_cr_stop.clicked.connect(lambda: self.on_fork_cr_change(0.0))
        fork_layout.addWidget(self.btn_fork_cr_stop, row, 2)
        
        self.btn_fork_cr_plus = QPushButton("Cr+ →")
        self.btn_fork_cr_plus.clicked.connect(lambda: self.on_fork_cr_change(0.1))
        fork_layout.addWidget(self.btn_fork_cr_plus, row, 3)
        
        # 压力控制
        row += 1
        fork_layout.addWidget(QLabel("Pressure:"), row, 0)
        
        self.btn_pressure_minus = QPushButton("Pressure-")
        self.btn_pressure_minus.clicked.connect(lambda: self.on_pressure_change(-10.0))
        fork_layout.addWidget(self.btn_pressure_minus, row, 1)
        
        self.btn_pressure_stop = QPushButton("Pressure 0")
        self.btn_pressure_stop.clicked.connect(lambda: self.on_pressure_change(0.0))
        fork_layout.addWidget(self.btn_pressure_stop, row, 2)
        
        self.btn_pressure_plus = QPushButton("Pressure+")
        self.btn_pressure_plus.clicked.connect(lambda: self.on_pressure_change(10.0))
        fork_layout.addWidget(self.btn_pressure_plus, row, 3)
        
        # 停止所有叉臂运动按钮
        row += 1
        self.btn_fork_all_stop = QPushButton("STOP ALL FORK")
        self.btn_fork_all_stop.clicked.connect(self.on_fork_stop)
        self.btn_fork_all_stop.setStyleSheet("QPushButton { background-color: orange; color: white; font-weight: bold; }")
        fork_layout.addWidget(self.btn_fork_all_stop, row, 0, 1, 4)
        
        layout.addWidget(fork_group)
        return panel

    def show_help_dialog(self):
        """显示帮助对话框"""
        from PyQt5.QtWidgets import QDialog, QVBoxLayout, QTextEdit, QPushButton, QTabWidget, QWidget
        
        help_dialog = QDialog(self)
        help_dialog.setWindowTitle("Vehicle Control Help")
        help_dialog.setMinimumWidth(600)
        help_dialog.setMinimumHeight(500)
        
        layout = QVBoxLayout(help_dialog)
        
        # 创建选项卡控件
        tab_widget = QTabWidget()
        
        # 键盘控制选项卡
        keyboard_tab = QWidget()
        keyboard_layout = QVBoxLayout(keyboard_tab)
        
        keyboard_text = QTextEdit()
        keyboard_text.setReadOnly(True)
        keyboard_text.setHtml("""
        <h2>Keyboard Controls</h2>
        <h3>Vehicle Movement:</h3>
        <ul>
        <li><b>W / ↑</b>: Forward acceleration</li>
        <li><b>S / ↓</b>: Backward acceleration</li>
        <li><b>A / ←</b>: Turn left</li>
        <li><b>D / →</b>: Turn right</li>
        <li><b>Space</b>: Emergency stop vehicle</li>
        <li><b>R</b>: Reset all controls</li>
        </ul>
        
        <h3>Fork Multi-Axis Control:</h3>
        <ul>
        <li><b>I</b>: Fork forward (X+)</li>
        <li><b>K</b>: Fork backward (X-)</li>
        <li><b>J</b>: Fork left (Y-)</li>
        <li><b>L</b>: Fork right (Y+)</li>
        <li><b>U</b>: Fork up (Z+)</li>
        <li><b>O</b>: Fork down (Z-)</li>
        <li><b>8</b>: Increase pitch angle</li>
        <li><b>2</b>: Decrease pitch angle</li>
        <li><b>4</b>: Increase roll angle</li>
        <li><b>6</b>: Decrease roll angle</li>
        <li><b>7</b>: Tighten left clamp</li>
        <li><b>9</b>: Loosen left clamp</li>
        <li><b>1</b>: Tighten right clamp</li>
        <li><b>3</b>: Loosen right clamp</li>
        <li><b>+</b>: Increase clamp pressure</li>
        <li><b>-</b>: Decrease clamp pressure</li>
        <li><b>F</b>: Stop all fork movement</li>
        </ul>
        
        <h3>Control Modes:</h3>
        <ul>
        <li><b>M</b>: Switch to Manual mode</li>
        <li><b>K</b>: Switch to Keyboard mode</li>
        <li><b>A</b>: Switch to Auto mode</li>
        </ul>
        """)
        keyboard_layout.addWidget(keyboard_text)
        
        # 界面说明选项卡
        ui_tab = QWidget()
        ui_layout = QVBoxLayout(ui_tab)
        
        ui_text = QTextEdit()
        ui_text.setReadOnly(True)
        ui_text.setHtml("""
        <h2>User Interface Guide</h2>
        
        <h3>Control Panel:</h3>
        <ul>
        <li><b>Vehicle Movement</b>: Speed and steering controls with visual buttons</li>
        <li><b>Control Mode</b>: Switch between Manual, Keyboard, and Auto modes</li>
        <li><b>Fork Multi-Axis Control</b>: Precise control of all fork movements</li>
        <li><b>Target Controls</b>: Set precise target values for speed, steering, and fork Z position</li>
        </ul>
        
        <h3>Status Panel:</h3>
        <ul>
        <li><b>Vehicle Status</b>: Real-time speed, steering, and wheel speeds</li>
        <li><b>Fork Status</b>: Current position and orientation of the fork</li>
        <li><b>IMU Data</b>: Acceleration and gyroscope readings</li>
        <li><b>Switch Status</b>: Limit switch states</li>
        <li><b>System Status</b>: Torque, adjustment control, and data recording status</li>
        </ul>
        
        <h3>Connection Status:</h3>
        <ul>
        <li><b>Green</b>: Connected to simulation server</li>
        <li><b>Red</b>: Disconnected from simulation server</li>
        <li><b>Timestamp</b>: Last received status update time</li>
        <li><b>Current Mode</b>: Active control mode with color coding</li>
        </ul>
        """)
        ui_layout.addWidget(ui_text)
        
        # 系统信息选项卡
        system_tab = QWidget()
        system_layout = QVBoxLayout(system_tab)
        
        system_text = QTextEdit()
        system_text.setReadOnly(True)
        system_text.setHtml("""
        <h2>System Information</h2>
        
        <h3>Version:</h3>
        <p>Vehicle Control System v1.0</p>
        
        <h3>Features:</h3>
        <ul>
        <li>Real-time vehicle control interface</li>
        <li>Multi-axis fork control with 6 degrees of freedom</li>
        <li>Keyboard and manual control modes</li>
        <li>Real-time status monitoring</li>
        <li>eCAL communication protocol</li>
        <li>Compatible with Isaac Sim simulation</li>
        </ul>
        
        <h3>Connection:</h3>
        <p>Using eCAL for real-time communication with simulation server</p>
        
        <h3>Shortcuts:</h3>
        <p>Press 'H' to show this help dialog at any time</p>
        """)
        system_layout.addWidget(system_text)
        
        # 添加选项卡
        tab_widget.addTab(keyboard_tab, "Keyboard Controls")
        tab_widget.addTab(ui_tab, "UI Guide")
        tab_widget.addTab(system_tab, "System Info")
        
        layout.addWidget(tab_widget)
        
        # 关闭按钮
        close_button = QPushButton("Close")
        close_button.clicked.connect(help_dialog.accept)
        layout.addWidget(close_button)
        
        help_dialog.exec_()
