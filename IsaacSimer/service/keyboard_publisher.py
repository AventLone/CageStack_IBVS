import sys
import ecal.core.core as ecal_core
from ecal.core.publisher import ProtoPublisher
from ecal.core.subscriber import ProtoSubscriber
from pynput import keyboard
import math
from protos import vehicle_state_msg_pb2
import asyncio
from service.service_manager import ControlMode
import time
import threading


class KeyBoardPublisher:
    """键盘控制发布器类

    提供车辆和货叉的键盘控制功能，支持多种控制模式切换。

    键盘按键映射说明:
    
    === 车辆控制 ===
    W - 前进 (drive_wheels_velocity: 2π)
    S - 后退 (drive_wheels_velocity: -2π)
    A - 向左转 (steer_wheels_angle_speed: 0.02)
    D - 向右转 (steer_wheels_angle_speed: -0.02)
    Esc - 重置所有控制状态（速度、转向角、货叉位置等）
    
    === 货叉控制 ===
    上箭头 - 货叉上升 (fork_z_speed: 0.003)
    下箭头 - 货叉下降 (fork_z_speed: -0.003)
    左箭头 - 货叉左移 (fork_y_speed: 0.01)
    右箭头 - 货叉右移 (fork_y_speed: -0.01)
    Home - 货叉俯仰角增加 (fork_pitch_speed: 0.001)
    End - 货叉俯仰角减少 (fork_pitch_speed: -0.001)
    
    === 控制模式切换 ===
    1 - 键盘控制模式 (ControlMode.KEYBOARD_CONTROL)
    2 - UI控制模式 (ControlMode.UI_CONTROL) - 同时具备智能唤起UI功能
    3 - 自适应控制模式 (ControlMode.ADAPTIVE_CONTROL)
    4 - AGV控制模式 (ControlMode.AGV_CONTROL)
    E - 自适应控制模式 (兼容原有快捷键)
    R - 键盘控制模式 (兼容原有快捷键)
    
    === 其他控制 ===
    N - 开始记录数据
    M - 停止记录数据并重置为键盘控制模式
    
    注意：所有控制键释放时会自动将对应速度置为0。
    """
    # 控制指令映射 - 便于集中管理和维护
    VEHICLE_CONTROL_MAP = {
        # 'w': ('drive_wheels_velocity', 1),
        # 's': ('drive_wheels_velocity', -1),
        'w': ('steer_wheels_accelerate', 0.15),
        's': ('steer_wheels_accelerate', -0.15),
        'a': ('steer_wheels_angle_speed', 0.01),
        'd': ('steer_wheels_angle_speed', -0.01)
    }

    # MAX_SPEED_MAP = {
    #     'w': ('drive_wheels_max_speed', 1.0),
    #     's': ('drive_wheels_max_speed', -1.0),
    # }

    FORK_CONTROL_MAP = {
        keyboard.Key.up: ('fork_z_speed', 0.1),
        keyboard.Key.down: ('fork_z_speed', -0.1),
        keyboard.Key.left: ('fork_y_speed', 0.01),
        keyboard.Key.right: ('fork_y_speed', -0.01),
        keyboard.Key.home: ('fork_pitch_speed', 0.001),
        keyboard.Key.end: ('fork_pitch_speed', -0.001)
    }
    
    MODE_SWITCH_MAP = {
        '1': ControlMode.KEYBOARD_CONTROL,
        '2': ControlMode.UI_CONTROL, 
        '3': ControlMode.ADAPTIVE_CONTROL,
        '4': ControlMode.AGV_CONTROL,
        '5': ControlMode.VLA_CONTROL,
        # 'e': ControlMode.ADAPTIVE_CONTROL,  # 兼容原有快捷键
        # 'r': ControlMode.KEYBOARD_CONTROL    # 兼容原有快捷键
    }
    
    RELEASE_CONTROL_MAP = {
        # 车辆控制释放
        'w': 'drive_wheels_velocity',
        's': 'drive_wheels_velocity',
        'a': 'steer_wheels_angle_speed',
        'd': 'steer_wheels_angle_speed',
        # 货叉控制释放
        keyboard.Key.up: 'fork_z_speed',
        keyboard.Key.down: 'fork_z_speed',
        keyboard.Key.left: 'fork_y_speed',
        keyboard.Key.right: 'fork_y_speed',
        keyboard.Key.home: 'fork_pitch_speed',
        keyboard.Key.end: 'fork_pitch_speed'
    }

    def __init__(self, state_sub_name: str, cmd_pub_name: str, time_period : float = 0.05) -> None:
        if not ecal_core.is_initialized():
            ecal_core.initialize(sys.argv, "Keyboard command")

        self.stop = False
        self.pub_ok = asyncio.Event()
        self.sub_ok = asyncio.Event()
        self.data_lock = threading.Lock()  # 同步键盘时间回调线程以及数据发送协程数据

        # 控制状态初始化
        self.steer_wheels_accelerate = 0.0  #  对速度做个时间上的积分，使得速度值变化连续
        self.press_velocity_time = 0.0  #  按下速度键的系统时间
        self.release_velocity_time = 0.0 #  释放速度键的系统时间
        self.drive_wheels_velocity = 0.0
        self.drive_wheels_max_speed = 0
        self.steer_wheels_angle = 0.0
        self.steer_wheels_angle_speed = 0.0
        self.fork_z = 0.0
        self.fork_y = 0.0
        self.fork_pitch = 0.0
        self.fork_z_speed = 0.0
        self.fork_y_speed = 0.0
        self.fork_pitch_speed = 0.0
        self.record = False
        self.control_mode = ControlMode.KEYBOARD_CONTROL  # 默认键盘控制模式
        self._is_switching_mode = False  # 标记是否正在主动切换模式
        self._target_mode = None  # 目标控制模式，用于切换确认

        # 有w/s键按下时更新线速度；只有当速度键（w/s）与旋转键(a/d)同时按下时才更新舵角
        self.vehicle_control_holder = set()
        self.has_speed = False

        # 移除ServiceManager集成，服务脚本应该独立运行
        # 订阅控制模式变更
        self.mode_subscriber = ProtoSubscriber(
            "control/mode", 
            vehicle_state_msg_pb2.VehicleStateMsg
        )
        self.mode_subscriber.set_callback(self._on_control_mode_changed)

        self.time_period = time_period

        # 通信组件初始化
        self.state_sub = ProtoSubscriber(state_sub_name,
                                         vehicle_state_msg_pb2.VehicleStateMsg)
        self.cmd_pub = ProtoPublisher(cmd_pub_name,
                                      vehicle_state_msg_pb2.VehicleStateMsg)
        
        self.state_sub.set_callback(self._receiveState)

        # 键盘监听器
        listener = keyboard.Listener(on_press=self.onPress,
                                     on_release=self.onRelease)
        listener.daemon = True
        listener.start()

    def __del__(self):
        self.stop = True
        if ecal_core.is_initialized():
            ecal_core.finalize()
    
    def _on_control_mode_changed(self, topic_name, msg, time):
        """处理控制模式变更"""
        new_mode = msg.control_mode
        if new_mode != self.control_mode:
            print(f"键盘控制服务: 控制模式变更 {ControlMode.get_mode_name(self.control_mode)} -> {ControlMode.get_mode_name(new_mode)}")
            self.control_mode = new_mode
            
            # 维持原来的状态，不重置控制状态
            # 只在键盘控制模式下发送键盘命令
            print(f"控制模式已变更，当前模式: {ControlMode.get_mode_name(new_mode)}")

    def run(self) -> None:
        # asyncio.run(self._tasks())
        asyncio.run(self._pubCmd())

    def _receiveState(self, topic_name, msg, msg_time):
        """
        状态接收回调函数 - 标准的ecal回调格式
        
        Args:
            topic_name: 主题名称
            msg: 接收到的消息对象
            msg_time: 消息时间戳
        """
        try:
            # 直接更新车辆状态数据
            self.steer_wheels_angle = msg.steer_angle
            self.fork_z = msg.fork_z
            self.fork_y = msg.fork_y
            self.fork_pitch = msg.fork_pitch
            
            # 控制模式同步逻辑
            if hasattr(msg, 'control_mode'):
                if msg.control_mode != self.control_mode:
                    # 如果键盘正在主动切换模式，检查是否切换完成
                    if self._is_switching_mode:
                        # 如果收到的状态与目标模式一致，说明切换完成
                        if msg.control_mode == self._target_mode:
                            self.switch_control_mode(msg.control_mode, "切换确认")
                            self._is_switching_mode = False
                            self._target_mode = None
                    else:
                        # 键盘被动接收状态同步
                        self.switch_control_mode(msg.control_mode, "状态同步")
        except Exception as e:
            print(f"处理状态消息时出错: {e}")

    async def _pubCmd(self):
        # await self.sub_ok.wait()
        next_t = time.perf_counter()

        while not self.stop:

            with self.data_lock:
                now = time.time()

                if self.has_speed:
                    vel_time = now - self.press_velocity_time
                    self.drive_wheels_velocity += self.steer_wheels_accelerate * vel_time
                    if self.steer_wheels_accelerate>0: #对应按下w键, 速度范围从0到1
                        self.drive_wheels_velocity = min(self.drive_wheels_velocity, 0.3)
                    else: #对应按下s键, 速度范围从-1到0
                        self.drive_wheels_velocity = max(self.drive_wheels_velocity, -0.3)
                else:
                    vel_time = now - self.release_velocity_time
                    self.drive_wheels_velocity -= self.steer_wheels_accelerate * vel_time
                    if self.steer_wheels_accelerate>0: #对应释放w键, 速度不能为负值
                        self.drive_wheels_velocity = max(self.drive_wheels_velocity, 0)
                    else: #对应释放w键, 速度不能为正值
                        self.drive_wheels_velocity = min(self.drive_wheels_velocity, 0)

                self.fork_z += self.fork_z_speed
                self.fork_y += self.fork_y_speed
                self.fork_pitch += self.fork_pitch_speed
                self.steer_wheels_angle += self.steer_wheels_angle_speed

            vehicle_msg = vehicle_state_msg_pb2.VehicleStateMsg()
            vehicle_msg.drive_velocity = self.drive_wheels_velocity
            vehicle_msg.steer_angle = self.steer_wheels_angle
            vehicle_msg.fork_x = 0
            vehicle_msg.fork_y = self.fork_y_speed
            vehicle_msg.fork_z = self.fork_z_speed
            vehicle_msg.fork_pitch = self.fork_pitch_speed
            vehicle_msg.data_record = self.record
            # 注意：_pubCmd不再发送control_mode，模式切换完全由按键处理

            self.cmd_pub.send(vehicle_msg)
            # 固定周期调度，避免频率漂移
            next_t += self.time_period
            dt = next_t - time.perf_counter()
            if dt > 0:
                await asyncio.sleep(dt)
            else:
                # 掉帧时重置起点，避免一直为负
                next_t = time.perf_counter()


    def onPress(self, key):
        with self.data_lock:
            try:
                k = key.char.lower()
            except AttributeError:
                k = None

            # 智能唤起功能：数字键2
            if k == '2':
                self.handle_smart_activation()
                return

            # 模式切换快捷键处理（排除数字键2）
            if k in self.MODE_SWITCH_MAP and k != '2':
                self.send_mode_switch_command(self.MODE_SWITCH_MAP[k], ControlMode.get_mode_name(self.MODE_SWITCH_MAP[k]))
                return

            # 车辆控制指令处理
            if k in self.VEHICLE_CONTROL_MAP:
                attr_name, value = self.VEHICLE_CONTROL_MAP[k]
                setattr(self, attr_name, value)

                self.vehicle_control_holder.add(k)
                self.has_speed = bool(self.vehicle_control_holder & {'w', 's'})

                # 记录按下速度键的时间
                if self.press_velocity_time == 0:
                    self.press_velocity_time = time.time()
                    self.release_velocity_time = 0
                return

            # 货叉控制指令处理
            if key in self.FORK_CONTROL_MAP:
                attr_name, value = self.FORK_CONTROL_MAP[key]
                setattr(self, attr_name, value)
                return

            if k == 'n':
                self.record = True
            elif k == 'm':
                self.record = False
                self.control_mode = ControlMode.KEYBOARD_CONTROL  # 重置为键盘控制模式

            if key == keyboard.Key.esc:  # Resotre
                self.steer_wheels_angle = 0.0
                self.fork_z = 0.0
                self.fork_y = 0.0
                self.fork_pitch = 0.0
                self.drive_wheels_velocity = 0.0
                self.record = False
                self.control_mode = ControlMode.KEYBOARD_CONTROL  # 重置为键盘控制模式

    def send_mode_switch_command(self, new_mode: int, mode_name: str):
        """
        发送模式切换指令到vehicle_state_server
        
        Args:
            new_mode: 新的控制模式
            mode_name: 模式名称（用于显示）
            
        Returns:
            bool: 模式切换指令是否发送成功
        """
        print(f"🔄 键盘请求模式切换: {ControlMode.get_mode_name(self.control_mode)} -> {ControlMode.get_mode_name(new_mode)}")
        
        # 设置模式切换标志和目标模式
        self._is_switching_mode = True
        self._target_mode = new_mode
        
        # 创建模式切换消息
        mode_msg = vehicle_state_msg_pb2.VehicleStateMsg()
        mode_msg.drive_velocity = self.drive_wheels_velocity
        mode_msg.steer_angle = self.steer_wheels_angle
        mode_msg.fork_x = 0
        mode_msg.fork_y = self.fork_y_speed
        mode_msg.fork_z = self.fork_z_speed
        mode_msg.fork_pitch = self.fork_pitch_speed
        mode_msg.data_record = self.record
        mode_msg.control_mode = new_mode
        
        # 发送模式切换指令
        try:
            self.cmd_pub.send(mode_msg)
            print(f"📤 已发送模式切换指令到vehicle_state_server")
            # 注意：实际模式切换结果由vehicle_state_server同步回来
            # 这里不立即更新self.control_mode，而是通过状态同步更新
            return True
        except Exception as e:
            print(f"❌ 发送模式切换指令失败: {e}")
            # 发送失败时清除标志
            self._is_switching_mode = False
            self._target_mode = None
            return False
        

    
    def switch_control_mode(self, new_mode: int, mode_name: str):
        """
        内部模式切换方法 - 仅用于状态同步，不发送指令
        
        Args:
            new_mode: 新的控制模式
            mode_name: 模式名称（用于显示）
        """
        print(f"🔄 键盘同步控制模式: {ControlMode.get_mode_name(self.control_mode)} -> {ControlMode.get_mode_name(new_mode)}")
        
        # 更新控制模式
        self.control_mode = new_mode
        
        print(f"✅ 键盘已同步到{ControlMode.get_mode_name(new_mode)}模式")
        print("=" * 50)


    def handle_smart_activation(self):
        """智能唤起功能：按2键智能处理UI状态"""
        print("🔍 按2键智能唤起UI...")
        
        # 移除ServiceManager依赖，智能唤起功能暂时简化
        # 直接发送UI控制模式切换命令
        print("🎯 发送UI控制模式切换命令")
        success = self.send_mode_switch_command(ControlMode.UI_CONTROL, ControlMode.get_mode_name(ControlMode.UI_CONTROL))
        
        if success:
            print("✅ UI智能唤起成功")
        else:
            print("❌ UI智能唤起失败")

    def onRelease(self, key):
        with self.data_lock:
            try:
                k = key.char.lower()
            except AttributeError:
                k = None

            # 处理按键释放
            if k in self.RELEASE_CONTROL_MAP:
                if self.RELEASE_CONTROL_MAP[k] != 'drive_wheels_velocity':
                    setattr(self, self.RELEASE_CONTROL_MAP[k], 0.0)  # 只释放线速度以外的值

                # 特殊处理速度键和方向键
                if k in set(self.VEHICLE_CONTROL_MAP.keys()):
                    self.vehicle_control_holder.discard(k)
                    if k in {"w", "s"}:
                        self.has_speed = False
                        if self.release_velocity_time==0:
                            self.release_velocity_time = time.time()
                            self.press_velocity_time = 0
                    elif k in {"a", "d"}:
                        self.steer_wheels_angle_speed = 0


if __name__ == "__main__":
    ecal_pub = KeyBoardPublisher("vehicle/status", "keyboard/cmd")
    # ecal_pub = KeyBoardPublisher("vehicl__state", "keyboard/cmd")
    ecal_pub.run()
