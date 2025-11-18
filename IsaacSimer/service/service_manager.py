#!/usr/bin/env python3
"""
Service Manager - 最小可行服务管理中心（单例模式）

采用"单管理进程 + 多业务子进程"模式：
- Launcher进程：初始化唯一的ServiceManager单例，负责服务生命周期管理
- 业务子进程：仅处理业务逻辑，通过环境变量获取控制信息
- 避免多进程共享变量，使用简单IPC通信
"""

import os
import sys
import time
import threading
import subprocess
from typing import Dict, Any, Optional, List
import signal
import platform
import ecal.core.core as ecal_core
from ecal.core.subscriber import ProtoSubscriber
from protos import vehicle_state_msg_pb2

# 控制模式枚举
class ControlMode:
    """控制模式枚举"""
    KEYBOARD_CONTROL = 1
    UI_CONTROL = 2
    ADAPTIVE_CONTROL = 3
    AGV_CONTROL = 4
    VLA_CONTROL = 5

    @staticmethod
    def get_mode_name(mode: int) -> str:
        """获取控制模式的字符串名称"""
        mode_names = {
            1: "键盘控制",
            2: "UI控制", 
            3: "自适应控制",
            4: "AGV控制",
            5: "VLA控制"
        }
        return mode_names.get(mode, f"未知模式({mode})")

# 服务配置类
class ServiceConfig:
    """服务配置信息"""
    def __init__(self, name: str, script_path: str, control_mode: int = 0,
                 env_vars: Dict[str, str] = None, start_delay: float = 0.0, args: List[str] = None):
        self.name = name
        self.script_path = script_path
        self.control_mode = control_mode  # 0表示不受控制模式影响
        self.env_vars = env_vars or {}
        self.start_delay = start_delay
        self.args = args or []  # 添加args参数

class ServiceManager:
    """最小可行服务管理中心 - 单例模式实现"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """单例模式实现"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(ServiceManager, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """初始化单例（确保只初始化一次）"""
        if self._initialized:
            return
            
        # 基本属性
        self._node_name = "ServiceManager"
        
        # 服务管理 - 使用普通字典，避免多进程共享问题
        self._services: Dict[str, ServiceConfig] = {}
        self._service_states: Dict[str, Dict[str, Any]] = {}
        self._processes: Dict[str, subprocess.Popen] = {}
        
        # 控制状态
        self._current_control_mode = ControlMode.KEYBOARD_CONTROL
        self._last_mode_switch_time = 0
        self._mode_switch_cooldown = 1.0  # 模式切换冷却时间(秒)
        
        # 监控线程
        self._monitor_thread = None
        self._monitor_running = False
        
        # 线程锁 - 使用可重入锁避免嵌套锁死锁问题
        self._instance_lock = threading.RLock()
        
        # 初始化默认服务
        self._initialize_default_services()
        
        # 初始化ECAL订阅器，监听control mode topic
        self._init_control_mode_subscriber()
        
        self._initialized = True
        print(f"[{self._node_name}] 单例服务管理器初始化完成")
        
    def _initialize_default_services(self):
        """初始化默认服务配置"""
        # 键盘控制服务
        self.add_service("keyboard_control", "service/keyboard_publisher.py",
                        ControlMode.KEYBOARD_CONTROL)
        
        # UI控制服务 - 使用ecal_ui_service.py
        self.add_service("ui_control", "service/ecal_ui_service.py", 
                        ControlMode.UI_CONTROL)
        
        # 自适应控制服务 - 使用adjust_control_publisher.py
        self.add_service("adaptive_control", "service/adjust_control_publisher.py", 
                        ControlMode.ADAPTIVE_CONTROL)
        
        # AGV控制服务 - 使用serial_service.py
        self.add_service("agv_control", "service/serial_service.py", 
                        ControlMode.AGV_CONTROL)
        
        # 仿真执行器服务 - 使用apps/simulation_excutor.py（不受控制模式影响）
        # self.add_service("simulation_executor", "apps/simulation_excutor.py",
        #                 5)
    
    def add_service(self, name: str, script_path: str, control_mode: int = 0,
                   env_vars: Dict[str, str] = None, start_delay: float = 0.0, args: List[str] = None):
        """添加服务配置"""
        with self._instance_lock:
            if name in self._services:
                print(f"[{self._node_name}] 警告: 服务 '{name}' 已存在，将被覆盖")
            
            # 创建服务配置
            service_config = ServiceConfig(name, script_path, control_mode, env_vars, start_delay, args)
            self._services[name] = service_config
            
            # 初始化服务状态
            self._service_states[name] = {
                "name": name,
                "pid": -1,  # 当前进程PID，-1表示未运行
                "start_time": 0.0,  # 启动时间戳
                "running": False  # 运行状态
            }
            
            print(f"[{self._node_name}] 添加服务: {name} -> {script_path}")
    
    def start_service(self, service_name: str) -> bool:
        """启动指定服务"""
        self.print_service_statuses()
        with self._instance_lock:
            if service_name not in self._services:
                print(f"[{self._node_name}] 错误: 服务 '{service_name}' 不存在")
                return False
            
            state = self._service_states[service_name]
            if state["running"]:
                print(f"[{self._node_name}] 服务 '{service_name}' 已在运行")
                return True
            
            # 获取服务配置
            service_config = self._services[service_name]
            
            # 设置环境变量
            env = os.environ.copy()
            env.update(service_config.env_vars)
            env['SERVICE_NAME'] = service_name
            env['CONTROL_MODE'] = str(self._current_control_mode)
            env['IS_LAUNCHER_PROCESS'] = 'false'  # 标记为子进程
            
            try:
                apps_dir    = os.path.dirname(os.path.abspath(service_config.script_path))
                module_name = os.path.splitext(os.path.basename(service_config.script_path))[0]
                cmd = [
                    sys.executable, '-c',
                    f'import sys,os;sys.path.insert(0,r"{apps_dir}");import {module_name};{module_name}.main()'
                ]
                if service_config.args:
                    cmd.extend(service_config.args)
                
                # 启动服务进程 - 使用subprocess.DEVNULL避免管道阻塞
                process = subprocess.Popen(
                    cmd,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    bufsize=1,
                    text=True,
                    universal_newlines=True
                )
                
                # 记录进程信息
                self._processes[service_name] = process
                state["pid"] = process.pid
                state["start_time"] = time.time()
                state["running"] = True
                
                # 启动输出监控线程
                threading.Thread(
                    target=self._monitor_service_output,
                    args=(service_name, process),
                    daemon=True
                ).start()
                
                print(f"[{self._node_name}] 启动服务: {service_name} (PID: {process.pid})")
                
                # 延迟启动
                if service_config.start_delay > 0:
                    time.sleep(service_config.start_delay)
                
                self.print_service_statuses()
                return True
                
            except Exception as e:
                print(f"[{self._node_name}] 启动服务 '{service_name}' 失败: {e}")
                return False
        

    def stop_service(self, service_name: str) -> bool:
        """停止指定服务"""
        with self._instance_lock:
            if service_name not in self._service_states:
                print(f"[{self._node_name}] 错误: 服务 '{service_name}' 不存在")
                return False
            
            state = self._service_states[service_name]
            if not state["running"]:
                print(f"[{self._node_name}] 服务 '{service_name}' 未在运行")
                return True
            
            try:
                process = self._processes.get(service_name)
                if not process:
                    print(f"[{self._node_name}] 服务 '{service_name}' 进程未找到")
                    return False
                
                # 尝试优雅关闭
                if platform.system() == "Windows":
                    process.terminate()
                else:
                    os.kill(state["pid"], signal.SIGTERM)
                
                # 等待进程结束，最多等待3秒
                wait_time = 0
                while wait_time < 3 and process.poll() is None:
                    time.sleep(0.1)
                    wait_time += 0.1
                
                # 如果进程仍在运行，强制终止
                if process.poll() is None:
                    if platform.system() == "Windows":
                        process.kill()
                    else:
                        os.kill(state["pid"], signal.SIGKILL)
                
                # 更新服务状态
                state["pid"] = -1
                state["start_time"] = 0.0
                state["running"] = False
                
                # 从进程字典中移除
                if service_name in self._processes:
                    del self._processes[service_name]
                
                print(f"[{self._node_name}] 停止服务: {service_name}")
                return True
                
            except Exception as e:
                print(f"[{self._node_name}] 停止服务 '{service_name}' 失败: {e}")
                return False
    
    def start_services(self, service_names: List[str] = None) -> bool:
        """启动指定服务列表，如果service_names为None则启动所有服务"""
        success = True
        
        with self._instance_lock:
            # 如果没有指定服务列表，则启动所有服务
            if service_names is None:
                service_names = list(self._services.keys())
                
                # 启动所有服务时，忽略控制模式限制，启动所有服务
                for service_name in service_names:
                    if service_name not in self._services:
                        print(f"[{self._node_name}] 错误: 服务 '{service_name}' 不存在")
                        success = False
                        continue
                    
                    # 检查服务是否已在运行
                    state = self._service_states[service_name]
                    if state["running"]:
                        print(f"[{self._node_name}] 服务 '{service_name}' 已在运行")
                        continue
                    
                    # 获取服务配置
                    service_config = self._services[service_name]
                    
                    # 应用启动延迟
                    if service_config.start_delay > 0:
                        time.sleep(service_config.start_delay)
                    
                    # 启动服务
                    if not self.start_service(service_name):
                        success = False
            else:
                # 如果指定了服务列表，直接启动这些服务，不进行控制模式过滤
                for service_name in service_names:
                    if service_name not in self._services:
                        print(f"[{self._node_name}] 错误: 服务 '{service_name}' 不存在")
                        success = False
                        continue
                    
                    # 检查服务是否已在运行
                    state = self._service_states[service_name]
                    if state["running"]:
                        print(f"[{self._node_name}] 服务 '{service_name}' 已在运行")
                        continue
                    
                    # 获取服务配置
                    service_config = self._services[service_name]
                    
                    # 应用启动延迟
                    if service_config.start_delay > 0:
                        time.sleep(service_config.start_delay)
                    
                    # 启动服务
                    if not self.start_service(service_name):
                        success = False
        
        return success
    
    def start_all_services(self) -> bool:
        """启动所有服务"""
        return self.start_services()
    
    def stop_all_services(self) -> bool:
        """停止所有服务"""
        success = True
        
        with self._instance_lock:
            service_names = list(self._services.keys())
            
            for service_name in reversed(service_names):
                if not self.stop_service(service_name):
                    success = False
        
        return success
    
    def set_control_mode(self, mode: int) -> bool:
        """设置控制模式"""
        # 检查模式是否有效
        valid_modes = [ControlMode.KEYBOARD_CONTROL, ControlMode.UI_CONTROL, 
                      ControlMode.ADAPTIVE_CONTROL, ControlMode.AGV_CONTROL]
        
        if mode not in valid_modes:
            print(f"[{self._node_name}] 错误: 无效的控制模式: {mode}")
            return False
        
        current_time = time.time()
        
        # 检查模式切换冷却时间
        if current_time - self._last_mode_switch_time < self._mode_switch_cooldown:
            remaining = self._mode_switch_cooldown - (current_time - self._last_mode_switch_time)
            print(f"[{self._node_name}] 模式切换冷却中，请等待 {remaining:.1f}秒")
            return False
        
        with self._instance_lock:
            if mode == self._current_control_mode:
                print(f"[{self._node_name}] 控制模式已经是 {ControlMode.get_mode_name(mode)}")
                return True
            
            old_mode = self._current_control_mode
            self._current_control_mode = mode
            self._last_mode_switch_time = current_time
            
            print(f"[{self._node_name}] 控制模式从 {ControlMode.get_mode_name(old_mode)} 切换到 {ControlMode.get_mode_name(mode)}")
            
            # 停止不需要的服务
            for service_name in self._services.keys():
                service_config = self._services[service_name]
                state = self._service_states[service_name]
                
                # 检查服务是否应该启动
                should_start = self._should_start_service(service_name)
                
                if not should_start and state["running"]:
                    self.stop_service(service_name)
            
            # 启动需要的服务
            for service_name in self._services.keys():
                service_config = self._services[service_name]
                state = self._service_states[service_name]
                
                # 检查服务是否应该启动
                should_start = self._should_start_service(service_name)
                
                if should_start and not state["running"]:
                    self.start_service(service_name)
        
        return True
    
    def _should_start_service(self, service_name: str) -> bool:
        """判断服务是否应该启动"""
        if service_name not in self._services:
            return False
        
        service_config = self._services[service_name]
        
        # 如果服务没有控制模式关联，总是启动
        if service_config.control_mode == 0:
            return True
        
        # 如果服务有控制模式关联，检查是否与当前控制模式匹配
        return service_config.control_mode == self._current_control_mode
    
    def get_service_status(self, service_name: str) -> Dict[str, Any]:
        """获取服务状态"""
        with self._instance_lock:
            if service_name in self._service_states:
                return self._service_states[service_name].copy()
            return {"error": "Service not found"}
    
    def get_all_service_statuses(self) -> Dict[str, Dict[str, Any]]:
        """获取所有服务状态"""
        with self._instance_lock:
            return {name: state.copy() for name, state in self._service_states.items()}
    
    def print_service_statuses(self):
        """打印所有服务状态"""
        statuses = self.get_all_service_statuses()
        print(f"\n[{self._node_name}] === 服务状态 ===")
        for service_name, status in statuses.items():
            running_status = "运行中" if status["running"] else "已停止"
            pid_info = f"(PID: {status['pid']})" if status["running"] else ""
            print(f"  {service_name}: {running_status} {pid_info}")
        print("=======================\n")
    
    def start_monitoring(self):
        """启动服务监控"""
        with self._instance_lock:
            if self._monitor_running:
                print(f"[{self._node_name}] 服务监控已在运行")
                return
            
            self._monitor_running = True
            self._monitor_thread = threading.Thread(
                target=self._monitor_loop,
                daemon=True
            )
            self._monitor_thread.start()
            print(f"[{self._node_name}] 启动服务监控")
    
    def stop_monitoring(self):
        """停止服务监控"""
        with self._instance_lock:
            if not self._monitor_running:
                print(f"[{self._node_name}] 服务监控未在运行")
                return
            
            self._monitor_running = False
            if self._monitor_thread:
                self._monitor_thread.join(timeout=2.0)
                self._monitor_thread = None
            print(f"[{self._node_name}] 停止服务监控")
    
    def _monitor_loop(self):
        """服务监控主循环"""
        while self._monitor_running:
            try:
                # 检查服务状态
                self._check_services()
                
                # 等待心跳间隔
                time.sleep(1.0)
                
            except Exception as e:
                print(f"[{self._node_name}] 监控循环异常: {e}")
    
    def _check_services(self):
        """检查所有服务的状态 - 只监控进程是否意外结束"""
        with self._instance_lock:
            for service_name in self._services:
                state = self._service_states[service_name]
                
                # 如果服务在运行，检查进程是否意外结束
                if state["running"]:
                    process = self._processes.get(service_name)
                    if process and process.poll() is not None:
                        # 进程已结束
                        state["pid"] = -1
                        state["start_time"] = 0.0
                        state["running"] = False
                        print(f"[{self._node_name}] 服务 '{service_name}' 意外结束")
                        del self._processes[service_name]
    
    def _monitor_service_output(self, service_name: str, process: subprocess.Popen):
        """监控服务输出，解析状态切换命令"""
        try:
            # 读取标准输出（先检查stdout是否为None）
            if process.stdout is not None:
                for line in process.stdout:
                    if line.strip():
                        print(f"[{service_name}] stdout: {line.strip()}")
                        # 检查是否包含状态切换命令
                        self._parse_service_command(service_name, line.strip())
            
            # 读取标准错误（先检查stderr是否为None）
            if process.stderr is not None:
                for line in process.stderr:
                    if line.strip():
                        print(f"[{service_name}] stderr: {line.strip()}")
                        # 检查错误输出中是否包含状态切换命令
                        self._parse_service_command(service_name, line.strip())
                    
        except Exception as e:
            print(f"[{self._node_name}] 监控服务 '{service_name}' 输出异常: {e}")
            
        # 检查进程是否已结束
        exit_code = process.poll()
        if exit_code is not None:
            print(f"[{self._node_name}] 服务 '{service_name}' 进程结束，退出码: {exit_code}")
            # 服务意外结束时，尝试重新启动符合当前控制模式的服务
            self._check_and_restart_services()

    def _parse_service_command(self, service_name: str, message: str):
        """解析服务发送的命令消息"""
        # 检查是否包含模式切换命令格式: [SWITCH_MODE: X]
        if "[SWITCH_MODE:" in message:
            try:
                # 提取模式数字
                mode_str = message.split("[SWITCH_MODE:")[1].split("]")[0].strip()
                mode = int(mode_str)
                print(f"[{self._node_name}] 收到来自 '{service_name}' 的模式切换命令: {mode}")
                # 执行模式切换
                self.set_control_mode(mode)
            except (ValueError, IndexError):
                print(f"[{self._node_name}] 解析来自 '{service_name}' 的模式切换命令失败: {message}")
        
        # 检查是否包含服务状态查询命令: [QUERY_STATUS]
        elif "[QUERY_STATUS]" in message:
            print(f"[{self._node_name}] 收到来自 '{service_name}' 的状态查询命令")
            self.print_service_statuses()
        
        # 检查是否包含特定服务启动命令: [START_SERVICE: service_name]
        elif "[START_SERVICE:" in message:
            try:
                target_service = message.split("[START_SERVICE:")[1].split("]")[0].strip()
                print(f"[{self._node_name}] 收到来自 '{service_name}' 的启动服务命令: {target_service}")
                self.start_service(target_service)
            except (IndexError):
                print(f"[{self._node_name}] 解析来自 '{service_name}' 的启动服务命令失败: {message}")
        
        # 检查是否包含特定服务停止命令: [STOP_SERVICE: service_name]
        elif "[STOP_SERVICE:" in message:
            try:
                target_service = message.split("[STOP_SERVICE:")[1].split("]")[0].strip()
                print(f"[{self._node_name}] 收到来自 '{service_name}' 的停止服务命令: {target_service}")
                self.stop_service(target_service)
            except (IndexError):
                print(f"[{self._node_name}] 解析来自 '{service_name}' 的停止服务命令失败: {message}")

    def _check_and_restart_services(self):
        """检查并重新启动应该运行的服务"""
        with self._instance_lock:
            for service_name in self._services:
                # 如果服务应该在当前控制模式下运行但未运行，则重新启动
                if self._should_start_service(service_name):
                    state = self._service_states[service_name]
                    if not state["running"]:
                        print(f"[{self._node_name}] 检测到服务 '{service_name}' 应该运行但未运行，正在重新启动...")
                        self.start_service(service_name)

    def _init_control_mode_subscriber(self):
        """初始化控制模式主题订阅器"""
        try:
            # 初始化ECAL核心
            if not ecal_core.initialize(sys.argv, "ServiceManager"):
                print(f"[{self._node_name}] 警告: 无法初始化ECAL核心，将使用默认控制模式")
                return
            
            # 订阅控制模式主题
            self.control_mode_sub = ProtoSubscriber("vehicle_state", vehicle_state_msg_pb2.VehicleStateMsg)
            self.control_mode_sub.set_callback(self._control_mode_callback)
            print(f"[{self._node_name}] 已订阅控制模式主题: vehicle_state")
        except Exception as e:
            print(f"[{self._node_name}] 初始化控制模式订阅器失败: {e}")
    
    def _control_mode_callback(self, topic_name, msg, msg_time):
        """处理控制模式消息的回调函数"""
        try:
            # 检查消息中是否包含控制模式
            if hasattr(msg, 'control_mode') and msg.control_mode > 0:
                new_mode = msg.control_mode
                if new_mode != self._current_control_mode:
                    print(f"[{self._node_name}] 收到控制模式切换指令: {ControlMode.get_mode_name(self._current_control_mode)} -> {ControlMode.get_mode_name(new_mode)}")
                    # 执行模式切换
                    result = self.set_control_mode(new_mode)
                    if result:
                        # 模式切换成功后，检查并重启相关服务
                        self._check_and_restart_services()
        except Exception as e:
            print(f"[{self._node_name}] 处理控制模式消息失败: {e}")
            

# 全局获取单例实例的函数
def get_service_manager() -> ServiceManager:
    """获取ServiceManager单例实例"""
    return ServiceManager()

# 测试代码
if __name__ == "__main__":
    # 测试单例模式
    manager1 = get_service_manager()
    manager2 = get_service_manager()
    
    print(f"单例测试: manager1 is manager2 = {manager1 is manager2}")
    
    # 启动所有服务
    manager1.start_all_services()
    
    # 启动监控
    manager1.start_monitoring()
    
    # 打印服务状态
    manager1.print_service_statuses()
    
    try:
        # 保持主线程运行
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n正在关闭服务...")
        manager1.stop_all_services()
        manager1.stop_monitoring()
        print("服务管理器已停止。")