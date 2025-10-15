import subprocess
import os
import signal
import time
import threading
import argparse  # 添加argparse模块
from pathlib import Path
from dataclasses import dataclass
from typing import List, Callable, Optional


@dataclass
class ScriptConfig:
    """脚本配置信息"""
    name: str  # 脚本名称
    path: str  # 脚本路径
    args: List[str] = None  # 命令行参数
    env_vars: dict = None  # 环境变量
    startup_delay: float = 0  # 启动延迟（秒）
    pre_start: Optional[Callable] = None  # 启动前回调
    post_start: Optional[Callable] = None  # 启动后回调


class ModularLauncher:
    def __init__(self):
        self.processes = {}  # 存储进程: {name: (process, thread)}
        self.running = False
        self.base_env = self._setup_base_environment()

    def _setup_base_environment(self):
        """设置基础环境变量"""
        env = os.environ.copy()

        # GPU优化设置
        env["__NV_PRIME_RENDER_OFFLOAD"] = "1"
        env["__GLX_VENDOR_LIBRARY_NAME"] = "nvidia"
        env["MESA_GL_VERSION_OVERRIDE"] = "4.6"

        # 输出缓冲设置
        env["PYTHONUNBUFFERED"] = "1"

        # eCAL核心配置 - 确保进程间可见
        env["ECAL_DISCOVERY_IP"] = "127.0.0.1"  # 本地通讯使用回环地址
        env["ECAL_SHM_ENABLED"] = "1"  # 启用共享内存
        env["ECAL_NET_ENABLED"] = "0"  # 禁用网络通讯（本地进程间不需要）
        env["ECAL_LOG_LEVEL"] = "3"  # 只显示警告和错误

        # 强制设置Python编码
        env["PYTHONUTF8"] = "1"
        env["LC_ALL"] = "en_US.UTF-8"
        env["LANG"] = "en_US.UTF-8"
        env["PYTHONIOENCODING"] = "utf-8"

        # GPU优化设置
        env["__NV_PRIME_RENDER_OFFLOAD"] = "1"
        env["__GLX_VENDOR_LIBRARY_NAME"] = "nvidia"

        # 输出缓冲设置
        env["PYTHONUNBUFFERED"] = "1"

        return env

    def _get_environment(self, custom_env: dict = None):
        """合并基础环境变量和自定义环境变量"""
        env = self.base_env.copy()
        if custom_env:
            env.update(custom_env)
        return env

    def _read_process_output(self, process: subprocess.Popen, name: str):
        """非阻塞读取进程输出"""
        while self.running and process.poll() is None:
            try:
                output = process.stdout.readline()
                if output:
                    print(f"[{name}] {output.strip()}")
                else:
                    time.sleep(0.1)
            except Exception as e:
                print(f"[{name}] 输出读取错误: {e}")
                break

    def _start_script(self, config: ScriptConfig):
        """启动单个脚本"""
        try:
            # 检查脚本是否存在
            script_path = Path(config.path).resolve()
            if not script_path.exists():
                print(f"[{config.name}] 脚本不存在: {script_path}")
                return False

            # 执行启动前回调
            if config.pre_start:
                config.pre_start()

            # 构建命令
            cmd = ["python", str(script_path)]
            if config.args:
                cmd.extend(config.args)

            # 准备环境变量
            env = self._get_environment(config.env_vars)

            print(f"[{config.name}] 启动命令: {' '.join(cmd)}")

            # 启动进程
            process = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )

            # 启动输出处理线程
            output_thread = threading.Thread(
                target=self._read_process_output,
                args=(process, config.name),
                daemon=True
            )
            output_thread.start()

            # 存储进程信息
            self.processes[config.name] = (process, output_thread)

            # 执行启动后回调
            if config.post_start:
                config.post_start()

            return True

        except Exception as e:
            print(f"[{config.name}] 启动失败: {e}")
            return False

    def add_script(self, config: ScriptConfig):
        """添加脚本配置（在启动前调用）"""
        if not hasattr(self, 'scripts'):
            self.scripts = []
        self.scripts.append(config)

    def start(self):
        """启动所有脚本"""
        if not hasattr(self, 'scripts') or not self.scripts:
            print("没有配置任何脚本")
            return False

        self.running = True

        # 按顺序启动所有脚本
        for config in self.scripts:
            # 应用启动延迟
            if config.startup_delay > 0:
                print(f"等待 {config.startup_delay} 秒后启动 {config.name}...")
                time.sleep(config.startup_delay)

            # 启动脚本
            if not self._start_script(config):
                print(f"启动 {config.name} 失败，停止所有进程")
                self.stop()
                return False

        print("所有脚本启动完成！按Ctrl+C退出")
        return True

    def stop_script(self, name: str):
        """停止单个脚本"""
        if name not in self.processes:
            return

        process, thread = self.processes[name]

        if process and process.poll() is None:
            print(f"[{name}] 正在停止...")
            process.terminate()

            try:
                # 等待优雅关闭
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                print(f"[{name}] 强制关闭...")
                process.kill()

        # 移除进程信息
        del self.processes[name]

    def stop(self):
        """停止所有脚本"""
        self.running = False
        print("\n开始关闭所有脚本...")

        # 按逆序停止进程
        for config in reversed(getattr(self, 'scripts', [])):
            self.stop_script(config.name)

        print("所有脚本已关闭")


def main():
    """
    模块化启动器主入口
    
    新架构启动顺序：
    1. 统一Serial Service - 处理所有通信协调
    2. 仿真执行器 - Isaac Sim仿真环境
    3. UI客户端 - 车辆控制界面
    4. 键盘控制 - 可选的键盘输入
    5. 调节控制 - 可选的参数调节
    6. AGV控制 - 可选的硬件连接
    """
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Isaac Sim模块化启动器')
    parser.add_argument('--modules', '-m', nargs='+', choices=[
        'all', 'serial', 'simulation', 'ui', 'keyboard', 'adjust', 'agv'
    ], default=['all'], help='选择要启动的模块: all(全部), serial(串口服务), simulation(仿真), ui(界面), keyboard(键盘控制), adjust(调节控制), agv(AGV控制)')
    parser.add_argument('--config', '-c', default='../configs/st_test.yaml', help='配置文件路径')
    
    args = parser.parse_args()
    
    # 创建启动器实例
    launcher = ModularLauncher()
    current_dir = Path(__file__).parent
    print("current_dir is", current_dir)
    
    # 配置并添加脚本（根据选择的模块）
    scripts_to_add = []

    # 1. Isaac Sim仿真执行器配置
    if 'all' in args.modules or 'simulation' in args.modules:
        def isaac_pre_start():
            print("准备启动Isaac Sim仿真执行器，确保GPU驱动正常...")

        scripts_to_add.append(ScriptConfig(
            name="SimulationExecutor",
            path=str(current_dir / "simulation_excutor.py"),
            args=["--config", args.config],
            startup_delay=0,  # 等待Serial Service启动
            pre_start=isaac_pre_start,
            post_start=lambda: print("仿真执行器启动完成，等待初始化...")
        ))

    # 2. UI客户端配置
    if 'all' in args.modules or 'ui' in args.modules:
        def ui_pre_start():
            print("准备启动UI客户端...")

        scripts_to_add.append(ScriptConfig(
            name="UIClient",
            path=str(current_dir / "ui_client.py"),
            args=["--config", args.config],  # 添加配置文件参数
            startup_delay=8,  # 等待服务启动
            pre_start=ui_pre_start,
            post_start=lambda: print("UI客户端启动完成，可以开始控制车辆...")
        ))

    # 3. 键盘控制脚本配置（可选）
    if 'all' in args.modules or 'keyboard' in args.modules:
        scripts_to_add.append(ScriptConfig(
            name="KeyboardControl",
            path=str(current_dir / "keyboard_ctrl.py"),
            args=["--config", args.config],
            startup_delay=1,  # 在仿真器启动后
            post_start=lambda: print("键盘控制脚本启动完成，开始监听输入...")
        ))

    # 4. adjust control（可选）
    if 'all' in args.modules or 'adjust' in args.modules:
        scripts_to_add.append(ScriptConfig(
            name="AdjustControl",
            path=str(current_dir / "adjust_ctrl.py"),
            args=["--config", args.config],
            env_vars={"COMM_MODE": "zmq"},
            startup_delay=1,
            post_start=lambda: print("adjust_ctrl脚本启动完成，开始监听输入...")
        ))

    # 5. AGV control（可选）
    if 'all' in args.modules or 'agv' in args.modules:
        scripts_to_add.append(ScriptConfig(
            name="AGVControl",
            path=str(current_dir / "agv_server.py"),
            args=["--config", args.config],
            env_vars={},
            startup_delay=1,
            post_start=lambda: print("AGV控制脚本启动完成，开始监听输入...")
        ))

    # 添加所有选择的脚本
    for script in scripts_to_add:
        launcher.add_script(script)
    
    # 显示启动信息
    print(f"启动配置: 模块={args.modules}, 配置文件={args.config}")
    print("=" * 50)

    # 注册信号处理
    def handle_signal(signum, frame):
        launcher.stop()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    # 启动所有脚本
    try:
        if launcher.start():
            # 保持主进程运行
            while launcher.running:
                time.sleep(1)
    except Exception as e:
        print(f"启动器错误: {e}")
        launcher.stop()


if __name__ == "__main__":
    main()
