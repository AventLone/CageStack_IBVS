#!/usr/bin/env python3
"""
基础服务类 - 子进程只负责业务逻辑，不创建服务管理实例
"""

import os
import sys
import time
import signal
import threading
from typing import Optional, Dict, Any

class BaseService:
    """基础服务类 - 所有子进程服务的基类"""
    
    def __init__(self, service_name: str):
        self.service_name = service_name
        self.is_running = False
        self.shutdown_event = threading.Event()
        
        # 从环境变量获取服务信息
        self.control_mode = int(os.environ.get("CONTROL_MODE", "0"))
        
        # 设置信号处理
        self._setup_signal_handlers()
        
    def _setup_signal_handlers(self):
        """设置信号处理器"""
        def signal_handler(signum, frame):
            print(f"[{self.service_name}] 收到信号 {signum}，准备退出...")
            self.stop()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def initialize(self) -> bool:
        """初始化服务"""
        try:
            print(f"[{self.service_name}] 初始化服务...")
            self.is_running = True
            return True
        except Exception as e:
            print(f"[{self.service_name}] 初始化失败: {e}")
            return False
    
    def run(self) -> None:
        """运行服务主循环"""
        if not self.initialize():
            return
            
        print(f"[{self.service_name}] 服务启动，控制模式: {self.control_mode}")
        
        try:
            while self.is_running and not self.shutdown_event.is_set():
                # 服务主循环逻辑
                self._main_loop()
                time.sleep(0.1)  # 避免CPU占用过高
                
        except KeyboardInterrupt:
            print(f"[{self.service_name}] 收到键盘中断信号")
        except Exception as e:
            print(f"[{self.service_name}] 服务运行异常: {e}")
        finally:
            self.cleanup()
    
    def _main_loop(self):
        """服务主循环 - 子类需要重写此方法"""
        # 基础心跳机制
        if int(time.time()) % 10 == 0:  # 每10秒打印一次心跳
            print(f"[{self.service_name}] 服务运行中...")
    
    def stop(self) -> None:
        """停止服务"""
        print(f"[{self.service_name}] 停止服务...")
        self.is_running = False
        self.shutdown_event.set()
    
    def cleanup(self) -> None:
        """清理资源"""
        print(f"[{self.service_name}] 清理资源...")
        self.is_running = False
    
    def get_status(self) -> Dict[str, Any]:
        """获取服务状态"""
        return {
            "service_name": self.service_name,
            "is_running": self.is_running,
            "control_mode": self.control_mode,
            "pid": os.getpid()
        }


def run_service(service_class, service_name: str):
    """运行服务的便捷函数"""
    service = service_class(service_name)
    service.run()


# 示例服务类
class ExampleService(BaseService):
    """示例服务类"""
    
    def __init__(self, service_name: str):
        super().__init__(service_name)
        self.counter = 0
    
    def _main_loop(self):
        """示例主循环"""
        self.counter += 1
        if self.counter % 20 == 0:  # 每2秒打印一次
            print(f"[{self.service_name}] 计数器: {self.counter}")


if __name__ == "__main__":
    # 测试示例服务
    service = ExampleService("ExampleService")
    service.run()