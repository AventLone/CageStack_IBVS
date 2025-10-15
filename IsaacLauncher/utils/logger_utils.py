import logging
import os
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from typing import Optional, Dict, Any


class LoggerUtil:
    """
    日志工具类，支持以下功能：
    - 控制台和文件双重输出
    - 日志按级别分离（普通日志/错误日志）
    - 日志文件轮转（按大小或时间）
    - 异常堆栈追踪记录
    - 支持自定义日志格式和级别
    """
    _loggers: Dict[str, logging.Logger] = {}  # 缓存已创建的logger实例

    @classmethod
    def get_logger(
            cls,
            name: str = "root",
            log_dir: str = "logs",
            level: int = logging.INFO,
            rotate_by_size: bool = True,
            max_bytes: int = 10 * 1024 * 1024,  # 10MB
            backup_count: int = 5,
            when: str = "midnight",  # 按时间轮转的单位
            interval: int = 1,
            fmt: Optional[str] = None,
            datefmt: str = "%Y-%m-%d %H:%M:%S"
    ) -> logging.Logger:
        """
        获取配置好的logger实例

        :param name: logger名称，建议使用模块名
        :param log_dir: 日志存储目录
        :param level: 日志级别
        :param rotate_by_size: 是否按大小轮转日志
        :param max_bytes: 单文件最大大小（仅按大小轮转时有效）
        :param backup_count: 备份文件数量
        :param when: 时间轮转单位（S/秒, M/分, H/时, D/天, midnight/午夜）
        :param interval: 轮转时间间隔（仅按时间轮转时有效）
        :param fmt: 日志格式
        :param datefmt: 日期格式
        :return: 配置好的logger
        """
        # 若已存在则直接返回
        if name in cls._loggers:
            return cls._loggers[name]

        # 创建日志目录
        os.makedirs(log_dir, exist_ok=True)

        # 创建logger
        logger = logging.getLogger(name)
        logger.setLevel(level)
        logger.propagate = False  # 防止日志向上传播到root logger

        # 日志格式
        if fmt is None:
            fmt = (
                "%(asctime)s - %(name)s - %(levelname)s - "
                "%(filename)s:%(lineno)d - %(process)d:%(thread)d - %(message)s"
            )
        formatter = logging.Formatter(fmt, datefmt=datefmt)

        # 1. 控制台处理器（INFO级别及以上）
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # 2. 普通日志文件处理器
        log_file = os.path.join(log_dir, f"{name}.log")
        if rotate_by_size:
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding="utf-8"
            )
        else:
            file_handler = TimedRotatingFileHandler(
                log_file,
                when=when,
                interval=interval,
                backupCount=backup_count,
                encoding="utf-8"
            )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # 3. 错误日志文件处理器（单独记录ERROR及以上级别）
        error_log_file = os.path.join(log_dir, f"{name}_error.log")
        if rotate_by_size:
            error_handler = RotatingFileHandler(
                error_log_file,
                maxBytes=max_bytes // 2,  # 错误日志文件小一点
                backupCount=backup_count,
                encoding="utf-8"
            )
        else:
            error_handler = TimedRotatingFileHandler(
                error_log_file,
                when=when,
                interval=interval,
                backupCount=backup_count,
                encoding="utf-8"
            )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(formatter)
        logger.addHandler(error_handler)

        # 缓存logger实例
        cls._loggers[name] = logger
        return logger

    @classmethod
    def log_exception(
            cls,
            logger: logging.Logger,
            message: str,
            exc_info: Optional[Any] = None
    ) -> None:
        """
        记录异常信息（包含堆栈跟踪）

        :param logger: 日志实例
        :param message: 错误消息
        :param exc_info: 异常信息，默认为当前异常
        """
        if exc_info is None:
            # 如果未提供异常信息，自动获取当前异常
            logger.error(f"{message}\n详细异常:", exc_info=True)
        else:
            logger.error(f"{message}\n详细异常:", exc_info=exc_info)

    @classmethod
    def clear_loggers(cls) -> None:
        """清除所有缓存的logger实例（主要用于测试）"""
        for logger in cls._loggers.values():
            for handler in logger.handlers[:]:
                handler.close()
                logger.removeHandler(handler)
        cls._loggers.clear()


if __name__ == "__main__":
    # 1. 获取默认配置的logger
    default_logger = LoggerUtil.get_logger("demo")

    # 2. 测试不同级别日志
    default_logger.debug("这是调试信息（仅文件输出）")
    default_logger.info("程序启动成功")
    default_logger.warning("内存使用率超过80%")
    default_logger.error("配置文件解析失败")
    default_logger.critical("数据库连接超时，程序无法继续运行")

    # 3. 测试异常日志
    try:
        1 / 0
    except ZeroDivisionError:
        LoggerUtil.log_exception(default_logger, "发生除零错误")

    # 4. 获取自定义配置的logger（按时间轮转）
    time_rotate_logger = LoggerUtil.get_logger(
        name="time_rotate_demo",
        rotate_by_size=False,
        when="S",  # 每秒轮转（仅作测试，实际常用midnight）
        interval=10,
        level=logging.DEBUG
    )
    time_rotate_logger.debug("这是按时间轮转的调试日志")
