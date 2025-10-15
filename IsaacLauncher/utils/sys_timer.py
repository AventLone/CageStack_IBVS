import time


class SysTimer:
    _start_time = None
    _base_timestamp = 1640966400000  # 2022-01-01 00:00:00 UTC
    _eight_hour = 28800_000

    @staticmethod
    def initialize():
        """初始化时间基准"""
        if SysTimer._start_time is None:
            SysTimer._start_time = time.time()
            print(f"============== SysTimer initialize : {SysTimer._start_time} ================")
        else:
            raise RuntimeError("SysTimer has already been initialized.")
        
    @staticmethod
    def get_timestamp() -> int: 
        """获取当前时间戳，单位毫秒"""
        if SysTimer._start_time is None:
            raise RuntimeError("SysTimer is not initialized.")
        current_time = time.time()
        elapsed_time_ms = int((current_time - SysTimer._start_time) * 1000)
        return elapsed_time_ms + SysTimer._base_timestamp

    @staticmethod
    def get_timestamp_plus8() -> int:
        """获取当前时间戳，单位毫秒"""
        if SysTimer._start_time is None:
            raise RuntimeError("SysTimer is not initialized.")
        current_time = time.time()
        elapsed_time_ms = int((current_time - SysTimer._start_time) * 1000)
        return elapsed_time_ms + SysTimer._base_timestamp + SysTimer._eight_hour
    
    @staticmethod
    def get_time_seqnum() -> int:
        """获取自初始化以来的时间序列号，单位10毫秒"""
        if SysTimer._start_time is None:
            raise RuntimeError("SysTimer is not initialized.")
        current_time = time.time()
        elapsed_time_ms = (current_time - SysTimer._start_time) * 1000
        elapsed_time_seq = int(elapsed_time_ms / 10)  # 转换为10毫秒为单位
        return elapsed_time_seq
    
    @staticmethod
    def get_timestamp_us() -> int: 
            """获取当前时间戳，单位微秒"""
            if SysTimer._start_time is None:
                raise RuntimeError("SysTimer is not initialized.")
            current_time = time.time()
            elapsed_time_us = int((current_time - SysTimer._start_time) * 1_000_000)
            return elapsed_time_us + SysTimer._base_timestamp * 1000  # 微秒基准

    @staticmethod
    def get_timestamp_us_plus8() -> int:
            """获取当前时间戳，单位微秒"""
            if SysTimer._start_time is None:
                raise RuntimeError("SysTimer is not initialized.")
            current_time = time.time()
            elapsed_time_us = int((current_time - SysTimer._start_time) * 1_000_000)
            return elapsed_time_us + SysTimer._base_timestamp * 1000  + SysTimer._eight_hour * 1000# 微秒基准