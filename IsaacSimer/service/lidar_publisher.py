import time
import numpy as np
import ecal.core.core as ecal_core
import sys
from typing import Optional

from ecal.core.publisher import ProtoPublisher
from protos.point_cloud2_pb2 import Header, PointField, PointCloud2

# 导入 Isaac Sim 激光雷达类和点云元数据类
from devices.lidar import  PointCloudWithMeta
from utils import LoggerUtil, SimulationTimer


def convert_isaac_to_pb(isaac_cloud: PointCloudWithMeta) -> PointCloud2:
    """将 Isaac Sim 点云转换为 Protobuf PointCloud2 格式"""
    # 1. 处理空点云
    if isaac_cloud.points.shape[0] == 0:
        return PointCloud2()

    # 2. 构建 Header（消息头）
    header = Header()
    header.seq = int(SimulationTimer.get_timestamp() % 10000)  # 简单自增序列号
    header.timestamp = SimulationTimer.get_timestamp()  # 纳秒级时间戳
    header.frame_id = "mid360_lidar"  # 坐标系 ID（需与接收端一致）

    # 3. 构建 PointField（点云字段定义：x/y/z 为 float32，强度为 uint8）
    fields = [
        PointField(name="x", offset=0, datatype=7, count=1),  # 7=float32
        PointField(name="y", offset=4, datatype=7, count=1),  # 偏移 4 字节（float32 占 4 字节）
        PointField(name="z", offset=8, datatype=7, count=1),  # 偏移 8 字节
        PointField(name="intensity", offset=12, datatype=1, count=1)  # 1=uint8，偏移 12 字节
    ]

    # 4. 构建 PointCloud2 主体
    pb_cloud = PointCloud2()
    pb_cloud.header.CopyFrom(header)  # 复制 Header 到嵌套字段
    pb_cloud.height = 1  # 非组织化点云（height=1，width=总点数）
    pb_cloud.width = isaac_cloud.points.shape[0]  # 点云总数量
    pb_cloud.fields.extend(fields)  # 添加字段定义
    pb_cloud.is_bigendian = False  # 小端字节序（主流系统默认）
    pb_cloud.point_step = 13  # 单个点字节数：3*4（x/y/z）+1（强度）=13
    pb_cloud.row_step = pb_cloud.point_step * pb_cloud.width  # 一行总字节数
    pb_cloud.is_dense = False  # 允许包含无效点（如雷达盲区）

    # 5. 打包点云二进制数据（x/y/z + 强度）
    points = isaac_cloud.points.astype(np.float32)  # 转换为 float32
    intensities = isaac_cloud.intensities.astype(np.uint8)  # 转换为 uint8
    data = b""
    for i in range(pb_cloud.width):
        data += points[i].tobytes()  # x/y/z 字节（12 字节）
        data += intensities[i:i + 1].tobytes()  # 强度字节（1 字节）
    pb_cloud.data = data

    return pb_cloud

class EcalLidarPublisher:
    """
    激光雷达点云 eCAL 发布器类
    新增：eCAL 全生命周期异常管理（初始化、发布、释放）、资源自动回收
    """
    def __init__(self, ecal_topic: str = "/sensors/lidar/mid360/pointcloud2"):
        self.ecal_topic = ecal_topic  # eCAL 发布话题（需与接收端一致）
        self.publisher: Optional[ProtoPublisher] = None  # eCAL Protobuf 发布器
        self.ecal_initialized = False  # eCAL 核心初始化状态标记
        self.seq = 0  # 消息序列号（替换原时间戳取模，确保严格自增）

    def init_ecal(self) -> bool:
        """
        初始化 eCAL 核心与发布器
        :return: True（初始化成功）/ False（初始化失败）
        """
        try:
            # 1. 初始化 eCAL 核心（避免重复初始化）
            if not ecal_core.is_initialized():
                # 传入进程名（便于 eCAL Monitor 识别）
                ecal_core.initialize(sys.argv, "IsaacSim_Mid360_Lidar_Publisher")
                self.ecal_initialized = True
                print(f"[INFO] eCAL 核心初始化成功（进程名：IsaacSim_Mid360_Lidar_Publisher）", file=sys.stdout)
            else:
                print(f"[INFO] eCAL 核心已初始化，跳过重复操作", file=sys.stdout)

            # 2. 创建 Protobuf 发布器（绑定话题与消息类型）
            self.publisher = ProtoPublisher(self.ecal_topic, PointCloud2)
            if not self.publisher:
                raise RuntimeError("Protobuf 发布器创建返回空对象")

            # 3. 验证发布器状态（部分版本需显式检查是否创建成功）
            if not hasattr(self.publisher, "send") or not callable(getattr(self.publisher, "send")):
                raise RuntimeError("发布器对象缺少 send 方法，创建异常")

            print(f"[INFO] eCAL Protobuf 发布器创建成功（话题：{self.ecal_topic}）", file=sys.stdout)
            return True

        except Exception as e:
            print(f"[ERROR] eCAL 初始化失败：{str(e)}，详细栈信息：", file=sys.stderr)
            # 初始化失败时清理资源
            self.cleanup()
            return False

    def publish_pointcloud(self, isaac_cloud: PointCloudWithMeta) -> bool:
        """
        发布 Isaac Sim 点云（先转换为 Protobuf，再通过 eCAL 发布）
        :param isaac_cloud: Isaac Sim 输出的带元信息点云
        :return: True（发布成功）/ False（发布失败）
        """
        # 前置校验：发布器已初始化
        if not self.publisher or not self.ecal_initialized:
            print(f"[ERROR] eCAL 发布器未初始化，无法发布点云", file=sys.stderr)
            return False

        try:
            # 1. 转换点云格式（调用完善后的转换函数）
            pb_cloud = convert_isaac_to_pb(isaac_cloud)
            if pb_cloud is None:
                print(f"[ERROR] 点云转换失败，跳过发布", file=sys.stderr)
                return False

            # 2. 覆盖序列号（确保严格自增，避免时间戳取模重复）
            pb_cloud.header.seq = self.seq
            self.seq += 1

            # 3. 发送 Protobuf 消息（捕获发送异常）
            send_success = self.publisher.send(pb_cloud)
            if not send_success:
                raise RuntimeError("发布器 send 方法返回 False")

            # 4. 打印发布日志（仅打印关键信息，避免刷屏）
            print(f"[INFO] 点云发布成功 | 点数：{pb_cloud.width} | 序列号：{pb_cloud.header.seq} | 时间戳：{pb_cloud.header.timestamp} ", file=sys.stdout)
            return True

        except Exception as e:
            print(f"[ERROR] 点云发布失败：{str(e)}，详细栈信息：", file=sys.stderr)
            # 发布失败不强制退出，尝试下一次发布
            return False

    def cleanup(self):
        """
        清理 eCAL 资源（核心+发布器），确保程序退出时无资源泄漏
        建议在程序退出前、异常捕获后调用
        """
        print(f"[INFO] 开始清理 eCAL 资源...", file=sys.stdout)
        try:
            # 1. 销毁发布器（优先释放发布器资源）
            if self.publisher:
                self.publisher.destroy()
                self.publisher = None
                print(f"[INFO] eCAL Protobuf 发布器已销毁", file=sys.stdout)

            # 2. 终止 eCAL 核心（仅在当前进程初始化时终止）
            if self.ecal_initialized and ecal_core.is_initialized():
                ecal_core.finalize()
                self.ecal_initialized = False
                print(f"[INFO] eCAL 核心已终止", file=sys.stdout)

        except Exception as e:
            print(f"[WARNING] 清理 eCAL 资源时发生异常：{str(e)}", file=sys.stderr)

    def __del__(self):
        """
        析构函数：自动清理资源（避免用户忘记调用 cleanup）
        作为最后一道保障，建议仍显式调用 cleanup
        """
        self.cleanup()
