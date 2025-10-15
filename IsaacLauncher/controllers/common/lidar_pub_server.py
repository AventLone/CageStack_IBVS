import time
import threading
from typing import Optional, Tuple, Dict

import numpy as np
import ecal.core.core as ecal_core
from ecal.core.publisher import ProtoPublisher

from protos import point_cloud2_pb2 as pb
from utils import LoggerUtil, SysTimer

# 与 PointField.datatype 对齐
DATATYPE_INT8    = 1
DATATYPE_UINT8   = 2
DATATYPE_INT16   = 3
DATATYPE_UINT16  = 4
DATATYPE_INT32   = 5
DATATYPE_UINT32  = 6
DATATYPE_FLOAT32 = 7
DATATYPE_FLOAT64 = 8

logger = LoggerUtil.get_logger("lidar_pc2_pub")


class LidarPC2Publisher(threading.Thread):
    """
    将 VN_Mid360 点云以 pb.PointCloud2 话题发布。

    约定：
    - 点布局：x, y, z, (intensity), (label), (time)，均为 float32 小端。
    - header.timestamp: 毫秒（uint64），通常取消息发送瞬时的 epoch ms。
    - per-point 'time' 字段：float32，**保持上游给出的单位与含义**（这里默认也是毫秒）。
    - 高度=1、宽度=N 的非组织化点云。
    """

    def __init__(self,
                 lidar,
                 topic: str,
                 frame_id: str = "mid360",
                 period: float = 0.05,         # 发布频率：默认 20Hz
                 publish_empty: bool = False   # 是否发布空帧作为心跳
                 ):
        super().__init__(daemon=True)
        self.lidar = lidar
        self.topic = topic
        self.frame_id = frame_id
        self.period = max(0.005, float(period))
        self.publish_empty = publish_empty

        self._pub = ProtoPublisher(topic, pb.PointCloud2)
        self._stop = threading.Event()
        self._seq = 0

        # 缓存不同通道组合的 fields 与 point_step，避免每帧重复构造
        self._fields_cache: Dict[Tuple[bool, bool, bool], Tuple[list, int]] = {}

    # ---------- 工具 ----------

    def _np_dtype_and_size(self, datatype: int):
        if datatype == DATATYPE_FLOAT32: return np.float32, 4
        if datatype == DATATYPE_FLOAT64: return np.float64, 8
        if datatype == DATATYPE_UINT8:   return np.uint8, 1
        if datatype == DATATYPE_INT8:    return np.int8, 1
        if datatype == DATATYPE_UINT16:  return np.uint16, 2
        if datatype == DATATYPE_INT16:   return np.int16, 2
        if datatype == DATATYPE_UINT32:  return np.uint32, 4
        if datatype == DATATYPE_INT32:   return np.int32, 4
        raise ValueError(f"Unsupported datatype: {datatype}")

    def make_fields_xyz_intensity_time_label(self, has_intensity: bool, has_time: bool, has_label: bool):
        """
        生成 PointField 列表（小端），并返回 (fields, point_step)。
        使用缓存以减少分配。
        """
        key = (has_intensity, has_time, has_label)
        if key in self._fields_cache:
            return self._fields_cache[key]

        fields = []
        offset = 0
        for name, dtype in [
            ("x", DATATYPE_FLOAT32),
            ("y", DATATYPE_FLOAT32),
            ("z", DATATYPE_FLOAT32),
        ]:
            fields.append(pb.PointField(name=name, offset=offset, datatype=dtype, count=1))
            _, sz = self._np_dtype_and_size(dtype)
            offset += sz

        if has_intensity:
            fields.append(pb.PointField(name="intensity", offset=offset,
                                        datatype=DATATYPE_FLOAT32, count=1))
            offset += 4

        if has_label:
            fields.append(pb.PointField(name="label", offset=offset,
                                        datatype=DATATYPE_UINT32, count=1))
            offset += 4

        if has_time:
            fields.append(pb.PointField(name="time", offset=offset,
                                        datatype=DATATYPE_FLOAT64, count=1))
            offset += 8

        point_step = offset
        self._fields_cache[key] = (fields, point_step)
        return fields, point_step

    def pack_pointcloud2(self,
                         points_xyz: Optional[np.ndarray],
                         intensities: Optional[np.ndarray] = None,
                         times: Optional[np.ndarray] = None,
                         labels: Optional[np.ndarray] = None,
                         frame_id: str = "mid360",
                         seq: int = 0,
                         stamp_ms: Optional[int] = None) -> pb.PointCloud2:
        """
        将 (N,3) XYZ 与可选 intensity/time/label 打包为 PointCloud2。
        - points_xyz:  (N,3) float32
        - intensities: (N,)  float32
        - times:       (N,)  float32（此处保留上游单位与语义，常为毫秒）
        - labels:      (N,)  uint32
        """
        if points_xyz is None:
            pts = np.empty((0, 3), dtype=np.float32)
        else:
            pts = np.asarray(points_xyz, dtype=np.float32)

        if not (pts.ndim == 2 and pts.shape[1] == 3):
            raise ValueError(f"points shape must be (N,3), got {pts.shape}")
        N = pts.shape[0]

        has_intensity = intensities is not None and len(intensities) == N
        has_time = times is not None and len(times) == N
        has_label = labels is not None and len(labels) == N

        # 如果 labels 为 None，强制为默认值，例如0
        if not has_label:
            labels = np.zeros(N, dtype=np.uint32)

        fields, point_step = self.make_fields_xyz_intensity_time_label(has_intensity, has_time, has_label)

        # 与 fields 的 point_step 一致（无 padding）
        names = ["x", "y", "z"]
        formats = ["<f4", "<f4", "<f4"]
        offsets = [0, 4, 8]
        offset = 12

        if has_intensity:
            names.append("intensity")
            formats.append("<f4")
            offsets.append(offset)
            offset += 4

        if has_label:
            names.append("label")
            formats.append("<u4")
            offsets.append(offset)
            offset += 4

        if has_time:
            names.append("time")
            formats.append("<f8")
            offsets.append(offset)
            offset += 8

        dtype = np.dtype({
            "names": names,
            "formats": formats,
            "offsets": offsets,
            "itemsize": point_step,  # 显式指定！防止出现 (N,0) 的字段形状
        }, align=False)

        buf = np.empty(N, dtype=dtype)
        if N:
            buf["x"] = pts[:, 0]
            buf["y"] = pts[:, 1]
            buf["z"] = pts[:, 2]
            if has_intensity:
                buf["intensity"] = np.asarray(intensities, dtype=np.float32)

            if has_label:
                buf["label"] = np.asarray(labels, dtype=np.uint32)

            if has_time:
                buf["time"] = np.asarray(times, dtype=np.float64)

        msg = pb.PointCloud2()
        msg.header.seq = seq
        msg.header.timestamp = int(stamp_ms if stamp_ms is not None else SysTimer.get_timestamp())
        msg.header.frame_id = frame_id

        msg.height = 1
        msg.width = N
        msg.fields.extend(fields)
        msg.is_bigendian = False
        msg.point_step = point_step
        msg.row_step = point_step * N
        msg.data = buf.tobytes(order="C")
        msg.is_dense = (N > 0) and (not np.isnan(pts).any())

        return msg

    # ---------- 线程控制 ----------
    def stop(self):
        """请求停止发布线程。"""
        self._stop.set()

    def run(self):
        logger.info("LidarPC2Publisher start -> topic=%s", self.topic)
        next_t = time.perf_counter()

        while not self._stop.is_set():
            try:
                # 期望上游返回类似：pc.points (N,3) / pc.timestamps (N,) / pc.intensities (N,) / pc.labels (N,)
                pc = self.lidar.get_pointcloud()
                pts = None if pc is None else pc.points
                N = 0 if pts is None else len(pts)

                if N == 0 and not self.publish_empty:
                    # 跳过空帧以省带宽
                    pass
                else:
                    # 时间戳：保留上游 per-point 时间（通常毫秒）；长度不匹配则忽略
                    times = None
                    if pc is not None and pc.timestamps is not None and len(pc.timestamps) == N:
                        times = np.asarray(pc.timestamps, dtype=np.float64)

                    intens = None
                    if pc is not None and pc.intensities is not None and len(pc.intensities) == N:
                        intens = pc.intensities

                    labels = None
                    if pc is not None and pc.labels is not None and len(pc.labels) == N:
                        labels = pc.labels
                    # 如果 labels 为 None，使用默认值
                    if labels is None:
                        labels = np.zeros(N, dtype=np.uint32)  # 默认将所有点的 label 设为 0

                    msg = self.pack_pointcloud2(
                        points_xyz=pts if N > 0 else np.empty((0, 3), dtype=np.float32),
                        intensities=intens,
                        times=times,
                        labels=labels,
                        frame_id=self.frame_id,
                        seq=self._seq,
                        stamp_ms=SysTimer.get_timestamp()
                    )
                    self._pub.send(msg)
                    self._seq += 1

            except Exception:
                # 打印完整堆栈，便于定位数据/协议问题
                logger.exception("publish error")

            # 固定周期调度，避免频率漂移
            next_t += self.period
            dt = next_t - time.perf_counter()
            if dt > 0:
                time.sleep(dt)
            else:
                # 掉帧时重置起点，避免一直为负
                next_t = time.perf_counter()

        logger.info("LidarPC2Publisher exit")
