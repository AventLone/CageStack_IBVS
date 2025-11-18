import ecal.core.core as ecal_core
from ecal.core.publisher import ProtoPublisher
from ecal.core.subscriber import ProtoSubscriber
import websockets
import asyncio
import cv2
import numpy as np
import json
from typing import Dict, List, Optional
import time
import threading
import base64
import msgpack
from protos import vehicle_state_msg_pb2, RawImage_pb2
from queue import Queue, Empty
import datetime


# -------------------------- 全局常量与数据结构定义 --------------------------

# 状态参数常量（与 param.h 对应）
STEER_SPEED = "steer_speed"
STEER_YAW = "steer_yaw"
FORK_HEIGHT = "fork_height"
FORK_X_HEIGHT = "fork_x"
FORK_Y_HEIGHT = "fork_y"

# 动作数据结构（对应 AGF_Action）
class AGF_Action:
    def __init__(self):
        self.timestamp: int = 0  # 毫秒级时间戳
        self.fork_z: float = 0.0  # 货叉Z轴高度
        self.fork_y: float = 0.0  # 货叉Y轴位置
        self.fork_x: float = 0.0  # 货叉X轴位置
        self.steering_angle: float = 0.0  # 转向角
        self.agf_speed: float = 0.0  # 车速
        self.stop_ratio: float = 0.0  # 停止比例（0~1，1表示任务完成）

# 图像状态数据结构（对应 ImageStatus）
class ImageStatus:
    def __init__(self):
        self.image: RawImage_pb2.RawImage = RawImage_pb2.RawImage()  # 原始图像proto
        self.timestamp: int = 0  # 图像时间戳（毫秒）
        self.is_reflash: bool = False  # 是否更新过

# VLA 数据结构（对应 VlaData）
class VlaData:
    def __init__(self):
        self.images: Dict[str, np.ndarray] = {}  # 相机图像（话题→cv2矩阵）
        self.state: List[float] = []  # 车体状态 [fork_height, fork_y, 0, steer_yaw, steer_speed]
        self.language_instruction: str = ""  # 任务指令（prompt）
        self.timestamp: int = 0  # 数据时间戳（毫秒）

class VlaInferenceServer:
    def __init__(self,
                 vehicle_state_topic: str = "vehicle/status",
                 vehicle_action_topic: str = "vla/cmd",
                 camera_topics=None
                 ):
        # 1. 初始化lock成员变量
        self.m_data_mutex = threading.Lock()  # 数据保护锁
        self.m_connection_mutex = threading.Lock()  # WebSocket 连接锁
        self.m_actions_mutex = threading.Lock()  # 动作列表锁

        # 2.发布动作到isaacsim
        self.m_vehicle_action = AGF_Action()
        self.vehicle_action_pub = ProtoPublisher(vehicle_action_topic, vehicle_state_msg_pb2.VehicleStateMsg)

        # 3. 订阅机器人状态（解析速度、舵角、货叉位姿）
        self.m_vehicle_state = AGF_Action()
        self.vehicle_state_sub = ProtoSubscriber(vehicle_state_topic, vehicle_state_msg_pb2.VehicleStateMsg)
        self.vehicle_state_sub.set_callback(self.status_callback)

        # 订阅相机图像
        if camera_topics is None:
            camera_topics = ['isaacsim/camera/fork_camera',
                             'isaacsim/camera/front_camera',
                             'isaacsim/camera/back_camera']
        # 图像缓存（话题→ImageStatus）
        self.m_image_map: Dict[str, ImageStatus] = {
            topic: ImageStatus() for topic in camera_topics
        }

        self.camera_sub = []
        self.camera_topics = camera_topics
        for cam_topic in camera_topics:
            camera_sub = ProtoSubscriber(cam_topic, RawImage_pb2.RawImage)
            camera_sub.set_callback(self.image_callback)
            self.camera_sub.append(camera_sub)


        # 最新 VLA 数据（待发送给 Python 推理服务）
        self.m_last_data = VlaData()

        # WebSocket 相关
        self.websocket: Optional[websockets.WebSocketClientProtocol] = None
        self.ws_uri: str = ""
        self.m_data_thread_stop = False  # 数据发送线程停止标志
        self.m_ctrl_thread_stop = False  # 动作控制线程停止标志
        self.m_data_thread: Optional[threading.Thread] = None  # 数据发送线程
        self.m_ctrl_thread: Optional[threading.Thread] = None  # 动作控制线程

        # 动作相关
        self.m_action_list: List[AGF_Action] = []  # 动作序列缓存
        self.m_is_finish = False  # 任务完成标志
        self.task_information: str = ""  # 任务指令（prompt）

        # 时间统计相关
        self.m_time_initialize = False  # 时间初始化标志
        self.m_last_time: float = 0.0  # 上次接收推理结果的时间

        # 子线程data_loop只负责往队列扔数据，数据发送在ws_connect中进行
        self.send_queue = Queue(maxsize=100)


    # -------------------------- eCAL 回调函数 --------------------------
    def status_callback(self, topic_name, msg, msg_time) -> None:
        """
        解析 Webots 发布的机器人状态（速度、舵角、货叉位姿）
        对应原 C++ getStatusMsg
        """
        with self.m_data_mutex:
            self.m_vehicle_state.timestamp = msg.timestamp
            self.m_vehicle_state.velocity = msg.drive_velocity
            self.m_vehicle_state.steering_angle = msg.steer_angle
            self.m_vehicle_state.fork_z = msg.fork_z


    def image_callback(self, topic_name, msg, msg_time) -> None:
        """
        解析 Webots 发布的相机图像，缓存并检查是否满足3路图像同步
        对应原 C++ imageCallback
        """
        with self.m_data_mutex:
            if topic_name not in self.m_image_map:
                self.m_image_map[topic_name] = ImageStatus()
                print(f"Insert new camera topic: {topic_name}")

            self.m_image_map[topic_name].image.CopyFrom(msg)
            self.m_image_map[topic_name].is_reflash = True

            # 记录当前时间戳（毫秒）
            now = int(time.time() * 1000)  # 转换为毫秒
            self.m_image_map[topic_name].timestamp = now

            # 检查是否所有相机都已更新（至少3路）
            all_reflash = all(
                status.is_reflash for status in self.m_image_map.values()
            )
            if not all_reflash or len(self.m_image_map) < 3:
                return

            # 无任务指令时跳过
            if not self.task_information:
                return

            # 收集所有相机图像
            new_timestamp = 0
            for cam_topic, status in self.m_image_map.items():
                # 从 proto 提取图像数据
                img_data = status.image.data
                height = status.image.height
                width = status.image.width
                # 转换为 cv2 矩阵
                bgr_mat = np.frombuffer(img_data, dtype=np.uint8).reshape((height, width, 3)) # IsaacSim图像数据为3通道而不是webots的4通道
                self.m_last_data.images[cam_topic] = bgr_mat

                # 取最大时间戳作为数据时间戳
                if status.timestamp > new_timestamp:
                    new_timestamp = status.timestamp

            # 收集车体状态
            self.m_last_data.state = [
                self.m_vehicle_state.fork_z,
                self.m_vehicle_state.fork_y,
                0.0,  # 原代码中固定为0的占位符
                self.m_vehicle_state.steering_angle,
                self.m_vehicle_state.agf_speed
            ]
            self.m_last_data.language_instruction = self.task_information
            self.m_last_data.timestamp = new_timestamp

            # 重置图像更新标志（准备下一轮）
            for status in self.m_image_map.values():
                status.is_reflash = False
                status.image.Clear()
            time.sleep(0.05)

    async def send_from_queue(self):
        """在主事件循环中运行的发送循环，从队列取数据并发送"""
        while True:
            try:
                # 从队列获取数据（非阻塞，超时0.1秒避免无数据时阻塞）
                data = self.send_queue.get(block=False, timeout=0.1)
                # 检查终止信号
                if data is None:
                    break

                # 发送数据（带超时控制）
                if self.websocket and self.websocket.open:
                    await self.websocket.send(data)
                    # await asyncio.wait_for(self.websocket.send(data), timeout=5.0)
                    await asyncio.sleep(0.1)
                    # timestamp_now = int(time.time() * 1000)
                    # print("send_from_queue send data time:", timestamp_now)

            except Empty:
                # 队列为空时继续循环（避免CPU空转）
                await asyncio.sleep(0.01)
                continue
            except asyncio.TimeoutError:
                print("发送超时，WebSocket无响应")
            except Exception as e:
                print(f"发送循环出错: {e}")

    # -------------------------- WebSocket 相关函数 --------------------------
    async def ws_connect(self, uri: str) -> None:
        """
        WebSocket 异步连接（内部调用，由 runVla 启动）
        对应原 C++ runVla
        """
        self.ws_uri = uri
        try:
            async with websockets.connect(uri) as websocket:
                self.websocket = websocket
                print(f"WebSocket connected to {uri}")

                # 启动数据发送和动作控制线程
                self.m_data_thread_stop = False
                self.m_ctrl_thread_stop = False
                self.m_data_thread = threading.Thread(target=self.data_loop)
                self.m_ctrl_thread = threading.Thread(target=self.ctrl_loop)
                self.m_data_thread.start()
                self.m_ctrl_thread.start()

                # 启动发送循环协程
                send_task = asyncio.create_task(self.send_from_queue())

                # 持续接收 WebSocket 消息（推理结果）
                async for msg in websocket:
                    await self.on_message(msg)

        except websockets.exceptions.ConnectionClosed as e:
            print(f"WebSocket closed: {e}")
        except Exception as e:
            print(f"WebSocket error: {str(e)}")
        finally:
            self.send_queue.put(None)
            # 关闭时清理线程
            await self.on_close()

    def runVla(self, uri: str) -> None:
        """启动 WebSocket 连接（对外接口）"""
        asyncio.run(self.ws_connect(uri))

    async def on_close(self) -> None:
        """WebSocket 关闭回调，清理线程"""
        self.m_data_thread_stop = True
        self.m_ctrl_thread_stop = True

        # 等待线程退出
        if self.m_data_thread and self.m_data_thread.is_alive():
            self.m_data_thread.join()
        if self.m_ctrl_thread and self.m_ctrl_thread.is_alive():
            self.m_ctrl_thread.join()

        self.websocket = None
        print("WebSocket connection closed (cleanup done)")


    async def on_message(self, msg: bytes) -> None:
        """
        接收 WebSocket 消息（Python 推理服务的动作序列）
        对应原 C++ on_message
        """
        # 统计推理延迟
        now = time.time()
        if self.m_time_initialize:
            delay_ms = (now - self.m_last_time) * 1000
            print(f"Infer delay: {delay_ms:.1f} ms")
        self.m_time_initialize = True
        self.m_last_time = now

        # 解析 JSON 消息（原 C++ 中为 Msgpack，此处兼容 JSON）
        # msg_str = msg.decode("utf-8")  #硬解utf-8会报错
        msg_str = msg.decode("latin-1")
        if "actions" not in msg_str:
            return

        try:
            json_data = json.loads(msg_str)
            actions = json_data["actions"]  # 动作序列（列表）
            timestamp = json_data["timestamp"]  # 动作基准时间戳
            self.update_actions(actions, timestamp)
        except json.JSONDecodeError as e:
            print(f"JSON parse error: {str(e)}")
        except Exception as e:
            print(f"Message processing error: {str(e)}")


    # -------------------------- 数据发送与动作控制线程 --------------------------
    def data_loop(self) -> None:
        """
        数据发送线程：将图像、状态、指令打包发送给 Python 推理服务
        """
        while not self.m_data_thread_stop:
            # 获取最新数据（线程安全）
            last_info = VlaData()
            self.get_last_data(last_info)

            # 无数据时跳过
            if not last_info.images or not last_info.state:
                time.sleep(0.01)  # 10ms 等待
                continue

            # 构建 JSON 消息（图像转 Base64）
            json_data = {
                "observation/front_image": self.cvMatToBase64(last_info.images[self.camera_topics[0]]),
                "observation/bev_image": self.cvMatToBase64(last_info.images[self.camera_topics[1]]),
                "observation/fork_image": self.cvMatToBase64(last_info.images[self.camera_topics[1]]),
                "observation/state": last_info.state,
                "prompt": last_info.language_instruction,
                "timestamp": last_info.timestamp
            }

            # 转换为 Msgpack 格式（与原 C++ 一致）
            # byte_data = json.dumps(json_data).encode("utf-8")  # 若需严格 Msgpack，可改用 `orjson` 或 `msgpack` 库
            msgpack_data = msgpack.packb(json_data)
            # 发送 WebSocket 消息（线程安全）
            try:
                # asyncio.run: 会导致旧连接失效, 导致报错.
                # asyncio.run(self._async_send(msgpack_data))
                # 将数据存入发送队列（线程安全操作）
                self.send_queue.put(msgpack_data)
                timestamp_now = int(time.time() * 1000)
            except Exception as e:
                print(f"Data send error: {str(e)}")

            # 控制发送频率
            time.sleep(0.1)

    async def _async_send(self, data: bytes) -> None:
        """异步发送 WebSocket 消息（内部调用，解决线程异步问题）"""
        try:
            if self.websocket and self.websocket.open:
                await self.websocket.send(data)
        except Exception as e:
            print(f"发送失败: {e}")

    def ctrl_loop(self) -> None:
        """
        动作控制线程：根据当前时间插值动作，发布给 Webots
        对应原 C++ ctrl_loop
        """
        while not self.m_ctrl_thread_stop:
            # 线程安全获取动作列表
            action_list = []
            with self.m_actions_mutex:
                action_list = self.m_action_list.copy()

            # 动作列表长度不足时跳过
            if len(action_list) < 2:
                continue

            # 获取当前系统时间（毫秒）
            timestamp_now = int(time.time() * 1000)

            # 插值找到当前应执行的动作
            target_action = None
            for i in range(len(action_list) - 1):
                action_0 = action_list[i]
                action_1 = action_list[i + 1]

                # 情况1：当前时间在两个动作之间 → 线性插值
                if action_0.timestamp <= timestamp_now < action_1.timestamp:
                    # 计算插值比例
                    time_diff = action_1.timestamp - action_0.timestamp
                    ratio_0 = (action_1.timestamp - timestamp_now) / time_diff  # 前动作权重
                    ratio_1 = (timestamp_now - action_0.timestamp) / time_diff  # 后动作权重

                    # 线性插值计算目标动作
                    target_action = AGF_Action()
                    target_action.timestamp = timestamp_now
                    target_action.fork_z = ratio_0 * action_0.fork_z + ratio_1 * action_1.fork_z
                    target_action.fork_y = ratio_0 * action_0.fork_y + ratio_1 * action_1.fork_y
                    target_action.fork_x = ratio_0 * action_0.fork_x + ratio_1 * action_1.fork_x
                    target_action.steering_angle = ratio_0 * action_0.steering_angle + ratio_1 * action_1.steering_angle
                    target_action.agf_speed = ratio_0 * action_0.agf_speed + ratio_1 * action_1.agf_speed
                    target_action.stop_ratio = ratio_0 * action_0.stop_ratio + ratio_1 * action_1.stop_ratio
                    break

                # 情况2：当前时间早于第一个动作 → 执行第一个动作
                elif i == 0 and timestamp_now < action_0.timestamp:
                    target_action = action_0
                    break

            # 发布目标动作到 Isaac-Sim
            if target_action:
                self.send_action(target_action)

            # 控制动作发布频率（100ms 一次）
            time.sleep(0.1)

    # -------------------------- 动作处理函数 --------------------------
    def update_actions(self, actions: List[float], timestamp: int) -> None:
        """
        处理推理服务返回的动作序列，缓存并平滑
        对应原 C++ update_actions
        """
        print(f"Update actions: size={len(actions)}, base timestamp={timestamp}")

        # 过滤出当前时间之后的动作（步长 100ms）
        timestamp_now = int(time.time() * 1000)
        action_list = []
        data_size = 6  # 每个动作6个参数：fork_z, fork_y, fork_x, steering_angle, agf_speed, stop_ratio
        for i in range(0, len(actions), data_size):
            # 计算动作的执行时间戳（原 C++ 中 i+1 逻辑保留）
            action_ts = timestamp + 100 * (i // data_size)

            # # 跳过已过期的动作
            # if action_ts < timestamp_now - 200:  # 允许 200ms 误差
            #     continue

            # 构建动作对象
            action = AGF_Action()
            action.timestamp = action_ts
            action.fork_z = actions[i]
            action.fork_y = actions[i + 1]
            action.fork_x = actions[i + 2]
            action.steering_angle = actions[i + 3]
            action.agf_speed = actions[i + 4]
            action.stop_ratio = actions[i + 5]
            action_list.append(action)

        # 无有效动作时返回
        if not action_list:
            print(f"No valid actions (now={timestamp_now}, first action ts={timestamp})")
            return

        # 打印第一个动作信息
        first_action = action_list[0]
        print(f"First action: fork_z={first_action.fork_z:.2f}, "
              f"steering_angle={first_action.steering_angle:.2f}, "
              f"agf_speed={first_action.agf_speed:.2f}")

        # # 保存动作到文件（用于分析）
        # with open("action_chunks.txt", "a", encoding="utf-8") as f:
        #     for action in action_list:
        #         f.write(f"{action.timestamp} {action.agf_speed} {action.steering_angle} "
        #                 f"{action.fork_z} {action.stop_ratio}\n")

        # 动作平滑（与历史动作加权融合，权重 0.7 新动作 + 0.3 历史动作）
        action_ratio = 0.7
        action_index = 0
        if len(self.m_action_list) >= 2:
            for i in range(len(self.m_action_list) - 1):
                old_0 = self.m_action_list[i]
                old_1 = self.m_action_list[i + 1]
                new_action = action_list[action_index]
                if old_0.timestamp <= new_action.timestamp < old_1.timestamp:
                    # 计算历史动作的权重
                    time_diff = old_1.timestamp - old_0.timestamp
                    old_ratio_0 = (1 - action_ratio) * (old_1.timestamp - new_action.timestamp) / time_diff
                    old_ratio_1 = (1 - action_ratio) * (new_action.timestamp - old_0.timestamp) / time_diff

                    # 加权融合
                    new_action.fork_z = action_ratio * new_action.fork_z + old_ratio_0 * old_0.fork_z + old_ratio_1 * old_1.fork_z
                    new_action.fork_y = action_ratio * new_action.fork_y + old_ratio_0 * old_0.fork_y + old_ratio_1 * old_1.fork_y
                    new_action.fork_x = action_ratio * new_action.fork_x + old_ratio_0 * old_0.fork_x + old_ratio_1 * old_1.fork_x
                    new_action.steering_angle = action_ratio * new_action.steering_angle + old_ratio_0 * old_0.steering_angle + old_ratio_1 * old_1.steering_angle
                    new_action.agf_speed = action_ratio * new_action.agf_speed + old_ratio_0 * old_0.agf_speed + old_ratio_1 * old_1.agf_speed
                    new_action.stop_ratio = action_ratio * new_action.stop_ratio + old_ratio_0 * old_0.stop_ratio + old_ratio_1 * old_1.stop_ratio

                    action_list[action_index] = new_action
                    action_index += 1


        # 线程安全更新动作列表
        with self.m_actions_mutex:
            self.m_action_list = action_list.copy()

    def send_action(self, action: AGF_Action) -> None:
        """
        将动作发布给 Webots（通过 eCAL）
        对应原 C++ updateAction（AGF_Action 重载版）
        """
        if self.m_is_finish:
            return

        # 检查任务是否完成（stop_ratio ≥ 0.98）
        if action.stop_ratio >= 0.98:
            print("Task finished!")
            self.m_is_finish = True
            return

        # 填充动作 protobuf 消息
        action2send = vehicle_state_msg_pb2.VehicleStateMsg()
        action2send.fork_z = action.fork_z
        action2send.fork_y = action.fork_y
        action2send.fork_x = action.fork_x
        action2send.fork_pitch = 0
        action2send.steer_angle = action.steering_angle
        action2send.drive_velocity = action.agf_speed

        # 序列化并发布到 eCAL
        self.vehicle_action_pub.send(action2send)

    # -------------------------- 辅助函数 --------------------------
    def cvMatToBase64(self, mat: np.ndarray) -> str:
        """
        将 cv2 图像矩阵转换为 Base64 字符串（JPG 压缩）
        对应原 C++ cvMatToBase64
        """
        # 压缩为 JPG 格式
        is_success, buffer = cv2.imencode(".jpg", mat)
        if not is_success:
            raise ValueError("Failed to encode image to JPG")
        # 转换为 Base64
        return base64.b64encode(buffer).decode("utf-8")

    def set_task_instruction(self, information: str) -> None:
        """设置任务指令（prompt）"""
        self.task_information = information
        self.m_is_finish = False  # 重置任务完成标志

    def get_last_data(self, data: VlaData) -> None:
        """
        线程安全获取最新 VLA 数据
        对应原 C++ get_last_data
        """
        with self.m_data_mutex:
            # 复制图像（深拷贝避免线程冲突）
            data.images = {
                topic: mat.copy() for topic, mat in self.m_last_data.images.items()
            }
            # 复制状态和指令
            data.state = self.m_last_data.state.copy()
            data.language_instruction = self.m_last_data.language_instruction
            data.timestamp = self.m_last_data.timestamp

            # 清空缓存（准备下一轮）
            self.m_last_data.images.clear()
            self.m_last_data.state.clear()
            self.m_last_data.language_instruction = ""


# -------------------------- 测试与使用示例 --------------------------
if __name__ == "__main__":
    import sys
    # 初始化 eCAL
    ecal_core.initialize(sys.argv, "Serial service")

    # 1. 创建 VLA 推理实例
    vla_infer = VlaInferenceServer()

    # 2. 设置任务指令（例如："Move the fork to height 1.0 and drive forward"）
    vla_infer.set_task_instruction("Move fork to 0.8m and drive at 0.5m/s")

    # 3. 连接 Python 推理服务的 WebSocket 地址（例如：ws://127.0.0.1:8765）
    ws_uri = "ws://10.10.20.25:8080"
    try:
        vla_infer.runVla(ws_uri)
    except KeyboardInterrupt:
        print("Program interrupted by user")
    finally:
        # 清理资源
        del vla_infer

    while ecal_core.ok():
        time.sleep(0.001)

    # 关闭 eCAL
    ecal_core.finalize()