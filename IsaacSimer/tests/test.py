from utils.common import openSimuApp, loadConfig
simulation_app = openSimuApp("configs/e_test_2.yaml")

# Settings
import carb.settings
# Set DLSS to Quality mode (2) for best SDG results (Options: 0 (Performance), 1 (Balanced), 2 (Quality), 3 (Auto)
carb.settings.get_settings().set("rtx/post/dlss/execMode", 0)

from isaacsim.core.utils.extensions import enable_extension
enable_extension("isaacsim.ros2.bridge")   # enable ROS2 bridge extension
simulation_app.update()

import asyncio, time, threading, sys
import numpy as np
from isaacsim.core.api import World

from devices.vehicles import Vehicle
from utils.common import SimTimer
from isaacsim.sensors.rtx.impl import LidarRtx
from isaacsim.sensors.camera.camera import Camera
from omni.kit.async_engine import run_coroutine
from concurrent.futures import ThreadPoolExecutor, Future, ProcessPoolExecutor


class SensorSet:
    def __init__(self) -> None:
        self._cameras: dict[str, Camera] = dict()
        # self._lidars: dict[str, LidarRtx] = dict()
        for name in ["left_camera", "right_camera", "fork_camera"]:
            if name == "fork_camera":
                self._cameras[name] = Camera(
                    prim_path=f"/World/E_car_finish/whole_fork/center_camera/SG3S_ISX031C_GMSL2F_H190XA_01",
                    name=name,resolution=(960, 786))
            else:
                self._cameras[name] = Camera(
                    prim_path=f"/World/E_car_finish/body/cameras/{name}/SG3S_ISX031C_GMSL2F_H190XA_01",
                    name=name,resolution=(960, 786))
            self._cameras[name].initialize()
            self._cameras[name].add_distance_to_image_plane_to_frame()
            self._cameras[name].add_semantic_segmentation_to_frame()

        # self._lidar = LidarRtx(prim_path="/World/E_car_finish/whole_fork/lidar", name="fork_lidar")

        # for name in ["left_lidar","right_lidar"]:
        #     self._lidars[name] = LidarRtx(prim_path="", name=name)

    def get_camera_rgb(self, name: str):
        return self._cameras[name].get_rgb(device="cpu")

    def get_camera_depth(self, name: str):
        return self._cameras[name].get_depth(device="cpu")

    def get_camera_semantics(self, name: str):
        camera_current_frame = self._cameras[name].get_current_frame()
        # semantic_data = camera_current_frame["semantic_segmentation"]
        # if semantic_data is None:
        #     return None
        # return semantic_data["data"], semantic_data["info"]
        return camera_current_frame["semantic_segmentation"]



import rclpy
from rclpy.node import Node
import sensor_msgs.msg


class Test(Node):
    def __init__(self) -> None:
        super().__init__('test_node')
        config = loadConfig("configs/e_test_2.yaml")
        self._world: World = World()
        self._world.reset()

        self._forklift = Vehicle(self._world, config["vehicle"])
        self._sensors = SensorSet()

        self._simu_timer = SimTimer(self._world)
        self._executor = ThreadPoolExecutor(max_workers=16)  # 程序全局创建一次

        self._fork_rgb_pub = self.create_publisher(msg_type=sensor_msgs.msg.Image, topic="fork_rgb", qos_profile=2)
        self._fork_depth_pub = self.create_publisher(msg_type=sensor_msgs.msg.Image, topic="fork_depth", qos_profile=2)
        self._fork_semantics_pub = self.create_publisher(msg_type=sensor_msgs.msg.Image,
                                                         topic="fork_semantics",
                                                         qos_profile=2)


        self._left_rgb_pub = self.create_publisher(msg_type=sensor_msgs.msg.Image, topic="left_rgb", qos_profile=2)
        self._left_depth_pub = self.create_publisher(msg_type=sensor_msgs.msg.Image, topic="left_depth", qos_profile=2)
        self._left_semantics_pub = self.create_publisher(msg_type=sensor_msgs.msg.Image,
                                                         topic="left_semantics",
                                                         qos_profile=2)

        self._right_rgb_pub = self.create_publisher(msg_type=sensor_msgs.msg.Image, topic="right_rgb", qos_profile=2)
        self._right_depth_pub = self.create_publisher(msg_type=sensor_msgs.msg.Image,
                                                      topic="right_depth",
                                                      qos_profile=2)
        self._right_semantics_pub = self.create_publisher(msg_type=sensor_msgs.msg.Image,
                                                          topic="right_semantics",
                                                          qos_profile=2)

    def _to_img_msg(self, arr: np.ndarray, encoding="rgb8"):
        h, w = arr.shape[:2]
        msg = sensor_msgs.msg.Image()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.height = int(h)
        msg.width = int(w)
        msg.encoding = encoding           # 或 'bgr8'，看下你的数组通道顺序
        msg.is_bigendian = 0
        if len(arr.shape) == 3:
            msg.step = int(w * arr.shape[2])  # bytes per row
        elif encoding == "mono16":
            msg.step = int(w * 2)
        else:
            msg.step = arr.shape[1]
        msg.data = arr.tobytes()
        return msg

    async def _pub_images(self):
        pub_period = 0.02
        next_time = time.perf_counter() + pub_period

        has_logged = False
        # loop = asyncio.get_event_loop()
        while simulation_app.is_running():
            # now_ = time.time()
            fork_rgb = self._sensors.get_camera_rgb("fork_camera")
            left_rgb = self._sensors.get_camera_rgb("left_camera")
            right_rgb = self._sensors.get_camera_rgb("right_camera")

            if fork_rgb is None or left_rgb is None or right_rgb is None:
                await asyncio.sleep(0.01)
                continue

            fork_depth = self._sensors.get_camera_depth("fork_camera")
            left_depth = self._sensors.get_camera_depth("left_camera")
            right_depth = self._sensors.get_camera_depth("right_camera")

            if left_depth is None or right_depth is None or fork_depth is None:
                await asyncio.sleep(0.01)
                continue

            fork_semantics = self._sensors.get_camera_semantics("fork_camera")
            left_semantics = self._sensors.get_camera_semantics("left_camera")
            right_semantics = self._sensors.get_camera_semantics("right_camera")

            if left_semantics is None or right_semantics is None or fork_semantics is None:
                await asyncio.sleep(0.01)
                continue

            fork_semantic_img:np.ndarray = fork_semantics["data"]
            fork_semantic_labels: dict = fork_semantics["info"]

            left_semantic_img: np.ndarray = left_semantics["data"]
            left_semantic_labels: dict = left_semantics["info"]

            right_semantic_img: np.ndarray = right_semantics["data"]
            right_semantic_labels: dict = right_semantics["info"]

            timestamp = self.get_clock().now().to_msg()

            fork_rgb_msg = self._to_img_msg(fork_rgb)
            left_rgb_msg = self._to_img_msg(left_rgb)
            right_rgb_msg = self._to_img_msg(right_rgb)

            fork_depth = np.clip(fork_depth, 0, 65535)
            left_depth = np.clip(left_depth, 0, 65535)
            right_depth = np.clip(right_depth, 0, 65535)

            fork_depth_msg = self._to_img_msg((fork_depth * 1000).astype(np.uint16), "mono16")
            left_depth_msg = self._to_img_msg((left_depth * 1000).astype(np.uint16), "mono16")
            right_depth_msg = self._to_img_msg((right_depth * 1000).astype(np.uint16), "mono16")
            fork_semantic_msg = self._to_img_msg(fork_semantic_img.astype(np.uint8), encoding="mono8")
            left_semantic_msg = self._to_img_msg(left_semantic_img.astype(np.uint8), encoding="mono8")
            right_semantic_msg = self._to_img_msg(right_semantic_img.astype(np.uint8), encoding="mono8")

            # left_rgb_msg.header.stamp = timestamp
            # right_rgb_msg.header.stamp = timestamp
            fork_depth_msg.header.stamp = timestamp
            left_depth_msg.header.stamp = timestamp
            right_depth_msg.header.stamp = timestamp
            fork_semantic_msg.header.stamp = timestamp
            left_semantic_msg.header.stamp = timestamp
            right_semantic_msg.header.stamp = timestamp

            self._fork_rgb_pub.publish(fork_rgb_msg)
            self._left_rgb_pub.publish(left_rgb_msg)
            self._right_rgb_pub.publish(right_rgb_msg)
            self._fork_depth_pub.publish(fork_depth_msg)
            self._left_depth_pub.publish(left_depth_msg)
            self._right_depth_pub.publish(right_depth_msg)
            self._fork_semantics_pub.publish(fork_semantic_msg)
            self._left_semantics_pub.publish(left_semantic_msg)
            self._right_semantics_pub.publish(right_semantic_msg)


            now = time.perf_counter()
            sleep_duration = next_time - now
            if sleep_duration > 0:
                await asyncio.sleep(sleep_duration)
            else:
                await asyncio.sleep(0.005)
            next_time += pub_period

            if not has_logged:
                print(f"fork_semantic_labels: {fork_semantic_labels}")
                print(f"left_semantic_labels: {left_semantic_labels}")
                print(f"right_semantic_labels: {right_semantic_labels}")
                has_logged = True


    # async def _simulate(self):
    #     try:
    #         require_reset = False
    #         while simulation_app.is_running():
    #             self._world.step(render=True)
    #             if self._world.is_stopped() and not require_reset:   # 播放/暂停与重置逻辑
    #                 require_reset = True

    #             if self._world.is_playing():
    #                 if require_reset:
    #                     self._world.reset()
    #                     require_reset = False

    #             await asyncio.sleep(0.01)
    #     except Exception as e:
    #         print(e)

    def _simulate(self):
        require_reset = False
        while simulation_app.is_running():
            self._world.step(render=True)
            if self._world.is_stopped() and not require_reset:   # 播放/暂停与重置逻辑
                require_reset = True

            if self._world.is_playing():
                if require_reset:
                    self._world.reset()
                    require_reset = False
            # simulation_app.update()


    def run(self):
        run_coroutine(self._pub_images())
        # print(self._sensors._cameras["left_camera"].get_intrinsics_matrix())
        self._simulate()
        self.destroy_node()
        self._executor.shutdown()
        simulation_app.close()


if __name__ == "__main__":
    rclpy.init()
    test = Test()
    test.run()
    rclpy.shutdown()
