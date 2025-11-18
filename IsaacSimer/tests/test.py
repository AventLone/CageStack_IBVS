from utils.common import openSimuApp, loadConfig
simulation_app = openSimuApp("configs/e_test_2.yaml")

# Settings
import carb.settings
# Set DLSS to Quality mode (2) for best SDG results (Options: 0 (Performance), 1 (Balanced), 2 (Quality), 3 (Auto)
carb.settings.get_settings().set("rtx/post/dlss/execMode", 2)

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
from concurrent.futures import ThreadPoolExecutor, Future


class SensorSet:
    def __init__(self) -> None:
        self._cameras: dict[str, Camera] = dict()
        self._lidars: dict[str, LidarRtx] = dict()
        for name in ["left_camera", "right_camera"]:
            self._cameras[name] = Camera(
                prim_path=f"/World/E_car_finish/body/cameras/{name}/SG3S_ISX031C_GMSL2F_H190XA_01",
                name=name,resolution=(960, 786))
            self._cameras[name].initialize()
            self._cameras[name].add_distance_to_image_plane_to_frame()
            self._cameras[name].add_semantic_segmentation_to_frame()

        # for name in ["left_lidar","right_lidar"]:
        #     self._lidars[name] = LidarRtx(prim_path="", name=name)

    def get_camera_rgb(self, name: str):
        return self._cameras[name].get_rgb(device="cpu")

    def get_camera_depth(self, name: str):
        return self._cameras[name].get_depth(device="cpu")

    def get_camera_semantics(self, name: str):
        camera_current_frame = self._cameras[name].get_current_frame()
        semantic_data = camera_current_frame["semantic_segmentation"]
        return semantic_data["data"], semantic_data["info"]


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
        self._executor = ThreadPoolExecutor(max_workers=3)  # 程序全局创建一次

        self._left_rgb_pub = self.create_publisher(msg_type=sensor_msgs.msg.Image, topic="left_rgb", qos_profile=1)
        self._left_depth_pub = self.create_publisher(msg_type=sensor_msgs.msg.Image, topic="left_depth", qos_profile=1)
        self._left_semantics_pub = self.create_publisher(msg_type=sensor_msgs.msg.Image, topic="left_semantics", qos_profile=1)

        self._right_rgb_pub = self.create_publisher(msg_type=sensor_msgs.msg.Image, topic="right_rgb", qos_profile=1)
        self._right_depth_pub = self.create_publisher(msg_type=sensor_msgs.msg.Image, topic="right_depth", qos_profile=1)
        self._right_semantics_pub = self.create_publisher(msg_type=sensor_msgs.msg.Image, topic="right_semantics", qos_profile=1)

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
        else:
            msg.step = arr.shape[1]
        msg.data = arr.tobytes()
        return msg
    
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
        else:
            msg.step = arr.shape[1]
        msg.data = arr.tobytes()
        return msg

    async def _pub_images(self):
        while simulation_app.is_running():
            left_rgb = self._sensors.get_camera_rgb("left_camera")
            right_rgb = self._sensors.get_camera_rgb("right_camera")

            left_depth = self._sensors.get_camera_depth("left_camera")
            right_depth = self._sensors.get_camera_depth("right_camera")

            if left_rgb is None or right_rgb is None:
                await asyncio.sleep(0.01)
                continue
            # left_semantics, _ = self._sensors.get_camera_semantics("left_camera")
            # right_semantics, _ = self._sensors.get_camera_semantics("right_camera")
            self._left_rgb_pub.publish(self._to_img_msg(left_rgb))
            self._right_rgb_pub.publish(self._to_img_msg(right_rgb))
            self._left_depth_pub.publish(self._to_img_msg(left_depth, "32FC1"))

            print(f"left_depth.dtype {left_depth.dtype}, left_depth.shape {left_depth.shape}, len(left_depth.shape) {len(left_depth.shape)}")

            await asyncio.sleep(1.2)

    def run(self):
        # run_coroutine(self._collect_imgs())
        # self._executor.submit(self._show_img)
        run_coroutine(self._pub_images())

        require_reset = False
        while simulation_app.is_running():
            self._world.step(render=True)
            if self._world.is_stopped() and not require_reset:   # 播放/暂停与重置逻辑
                require_reset = True

            if self._world.is_playing():
                if require_reset:
                    self._world.reset()
                    require_reset = False

        self._executor.shutdown()
        simulation_app.close()



if __name__ == "__main__":
    rclpy.init()
    test = Test()
    test.run()
    rclpy.shutdown()
