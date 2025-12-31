from utils.common import open_simu_app, load_config
simulation_app = open_simu_app("configs/e_test_2.yaml")

# Settings
import carb.settings
# Set DLSS to Quality mode (2) for best SDG results (Options: 0 (Performance), 1 (Balanced), 2 (Quality), 3 (Auto)
carb.settings.get_settings().set("rtx/post/dlss/execMode", 0)

from isaacsim.core.utils.extensions import enable_extension
enable_extension("isaacsim.ros2.bridge")   # enable ROS2 bridge extension
simulation_app.update()

import asyncio, time
from itertools import chain
import numpy as np
from isaacsim.core.api import World

from devices.vehicles import Vehicle
from utils.common import SimTimer
from isaacsim.sensors.rtx.impl import LidarRtx
from isaacsim.sensors.camera.camera import Camera
from omni.kit.async_engine import run_coroutine


class SensorSet:
    def __init__(self) -> None:
        self._cameras: dict[str, Camera] = dict()

        for name in ["left_camera", "right_camera", "fork_camera"]:
            if name == "fork_camera":
                self._cameras[name] = Camera(
                    prim_path=f"/World/E_car_finish/body/center_camera/SG3S_ISX031C_GMSL2F_H190XA_01",
                    name=name,resolution=(960, 786))
            else:
                self._cameras[name] = Camera(
                    prim_path=f"/World/E_car_finish/body/cameras/{name}/SG3S_ISX031C_GMSL2F_H190XA_01",
                    name=name,resolution=(960, 786))
            self._cameras[name].initialize()
            self._cameras[name].add_distance_to_image_plane_to_frame()
            self._cameras[name].add_semantic_segmentation_to_frame()

    def get_camera_rgb(self, name: str):
        return self._cameras[name].get_rgb(device="cpu")

    def get_camera_depth(self, name: str):
        return self._cameras[name].get_depth(device="cpu")

    def get_camera_semantics(self, name: str):
        camera_current_frame = self._cameras[name].get_current_frame()
        return camera_current_frame["semantic_segmentation"]



import rclpy
from rclpy.node import Node
import sensor_msgs.msg, std_msgs.msg


class Test(Node):
    def __init__(self) -> None:
        super().__init__('IsaacSim')
        config = load_config("configs/e_test_2.yaml")
        self._world: World = World()
        self._world.reset()

        self._forklift = Vehicle(self._world, config["vehicle"])
        self._sensors = SensorSet()

        self._simu_timer = SimTimer(self._world)

        self._fork_rgb_pub = self.create_publisher(msg_type=sensor_msgs.msg.Image, topic="fork_rgb", qos_profile=2)
        self._fork_depth_pub = self.create_publisher(msg_type=sensor_msgs.msg.Image, 
                                                     topic="sensor/camera/fork/depth", qos_profile=2)
        self._fork_semantics_pub = self.create_publisher(msg_type=sensor_msgs.msg.Image,
                                                         topic="sensor/camera/fork/semantics",
                                                         qos_profile=2)


        self._left_rgb_pub = self.create_publisher(msg_type=sensor_msgs.msg.Image, topic="left_rgb", qos_profile=2)
        self._left_depth_pub = self.create_publisher(msg_type=sensor_msgs.msg.Image, 
                                                     topic="sensor/camera/left/depth", qos_profile=2)
        self._left_semantics_pub = self.create_publisher(msg_type=sensor_msgs.msg.Image,
                                                         topic="sensor/camera/left/semantics", qos_profile=2)

        self._right_rgb_pub = self.create_publisher(msg_type=sensor_msgs.msg.Image, topic="right_rgb", qos_profile=2)
        self._right_depth_pub = self.create_publisher(msg_type=sensor_msgs.msg.Image,
                                                      topic="sensor/camera/right/depth", qos_profile=2)
        self._right_semantics_pub = self.create_publisher(msg_type=sensor_msgs.msg.Image,
                                                          topic="sensor/camera/right/semantics",
                                                          qos_profile=2)
        
        self._cmd_sub = self.create_subscription(msg_type=std_msgs.msg.Float64MultiArray, 
                                                  topic="/control/cmds",
                                                  callback=self._cmd_handler,
                                                  qos_profile=2)
        
    def _cmd_handler(self, cmd_msg: std_msgs.msg.Float64MultiArray):
        drive_velocity = cmd_msg.data[0]
        steer_velocity = cmd_msg.data[1]
        lift_velocity = cmd_msg.data[2]

        self._forklift.move(drive_velocity)
        self._forklift.steer(steer_velocity)
        self._forklift.moveFork([lift_velocity])

    async def _pub_images(self):
        def process_image(arr: np.ndarray, encoding="rgb8", scale=None):
            if scale is not None:
                arr = np.clip(arr * scale, 0, 65535).astype(np.uint16)
            h, w = arr.shape[:2]
            msg = sensor_msgs.msg.Image()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.height = int(h)
            msg.width = int(w)
            msg.encoding = encoding
            msg.is_bigendian = 0
            msg.step = w * arr.shape[2] if len(arr.shape) == 3 else (w * 2 if encoding == "mono16" else w)
            msg.data = arr.tobytes()
            return msg

        cameras = ["fork_camera", "left_camera", "right_camera"]
        is_logged = False
        while simulation_app.is_running():
            rgb_imgs, depth_imgs, semantics = list(), list(), list()

            for camera in cameras:
                rgb_imgs.append(self._sensors.get_camera_rgb(camera))
                depth_imgs.append(self._sensors.get_camera_depth(camera))
                semantics.append(self._sensors.get_camera_semantics(camera))

            if any(img is None for img in (rgb_imgs + depth_imgs + semantics)):
                await asyncio.sleep(0.005)
                continue

            timestamp = self.get_clock().now().to_msg()

            rgb_msg_list, depth_msg_list, semantic_msg_list = list(), list(), list()

            for i in range(3):
                rgb_msg_list.append(process_image(rgb_imgs[i]))
                depth_msg_list.append(process_image(depth_imgs[i], "mono16", 1000))
                semantic_msg_list.append(process_image(semantics[i]["data"].astype(np.uint8), "mono8"))

            for msg in chain(rgb_msg_list, depth_msg_list, semantic_msg_list):
                msg.header.stamp = timestamp

            self._fork_rgb_pub.publish(rgb_msg_list[0])
            self._left_rgb_pub.publish(rgb_msg_list[1])
            self._right_rgb_pub.publish(rgb_msg_list[2])
            self._fork_depth_pub.publish(depth_msg_list[0])
            self._left_depth_pub.publish(depth_msg_list[1])
            self._right_depth_pub.publish(depth_msg_list[2])
            self._fork_semantics_pub.publish(semantic_msg_list[0])
            self._left_semantics_pub.publish(semantic_msg_list[1])
            self._right_semantics_pub.publish(semantic_msg_list[2])

            if not is_logged:
                fork_semantics_info: dict = semantics[0]["info"]["idToLabels"]
                left_semantics_info: dict = semantics[1]["info"]["idToLabels"]
                right_semantics_info: dict = semantics[2]["info"]["idToLabels"]

                for key, value in fork_semantics_info.items():
                    if value == {"class" : "target_cage_post"} or value == {"class" : "target_cage_crossbeam"}:
                        print(f"fork {value} : {key}")

                for key, value in left_semantics_info.items():
                    if value == {"class" : "target_cage_post"}  or value == {"class" : "target_cage_crossbeam"}:
                        print(f"left {value} : {key}")

                for key, value in right_semantics_info.items():
                    if value == {"class" : "target_cage_post"}  or value == {"class" : "target_cage_crossbeam"}:
                        print(f"right {value} : {key}")

                is_logged = True

            await asyncio.sleep(0.01)


    def _simulate(self):
        require_reset = False
        while simulation_app.is_running():
            self._world.step(render=True)
            rclpy.spin_once(self, timeout_sec=0.01)
            if self._world.is_stopped() and not require_reset:
                require_reset = True
            if self._world.is_playing() and require_reset:
                self._world.reset()
                require_reset = False

    def run(self):
        run_coroutine(self._pub_images())
        self._simulate()
        self.destroy_node()
        simulation_app.close()


if __name__ == "__main__":
    rclpy.init()
    test = Test()
    test.run()
    rclpy.shutdown()
