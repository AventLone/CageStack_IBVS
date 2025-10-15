from devices import VN_Camera, VN_LidarRtx, VN_Mid360
import threading
import time

from utils import ConfigLoader, LoggerUtil, H5Writer
from utils.world_model import Pose
from typing import Optional
from pathlib import Path
import yaml
import numpy as np

class ST_SensorsThread(threading.Thread):
    """Sensors工作线程类，继承自threading.Thread"""
    def __init__(self, delay:float, config : Optional[dict]):
        super().__init__()
        self.delay = delay # s
        self.run_thread = True
        self.logger = LoggerUtil.get_logger("sensors")
        self.camera_keys = ['fork_camera', 'front_camera','top_camera']
        self.camera_devices = {}
        self.YAML_PATH = "../configs/st_vla.yaml"
        self.h5_writer = H5Writer(
            h5_file_path="camera_data.h5",  # 保存路径可从配置读取
            logger=self.logger
        )
        for camera_key in self.camera_keys:
            if camera_key in config.keys():
                #self.logger.info(f"add camera : {config['fork_camera']['prim_path']}")
                cam_config = self.load_single_camera_config(self.YAML_PATH, camera_key)
                camera= VN_Camera(prim_path=cam_config["prim_path"],config=cam_config["params"])
                self.camera_devices[camera_key] = camera

        self.pose_devices = {}
        if 'pose' in config:
            for pose_key, pose_config in config['pose'].items():
                self.logger.info(f"add pose sensor : {pose_config['prim_path']}")
                self.pose_devices[pose_key] = Pose(pose_config['prim_path'])

        # #  实例化MID360雷达
        self.mid360 = VN_Mid360(
            prim_path="/World/MID360_Lidar",  # 雷达在场景中的路径
            batch_size=20000,  # 每批射线数（根据性能调整）
            draw_points=True  # 启用点云可视化
        )

        self.mid360.start()



    def load_single_camera_config(self,yaml_path: str, camera_name: str) -> dict:
        """
        从YAML传感器配置中，提取单个相机的参数（prim_path + params）
        :param yaml_path: YAML配置文件路径
        :param camera_name: 目标相机名称（如"fork_camera"、"front_camera"）
        :return: 包含相机prim_path和params的字典
        """
        # 检查YAML文件是否存在
        yaml_file = Path(yaml_path)
        if not yaml_file.exists():
            raise FileNotFoundError(f"YAML配置文件不存在：{yaml_path}")

        # 读取YAML内容（传感器配置在"sensors"节点下）
        with open(yaml_file, "r", encoding="utf-8") as f:
            yaml_data = yaml.safe_load(f)

        # 校验传感器配置节点和目标相机是否存在
        sensors_config = yaml_data.get("sensors")
        if not sensors_config:
            raise KeyError("YAML中未找到'sensors'配置节点")

        camera_config = sensors_config.get(camera_name)
        if not camera_config:
            raise KeyError(f"YAML的sensors中未找到相机：{camera_name}")

        # 提取相机的prim_path和params，并处理参数映射（horizontal_fov -> fov）
        camera_params = camera_config.get("params", {})


        # 返回包含prim_path和处理后params的字典
        return {
            "prim_path": camera_config["prim_path"],
            "params": camera_params
        }

    def run(self) -> None:
        """线程启动时执行的方法"""
        print(f"start SensorsThread")
        while self.run_thread :
            start = time.perf_counter()
            multi_camera_frames =self.data_process()
            #self.h5_writer.save_multi_camera_data(multi_camera_frames)
            end = time.perf_counter()
            delay = self.delay - (end - start)
            if delay > 0:
                time.sleep(delay)
        self.run_thread = False

    def stop(self) -> None:
        self.run_thread = False

    def data_process(self):
        multi_camera_frames = []
        for key, camera in self.camera_devices.items():
            frame=camera.get_frame()
            try:
                # 统一提取图像数组
                if isinstance(frame, tuple):
                    rendering_time = frame[0]
                    img_array = frame[1]
                elif isinstance(frame, dict) and "rgb" in frame:
                    rendering_time = frame["rendering_time"]
                    img_array = frame["rgb"]
                else:
                    self.logger.warning(f"相机 {key} 帧格式未知，跳过")
                    continue

                # 后续操作（如打印 shape、转字节数据存 HDF5）
                print(f"相机 {key} 图像 shape：{img_array.shape}")
                img_bytes = img_array.tobytes()  # 转为字节数据，用于 HDF5 保存
  
                poses = {}
                for key, pose_sensor in self.pose_devices.items():
                    translation, orientation = pose_sensor.get_pose()
                    if translation is not None:
                        poses[key] = {'translation': translation, 'orientation': orientation}
                
                if 'vehicle_pose' in poses and 'pallet_pose' in poses:
                    # Simple relative position calculation (pallet_pos - vehicle_pos)
                    # This does not account for vehicle's rotation. 
                    # For a more accurate relative pose, a full transformation is needed.
                    vehicle_pos = np.array(poses['vehicle_pose']['translation'])
                    pallet_pos = np.array(poses['pallet_pose']['translation'])
                    relative_pos = pallet_pos - vehicle_pos
                    self.logger.info(f"Vehicle Pose: {poses['vehicle_pose']}")
                    self.logger.info(f"Pallet Pose: {poses['pallet_pose']}")
                    self.logger.info(f"Pallet relative position to vehicle: {relative_pos.tolist()}")

                pc = self.mid360.get_pointcloud()
                print("=======================the pointcloud is ", pc.points.shape)

                # 存入多相机列表
                multi_camera_frames.append({
                    "camera_key": key,
                    "rendering_time": rendering_time,
                    "img_bytes": img_bytes,
                    "width": img_array.shape[1],  # 图像宽（shape：高×宽×通道）
                    "height": img_array.shape[0],  # 图像高
                    "channels": img_array.shape[2]  # 通道数（如 RGBA 为4）
                })
            except Exception as e:
                self.logger.error(f"处理相机 {key} 数据失败：{str(e)}")

        return multi_camera_frames


class STController_VLA:
    def __init__(self, config : str):
        loader = ConfigLoader(config)
        self.config = loader.get()
        #print(self.config)
        self.sensors = ST_SensorsThread(0.1, self.config['sensors'])

    def start(self):
        self.sensors.start()

    def stop(self):
        self.sensors.stop()
        self.sensors.join()