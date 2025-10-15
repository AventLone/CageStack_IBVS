from isaacsim.sensors.camera import Camera
from typing import Optional
import numpy as np
import isaacsim.core.utils.numpy.rotations as rot_utils
from pxr import Gf
import math


class VN_Camera:
    """
    camera
    """
    def __init__(self, prim_path : str, config : Optional[dict]):
        """
        camera init
        :param prim_path:
        """
        self.prim_path = prim_path
        self.camera = Camera(prim_path=self.prim_path)
        self.camera.initialize()
        self.config_camera(config)

    def config_camera(self, config : Optional[dict]):
        """
        配置相机参数
        :param config: 配置字典，可包含width, height, fov, position, rotation等键
        """
        if not config:
            return
            
        # 设置分辨率
        if 'width' in config and 'height' in config:
            self.width = config['width']
            self.height = config['height']
            self.camera.set_resolution((config['width'], config['height']))
            print(f"相机分辨率设置为: {config['width']}x{config['height']}")

        if "fps" in config:
            self.camera.set_frequency(config['fps'])

        if "camera_matrix" in config:
            (self.fx, _, self.cx), (_, self.fy, self.cy), (_, _, _) = config["camera_matrix"]


        # 设置位置
        if 'position' in config:
            # 位置应是一个包含x, y, z的列表或元组
            position = np.array(config['position'], dtype=np.float32)
            if np.isnan(position).any():
                return
            self.camera.set_local_pose(translation=position,camera_axes="usd")
            #self.camera.set_local_pose(position=position)
            print(f"相机位置设置为: {position}")


        if "orientation" in config:
            rotation = np.array(config['orientation'], dtype=np.float32)
            if np.isnan(rotation).any():
                return
            R = [rotation[1], rotation[2], rotation[0]]   #循序YZX
            rotation2orientation = rot_utils.euler_angles_to_quats(R , degrees=True)  #欧拉转四元数
            orientation_wxyz= np.zeros(4)
            orientation_wxyz[0]= rotation2orientation[3]
            orientation_wxyz[1] = rotation2orientation[0]
            orientation_wxyz[2] = rotation2orientation[1]
            orientation_wxyz[3] = rotation2orientation[2]

            self.camera.set_local_pose(orientation=orientation_wxyz,camera_axes="usd")
            print(f"相机旋转设置为: {rotation}")


        if "vertical_aperture" in config:
            vertical_aperture = config['vertical_aperture']
            if np.isnan(vertical_aperture):
                return
            self.camera.set_vertical_aperture(vertical_aperture)

        if "focal_length" in config:
            self.focal_length = np.array(config['set_focal_length'], dtype=np.float32)
            self.camera.set_focal_length(self.focal_length)

        if "focus_distance" in config:
            # 1. 读取配置值（确保是数值或可转换为数值的类型）
            focus_distance = config['focus_distance']

            # 2. 处理可能的序列类型（如列表/数组）
            if isinstance(focus_distance, (list, np.ndarray)):
                focus_distance = focus_distance[0]  # 取第一个元素

            # 3. 转换为原生Python浮点数（关键修复）
            try:
                focus_distance = float(focus_distance)
            except ValueError:
                raise TypeError(f"focus_distance必须是数值类型，实际为{type(focus_distance)}")

            # 4. 调用相机接口（此时参数为原生float，符合要求）
            self.camera.set_focus_distance(focus_distance)
        if "pixel_size" in config:
            self.pixel_size = np.array(config['pixel_size'], dtype=np.float32)
            if 'width' in config and 'height' in config:
                self.horizontal_aperture = self.pixel_size * self.width * 1e-6
                self.vertical_aperture = self.pixel_size * self.height * 1e-6
                self.camera.set_horizontal_aperture( self.horizontal_aperture )
                self.camera.set_vertical_aperture(self.vertical_aperture)



            # 设置视场角 (FOV)
        if 'set_hfov' in config:
            # 假设set_fov方法接受角度值
            hfov = config['set_hfov']
            target_hfov_rad = math.radians(hfov)
            new_focal_length = self.horizontal_aperture / (2 * math.tan(target_hfov_rad / 2))
            self.fx = new_focal_length / (self.pixel_size * 1e-6)
            self.fy = self.fx * (self.height / self.width)

        if "set_focal_length" in config:
            set_focal_length = config['set_focal_length']
            if set_focal_length:
                focal_length_x = self.pixel_size * self.fx * 1e-6
                focal_length_y = self.pixel_size * self.fy * 1e-6  # convert to meters
                focal_length = (focal_length_x + focal_length_y) / 2  # convert to meters
                self.camera.set_focal_length(focal_length)
        if "distortion_coefficients" in config:
            self.distortion_coefficients = config['distortion_coefficients']
            if "distortion_way" in config:
                if config["distortion_way"] == "fish":
                    self.camera.set_opencv_fisheye_properties(cx=self.cx, cy=self.cy, fx=self.fx, fy=self.fy,
                                                              fisheye=self.distortion_coefficients)
                elif config["distortion_way"] == "pinhole":
                    self.camera.set_opencv_pinhole_properties(cx=self.cx, cy=self.cy,fx=self.fx, fy=self.fy,
                                                              pinhole=self.distortion_coefficients)



        if "f_stop" in config:
            f_stop = config['f_stop']
            if not isinstance(f_stop, (int, float)):
                return
            self.camera.set_lens_aperture(f_stop)




    def add_frame(self):
        self.camera.add_distance_to_image_plane_to_frame()
        self.camera.add_semantic_segmentation_to_frame()
        self.camera.add_instance_id_segmentation_to_frame()
        self.camera.add_instance_segmentation_to_frame()
        self.camera.add_pointcloud_to_frame()
        self.camera.add_normals_to_frame()

    def get_rendering_time(self):
        """
        :return:
        """
        camera_current_frame = self.camera.get_current_frame()
        # print(camera_current_frame)
        rendering_time = camera_current_frame['rendering_time']
        return rendering_time

    def get_pointcloud(self):
        pointcloud = self.camera.get_pointcloud()
        return pointcloud

    def get_depth(self):
        camera_current_frame = self.camera.get_current_frame()
        depth = camera_current_frame['depth']
        return depth

    def get_normals(self):
        camera_current_frame = self.camera.get_current_frame()
        normals = camera_current_frame['normals']
        return normals

    def get_instance_segmentation(self):
        camera_current_frame = self.camera.get_current_frame()
        instance_segmentation = camera_current_frame['instance_segmentation']
        return instance_segmentation

    def get_semantic_segmentation(self):
        camera_current_frame = self.camera.get_current_frame()
        semantic_segmentation = camera_current_frame['semantic_segmentation']
        return semantic_segmentation

    def get_color(self):
        color = self.camera.get_rgba()
        return color

    def get_depth(self):
        depth = self.camera.get_depth()
        return depth

    def get_world_position(self):
        """获取相机当前世界位姿"""
        position, orientation = self.camera.get_world_pose()
        return position, orientation

    def get_local_position(self, camera_axi:str):
        """获取相机相對父級位姿"""
        """Gets prim's pose with respect to the local frame (the prim's parent frame in the world axes).

                Args:
                    camera_axes (str, optional): camera axes, world is (+Z up, +X forward), ros is (+Y up, +Z forward) and usd is (+Y up and -Z forward). Defaults to "world".

                Returns:
                    Tuple[np.ndarray, np.ndarray]: first index is position in the local frame of the prim. shape is (3, ).
                                                   second index is quaternion orientation in the local frame of the prim.
                                                   quaternion is scalar-first (w, x, y, z). shape is (4, ).
        """
        translate,orientation = self.camera.get_local_position(camera_axi=camera_axi)
        return translate, orientation
        return translate,orientation

