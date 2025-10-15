import numpy as np
import math
import transforms3d as tf
from scipy.spatial.transform import Rotation


class Pose:
    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0, roll: float = 0.0, pitch: float = 0.0,
                 yaw: float = 0.0):
        """
        位姿，包含位移信息和旋转信息

        :param x: 长度
        :param y:
        :param z:
        :param roll: 角度
        :param pitch:
        :param yaw:
        """
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.roll = float(roll)
        self.pitch = float(pitch)
        self.yaw = float(yaw)

    def update(self, x: float, y: float, z: float, axis_x: float, axis_y: float, axis_z: float, w: float):
        """
        位姿，包含位移信息和旋转信息

        :param x: 长度
        :param y:
        :param z:
        :param roll: 角度
        :param pitch:
        :param yaw:
        """
        # 归一化旋转轴
        axis = np.array([axis_x, axis_y, axis_z])
        axis = np.array(axis) / np.linalg.norm(axis)
        # 创建旋转对象
        r = Rotation.from_rotvec(w * axis)
        # 获取欧拉角（以弧度为单位）
        euler_angles_rad = r.as_euler('xyz', degrees=True)
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.roll = float(euler_angles_rad[0])
        self.pitch = float(euler_angles_rad[1])
        self.yaw = float(euler_angles_rad[2])

    def printself(self, name: str = "pose"):
        print(name, " : x=", self.x, " y=", self.y, " z=", self.z, \
              " roll=", self.roll, " pitch=", self.pitch, " yaw=", self.yaw)


class PoseTrans:
    @staticmethod
    def eulerAngles2rotationMat(theta, format='degree'):
        """
        Calculates Rotation Matrix given euler angles.
        :param theta: 1-by-3 list [rx, ry, rz] angle in degree
        :return:
        RPY角，是ZYX欧拉角，依次 绕定轴XYZ转动[rx, ry, rz]
        """
        if format == 'degree':
            theta = [i * math.pi / 180.0 for i in theta]

        R_x = np.array([[1, 0, 0],
                        [0, math.cos(theta[0]), -math.sin(theta[0])],
                        [0, math.sin(theta[0]), math.cos(theta[0])]
                        ])

        R_y = np.array([[math.cos(theta[1]), 0, math.sin(theta[1])],
                        [0, 1, 0],
                        [-math.sin(theta[1]), 0, math.cos(theta[1])]
                        ])

        R_z = np.array([[math.cos(theta[2]), -math.sin(theta[2]), 0],
                        [math.sin(theta[2]), math.cos(theta[2]), 0],
                        [0, 0, 1]
                        ])
        R = np.dot(R_z, np.dot(R_y, R_x))  # xyz 定轴
        # R = np.dot(R_x, np.dot(R_y, R_z))  # xyz 动轴
        return R

    @staticmethod
    def get_a_c_pose(a_b: Pose, b_c: Pose) -> Pose:
        """
        已知 a 坐标系下 b 位姿，已知 b 坐标系下 c 位姿，获取 a 坐标系下 c 点位姿

        :param a_b:
        :param b_c:
        :return:
        """
        a_c = Pose()
        r_a_b = PoseTrans.eulerAngles2rotationMat([a_b.roll, a_b.pitch, a_b.yaw])
        t_a_b = np.array([a_b.x, a_b.y, a_b.z])
        r_b_c = PoseTrans.eulerAngles2rotationMat([b_c.roll, b_c.pitch, b_c.yaw])
        t_b_c = np.array([b_c.x, b_c.y, b_c.z])

        r_a_c = np.dot(r_a_b, r_b_c)
        t_a_c = np.dot(r_a_b, t_b_c) + t_a_b

        a_c.x, a_c.y, a_c.z = t_a_c

        a_c.roll, a_c.pitch, a_c.yaw = np.degrees(tf.euler.mat2euler(r_a_c, 'sxyz'))
        return a_c

    @staticmethod
    def get_c_b_pose(a_b: Pose, a_c: Pose):
        """
        已知 a 坐标系下 b 位姿，已知 a 坐标系下 c 位姿，获取 c 坐标系下 b 点位姿
        当 a_b 全为 0 时，可以获取 c 坐标系下 a 位姿

        :param a_b:
        :param a_c:
        :return:
        """
        # 将位移向量和欧拉角转换为 numpy 数组
        a_b_translation = np.array([a_b.x, a_b.y, a_b.z])
        a_b_euler = np.array([a_b.roll, a_b.pitch, a_b.yaw])
        a_c_translation = np.array([a_c.x, a_c.y, a_c.z])
        a_c_euler = np.array([a_c.roll, a_c.pitch, a_c.yaw])

        # 构造旋转矩阵
        a_b_rotation_matrix = Rotation.from_euler('xyz', a_b_euler, degrees=True).as_matrix()
        a_c_rotation_matrix = Rotation.from_euler('xyz', a_c_euler, degrees=True).as_matrix()

        # 计算点 A 相对于以点 B 为原点的坐标系中的位移
        a_b_relative_translation = a_b_translation - a_c_translation

        # 计算点 A 相对于以点 B 为原点的坐标系中的旋转矩阵
        relative_rotation_matrix = np.dot(a_c_rotation_matrix.T, a_b_rotation_matrix)

        # 将旋转矩阵转换为欧拉角
        relative_euler = Rotation.from_matrix(relative_rotation_matrix).as_euler('xyz', degrees=True)
        c_b = Pose(x=a_b_relative_translation.tolist()[0],
                   y=a_b_relative_translation.tolist()[1],
                   z=a_b_relative_translation.tolist()[2],
                   roll=relative_euler.tolist()[0],
                   pitch=relative_euler.tolist()[1],
                   yaw=relative_euler.tolist()[2])

        return c_b

    @staticmethod
    def get_pallet2agv_pose(pallet_front: Pose, pallet2world: Pose, agv2world: Pose):
        """
        获取托盘前端面在agv坐标系下的位姿， 即感知结果
        pallet_front:托盘前端面在托盘坐标系下的位姿
        paller2world：托盘在世界坐标系下的位姿
        agv2world：叉车在世界坐标系下的位姿
        """
        # 将位移向量和欧拉角转换为numpy数组
        pallet_front_translation = np.array([pallet_front.x, pallet_front.y, pallet_front.z])
        pallet_front_euler = np.array([pallet_front.roll, pallet_front.pitch, pallet_front.yaw])

        pallet2world_translation = np.array([pallet2world.x, pallet2world.y, pallet2world.z])
        pallet2world_euler = np.array([pallet2world.roll, pallet2world.pitch, pallet2world.yaw])

        agv2world_translation = np.array([agv2world.x, agv2world.y, agv2world.z])
        agv2world_euler = np.array([agv2world.roll, agv2world.pitch, agv2world.yaw])

        # print("pallet_front_translation:", pallet_front_translation)
        # print("pallet2world_translation:", pallet2world_translation)
        # print("agv2world_translation:", agv2world_translation)

        # 构造旋转矩阵
        pallet_front_rotation_matrix = Rotation.from_euler('xyz', pallet_front_euler, degrees=True).as_matrix()
        pallet2world_rotation_matrix = Rotation.from_euler('xyz', pallet2world_euler, degrees=True).as_matrix()
        agv2world_rotation_matrix = Rotation.from_euler('xyz', agv2world_euler, degrees=True).as_matrix()
        # print("pallet_front_rotation_matrix:", pallet_front_rotation_matrix)
        # print("pallet2world_rotation_matrix:", pallet2world_rotation_matrix)
        # print("agv2world_rotation_matrix:", agv2world_rotation_matrix)

        relative_rotation_matrix = np.dot(pallet2world_rotation_matrix, pallet_front_rotation_matrix)
        relative_rotation_matrix = np.dot(agv2world_rotation_matrix.T, relative_rotation_matrix)

        relative_translation = np.dot(pallet2world_rotation_matrix, pallet_front_translation) + pallet2world_translation
        # print("relative_translation:", relative_translation)
        # relative_translation = np.dot(agv2world_rotation_matrix.T, relative_translation) - agv2world_translation
        relative_translation = np.dot(agv2world_rotation_matrix.T, relative_translation - agv2world_translation)
        # print("relative_translation:", relative_translation)

        # 将旋转矩阵转换为欧拉角
        relative_euler = Rotation.from_matrix(relative_rotation_matrix).as_euler('xyz', degrees=True)
        pallet2agv = Pose(
            x=relative_translation.tolist()[0],
            y=relative_translation.tolist()[1],
            z=relative_translation.tolist()[2],
            roll=relative_euler.tolist()[0],
            pitch=relative_euler.tolist()[1],
            yaw=relative_euler.tolist()[2])
        return pallet2agv

    @staticmethod
    def axis_angle_to_euler(axis, angle):
        # 归一化旋转轴
        axis = np.array(axis) / np.linalg.norm(axis)
        # 创建旋转对象
        r = Rotation.from_rotvec(angle * axis)
        # 获取欧拉角（以弧度为单位）
        euler_angles_rad = r.as_euler('xyz', degrees=False)
        # 将欧拉角转换为度
        euler_angles_deg = np.rad2deg(euler_angles_rad)
        return euler_angles_deg

    @staticmethod
    def axis_angle_to_rotation_matrix(axis, angle):
        # 归一化旋转轴
        axis = np.array(axis) / np.linalg.norm(axis)
        # 创建旋转对象
        r = Rotation.from_rotvec(angle * axis)
        # 转为旋转矩阵
        rotation_matrix = r.as_matrix()
        return rotation_matrix

    @staticmethod
    def euler_to_rotation_matrix(euler_angles):
        # 创建一个rotation对象
        r = Rotation.from_euler('xyz', euler_angles, degrees=True)
        # 将欧拉角转为旋转矩阵
        rotation_matrix = r.as_matrix()
        return rotation_matrix

    @staticmethod
    def pointcloud_trans(pc: np.ndarray, extrinsic: Pose):
        assert (pc.shape[-1], 3)

        # 平移向量 欧拉角
        agv_translation = np.array([extrinsic.x, extrinsic.y, extrinsic.z])
        agv_euler = np.array([extrinsic.roll, extrinsic.pitch, extrinsic.yaw])

        # 旋转矩阵
        agv_rotation_matrix = Rotation.from_euler('xyz', agv_euler, degrees=True).as_matrix()

        # 点云旋转
        pc_trans = np.dot(pc, agv_rotation_matrix.T) + np.expand_dims(agv_translation, 0).repeat(pc.shape[0], axis=0)

        return pc_trans

    @staticmethod
    def quaternion_to_euler(x, y, z, w, order='xyz', degrees=False):
        """
        使用scipy将四元数转换为欧拉角

        参数:
            x, y, z, w: 四元数分量 (x, y, z, w)
            order: 旋转顺序，默认为'xyz'，支持所有常见顺序
            degrees: 是否返回角度（True）还是弧度（False），默认False

        返回:
            三个欧拉角（滚转角roll, 俯仰角pitch, 偏航角yaw）
        """
        # 创建Rotation对象，注意scipy默认四元数格式是(w, x, y, z)
        rotation = Rotation.from_quat([x, y, z, w])  # from_quat接受[x, y, z, w]格式

        # 转换为欧拉角
        euler_angles = rotation.as_euler(order, degrees=degrees)

        return euler_angles[0], euler_angles[1], euler_angles[2]  # roll, pitch, yaw