import ecal.core.core as ecal_core
from ecal.core.publisher import StringPublisher
from ecal.core.media import ImagePublisher, PointCloudPublisher  
import numpy as np
import time
import cv2  # 用于 RGB 图像格式转换
from devices.camera import VN_Camera


def init_ecal_publishers():
    """初始化 eCAL 发布器：分别发布 RGB、深度、点云数据"""
    # 1. RGB 图像发布器（ecal 内置 ImagePublisher，支持 OpenCV 格式）
    rgb_pub = ImagePublisher("camera/rgb")  # 话题名：camera/rgb

    # 2. 深度图像发布器（转为字符串发布，或用 ImagePublisher 按单通道处理）
    # 此处选择 StringPublisher，将深度数据转为二进制字符串（节省带宽）
    depth_pub = StringPublisher("camera/depth")  # 话题名：camera/depth

    # 3. 点云发布器（ecal 内置 PointCloudPublisher，支持 PCL 格式）
    pointcloud_pub = PointCloudPublisher("camera/pointcloud")  # 话题名：camera/pointcloud

    print("eCAL 发布器初始化完成，话题列表：")
    print(f"- RGB: camera/rgb")
    print(f"- 深度: camera/depth")
    print(f"- 点云: camera/pointcloud")
    return rgb_pub, depth_pub, pointcloud_pub


def format_rgb_data(rgba_data):
    """
    格式化 RGB 数据：将 Isaac Sim 返回的 RGBA 转为 OpenCV 兼容的 BGR 格式
    :param rgba_data: VN_Camera.get_color() 返回的 RGBA 数组（shape: [height, width, 4]）
    :return: OpenCV 格式的 BGR 图像（shape: [height, width, 3]）
    """
    if rgba_data is None or rgba_data.size == 0:
        raise ValueError("RGB 数据为空，检查相机配置或采集逻辑")

    # 1. 去掉 Alpha 通道（RGBA -> RGB）
    rgb_data = rgba_data[..., :3]  # 取前 3 通道
    # 2. OpenCV 默认用 BGR 格式，需转换（RGB -> BGR）
    bgr_data = cv2.cvtColor(rgb_data, cv2.COLOR_RGB2BGR)
    return bgr_data


def format_depth_data(depth_data, scale=1000.0):
    """
    格式化深度数据：转为 16 位整数（mm 单位，便于存储和传输）
    :param depth_data: VN_Camera.get_depth() 返回的深度数组（shape: [height, width]，单位：m）
    :param scale: 米 -> 毫米的缩放因子（1m = 1000mm）
    :return: 二进制格式的 16 位深度数据
    """
    if depth_data is None or depth_data.size == 0:
        raise ValueError("深度数据为空，检查相机配置或采集逻辑")

    # 1. 过滤无效深度值（如 NaN/Inf）
    depth_data = np.nan_to_num(depth_data, nan=0.0, posinf=0.0, neginf=0.0)
    # 2. 米 -> 毫米，转为 16 位整数（避免浮点数精度损失）
    depth_mm = (depth_data * scale).astype(np.uint16)
    # 3. 转为二进制字符串（ecal 传输二进制数据更高效）
    return depth_mm.tobytes()


def format_pointcloud_data(pointcloud_data):
    """
    格式化点云数据：适配 eCAL PointCloudPublisher 要求的格式
    :param pointcloud_data: VN_Camera.get_pointcloud() 返回的点云数组（shape: [N, 3]，x/y/z 单位：m）
    :return: eCAL 兼容的点云对象（包含 x/y/z 坐标）
    """
    if pointcloud_data is None or pointcloud_data.shape[0] == 0:
        raise ValueError("点云数据为空，检查相机配置或采集逻辑")

    # 过滤无效点（如包含 NaN 的点）
    valid_mask = ~np.any(np.isnan(pointcloud_data), axis=1)
    valid_pointcloud = pointcloud_data[valid_mask]

    # eCAL PointCloudPublisher 要求输入 (N, 3) 的 float32 数组
    return valid_pointcloud.astype(np.float32)


def main():
    # -------------------------- 1. 初始化 eCAL --------------------------
    ecal_core.initialize([], "camera_ecal_publisher")
    if not ecal_core.ok():
        print("eCAL 初始化失败，退出程序")
        return

    # -------------------------- 2. 初始化相机 --------------------------
    # 相机配置字典（根据实际硬件参数调整！）
    camera_config = {
        "width": 1280,  # 分辨率宽度
        "height": 720,  # 分辨率高度
        "fps": 30,  # 帧率
        "position": [0.0, 0.0, 1.5],  # 相机世界位置（x, y, z，单位：m）
        "orientation": [0.0, 0.0, 0.0, 1.0],  # 四元数（x, y, z, w）
        "focal_length": 0.015,  # 焦距（单位：m，根据相机参数调整）
        "pixel_size": 3.75e-6,  # 像素尺寸（单位：m，如 3.75μm）
        "camera_matrix": [  # 内参矩阵（fx, cx; fy, cy; 0,0,1）
            [800.0, 0.0, 640.0],
            [0.0, 800.0, 360.0],
            [0.0, 0.0, 1.0]
        ],
        "distortion_way": "pinhole",  # 畸变模型（pinhole/fish）
        "distortion_coefficients": [0.0, 0.0, 0.0, 0.0, 0.0]  # 畸变系数
    }

    # 创建 VN_Camera 实例
    try:
        vn_camera = VN_Camera(
            prim_path="/World/Camera",  # Isaac Sim 相机 Prim 路径（需与场景匹配）
            config=camera_config
        )
        # 添加相机帧（启用深度、点云等数据采集）
        vn_camera.add_frame()
        print("相机初始化完成，开始采集数据...")
    except Exception as e:
        print(f"相机初始化失败：{e}")
        ecal_core.finalize()
        return

    # -------------------------- 3. 初始化 eCAL 发布器 --------------------------
    rgb_pub, depth_pub, pointcloud_pub = init_ecal_publishers()

    # -------------------------- 4. 循环采集并发布数据 --------------------------
    try:
        while ecal_core.ok():
            # 记录采集开始时间（控制帧率）
            start_time = time.time()

            # -------------------------- 采集相机数据 --------------------------
            # 1. 采集 RGB 图像
            rgba_data = vn_camera.get_color()
            bgr_data = format_rgb_data(rgba_data)

            # 2. 采集深度数据
            depth_data = vn_camera.get_depth()
            depth_bytes = format_depth_data(depth_data)

            # 3. 采集点云数据
            pointcloud_data = vn_camera.get_pointcloud()
            valid_pointcloud = format_pointcloud_data(pointcloud_data)

            # -------------------------- 发布数据到 eCAL --------------------------
            # 1. 发布 RGB 图像（ecal ImagePublisher 需传入 BGR 数据、宽度、高度）
            rgb_pub.send_image(
                image=bgr_data,
                width=camera_config["width"],
                height=camera_config["height"],
                encoding="bgr8"  # 明确编码格式（OpenCV 标准）
            )

            # 2. 发布深度数据（二进制字符串）
            depth_pub.send(depth_bytes)

            # 3. 发布点云数据（ecal PointCloudPublisher 直接传入 (N,3) 的 float32 数组）
            pointcloud_pub.send(valid_pointcloud)

            # -------------------------- 控制帧率 --------------------------
            elapsed_time = time.time() - start_time
            target_fps = camera_config["fps"]
            sleep_time = max(0, 1.0 / target_fps - elapsed_time)
            time.sleep(sleep_time)

            # 打印状态（每 10 帧打印一次，避免日志过多）
            if int(time.time()) % 10 == 0:
                print(f"数据发布中 | FPS: {1.0 / (time.time() - start_time):.1f} "
                      f"| 点云点数: {valid_pointcloud.shape[0]}")

    except KeyboardInterrupt:
        print("用户中断，停止数据发布")
    except Exception as e:
        print(f"数据采集/发布出错：{e}")
    finally:
        # -------------------------- 资源清理 --------------------------
        ecal_core.finalize()
        print("eCAL 资源已释放，程序退出")


if __name__ == "__main__":
    main()