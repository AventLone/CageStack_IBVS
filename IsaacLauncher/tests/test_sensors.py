from isaacsim import SimulationApp
from pathlib import Path
import yaml
import asyncio

simulation_app = SimulationApp({"headless": False})

import isaacsim.core.utils.numpy.rotations as rot_utils
import os 
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from isaacsim.core.api import World
from devices import VN_Camera
from devices import VN_LidarRtx,VN_Mid360
from devices import VN_IMU
from service.lidar_publisher import convert_isaac_to_pb
import ecal.core.core as ecal_core
from ecal.core.publisher import ProtoPublisher
# from protos.point_cloud2_pb2 import Header, PointField, PointCloud2
from service.lidar_publisher import EcalLidarPublisher

def load_single_camera_config(yaml_path: str, camera_name: str) -> dict:
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
    # 将YAML中的"horizontal_fov"转换为VN_Camera需要的"fov"
    if "horizontal_fov" in camera_params:
        camera_params["fov"] = camera_params.pop("horizontal_fov")
    
    # 返回包含prim_path和处理后params的字典
    return {
        "prim_path": camera_config["prim_path"],
        "params": camera_params
    }

def load_single_imu_config(yaml_path: str, imu_name: str) -> dict:
    """
    从YAML传感器配置中，提取单个IMU的参数（prim_path + params）
    :param yaml_path: YAML配置文件路径
    :param imu_name: 目标IMU名称（如"imu"）
    :return: 包含IMU prim_path和params的字典
    """
    # 检查YAML文件是否存在
    yaml_file = Path(yaml_path)
    if not yaml_file.exists():
        raise FileNotFoundError(f"YAML配置文件不存在：{yaml_path}")
    
    # 读取YAML内容（传感器配置在"sensors"节点下）
    with open(yaml_file, "r", encoding="utf-8") as f:
        yaml_data = yaml.safe_load(f)
    
    # 校验传感器配置节点和目标IMU是否存在
    sensors_config = yaml_data.get("sensors")
    if not sensors_config:
        raise KeyError("YAML中未找到'sensors'配置节点")
    
    imu_config = sensors_config.get(imu_name)
    if not imu_config:
        raise KeyError(f"YAML的sensors中未找到IMU：{imu_name}")
    
    # 提取IMU的prim_path和params
    imu_params = imu_config.get("params", {})
    
    # 返回包含prim_path和params的字典
    return {
        "prim_path": imu_config["prim_path"],
        "params": imu_params
    }

def load_lidar_config(yaml_path: str, lidar_name: str) -> dict:
    """
    从YAML传感器配置中，提取单个lidar的参数（prim_path + params）
    :param yaml_path: YAML配置文件路径
    :param lidar_name: 目标lidar名称（
    :return: 包含lidar prim_path和params的字典
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

    lidar_config = sensors_config.get(lidar_name)
    if not lidar_config:
        raise KeyError(f"YAML的sensors中未找到相机：{lidar_name}")

    # 提取相机的prim_path和params，并处理参数映射（horizontal_fov -> fov）
    lidar_params = lidar_config.get("params", {})


    # 返回包含prim_path和处理后params的字典
    return {
        "prim_path": lidar_config["prim_path"],
        "params": lidar_params
    }

warehouse_asset_path = "/home/visionnav/isaac-sim/5.0_assets/Assets/Isaac/5.0/Isaac/Environments/Simple_Warehouse/warehouse_with_forklifts.usd"
fork_camera_prim_path = "/World/Forklift_01/S_ForkliftFork/fork_Camera"
lidar_prim_path = "/World/Forklift_01/XT32_SD10/PandarXT_32_10hz"
imu_prim_path = "/World/Forklift_01/Imu_Sensor"

my_world = World(stage_units_in_meters=1.0)
# my_world.scene.add_default_ground_plane()
my_world.set_simulation_dt(rendering_dt=0.1)

import isaacsim.core.utils.stage as stage_utils
# stage_utils.add_reference_to_stage(scene_path, "/World")
warehouse=stage_utils.add_reference_to_stage(
    warehouse_asset_path, "/World"
)

my_world.reset()

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
config_path = os.path.join(project_root, "configs", "st_vla.yaml")
YAML_PATH = config_path  # 替换为你的YAML文件路径
TARGET_CAMERA = "fork_camera"  # 目标相机名称（可改为"front_camera"）
    
# ---------------------- 2. 加载单个相机配置 ----------------------
try:
    cam_config = load_single_camera_config(YAML_PATH, TARGET_CAMERA)
except (FileNotFoundError, KeyError) as e:
    print(f"配置加载失败：{e}")
    exit()  # 加载失败则退出

# # 从cam_config中提取prim_path和处理后的params
# vn_camera = VN_Camera(
#     prim_path=cam_config["prim_path"],
#     config=cam_config["params"]  # 将params传给config_camera
# )
#
# vn_camera.add_frame()


target_lidar="lidar_1"
lidar_config = load_lidar_config(YAML_PATH, target_lidar)
#lidar = VN_LidarRtx(lidar_prim_path)
#  实例化MID360雷达
mid360 = VN_Mid360(
    prim_path=lidar_config["prim_path"],  # 雷达在场景中的路径
    config=lidar_config["params"]
)

mid360.start()

ecal_publisher = EcalLidarPublisher(ecal_topic="192.168.102.109")
if not ecal_publisher.init_ecal():
    print(f"[ERROR] eCAL 发布器初始化失败，程序退出", file=sys.stderr)
    sys.exit(1)

#imu = VN_IMU(my_world)

# ---------------------- 3. 加载单个IMU配置 ----------------------
TARGET_IMU = "imu"  # 目标IMU名称

try:
    imu_config = load_single_imu_config(YAML_PATH, TARGET_IMU)
except (FileNotFoundError, KeyError) as e:
    print(f"IMU配置加载失败：{e}")
    # 使用默认配置
    imu_config = {
        "prim_path": imu_prim_path,
        "params": {
            "frequency": 60,
            "translation": [0, 0, 0],
            "orientation": [1, 0, 0, 0]
        }
    }

# 从imu_config中提取prim_path和处理后的params
# imu = VN_IMU(
#     prim_path=imu_config["prim_path"],
#     config=imu_config["params"]  # 将params传给config参数
# )

# while simulation_app.is_running():
#     my_world.step(render=True)
#     # fork_camera.get_frame()
#     mid360.get_pointcloud()
#     imu.get_frame()

# 仿真循环（与Isaac Sim的主循环同步）
frame_count = 0
while simulation_app.is_running():
    # 步进仿真（关键：每帧必须调用，驱动物理和传感器更新）
    my_world.step(render=True)

    # 定期获取点云（例如每10帧获取一次，避免频繁打印）
    pc = mid360.get_pointcloud()
    ecal_publisher.publish_pointcloud(pc)


    # 获取相机和IMU数据
    #vn_camera.get_frame()
    #imu.get_frame()

# 6. 停止雷达
#mid360.stop()


my_world.stop()
simulation_app.update()
simulation_app.close()
