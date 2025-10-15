import sys
import os
import yaml
from pathlib import Path
# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(r'D:\Anaconda3\envs\isaac-sim\Lib\site-packages')


from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import carb
from isaacsim.core.utils.stage import open_stage
open_stage(r"E:\Isaac_sim\asset\ForkliftE_test.usd")
# open_stage("/home/vn/Documents/IsaacStages/ForkliftE_test.usd")

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


# # ---- Add a pallet to the stage for testing ----
# import omni.usd
# from pxr import Gf, UsdGeom
# stage = omni.usd.get_context().get_stage()
# pallet_path = "/World/warehouse/pallet_relative"
# if not stage.GetPrimAtPath(pallet_path):
#     print(f"'{pallet_path}' not found in stage, creating a new one for testing.")
#     # Create a cube to represent the pallet
#     pallet_prim = UsdGeom.Cube.Define(stage, pallet_path)
#     # Set its size (assuming meters)
#     pallet_prim.GetSizeAttr().Set(1.2) # Standard pallet is ~1.2m x 1.0m
#     # Set its position in the world
#     xform = UsdGeom.Xformable(pallet_prim)
#     transform = xform.AddTransformOp()
#     mat = Gf.Matrix4d()
#     # Place it 3 meters in front of the forklift's initial position
#     mat.SetTranslate(Gf.Vec3d(3.0, 0, 0.1))
#     transform.Set(mat)
# # -----------------------------------------

import asyncio
import math
import numpy as np
import yaml
from isaacsim.core.api.world import World
from controllers import ForkliftE
from devices import VN_IMU
from utils.world_model import ObjectTracker, PalletInitializer

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
config_path = os.path.join(project_root, "configs", "e_test.yaml")

# 加载配置
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

my_world = World()
my_world.reset()
forkliftE_simu = ForkliftE(world=my_world, yamlfile=config_path)

config_path = os.path.join(project_root, "configs", "st_vla.yaml")
YAML_PATH = config_path 
# ---------------------- 3. 加载单个IMU配置 ----------------------
TARGET_IMU = "imu"  # 目标IMU名称

try:
    imu_config = load_single_imu_config(YAML_PATH, TARGET_IMU)
except (FileNotFoundError, KeyError) as e:
    print(f"IMU配置加载失败：{e}")
    # 使用默认配置
    imu_config = {
        "prim_path": "/World/forklift_E/body/Imu_Sensor",
        "params": {
            "frequency": 60,
            "translation": [0, 0, 0],
            "orientation": [1, 0, 0, 0]
        }
    }

# 从imu_config中提取prim_path和处理后的params
imu = VN_IMU(
    prim_path=imu_config["prim_path"],
    config=imu_config["params"]  # 将params传给config参数
)

# 初始化 ObjectTracker
if "tracked_objects" in config:
    tracker = ObjectTracker(config["tracked_objects"])
else:
    tracker = None
    print("Warning: 'tracked_objects' not found in config. Pose tracking will be disabled.")
my_world.reset()
# 托盘初始化
if "pallet" in config:
        vehicle_prim_path = config.get("vehicle", {}).get("prim_path", "/World/forklift_E")
        pallet_initializer = PalletInitializer(vehicle_prim_path=vehicle_prim_path)
        print(f"Statically initializing pallet poses from configuration (vehicle path: {vehicle_prim_path})...")
        for pallet_name, pallet_config in config["pallet"].items():
            print(f"Initializing {pallet_name}...")
            success = pallet_initializer.initialize_pallet_statically(pallet_config)
            if success:
                print(f"Successfully initialized pallet '{pallet_name}'")
            else:
                print(f"Failed to initialize pallet '{pallet_name}'")
else:
    pallet_initializer = None
    print("Warning: 'pallet' not found in config. Pallet initialization will be disabled.")

require_reset = False
while simulation_app.is_running():
    my_world.step(render=True)

    if my_world.is_stopped() and not require_reset:   # 播放/暂停与重置逻辑
        require_reset = True

    if my_world.is_playing():
        if require_reset:
            my_world.reset()
            if tracker:
                tracker.reset() # 重置缓存
            require_reset = False
        
        # imu.get_frame()
        forkliftE_simu.run()

        # 获取并打印位姿
        # if False & tracker:
        if False:
            tracker.reset() # Reset cache every frame
            all_poses = tracker.get_all_poses()
            if all_poses and 'vehicle' in all_poses and 'pallet' in all_poses:
                relative_pose = ObjectTracker.get_relative_pose(all_poses['vehicle'], all_poses['pallet'])
                
                print("--- Pose Information ---")
                print(f"Vehicle World Pose: {all_poses['vehicle']}")
                print(f"Pallet World Pose:  {all_poses['pallet']}")
                print(f"Pallet relative to Vehicle: {relative_pose}")
                print("------------------------\n")

simulation_app.close()