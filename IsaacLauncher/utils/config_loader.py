import yaml
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional, Any
import os
from pathlib import Path


class RootConfig(BaseModel):
    asset_root: Optional[str] = Field(None, description="资产根目录路径")

class SimulateAppConfig(BaseModel):
    config: dict
    stage_file_path: str
    controller_class: str
    actuators_config: Optional[str] = None
    sensors_config: Optional[str] = None

class VehicleConfig(BaseModel):
    name : Optional[str]
    prim_path : str
    keyboard_topic: Optional[str]
    adjust_ctrl_topic: Optional[str]
    fork_joint_names: List[str]
    drive_wheel_joint_names: Optional[List[str]]
    steer_wheel_joint_names: Optional[List[str]]
    
    def __getitem__(self, key):
        """支持字典访问方式，实现向后兼容"""
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(f"'{key}' not found in VehicleConfig")
    


# 基础传感器参数类
class BaseSensorParams(BaseModel):
    prim_path: str = Field(..., description="传感器的USD路径")
    enabled: Optional[bool] = Field(True, description="是否启用传感器")

# 摄像头参数类
class CameraParams(BaseSensorParams):
    width: int = Field(..., gt=0, description="图像宽度")
    height: int = Field(..., gt=0, description="图像高度")
    fps: Optional[int] = Field(30, gt=0, description="帧率")
    horizontal_fov: Optional[float] = Field(70.0, gt=0.0, le=180.0, description="水平视场角(度)")
    
    @validator('horizontal_fov')
    def validate_horizontal_fov(cls, v):
        if v is not None and (v <= 0 or v > 180):
            raise ValueError('水平视场角必须在0到180度之间')
        return v

# 激光雷达参数类
class LidarParams(BaseSensorParams):
    range_min: Optional[float] = Field(0.1, ge=0.0, description="最小探测距离(m)")
    range_max: Optional[float] = Field(100.0, gt=0.0, description="最大探测距离(m)")
    scan_frequency: Optional[float] = Field(10.0, gt=0.0, description="扫描频率(Hz)")
    angular_resolution: Optional[float] = Field(0.25, gt=0.0, description="角度分辨率(度)")
    vertical_fov: Optional[float] = Field(30.0, gt=0.0, le=180.0, description="垂直视场角(度)")
    
    @validator('vertical_fov')
    def validate_vertical_fov(cls, v):
        if v is not None and (v <= 0 or v > 180):
            raise ValueError('垂直视场角必须在0到180度之间')
        return v
    
    @validator('range_max')
    def validate_range_max(cls, v, values):
        if v is not None and 'range_min' in values and values['range_min'] is not None:
            if v <= values['range_min']:
                raise ValueError('最大探测距离必须大于最小探测距离')
        return v

# IMU传感器参数类
class IMUParams(BaseSensorParams):
    sensor_name: Optional[str] = Field("imu", description="传感器名称")
    frequency: Optional[int] = Field(60, gt=0, description="采样频率(Hz)")
    translation: Optional[List[float]] = Field([0,0,0], description="传感器位置[x,y,z]")
    orientation: Optional[List[float]] = Field([1,0,0,0], description="传感器朝向四元数")
    linear_acceleration_filter_size: Optional[int] = Field(10, gt=0, description="线性加速度滤波器大小")
    angular_velocity_filter_size: Optional[int] = Field(10, gt=0, description="角速度滤波器大小")
    orientation_filter_size: Optional[int] = Field(10, gt=0, description="姿态滤波器大小")

# 为了向后兼容，保留原有的 SensorParams 类作为联合类型
from typing import Union
SensorParams = Union[CameraParams, LidarParams, IMUParams, BaseSensorParams]


class SensorConfig(BaseModel):
    type: Optional[str]
    prim_path: str
    params: Optional[SensorParams]
    
    def __init__(self, **data):
        # 根据传感器类型动态选择参数类
        if 'params' in data and 'type' in data:
            sensor_type = data['type']
            params_data = data['params']
            
            if params_data:
                # 确保 prim_path 存在于 params 中
                if 'prim_path' not in params_data:
                    params_data['prim_path'] = data['prim_path']
                
                # 根据传感器类型选择对应的参数类
                if sensor_type == 'camera.rgb':
                    data['params'] = CameraParams(**params_data)
                elif sensor_type in ['rtx_lidar', 'mid_360']:
                    data['params'] = LidarParams(**params_data)
                elif sensor_type == 'sensor.imu':
                    data['params'] = IMUParams(**params_data)
                else:
                    data['params'] = BaseSensorParams(**params_data)
        
        super().__init__(**data)


class AGVSimulationConfig(BaseModel):
    asset_root: Optional[str] = Field(None, description="资产根目录路径")
    simulation_app: SimulateAppConfig
    vehicle: VehicleConfig
    sensors: Dict[str, SensorConfig]  # 键为传感器名称（如lidar）
    
    def __getitem__(self, key):
        """支持字典访问方式，实现向后兼容"""
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(f"'{key}' not found in AGVSimulationConfig")


class ConfigLoader:
    def __init__(self, config_path : str):
        self.config_path = config_path
        self.config_dir = os.path.dirname(config_path)
        self.config = self.load()

    def _resolve_path(self, path: str, asset_root: Optional[str] = None) -> str:
        """解析路径，支持相对路径和绝对路径"""
        if os.path.isabs(path):
            return path
        
        # 如果有资产根目录配置，优先使用资产根目录 + 相对路径
        if asset_root and os.path.isabs(asset_root):
            return os.path.join(asset_root, path)
        
        # 如果既不是绝对路径也没有资产根目录，直接返回原路径
        return path

    def load(self) -> dict:
        try:
            # 1. 读取YAML文件
            with open(self.config_path, 'r', encoding='utf-8') as f:
                raw_config = yaml.safe_load(f)
            if not raw_config:
                raise ValueError("config error")
            
            # 2. 获取资产根目录
            asset_root = raw_config.get('asset_root')
            
            # 3. 解析路径 - 只有USD文件使用资产根目录，配置文件保持相对路径
            if 'simulation_app' in raw_config:
                sim_app = raw_config['simulation_app']
                if 'stage_file_path' in sim_app:
                    # 只有USD文件使用资产根目录解析
                    sim_app['stage_file_path'] = self._resolve_path(sim_app['stage_file_path'], asset_root)
                # actuators_config 和 sensors_config 保持相对路径，不进行资产根目录解析
            
            # 4. 解析传感器路径
            if 'sensors' in raw_config:
                for sensor_name, sensor_config in raw_config['sensors'].items():
                    if 'params' in sensor_config and sensor_config['params']:
                        params = sensor_config['params']
                        # if 'npy_path' in params:
                            # params['npy_path'] = self._resolve_path(params['npy_path'], asset_root)
            
            # 5. 校验配置结构和约束
            validated_config = AGVSimulationConfig(**raw_config)
            print("配置文件校验成功！")
            print(validated_config)
            return raw_config
        except yaml.YAMLError as e:
            raise ValueError(f"YAML格式错误: {str(e)}")
        except FileNotFoundError:
            raise FileNotFoundError(f"配置文件不存在: {self.config_path}")
        except Exception as e:
            raise ValueError(f"配置校验失败: {str(e)}")

    def get(self):
        """获取完整的配置对象"""
        return self.config
    
    def get_vehicle_config(self) -> VehicleConfig:
        """获取车辆配置"""
        return self.config.vehicle
    
    def get_sensor_configs(self) -> Dict[str, SensorConfig]:
        """获取传感器配置字典"""
        return self.config.sensors
    
    def get_sensor_config(self, sensor_name: str) -> Optional[SensorConfig]:
        """获取指定传感器的配置"""
        return self.config.sensors.get(sensor_name)
    
    def get_sensor_params(self, sensor_name: str) -> Optional[SensorParams]:
        """获取指定传感器的参数"""
        sensor_config = self.get_sensor_config(sensor_name)
        return sensor_config.params if sensor_config else None
    
    def get_camera_params(self, sensor_name: str) -> Optional[CameraParams]:
        """获取指定摄像头传感器的参数"""
        params = self.get_sensor_params(sensor_name)
        return params if isinstance(params, CameraParams) else None
    
    def get_lidar_params(self, sensor_name: str) -> Optional[LidarParams]:
        """获取指定激光雷达传感器的参数"""
        params = self.get_sensor_params(sensor_name)
        return params if isinstance(params, LidarParams) else None
    
    def get_imu_params(self, sensor_name: str) -> Optional[IMUParams]:
        """获取指定IMU传感器的参数"""
        params = self.get_sensor_params(sensor_name)
        return params if isinstance(params, IMUParams) else None



if __name__ == "__main__":
    loader = ConfigLoader("./configs/st_vla.yaml")
    print(loader.get())