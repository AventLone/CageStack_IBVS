import numpy as np
from typing import Optional
from isaacsim.sensors.physics import IMUSensor
import numpy as np

class VN_IMU:
    def __init__(self, prim_path: str, config: Optional[dict] = None):
        """
        IMU sensor initialization.

        :param prim_path: The prim path for the IMU sensor.
        :param config: A dictionary containing configuration parameters for the IMU.
                       Supported keys: 'frequency', 'translation', 'orientation'.
        """
        self.prim_path = prim_path
        self.config = config
        self.imu_sensor = None
        self.imu_init()
        self.config_imu(config)

    def config_imu(self, config: Optional[dict]):
        """
        Configures the IMU sensor based on a configuration dictionary.

        :param config: A dictionary with IMU parameters.
        """
        if not config:
            return

        # Set frequency
        if 'frequency' in config:
            self.imu_sensor.set_frequency(config['frequency'])
            print(f"IMU frequency set to: {config['frequency']} Hz")

        # Set position and orientation using set_local_pose
        if 'translation' in config and 'orientation' in config:
            translation = np.array(config.get('translation', [0.0, 0.0, 0.0]), dtype=np.float32)
            orientation = np.array(config.get('orientation', [1.0, 0.0, 0.0, 0.0]), dtype=np.float32) # Default to (w=1, x=0, y=0, z=0)

            self.imu_sensor.set_local_pose(translation=translation, orientation=orientation)
            print(f"IMU position set to: {translation}")
            print(f"IMU orientation set to: {orientation}")

    def imu_init(self):
        self.imu_sensor = IMUSensor(
            prim_path=self.prim_path,
            name=self.prim_path.replace("/", "_")  # Generate a name from prim_path
        )
        self.imu_sensor.initialize()

    def get_frame(self):
        """
        Retrieves the latest sensor data from the IMU.

        :return: A tuple containing linear acceleration, angular velocity, and orientation.
        """
        current_frame = self.imu_sensor.get_current_frame()
        lin_acc = current_frame.get("lin_acc")
        ang_vel = current_frame.get("ang_vel")
        orientation = current_frame.get("orientation")
        return lin_acc, ang_vel, orientation

if __name__ == "__main__":
    # 测试配置加载
    # 打印当前路径
    print(os.getcwd())
    # 构建配置文件的绝对路径
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(project_root, "configs", "st_vla.yaml")
    config_loader = ConfigLoader(config_path)
    # 使用专门的 IMU 参数获取方法
    imu_params = config_loader.get_imu_params("imu")
    print("IMU 传感器参数:", imu_params)
    print("imu_params.prim_path:", imu_params.prim_path)
    
    # 也可以使用通用方法（向后兼容）
    sensor_params = config_loader.get_sensor_params("imu")
    print("通用传感器参数:", sensor_params)
