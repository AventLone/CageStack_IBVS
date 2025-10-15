import time

from sympy.strategies.core import switch

from devices import VN_Camera, PrimPose, VN_Mid360, VN_IMU,LimitSwitchSensor
import threading
from utils import ConfigLoader, LoggerUtil
from typing import Optional
from devices import Vehicle
import numpy as np
import ecal.core.core as ecal_core
from ecal.core.subscriber import ProtoSubscriber
from controllers.common import VehicleStateServer, VLADataCollection,LidarPC2Publisher
import sys
from protos import vehicle_state_msg_pb2
from utils import LoggerUtil
import threading

logger = LoggerUtil.get_logger("controller")

class ForkliftST(threading.Thread):
    def __init__(self, world, config: dict, data_collect=False):
        super().__init__()
        if not ecal_core.is_initialized():  # True/False
            ecal_core.initialize(sys.argv, "IsaacLauncher")

        vehicle_config = config["vehicle"]
        self.vehicle = Vehicle(world=world, cfg=vehicle_config)


        sensor_config = config['sensors']
        self.camera_keys = ['fork_camera', 'front_camera', 'back_camera']
        self.camera_devices = {}
        for camera_key in self.camera_keys:
            if camera_key in sensor_config.keys():
                logger.info(f"add camera : {sensor_config[camera_key]['prim_path']}")
                camera = VN_Camera(sensor_config[camera_key]['prim_path'],
                                   sensor_config[camera_key]['params'])
                self.camera_devices[camera_key] = camera

        vehicle_config = config['vehicle']
        self.lidar_keys =["lidar_1", "lidar_2", "lidar_3"]
        self.lidar_devices = {}
        for lidar_key in self.lidar_keys:
            if lidar_key in sensor_config.keys():
                logger.info(f"add lidar : {sensor_config[lidar_key]['prim_path']}")
                mid360 = VN_Mid360(prim_path=sensor_config[lidar_key]["prim_path"],  # 雷达在场景中的路径
                    config=sensor_config[lidar_key]["params"],
                    vehicle_xform_path=vehicle_config["prim_path"])
                mid360.start()
                mid360.link_to_xform(sensor_config[lidar_key]["params"]["parent_xform_path"])
                self.lidar_devices[lidar_key] = mid360
                pub = LidarPC2Publisher(
                    lidar=mid360,
                    topic=sensor_config[lidar_key]['topic_name'],
                    frame_id="mid360",
                    period=0.1,  # 10 Hz
                )
                pub.start()


        if data_collect:

            self.data_collection = VLADataCollection(0.1, self.vehicle,
                                                     self.camera_devices,
                                                     self.lidar_devices)
            self.data_collection.start()
        else:
            self.data_collection = None

        if "imu" in config['sensors']:
            imu_config = config['sensors']['imu']
            logger.info(f"add imu : {imu_config['prim_path']}")
            self.imu = VN_IMU(imu_config["prim_path"], imu_config['params'])
        else:
            self.imu = None
        self.state_pub = VehicleStateServer(self.vehicle, self.imu, vehicle_config)
        self.state_pub.start()
        # self.state_pub.run()

        self.switch_keys = ['right_h_switch_limit', 'left_h_switch_limit', 'right_v_switch_limit', 'left_v_switch_limit']
        self.switches = {}
        for switch_key in self.switch_keys:
            if switch_key in config["sensors"].keys():
                logger.info(f"add switch : {config['sensors'][switch_key]['prim_path']}")
                switch = LimitSwitchSensor(config['sensors'][switch_key]['prim_path'])
                self.switches[switch_key] = switch


        # self.start()

        # self.tracking_obj_keys = ['vehicle', 'pallet']
        # obj_config = config['tracked_objects']
        # self.tracking_objs = {}
        # for obj_key in self.tracking_obj_keys:
        #     if obj_key in obj_config.keys():
        #         logger.info(f"add tracking_obj : {obj_config[obj_key]['prim_path']}")
        #         tracking_obj = PrimPose(obj_config[obj_key]['prim_path'])
        #         self.tracking_objs[obj_key] = tracking_obj
        #         print(tracking_obj.get_pose())

    def __del__(self):
        if ecal_core.is_initialized():
            ecal_core.finalize()

    def run(self) -> None:
        # if self.imu is not None:
        #     print(self.imu.get_frame())
        # data=self.switches['right_h_switch_limit'].get_force_data()
        # print("the right_h_switch_limit",data)
        #
        # data = self.switches['left_h_switch_limit'].get_force_data()
        # print("the left_h_switch_limit", data)
        #
        # data = self.switches['right_v_switch_limit'].get_force_data()
        # print("the right_v_switch_limit", data)
        #
        # data = self.switches['left_v_switch_limit'].get_force_data()
        # print("the left_v_switch_limit", data)

        time.sleep(0.1)

    def step(self) -> None:
        self.state_pub.step()

    # def keyboardCallback(self, topic_name, msg, msg_time) -> None:
    #     # self.cmd_data = np.fromstring(string=msg, dtype=np.float32, sep=" ")
    #     if self.data_collection is not None:
    #         if msg.data_record > 0:
    #             self.data_collection.start_record()
    #         else:
    #             self.data_collection.stop_record()
