from utils import LoggerUtil, H5Writer, VLADataFrame, VLADataset
import threading
from devices import AckermannVehicle
import time, h5py


logger = LoggerUtil.get_logger("data_collect")


class VLADataCollection(threading.Thread):
    """Sensors工作线程类，继承自threading.Thread"""

    def __init__(self, delay: float, vehicle: AckermannVehicle, cameras: dict, lidars : dict):
        super().__init__()
        self.delay = delay # s
        self.run_thread = True
        self.record = False
        self.vehicle = vehicle
        self.cameras = cameras
        self.lidars = lidars
        self.dataset = VLADataset()
        self.frame_index = 0
        self.dataset.task_id = 0
        self.dataset.language_instruction = "Test instruction for task {}".format(self.dataset.task_id)

    def run(self) -> None:
        """线程启动时执行的方法"""
        logger.info(f"start SensorsThread")
        while self.run_thread :
            start = time.perf_counter()
            self.data_process()
            end = time.perf_counter()
            delay = self.delay - (end - start)
            # logger.info(f"delay = {delay} end-start = {end - start}")
            if delay > 0:
                time.sleep(delay)
        self.run_thread = False


    def stop(self) -> None:
        self.run_thread = False

    def save_h5file(self):
        if self.frame_index == 0:
            return
        test_file = "test_h5_output.h5"
        print("保存数据集到HDF5文件...")
        with h5py.File(test_file, 'w') as f:
            H5Writer.save_dataset(self.dataset, f)
        self.dataset = VLADataset()
        self.frame_index = 0

    def data_process(self):
        if self.record is False:
            self.save_h5file()
            return
        imgs = []
        for key, camera in self.cameras.items():
            rendering_time, rgba = camera.get_frame()
            imgs.append(rgba)
        logger.info(f"steer_angle : {self.vehicle.steer_angle} drive_velocity : {self.vehicle.drive_velocity} fork_pose : {self.vehicle.fork_pose}")
        frame = VLADataFrame()
        frame.index = self.frame_index
        frame.timestamp = self.frame_index  # 模拟时间戳
        # 添加5个测试图像
        for img in imgs:
            img_bytes = img.tobytes()
            frame.images.append(img_bytes)
            # 添加状态向量（5个随机float值）
            frame.state = [self.vehicle.steer_angle, self.vehicle.drive_velocity, 0, 0, self.vehicle.fork_pose.z]
        self.dataset.records.append(frame)
        self.frame_index += 1

    def start_record(self):
        self.record = True

    def stop_record(self):
        self.record = False