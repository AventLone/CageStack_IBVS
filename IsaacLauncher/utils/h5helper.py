import h5py
import numpy as np
import time
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import tqdm
import os

class VLADataFrame:
    """数据帧结构，包含索引、时间戳、图像列表和状态向量"""

    def __init__(self):
        self.index = 0
        self.timestamp = 0  # 时间戳，long long类型
        self.images = []  # 存储图像数据的列表，每个元素是字节数组
        self.state = []  # 状态向量，长度为5的float列表


class VLADataset:
    """数据集结构，包含多条记录、语言指令、长度和任务ID"""

    def __init__(self):
        self.records = []  # DataFrame对象的列表
        self.language_instruction = ""  # 语言指令
        self.length = 0  # 记录数量
        self.task_id = 0  # 任务ID


class H5Writer:
    """HDF5操作辅助类，提供保存图像、数据帧和数据集的静态方法"""

    def __init__(self, h5_file_path: str, logger):
        # 初始化传入的参数
        self.h5_file_path = h5_file_path  # HDF5文件保存路径
        self.logger = logger  # 日志对象（用于打印信息/错误）

        # 初始化后续需要用到的实例属性
        self.frame_index = 0  # 数据帧索引（自增）
        self.dataset = VLADataset()  # 数据集对象（存储待保存的帧数据）
        # 初始化数据集元信息（可根据需求调整）
        self.dataset.task_id = 1
        self.dataset.language_instruction = "多相机数据采集任务"

    @staticmethod
    def save_image_group(parent_group: h5py.Group, img_data: bytes,
                         width: int, height: int, channels: int, img_idx: int) -> None:
        """
        保存图像到HDF5组

        参数:
            parent_group: 父HDF5组
            img_data: 图像数据字节数组
            width: 图像宽度
            height: 图像高度
            channels: 图像通道数
            img_idx: 图像索引
        """
        # 创建图像子组
        group_name = f"image_{img_idx}"
        img_group = parent_group.create_group(group_name)

        # 存储维度属性
        img_group.attrs["width"] = width
        img_group.attrs["height"] = height
        img_group.attrs["channels"] = channels

        # 将字节数据转换为numpy数组
        img_array = np.frombuffer(img_data, dtype=np.uint8)
        img_array = img_array.reshape((height, width, channels))

        # 创建数据集并写入图像数据，使用压缩
        img_group.create_dataset(
            "data",
            data=img_array,
            dtype=np.uint8,
            compression="gzip",
            compression_opts=6
        )

    @staticmethod
    def save_dataframe(task_group: h5py.Group, frame: VLADataFrame) -> None:
        """
        保存数据帧到HDF5组

        参数:
            task_group: 任务HDF5组
            frame: 要保存的数据帧对象
        """
        # 创建数据帧子组
        group_name = f"dataframe_{frame.index}"
        df_group = task_group.create_group(group_name)

        # 存储元数据
        df_group.attrs["index"] = frame.index
        df_group.attrs["timestamp"] = frame.timestamp

        # 存储状态向量
        if len(frame.state) != 5:
            raise ValueError("State vector must have exactly 5 elements")

        df_group.create_dataset(
            "state",
            data=np.array(frame.state, dtype=np.float32),
            dtype=np.float32
        )

        # 存储图像
        for img_idx, img_data in enumerate(frame.images):
            # 假设Webots相机的尺寸，实际使用时应替换为真实尺寸
            width, height, channels = 640, 480, 4
            H5Writer.save_image_group(df_group, img_data, width, height, channels, img_idx)

    @staticmethod
    def save_dataset(dataset: VLADataset, file: h5py.File) -> None:
        """
        保存数据集到HDF5文件，支持创建新组或追加到现有组

        参数:
            dataset: 要保存的数据集对象
            file: 已打开的HDF5文件对象
        """
        task_name = f"/task_{dataset.task_id}"

        # 检查组是否已存在
        group_exists = task_name in file

        try:
            if group_exists:
                task_group = file[task_name]
                print("Appending to existing task group")

                # 更新长度属性
                if "length" in task_group.attrs:
                    existing_length = task_group.attrs["length"]
                    new_length = existing_length + dataset.length
                    task_group.attrs["length"] = new_length
                else:
                    task_group.attrs["length"] = dataset.length
            else:
                # 创建新组
                task_group = file.create_group(task_name)
                print("Creating new task group")

                # 设置属性
                task_group.attrs["language_instruction"] = dataset.language_instruction
                task_group.attrs["length"] = dataset.length
                task_group.attrs["task_id"] = dataset.task_id

            # 追加所有数据帧
            for frame in dataset.records:
                if frame is not None:
                    H5Writer.save_dataframe(task_group, frame)

        except Exception as e:
            print(f"HDF5 Error: {str(e)}")

    def save_multi_camera_data(self, multi_frames):
        if not multi_frames:
            self.logger.warning("无有效相机数据，跳过保存")
            return

            # 创建数据帧对象
        frame = VLADataFrame()
        frame.index = self.frame_index
        frame.timestamp = int(time.time() * 1000)
        frame.state = [0.0, 0.0, 0.0, 0.0, 0.0]  # 状态向量

        # 收集图像数据（修正键名：img_bytes）
        for cam_frame in multi_frames:
            frame.images.append(cam_frame["img_bytes"])  # 键名改为img_bytes
            frame.width = cam_frame["width"]
            frame.height = cam_frame["height"]
            frame.channels = cam_frame["channels"]

        # 保存到数据集
        self.dataset.records.append(frame)
        self.dataset.length += 1

        # 打开HDF5文件并保存（修正文件对象传递）
        with h5py.File(self.h5_file_path, "a" if Path(self.h5_file_path).exists() else "w") as f:
            H5Writer.save_dataset(self.dataset, f)  # 传入打开的文件对象

        # 重置数据集（避免重复保存）
        self.dataset.records = []
        self.dataset.length = 0
        self.frame_index += 1
        self.logger.info(f"已保存第 {self.frame_index} 帧数据")


class H5DataAnalysis:
    def __init__(self, h5_dir="./", save_dir="./"):
        self.h5_dir = h5_dir
        self.save_dir = save_dir

    def setH5Dir(self, h5_dir):
        self.h5_dir = h5_dir

    def setSaveDir(self, save_dir):
        self.save_dir = save_dir

    def getFileList(self, dir, file_end=None):
        files = os.listdir(dir)
        file_list = []
        for file in files:
            if file.endswith(file_end):
                image_path = os.path.join(dir, file)
                file_list.append(image_path)
        return file_list

    def h5_reader(self, h5_file):
        gt_actions = []
        front_images = []
        bev_images = []
        fork_images = []
        with h5py.File(h5_file, 'r') as f:
            length = len(f["/task_0"])
            for idx in tqdm.tqdm(range(length - 1)):
                # parse state
                gt_action = f["/task_0/dataframe_" + str(idx + 1) + "/state"][:]
                gt_actions.append(gt_action)

                if "smooth" not in h5_file:
                    front_image = f["/task_0/dataframe_" + str(idx) + "/image_0/data"][:][..., :-1]
                    bev_image = f["/task_0/dataframe_" + str(idx) + "/image_1/data"][:][..., :-1]
                    fork_image = f["/task_0/dataframe_" + str(idx) + "/image_2/data"][:][..., :-1]
                else:
                    front_msg = f["/task_0/dataframe_" + str(idx) + "/image_0/data"][:][..., :-1]
                    bev_msg = f["/task_0/dataframe_" + str(idx) + "/image_1/data"][:][..., :-1]
                    fork_msg = f["/task_0/dataframe_" + str(idx) + "/image_2/data"][:][..., :-1]
                    front_image = cv2.imdecode(np.frombuffer(front_msg.tobytes(), np.uint8), cv2.IMREAD_COLOR)
                    bev_image = cv2.imdecode(np.frombuffer(bev_msg.tobytes(), np.uint8), cv2.IMREAD_COLOR)
                    fork_image = cv2.imdecode(np.frombuffer(fork_msg.tobytes(), np.uint8), cv2.IMREAD_COLOR)

                # front_image = cv2.cvtColor(front_image, cv2.COLOR_BGR2RGB)
                # bev_image = cv2.cvtColor(bev_image, cv2.COLOR_BGR2RGB)
                # fork_image = cv2.cvtColor(fork_image, cv2.COLOR_BGR2RGB)

                front_images.append(front_image)
                bev_images.append(bev_image)
                fork_images.append(fork_image)
        return  gt_actions, front_images, bev_images, fork_images

    def video_save(self, video_file, front_images, bev_images, fork_images):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        width = front_images[0].shape[1] * 3
        height = front_images[0].shape[0]
        video = cv2.VideoWriter(video_file, fourcc, 30, (width, height))
        for front_image, bev_image, fork_image in zip(front_images, bev_images, fork_images):
            image = np.hstack((front_image, bev_image, fork_image))
            video.write(image)

    def state_plot_save(self, plot_file, states):
        length = len(states)
        states = np.asarray(states)
        x = np.linspace(0, length - 1, length, endpoint=True)
        # fig = plt.figure(figsize=(10, 5))
        fig, axs = plt.subplots(5, 1, figsize=(10, 10))
        axs[0].plot(x, states[:, 0], color='green', linewidth=0.7, label="Ground Truth (gt)")
        axs[0].set_title("fork_z")
        axs[0].legend()

        axs[1].plot(x, states[:, 1], color='green', linewidth=0.7, label="Ground Truth (gt)")
        axs[1].set_title("fork_y")
        axs[1].legend()

        axs[2].plot(x, states[:, 2], color='green', linewidth=0.7, label="Ground Truth (gt)")
        axs[2].set_title("fork_x")
        axs[2].legend()

        axs[3].plot(x, states[:, 3], color='green', linewidth=0.7, label="Ground Truth (gt)")
        axs[3].set_title("angle")
        axs[3].legend()

        axs[4].plot(x, states[:, 4], color='green', linewidth=0.7, label="Ground Truth (gt)")
        axs[4].set_title("speed")
        axs[4].legend()

        plt.tight_layout()
        fig.text(0.5, 0.01, "next 0.1s action(Pred vs GT)", ha="center")
        plt.savefig(plot_file)


    def data_show(self):
        h5_file_list = self.getFileList(self.h5_dir, file_end=".h5")
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
        for h5_file in h5_file_list:
            print(h5_file)
            gt_actions, front_images, bev_images, fork_images = self.h5_reader(h5_file)
            video_file = h5_file.replace(self.h5_dir, self.save_dir).replace("h5", "mp4")
            plot_file = h5_file.replace(self.h5_dir, self.save_dir).replace("h5", "jpg")
            self.video_save(video_file, front_images, bev_images, fork_images)
            print(video_file)
            self.state_plot_save(plot_file, gt_actions)
            print(plot_file)

if __name__ == "__main__":
    height = 480
    width = 640
    channels = 4
    dataset = VLADataset()
    dataset.task_id = 0
    dataset.language_instruction = "Test instruction for task {}".format(dataset.task_id)
    for i in range(3):
        frame = VLADataFrame()
        frame.index = i
        frame.timestamp = 1620000000000 + i  # 模拟时间戳
        # 添加5个测试图像
        for _ in range(5):
            # 生成随机的uint8类型图像数据
            img_array = np.random.randint(0, 256, size=(height, width, channels), dtype=np.uint8)
            # 转换为字节流
            img_bytes = img_array.tobytes()
            frame.images.append(img_bytes)
        # 添加状态向量（5个随机float值）
        frame.state = np.random.rand(5).tolist()
        dataset.records.append(frame)
    dataset.length = 3

    test_file = "test_h5helper_output.h5"
    print("保存数据集到HDF5文件...")
    with h5py.File(test_file, 'w') as f:
        H5Writer.save_dataset(dataset, f)