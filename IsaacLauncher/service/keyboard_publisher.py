import sys
import ecal.core.core as ecal_core
from ecal.core.publisher import ProtoPublisher
from ecal.core.subscriber import ProtoSubscriber
from pynput import keyboard
import math
from protos import vehicle_state_msg_pb2
import asyncio, time


class KeyBoardPublisher:

    def __init__(self, state_sub_name: str, cmd_pub_name: str, time_period : float = 0.05) -> None:
        if not ecal_core.is_initialized():
            ecal_core.initialize(sys.argv, "Keyboard command")

        self.stop = False
        self.pub_ok = asyncio.Event()
        self.sub_ok = asyncio.Event()
        self.data_lock = asyncio.Lock()

        self.drive_wheels_velocity = 0.0
        self.steer_wheels_angle = 0.0
        self.steer_wheels_angle_speed = 0.0
        self.fork_z = 0.0
        self.fork_y = 0.0
        self.fork_pitch = 0.0
        self.fork_z_speed = 0.0
        self.fork_y_speed = 0.0
        self.fork_pitch_speed = 0.0
        self.record = False
        self.adjust_ctrl = False

        self.time_period = time_period

        self.state_sub = ProtoSubscriber(state_sub_name,
                                         vehicle_state_msg_pb2.VehicleStateMsg)
        self.cmd_pub = ProtoPublisher(cmd_pub_name,
                                      vehicle_state_msg_pb2.VehicleStateMsg)
        
        self.state_sub.set_callback(self._msgHandler)

        # 键盘线程
        listener = keyboard.Listener(on_press=self.onPress,
                                     on_release=self.onRelease)
        listener.daemon = True
        listener.start()

    def __del__(self):
        self.stop = True
        if ecal_core.is_initialized():
            ecal_core.finalize()

    def run(self) -> None:
        # asyncio.run(self._tasks())
        asyncio.run(self._pubCmd())

    def _msgHandler(self, topic_name, msg, msg_time):
        self.steer_wheels_angle = msg.steer_angle
        self.fork_z = msg.fork_z
        self.fork_y = msg.fork_y
        self.fork_pitch = msg.fork_pitch

    async def _pubCmd(self):
        # await self.sub_ok.wait()
        next_t = time.perf_counter()

        while not self.stop:

            # async with self.data_lock:
            self.steer_wheels_angle += self.steer_wheels_angle_speed
            self.fork_z += self.fork_z_speed
            self.fork_y += self.fork_y_speed
            self.fork_pitch += self.fork_pitch_speed

            vehicle_msg = vehicle_state_msg_pb2.VehicleStateMsg()
            vehicle_msg.drive_velocity = self.drive_wheels_velocity
            vehicle_msg.steer_angle = self.steer_wheels_angle
            vehicle_msg.fork_x = 0
            vehicle_msg.fork_y = self.fork_y
            vehicle_msg.fork_z = self.fork_z
            vehicle_msg.fork_pitch = self.fork_pitch
            vehicle_msg.data_record = self.record
            vehicle_msg.adjust_control = self.adjust_ctrl

            self.cmd_pub.send(vehicle_msg)
            # 固定周期调度，避免频率漂移
            next_t += self.time_period
            dt = next_t - time.perf_counter()
            if dt > 0:
                await asyncio.sleep(dt)
            else:
                # 掉帧时重置起点，避免一直为负
                next_t = time.perf_counter()


    def onPress(self, key):
        try:
            k = key.char.lower()
        except AttributeError:
            k = None

        if k == 'w':
            self.drive_wheels_velocity = math.pi * 2
        elif k == "s":
            self.drive_wheels_velocity = -math.pi * 2

        if k == 'a':
            self.steer_wheels_angle_speed = 0.02
        elif k == 'd':
            self.steer_wheels_angle_speed = -0.02

        if k == 'e':
            self.adjust_ctrl = True
        elif k == 'r':
            self.adjust_ctrl = False

        if key == keyboard.Key.up:
            self.fork_z_speed = 0.02
        elif key == keyboard.Key.down:
            self.fork_z_speed = -0.02

        if key == keyboard.Key.left:
            self.fork_y_speed = 0.02
        elif key == keyboard.Key.right:
            self.fork_y_speed = -0.02

        if key == keyboard.Key.home:
            self.fork_pitch_speed = 0.002
        elif key == keyboard.Key.end:
            self.fork_pitch_speed = -0.002

        if k == 'n':
            self.record = True
        elif k == 'm':
            self.record = False

        if key == keyboard.Key.esc:  # Resotre
            self.steer_wheels_angle = 0.0
            self.fork_z = 0.0
            self.fork_y = 0.0
            self.fork_pitch = 0.0
            self.drive_wheels_velocity = 0.0
            self.record = False

    def onRelease(self, key):
        try:
            k = key.char.lower()
        except AttributeError:
            k = None

        if k == 'w' or k == 's':
            self.drive_wheels_velocity = 0.0

        if k == 'a' or k == 'd':
            self.steer_wheels_angle_speed = 0.0

        if key == keyboard.Key.up or key == keyboard.Key.down:
            self.fork_z_speed = 0.0

        if key == keyboard.Key.left or key == keyboard.Key.right:
            self.fork_y_speed = 0.0

        if key == keyboard.Key.home or key == keyboard.Key.end:
            self.fork_pitch_speed = 0.0


if __name__ == "__main__":
    ecal_pub = KeyBoardPublisher("vehicle/status", "keyboard/cmd")
    # ecal_pub = KeyBoardPublisher("vehicl__state", "keyboard/cmd")
    ecal_pub.run()
