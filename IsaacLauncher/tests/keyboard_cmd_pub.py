import sys, time
import ecal.core.core as ecal_core
from ecal.core.publisher import StringPublisher
from pynput import keyboard
import math


class KeyboardCmdPub_E:
    def __init__(self, topic_name: str):
        if not ecal_core.is_initialized():
            ecal_core.initialize(sys.argv, "Keyboard command")
        self.pub = StringPublisher(topic_name)
        self.drive_wheels_velocity = 0.0
        self.steer_wheels_angle = 0.0
        self.fork_z = 0.0
        self.fork_y = 0.0
        self.fork_pitch = 0.0

        # self.restore = -1.0

        self.listener = keyboard.Listener(on_press=self.onPress, on_release=self.onRelease)
        self.listener.daemon = True

        self.drive_v = math.pi
        self.steer_v = math.radians(25.0)
        self.fork_pitch_v = math.radians(10.0)
        self.fork_z_v = 0.8
        self.fork_y_v = 0.5

    def __del__(self):
        if ecal_core.is_initialized():
            ecal_core.finalize()

    def start(self):
        self.listener.start()

        while ecal_core.ok():
            time.sleep(0.001)

    def onPress(self, key):

        try:
            k = key.char.lower()
        except AttributeError:
            k = None
        
        if k == 'w':
            self.drive_wheels_velocity = self.drive_v
        elif k == "s":
            self.drive_wheels_velocity = -self.drive_v

        if k == 'a':
            self.steer_wheels_angle = self.steer_v
        elif k == 'd':
            self.steer_wheels_angle = -self.steer_v


        if key == keyboard.Key.up:
            self.fork_z = self.fork_z_v
        elif key == keyboard.Key.down:
            self.fork_z = -self.fork_z_v

        if key == keyboard.Key.left:
            self.fork_y = self.fork_y_v
        elif key == keyboard.Key.right:
            self.fork_y = -self.fork_y_v

        if key==keyboard.Key.home:
            self.fork_pitch = self.fork_pitch_v
        elif key==keyboard.Key.end:
            self.fork_pitch = -self.fork_pitch_v

        # if key == keyboard.Key.esc:   # Resotre
        #     self.steer_wheels_angle = 0.0
        #     self.fork_z = 0.0
        #     self.fork_pitch = 0.0
        #     self.fork_y = 0.0

        msg = f"{self.drive_wheels_velocity} {self.steer_wheels_angle} {self.fork_z} {self.fork_y} {self.fork_pitch}"
        self.pub.send(msg)

    
    def onRelease(self, key):
        try:
            k = key.char.lower()
        except AttributeError:
            k = None

        if k == 'w' or k =='s':
            self.drive_wheels_velocity = 0.0

        if k == 'a' or k=='d':
            self.steer_wheels_angle = 0.0

        if key == keyboard.Key.up or key == keyboard.Key.down:
            self.fork_z = 0.0

        if key == keyboard.Key.left or key == keyboard.Key.right:
            self.fork_y = 0.0

        if key==keyboard.Key.home or key==keyboard.Key.end:
            self.fork_pitch = 0.0
        
        msg = f"{self.drive_wheels_velocity} {self.steer_wheels_angle} {self.fork_z} {self.fork_y} {self.fork_pitch}"
        self.pub.send(msg)


if __name__ == "__main__":
    keyboard_cmd_pub = KeyboardCmdPub_E("ackermann_control_cmd")
    keyboard_cmd_pub.start()
