import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

import sys, time
from pynput import keyboard
import math

def clamp(n, smallest, largest):
    return max(smallest, min(n, largest))

# --- ROS2 Node ---
class KeyboardJointPublisher(Node):
    DRIVE_SPEED = math.pi * 2
    STEER_SPEED = math.pi / 2
    LIFT_SPEED = 0.1
    SIDESHIFT_SPEED = 0.06

    DRIVE_JOINT_NAME = "drive_joint"
    STEER_JOINT_NAME = "steer_joint"
    LIFT_Z = "lift_z"
    LIFT_Y = "lift_y"

    def __init__(self):
        super().__init__('keyboard_joint_publisher')
        self.pub = self.create_publisher(JointState, '/lola/joint_command', 10)
        self.joint_names = ["drive_joint", "steer_joint", "lift_z", "lift_y"]
        self.init_velocities = [0.0 for _ in self.joint_names]
        self.init_positions = [0.0 for _ in self.joint_names]

        self.fork_z = 0.0
        self.fork_y = 0.0        

        self.step = 0.05  # radians per key press
        self.get_logger().info("Keyboard Joint Publisher started. Press Ctrl-C to exit.")

        self.keyboard_listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        self.keyboard_listener.daemon = True

    def run(self):
        self.keyboard_listener.start()

    def on_press(self, key: keyboard.KeyCode):
        try:
            k = key.char.lower()
        except AttributeError:
            k = None
        
        # if k not in {'w', 's', 'a', 'd', 
        #              keyboard.Key.up, keyboard.Key.down, keyboard.Key.left, keyboard.Key.right,
        #              keyboard.Key.esc}:
        #     return

        joint_cmd_msg = JointState()
        joint_cmd_msg.header.stamp = self.get_clock().now().to_msg()
        joint_cmd_msg.name = self.joint_names
        joint_cmd_msg.velocity = self.init_velocities

        if k == 'w':
            joint_cmd_msg.velocity[0] = KeyboardJointPublisher.DRIVE_SPEED
        elif k == "s":
            joint_cmd_msg.velocity[0] = -KeyboardJointPublisher.DRIVE_SPEED

        if k == 'a':
            joint_cmd_msg.velocity[1] = KeyboardJointPublisher.STEER_SPEED
        elif k == 'd':
            joint_cmd_msg.velocity[1] = -KeyboardJointPublisher.STEER_SPEED

        if key == keyboard.Key.up:
            self.fork_z += 0.01
            self.fork_z = clamp(self.fork_z, 0.0, 1.0)
        elif key == keyboard.Key.down:
            self.fork_z -= 0.01
            self.fork_z = clamp(self.fork_z, 0.0, 1.0)

        if key == keyboard.Key.left:
            self.fork_y -= 0.01
            self.fork_y = clamp(self.fork_y, -0.3, 0.3)
        elif key == keyboard.Key.right:
            self.fork_y += 0.01
            self.fork_y = clamp(self.fork_y, -0.3, 0.3)

        if key == keyboard.Key.esc:   # Resotre
            self.fork_z = 0.0
            self.fork_y = 0.0

        joint_cmd_msg.position = [0.0, 0.0, self.fork_z, self.fork_y]

        self.pub.publish(joint_cmd_msg)

    def on_release(self, key: keyboard.KeyCode):
        try:
            k = key.char.lower()
        except AttributeError:
            k = None

        if k not in {'w', 's', 'a', 'd'}:
            return

        joint_cmd_msg = JointState()
        joint_cmd_msg.header.stamp = self.get_clock().now().to_msg()

        if k == 'w' or k =='s':
            joint_cmd_msg.name.append(KeyboardJointPublisher.DRIVE_JOINT_NAME)
            joint_cmd_msg.velocity.append(0.0)

        if k == 'a' or k=='d':
            joint_cmd_msg.name.append(KeyboardJointPublisher.STEER_JOINT_NAME)
            joint_cmd_msg.velocity.append(0.0)

        # if key == keyboard.Key.up or key == keyboard.Key.down:
        #     self.fork_z = 0.0

        # if key == keyboard.Key.left or key == keyboard.Key.right:
        #     self.fork_y = 0.0

        # if key==keyboard.Key.home or key==keyboard.Key.end:
        #     self.fork_pitch = 0.0

        self.pub.publish(joint_cmd_msg)

def main(args=None):
    rclpy.init(args=args)
    node = KeyboardJointPublisher()
    node.run()
    while rclpy.ok():
        time.sleep(0.01)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()