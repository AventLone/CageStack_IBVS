import sys, os, tty, termios, select, signal, math
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from ackermann_msgs.msg import AckermannDriveStamped as Ack
from pynput import keyboard
from rclpy.clock import Clock, ClockType

class _RawTerminal:
    def __enter__(self):
        self._fd = sys.stdin.fileno()
        self._old = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)  # 非 canonical，低延迟
        return self
    def __exit__(self, exc_type, exc, tb):
        termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old)

def _read_keys():
    """非阻塞读尽键盘缓冲，返回字符列表。"""
    chars = []
    while True:
        r, _, _ = select.select([sys.stdin], [], [], 0)
        if not r:
            break
        ch = os.read(sys.stdin.fileno(), 1)
        if not ch:
            break
        chars.append(ch)
    return b"".join(chars)

class KeyTeleopAckermann(Node):
    def __init__(self):
        super().__init__('key_teleop_ackermann')
        # 参数
        self.max_speed   = float(self.declare_parameter('max_speed', 2.0).value)   # m/s
        self.max_steer   = float(self.declare_parameter('max_steer', math.pi / 2).value)   # rad
        self.accel_step  = float(self.declare_parameter('accel_step', 0.1).value)   # m/s per key
        self.steer_step  = float(self.declare_parameter('steer_step', 0.05).value)  # rad per key
        self.rate_hz     = float(self.declare_parameter('publish_rate', 50.0).value)  # Hz
        self.topic       = self.declare_parameter('topic', 'ackermann_cmd').value

        self.drive_wheels_velocity = 0.0
        self.steer_wheels_angle_speed = 0.0

        qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST
        )
        self.pub = self.create_publisher(Ack, self.topic, qos)
        period = 1.0 / max(1.0, self.rate_hz)
        self.timer = self.create_timer(period, self._timerCallback)

        self.msg = Ack()
        self.msg.drive.acceleration = math.pi * 10.0

        self.speed = 0.0
        self.steer = 0.0
        self.get_logger().info(
            "Keys: W/S speed, A/D steer, Arrows OK, SPACE brake, C center, Q/Esc quit\n"
            f"Params: max_speed={self.max_speed:.2f}, max_steer={self.max_steer:.2f}, "
            f"accel_step={self.accel_step:.2f}, steer_step={self.steer_step:.2f}, "
            f"rate={self.rate_hz:.1f}Hz, topic={self.topic}"
        )

         # 键盘线程
        listener = keyboard.Listener(on_press=self._onPress,
                                     on_release=self._onRelease)
        listener.daemon = True
        listener.start()

        # self.timer = self.create_timer(0.01, self._timerCallback, clock=Clock(clock_type=ClockType.STEADY_TIME))

    def _timerCallback(self):
        self.msg.header.stamp = self.get_clock().now().to_msg()
        self.msg.drive.speed = self.drive_wheels_velocity
        self.msg.drive.steering_angle += self.steer_wheels_angle_speed

        self.msg.drive.steering_angle = self.clamp(self.msg.drive.steering_angle, -self.max_steer, self.max_steer)

        self.pub.publish(self.msg)

    @staticmethod
    def clamp(input, lower, upper):
        return max(lower, min(input, upper))

    def _onPress(self, key):
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

    def _onRelease(self, key):
        try:
            k = key.char.lower()
        except AttributeError:
            k = None

        if k == 'w' or k == 's':
            self.drive_wheels_velocity = 0.0

        if k == 'a' or k == 'd':
            self.steer_wheels_angle_speed = 0.0


def main():
    # signal.signal(signal.SIGINT, lambda s, f: rclpy.shutdown())
    # rclpy.init()
    # with _RawTerminal():
    #     node = KeyTeleopAckermann()
    #     rclpy.spin(node)
    # # 退出后终端恢复
    # if rclpy.ok():
    #     rclpy.shutdown()
    rclpy.init()
    node = KeyTeleopAckermann()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
