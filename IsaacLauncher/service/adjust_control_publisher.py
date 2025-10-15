import numpy as np
from modules.planner import load_config, VehicleState, Trajectory, VehicleModel, \
    VelocityControllerConfig, VelocityController, PurePursuitController
import threading, time, sys
import ecal.core.core as ecal_core
from ecal.core.publisher import ProtoPublisher
from ecal.core.subscriber import ProtoSubscriber
from protos import vehicle_state_msg_pb2
from utils import PoseTrans, LoggerUtil

logger = LoggerUtil.get_logger("adjust_ctrl")

class AdjustControlPublisher(threading.Thread):
    def __init__(self, pub : str, sub : str, time_period : float = 0.01, wheelbase : float = 2.9):
        """
        AdjustControlPublisher init
        :param pub: pub topic name
        :param sub: sub topic name
        :param time_period: time_period for pub
        :param wheelbase:
        """
        super().__init__()
        self.time_period = time_period
        self.config = load_config()
        self.config.vehicle.wheelbase = wheelbase

        self.vehicle_state = VehicleState()
        self.vehicle_state_lock = threading.Lock()

        self.controller = None
        self.controller_lock = threading.Lock()

        self.pub = ProtoPublisher(pub, vehicle_state_msg_pb2.VehicleStateMsg)
        self.run_thread = True

        self.sub = ProtoSubscriber(sub, vehicle_state_msg_pb2.VehicleStateMsg)
        self.sub.set_callback(self.vehicle_state_callback)

        self.adjust_control = False
        self.fork_z = 0

    def __del__(self):
        self.run_thread = False
        if ecal_core.is_initialized():
            ecal_core.finalize()


    def run(self) -> None:
        """线程启动时执行的方法"""
        while self.run_thread:
            if self.get_controller() is None:
                continue
            steering, target_velocity = self.get_controller().compute_control(self.get_vehicle_state(), self.time_period)
            logger.info(f"target_velocity = {target_velocity} steering = {steering}")
            # pub_msg = f"{target_velocity} {steering}"
            pub_msg = vehicle_state_msg_pb2.VehicleStateMsg()
            pub_msg.drive_velocity = target_velocity
            pub_msg.steer_angle = -steering
            pub_msg.fork_z = self.fork_z
            self.pub.send(pub_msg)
            print(pub_msg)
            time.sleep(self.time_period)
        self.run_thread = False

    def stop(self) -> None:
        self.run_thread = False

    def set_vehicle_state(self, vehicle_state):
        self.vehicle_state_lock.acquire()
        self.vehicle_state = vehicle_state
        self.vehicle_state_lock.release()

    def get_vehicle_state(self):
        self.vehicle_state_lock.acquire()
        vehicle_state = self.vehicle_state
        self.vehicle_state_lock.release()
        return vehicle_state

    def vehicle_state_callback(self, topic_name, msg, msg_time) -> None:
        # print(msg)
        vehicle_state = VehicleState()
        vehicle_state.velocity = msg.drive_velocity
        vehicle_state.steering_angle = msg.steer_angle
        if msg.HasField("robot_pose"):
            vehicle_state.position_x = msg.robot_pose.position.x
            vehicle_state.position_y = msg.robot_pose.position.y
            roll, pitch, yaw = PoseTrans.quaternion_to_euler(msg.robot_pose.orientation.x, msg.robot_pose.orientation.y,\
                                                  msg.robot_pose.orientation.z, msg.robot_pose.orientation.w)
            vehicle_state.yaw_angle = yaw
        # print(vehicle_state)
        self.set_vehicle_state(vehicle_state)

        if self.adjust_control is False and msg.adjust_control is True and \
                msg.HasField("robot_pose") and msg.HasField("target_pose"):
            start_pose = np.array([vehicle_state.position_x, vehicle_state.position_y, vehicle_state.yaw_angle, 0])
            _, _, yaw = PoseTrans.quaternion_to_euler(msg.target_pose.orientation.x, msg.target_pose.orientation.y,\
                                                             msg.target_pose.orientation.z, msg.target_pose.orientation.w)
            target_pose = np.array([msg.target_pose.position.x, msg.target_pose.position.y, yaw, 0])
            logger.info(f"start_pose = {start_pose} target_pose = {target_pose}")
            self.start_planner(start_pose, target_pose)
            self.fork_z = msg.target_pose.position.z
            self.adjust_control = True

        if msg.adjust_control is False:
            self.adjust_control = False


    def set_controller(self, controller):
        self.controller_lock.acquire()
        self.controller = controller
        self.controller_lock.release()

    def get_controller(self):
        self.controller_lock.acquire()
        controller = self.controller
        self.controller_lock.release()
        return controller

    def start_planner(self, start_pose : np.ndarray, end_pose : np.ndarray):
        """
        start path planner
        :param start_pose:
            x_coords: List of X coordinates
            y_coords: List of Y coordinates
            yaw_angles: List of heading angles in radians
            directions: List of movement directions (default: all forward)
        :param end_pose:
            x_coords: List of X coordinates
            y_coords: List of Y coordinates
            yaw_angles: List of heading angles in radians
            directions: List of movement directions (default: all forward)
        :return:
        """
        trajectory = Trajectory(self.config.trajectory)
        trajectory.add_waypoint(start_pose[0], start_pose[1], start_pose[2], start_pose[3])
        trajectory.add_waypoint(end_pose[0], end_pose[1], end_pose[2], end_pose[3])

        # Create velocity controller optimized for reverse driving
        velocity_config = VelocityControllerConfig()
        velocity_controller = VelocityController(velocity_config)

        controller = PurePursuitController(
            wheelbase=self.config.vehicle.wheelbase,
            config=self.config.pure_pursuit,
            trajectory=trajectory,
            velocity_controller=velocity_controller,
        )
        self.set_controller(controller)



if __name__ == "__main__":
    # 初始化 eCAL
    ecal_core.initialize(sys.argv, "Keyboard command")
    # publish
    ecal_pub = AdjustControlPublisher(pub="cmd/adjust_ctrl", sub="vehicle/state")
    ecal_pub.start()
    ecal_pub.start_planner(np.array([0, 0, 0, 0], np.array([-3, 0.3, -0.1, -1])))

    while ecal_core.ok():
        time.sleep(0.001)

    # 关闭 eCAL
    ecal_core.finalize()