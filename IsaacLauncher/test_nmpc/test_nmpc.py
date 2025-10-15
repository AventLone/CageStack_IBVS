import ecal.core.core as ecal_core
from ecal.core.subscriber import ProtoSubscriber
from ecal.core.publisher import ProtoPublisher
from protos import VehicleControl_pb2
import math, sys, queue, asyncio, time

from test_nmpc.nmpc import NMPC
from collections import deque
from queue import Queue
import numpy as np
from threading import Thread, Lock
import multiprocessing
from multiprocessing import Process

class TestNMPC(Thread):
    def __init__(self) -> None:
        super().__init__(daemon=True)
        if not ecal_core.is_initialized():
            ecal_core.initialize(sys.argv, "Test NMPC")

        self.goal_subscriber = ProtoSubscriber("goal", VehicleControl_pb2.Pose)
        self.state_subscriber = ProtoSubscriber("vehicle/status", VehicleControl_pb2.State)
        self.cmd_publisher = ProtoPublisher("nmpc_cmd", VehicleControl_pb2.State)

        # self.goal_subscriber.set_callback(self.goalHandler)
        # self.state_subscriber.set_callback(self.stateHandler)

        self.nmpc = NMPC()

        self.cmd_queue = deque()
        self.lock = Lock()
        # self.cmd_queue = Queue()
        # self.cmd = deque(maxlen=1)
        self.goal = None
        self.state = None

        print("Task start...")

    def __del__(self):
        if ecal_core.is_initialized():
            ecal_core.finalize()

        print("Task done.")

    def shutdown(self):
        if ecal_core.is_initialized():
            ecal_core.finalize()
        self.join(timeout=2.0)
        print("Task done.")

    # def pubCmd(self):
    def run(self):
        while ecal_core.ok():

            # with self.lock:
            #     cmd:VehicleControl_pb2.State = self.cmd_queue.popleft()
            # self.cmd_publisher.send(cmd)
            if len(self.cmd_queue) > 0:
                cmd:VehicleControl_pb2.State = self.cmd_queue.popleft()
            else:
                cmd = VehicleControl_pb2.State()
            self.cmd_publisher.send(cmd)
            time.sleep(0.05)

    # def run(self):
    def pubCmd(self):
        while ecal_core.ok():
            t0 = time.time()
            _, goal, _ = self.goal_subscriber.receive(1)
            _, state, _ = self.state_subscriber.receive(1)

            # if self.goal is None or self.state is None:
            if goal.x == 0.0:
                print("Did not receive goal yet...")
                time.sleep(1.0)
                continue

            drive_velocity = state.drive_velocity
            steer_angle = state.steer_angle

            # input_goal = np.array([self.goal.x, self.goal.y, self.goal.yaw])
            # input_state = np.array([self.state.drive_velocity, self.state.steer_angle])

            input_goal = np.array([goal.x, goal.y, goal.yaw])
            input_state = np.array([state.drive_velocity, state.steer_angle])

            self.nmpc.setGoalAndState(input_goal, input_state)

            result = self.nmpc.solve()

            if result is not None:
                result_u, result_x = result
                cmd_que = deque()
                for u in result_u:
                    cmd = VehicleControl_pb2.State()
                    # cmd.drive_acc = u[0]
                    # drive_velocity += self.nmpc._params.dt * u[0]
                    cmd.drive_velocity = u[0]
                    cmd.steer_velocity = u[1]
                    steer_angle += self.nmpc._params.dt * u[1]
                    cmd.steer_angle = steer_angle
                    cmd_que.append(cmd)

                with self.lock:
                    self.cmd_queue = cmd_que
            t1 = time.time()

            print(f"Elapse (ms): {int((t1 - t0) * 1000)}.")
            time.sleep(0.001)

if __name__ == "__main__":

    test_nmpc = TestNMPC()
    test_nmpc.start()
    try:
        test_nmpc.pubCmd()   # 主线程跑发布循环
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        test_nmpc.shutdown()    # 永远走到这里做清理
