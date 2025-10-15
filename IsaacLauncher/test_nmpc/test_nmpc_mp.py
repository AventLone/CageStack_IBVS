import sys, time, numpy as np
import multiprocessing as mp
import queue as pyqueue
from collections import deque

import ecal.core.core as ecal_core
from ecal.core.publisher import ProtoPublisher
from ecal.core.subscriber import ProtoSubscriber

from protos import VehicleControl_pb2
from test_nmpc.nmpc import NMPC


class SolverProc(mp.Process):
    """在独立进程里跑订阅与NMPC求解，把'最新一批'控制计划放到队列"""
    def __init__(self, plan_q: mp.Queue, stop_evt: mp.Event):
        super().__init__(daemon=True)
        self.plan_q = plan_q
        self.stop_evt = stop_evt

    def run(self):
        ecal_core.initialize(sys.argv, "Test NMPC Solver")
        goal_sub  = ProtoSubscriber("goal", VehicleControl_pb2.Pose)
        state_sub = ProtoSubscriber("vehicle/status", VehicleControl_pb2.State)
        nmpc = NMPC()

        try:
            while not self.stop_evt.is_set() and ecal_core.ok():
                # 短超时，避免长时间阻塞
                _, goal,  _ = goal_sub.receive(1)
                _, state, _ = state_sub.receive(1)
                if goal is None or state is None or goal.x == 0.0:
                    print("Did not receive goal yet...")
                    time.sleep(1.0)
                    continue

                input_goal  = np.array([goal.x, goal.y, goal.yaw], dtype=float)
                input_state = np.array([state.drive_velocity, state.steer_angle], dtype=float)
                nmpc.setGoalAndState(input_goal, input_state)

                # t0 = time.time()
                result = nmpc.solve()
                # print(f"Solve ms: {(time.time()-t0)*1000:.1f}")

                if result is None:
                    continue

                result_u, _ = result
                steer = float(state.steer_angle)
                dt = float(nmpc._params.dt)

                # 仅传基本数值(元组)，跨进程更稳；主进程再组装为proto
                plan = []
                for u in result_u:
                    v = float(u[0]); w = float(u[1])
                    steer += dt * w
                    plan.append((v, w, steer))

                # 只保留“最新一批”：清空旧的再放新的（非阻塞）
                try:
                    while True:
                        self.plan_q.get_nowait()
                except pyqueue.Empty:
                    pass
                try:
                    self.plan_q.put_nowait(plan)
                except pyqueue.Full:
                    pass
        finally:
            if ecal_core.is_initialized():
                ecal_core.finalize()


def pub_loop(plan_q: mp.Queue, stop_evt: mp.Event):
    """主进程：固定20Hz发布；空档保持上一条"""
    ecal_core.initialize(sys.argv, "Test NMPC Pub")
    pub = ProtoPublisher("nmpc_cmd", VehicleControl_pb2.State)

    period = 0.1
    next_t = time.perf_counter()
    pending = deque()  # 待发的本地批
    last_msg = VehicleControl_pb2.State()  # 安全保持

    try:
        while not stop_evt.is_set() and ecal_core.ok():
            next_t += period

            # 尽量拿到“最新一批”，丢掉更旧的
            new_plan = None
            try:
                while True:
                    new_plan = plan_q.get_nowait()  # 取到最后一次
            except pyqueue.Empty:
                pass
            if new_plan:
                pending = deque(new_plan)

            # 取一条来发；空就保持上一条
            if pending:
                v, w, steer = pending.popleft()
                msg = VehicleControl_pb2.State()
                msg.drive_velocity = v
                msg.steer_velocity = w
                msg.steer_angle    = steer
                # last_msg = msg

                pub.send(msg)
            # else:
            #     msg = last_msg

            # pub.send(msg)

            # 精确对齐20Hz
            # sleep = next_t - time.perf_counter()
            # if sleep > 0:
            #     time.sleep(sleep)
            # else:
            #     next_t = time.perf_counter()
            time.sleep(period)
    finally:
        if ecal_core.is_initialized():
            ecal_core.finalize()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)  # 跨平台更稳
    plan_q  = mp.Queue(maxsize=1)             # 只存“最新一批”
    stop_evt = mp.Event()

    solver = SolverProc(plan_q, stop_evt)
    solver.start()
    print("Task begin...")
    try:
        pub_loop(plan_q, stop_evt)
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        stop_evt.set()
        solver.join(timeout=2.0)
        if solver.is_alive():
            solver.terminate()
        print("Task done.")
