import math
import casadi
from dataclasses import dataclass
import numpy as np

@dataclass
class Params:
    horizon: int = 50
    dt: float = 0.2

    max_acc: float = math.pi
    max_velocity: float = math.pi
    max_steer_velocity: float = math.pi / 6.0
    max_steer_angle: float = math.pi * 0.4

    wheel_base: float = 1.5
    wheel_radius: float = 0.3

# class NMPC:
#     def __init__(self) -> None:
#         self._params = Params()
#         self._NLP = casadi.Opti()
#         opts = {
#             "ipopt.sb": "yes",
#             "ipopt.print_level": 0,
#             "print_time": 0,
#             "ipopt.max_iter": 200,
#             "ipopt.acceptable_tol": 1e-3,
#             "ipopt.acceptable_obj_change_tol": 1e-3
#         }
#         self._NLP.solver("ipopt", opts)

#         self._goal = self._NLP.parameter(3)
#         self._x_0 = self._NLP.parameter(5)

#         self._u_s = self._NLP.variable(2, self._params.horizon)
#         self._x_s = self._NLP.variable(5, self._params.horizon + 1)

#         self._F = casadi.diag(casadi.MX([3.2, 600.7, 300.6, 3.2]))
#         self._Q = casadi.diag(casadi.MX([3.2, 600.7, 300.6, 3.2]))
#         self._R = casadi.diag(casadi.MX([0.01, 0.05]))

#         self.buildModel()

#     def setGoalAndState(self, goal, state):
#         self._NLP.set_value(self._goal, goal)
#         self._NLP.set_value(self._x_0, [0.0, 0.0, 0.0, state[0], state[1]])

#     def solve(self):
#         try:
#             solution = self._NLP.solve()
#             # return solution.value(self._u_s), solution.value(self._x_s)
#             result_u = solution.value(self._u_s)
#             result_x = solution.value(self._x_s)

#             # return np.vsplit(result_u.T, result_u.shape[1]), np.vsplit(result_x.T, result_x.shape[1])
#             return [
#                 r.squeeze(0) for r in np.vsplit(result_u.T, result_u.shape[1])
#             ], [
#                 r.squeeze(0) for r in np.vsplit(result_x.T, result_x.shape[1])
#             ]
#         except Exception as e:
#             print(e)
#             return None

#     def _sysFunc(self, vec_x:casadi.MX, vec_u:casadi.MX):
#         th = vec_x[2]
#         v = vec_x[3]
#         delta = vec_x[4]

#         a = vec_u[0]
#         w = vec_u[1]

#         return casadi.vertcat(v * self._params.wheel_radius * casadi.MX.cos(th),
#                           v * self._params.wheel_radius *  casadi.MX.sin(th),
#                           -v * self._params.wheel_radius / self._params.wheel_base * casadi.MX.tan(delta),
#                           a, w)

#     def _rk4(self, x_k: casadi.MX, u_k: casadi.MX, dt:float):
#         k1 = self._sysFunc(x_k, u_k)
#         k2 = self._sysFunc(x_k + 0.5 * dt * k1, u_k)
#         k3 = self._sysFunc(x_k + 0.5 * dt * k2, u_k)
#         k4 = self._sysFunc(x_k + dt * k3, u_k)
#         return x_k + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0

#     def buildModel(self):
#         # Control variables
#         a = self._u_s[0, :]
#         w = self._u_s[1, :]

#         # State variables
#         p_x = self._x_s[0, :]
#         velocity = self._x_s[3, :]
#         steer_angle = self._x_s[4, :]

#         self._NLP.subject_to(self._x_s[:, 0] == self._x_0)
#         self._NLP.subject_to(p_x <= self._goal[0])
#         self._NLP.subject_to([-self._params.max_acc <= a, a <= self._params.max_acc])
#         self._NLP.subject_to([-self._params.max_velocity <= velocity, velocity <= self._params.max_velocity])
#         self._NLP.subject_to([-self._params.max_steer_velocity <= w, w <= self._params.max_steer_velocity])
#         self._NLP.subject_to([-self._params.max_steer_angle <= steer_angle, steer_angle <= self._params.max_steer_angle])

#         cost = casadi.MX(0.0)

#         for k in range(self._params.horizon):
#             x_k = self._x_s[:, k]
#             u_k = self._u_s[:, k]

#             # Error
#             dx = self._goal[0] - x_k[0]
#             dy = self._goal[1] - x_k[1]
#             th = x_k[2]
#             dv = (0.5 * dx)**2 - x_k[3]**2
#             e_x = casadi.MX.cos(th) * dx + casadi.sin(th) * dy
#             e_y = -casadi.MX.sin(th) * dx + casadi.MX.cos(th) * dy
#             e_th = casadi.atan2(casadi.MX.sin(self._goal[2]-th), casadi.MX.cos(self._goal[2] - th))

#             e_k = casadi.vertcat(e_x, e_y, e_th, dv)

#             cost += e_k.T @ self._Q @ e_k + u_k.T @ self._R @ u_k

#             x_next = self._rk4(x_k, u_k, self._params.dt)
#             self._NLP.subject_to(self._x_s[:, k + 1] == x_next)

#         x_n = self._x_s[:, self._params.horizon]
#         dx = self._goal[0] - x_n[0]
#         dy = self._goal[1] - x_n[1]
#         th = x_n[2]
#         dv = (0.5 * dx)**2 - x_n[3]**2
#         e_x = casadi.MX.cos(th) * dx + casadi.sin(th) * dy
#         e_y = -casadi.MX.sin(th) * dx + casadi.MX.cos(th) * dy
#         e_th = casadi.atan2(casadi.MX.sin(self._goal[2]-th), casadi.MX.cos(self._goal[2] - th))

#         e_n = casadi.vertcat(e_x, e_y, e_th, dv)

#         cost += e_n.T @ self._F @ e_n

#         # self._NLP.set_initial(self._x_s, casadi.DM.zeros(5, self._params.horizon +1))
#         self._NLP.minimize(cost)


class NMPC:
    def __init__(self) -> None:
        self._params = Params()
        self._NLP = casadi.Opti()
        opts = {
            "ipopt.sb": "yes",
            "ipopt.print_level": 0,
            "print_time": 0,
            "ipopt.max_iter": 1000,
            "ipopt.acceptable_tol": 1e-3,
            "ipopt.acceptable_obj_change_tol": 1e-3,
            'jit': True, 'compiler': 'shell',
            'jit_options': {'flags': ['-O3','-march=native']}
        }
        self._NLP.solver("ipopt", opts)

        self._goal = self._NLP.parameter(3)
        self._x_0 = self._NLP.parameter(4)

        self._u_s = self._NLP.variable(2, self._params.horizon)
        self._x_s = self._NLP.variable(4, self._params.horizon + 1)

        self._F = casadi.diag(casadi.MX([3.2, 600.7, 3.6]))
        self._Q = casadi.diag(casadi.MX([3.2, 600.7, 3.6]))
        self._R = casadi.diag(casadi.MX([1.0, 1.0]))

        self.buildModel()

    def setGoalAndState(self, goal, state):
        self._NLP.set_value(self._goal, goal)
        self._NLP.set_value(self._x_0, [0.0, 0.0, 0.0, state[1]])

    def solve(self):
        try:
            solution = self._NLP.solve()
            # return solution.value(self._u_s), solution.value(self._x_s)
            result_u = solution.value(self._u_s)
            result_x = solution.value(self._x_s)

            # return np.vsplit(result_u.T, result_u.shape[1]), np.vsplit(result_x.T, result_x.shape[1])
            return [
                r.squeeze(0) for r in np.vsplit(result_u.T, result_u.shape[1])
            ], [
                r.squeeze(0) for r in np.vsplit(result_x.T, result_x.shape[1])
            ]
        except Exception as e:
            print(e)
            return None

    def _sysFunc(self, vec_x:casadi.MX, vec_u:casadi.MX):
        th = vec_x[2]
        # v = vec_x[3]
        delta = vec_x[3]

        v = vec_u[0]
        w = vec_u[1]

        return casadi.vertcat(v * self._params.wheel_radius * casadi.MX.cos(th),
                          v * self._params.wheel_radius *  casadi.MX.sin(th),
                          -v * self._params.wheel_radius / self._params.wheel_base * casadi.MX.tan(delta),
                          w)

    def _sysFunc2(self, vec_x:casadi.MX, vec_u:casadi.MX):
        th = vec_x[2]
        # v = vec_x[3]
        delta = vec_x[3]

        v = vec_u[0]
        w = vec_u[1]

        return vec_x + self._params.dt * casadi.vertcat(v * self._params.wheel_radius * casadi.MX.cos(th),
                          v * self._params.wheel_radius *  casadi.MX.sin(th),
                          -v * self._params.wheel_radius / self._params.wheel_base * casadi.MX.tan(delta),
                          w)

    def _rk4(self, x_k: casadi.MX, u_k: casadi.MX, dt:float):
        k1 = self._sysFunc(x_k, u_k)
        k2 = self._sysFunc(x_k + 0.5 * dt * k1, u_k)
        k3 = self._sysFunc(x_k + 0.5 * dt * k2, u_k)
        k4 = self._sysFunc(x_k + dt * k3, u_k)
        return x_k + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0

    def buildModel(self):
        # Control variables
        velocity = self._u_s[0, :]
        w = self._u_s[1, :]

        # State variables
        p_x = self._x_s[0, :]
        steer_angle = self._x_s[3, :]

        self._NLP.subject_to(self._x_s[:, 0] == self._x_0)
        self._NLP.subject_to(p_x <= self._goal[0])
        # self._NLP.subject_to([-self._params.max_acc <= a, a <= self._params.max_acc])
        self._NLP.subject_to([-self._params.max_velocity <= velocity, velocity <= self._params.max_velocity])
        self._NLP.subject_to([-self._params.max_steer_velocity <= w, w <= self._params.max_steer_velocity])
        self._NLP.subject_to([-self._params.max_steer_angle <= steer_angle, steer_angle <= self._params.max_steer_angle])

        cost = casadi.MX(0.0)

        for k in range(self._params.horizon):
            x_k = self._x_s[:, k]
            u_k = self._u_s[:, k]

            # Error
            dx = self._goal[0] - x_k[0]
            dy = self._goal[1] - x_k[1]
            th = x_k[2]
            # dv = (0.5 * dx)**2 - x_k[3]**2
            e_x = casadi.MX.cos(th) * dx + casadi.sin(th) * dy
            e_y = -casadi.MX.sin(th) * dx + casadi.MX.cos(th) * dy
            e_th = casadi.atan2(casadi.MX.sin(self._goal[2] - th), casadi.MX.cos(self._goal[2] - th))

            e_k = casadi.vertcat(e_x, e_y, e_th)

            cost += e_k.T @ self._Q @ e_k + u_k.T @ self._R @ u_k

            # x_next = self._rk4(x_k, u_k, self._params.dt)
            x_next = self._sysFunc2(x_k, u_k)
            self._NLP.subject_to(self._x_s[:, k + 1] == x_next)

        x_n = self._x_s[:, self._params.horizon]
        dx = self._goal[0] - x_n[0]
        dy = self._goal[1] - x_n[1]
        th = x_n[2]
        # dv = (0.5 * dx)**2 - x_n[3]**2
        e_x = casadi.MX.cos(th) * dx + casadi.sin(th) * dy
        e_y = -casadi.MX.sin(th) * dx + casadi.MX.cos(th) * dy
        e_th = casadi.atan2(casadi.MX.sin(self._goal[2] - th), casadi.MX.cos(self._goal[2] - th))

        e_n = casadi.vertcat(e_x, e_y, e_th)

        cost += e_n.T @ self._F @ e_n

        # self._NLP.set_initial(self._x_s, casadi.DM.zeros(5, self._params.horizon +1))
        self._NLP.minimize(cost)
