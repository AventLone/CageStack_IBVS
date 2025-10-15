import time
import casadi as ca
import logging
logging.basicConfig(level=logging.INFO)

def main():
    # 1) 构造 Opti 与求解器
    opti = ca.Opti()
    opts = {
        "ipopt.sb": "yes",
        "ipopt.print_level": 0,
        "print_time": 0,
        "ipopt.max_iter": 99,
        "ipopt.acceptable_tol": 1e-3,
        "ipopt.acceptable_obj_change_tol": 1e-3,
    }
    opti.solver("ipopt", opts)

    # 2) 变量/参数（列向量形状与 C++ 版一致）
    x = opti.variable(3, 1)           # 3x1
    p = opti.parameter(2, 1)          # 2x1

    # 3) 目标函数
    f = ca.mtimes(x.T, x)             # x^T x
    opti.minimize(f)

    # 4) 约束
    opti.subject_to(0 <= x)
    opti.subject_to(6*x[0] + 3*x[1] + 2*x[2] - p[0] == 0)
    opti.subject_to(p[1]*x[0] + x[1] - x[2] - 1 == 0)

    # 5) 赋值与初值
    opti.set_value(p, [5.0, 1.0])     # 或 [[5.0],[1.0]]，两者等价
    opti.set_initial(x, [0.15, 0.15, 0.0])

    # 6) 求解
    t0 = time.time()
    sol = opti.solve()
    t1 = time.time()

    # 7) 取解并打印
    x_opt = sol.value(x)              # DM 形状 3x1
    f_opt = sol.value(f)              # 1x1
    print("-------------------------------------------------")
    print("Optimal x:\n", x_opt)        # 转 numpy
    print("Optimal f:\n", f_opt)
    print("Elapse (ms):", int((t1 - t0) * 1000))

if __name__ == "__main__":
    main()
    # F = ca.diag(ca.MX([3.2, 600.7, 300.6, 3.2]))
    # print()
    logging.exception("No")