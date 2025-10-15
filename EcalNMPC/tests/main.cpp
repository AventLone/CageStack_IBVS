// #include <casadi/casadi.hpp>
// #include <chrono>
//
// int main()
// {
//     auto begin = std::chrono::system_clock::now();
//     casadi::Opti nlp; // Constructed NLP using CasADi
//     const casadi::Dict opts = {
//         {"ipopt.max_iter", 80},
//         {"ipopt.print_level", 0},
//         {"print_time", 0},
//         {"ipopt.acceptable_tol", 1e-3},
//         {"ipopt.acceptable_obj_change_tol", 1e-3}
//     };
//     nlp.solver("ipopt", opts); // Choose IPOPT as solver
//
//     const casadi::MX x = nlp.variable(); // Define optimization variable
//     const casadi::MX f = casadi::MX::sin(x); // Define objective function
//
//     nlp.subject_to(-casadi::pi <= x <= casadi::pi); // Define constraints
//     nlp.set_initial(x, -1); // Set initial value of decision variables
//
//     nlp.minimize(f); // Set objective
//
//     const casadi::OptiSol solution = nlp.solve(); // Solve the optimization problem
//
//     /* Get the optimal solution */
//     const auto x_optimal = solution.value(x).nonzeros()[0];
//     const auto f_optimal = solution.value(f).nonzeros()[0];
//
//     /* Print the optimal solution */
//     std::cout << "Optimal solution of x: " << x_optimal << std::endl;
//     std::cout << "Optimal solution of f: " << f_optimal << std::endl;
//
//     auto end = std::chrono::system_clock::now();
//
//     auto elapse = std::chrono::duration<std::chrono::milliseconds>(end - begin);
//
//     return 0;
// }

#include <casadi/casadi.hpp>

/********************************************************************************
 * @brief Round a number to disired decimal places
 * @param data A group of numbers to be rounded
 * @param precision The number of decimal places to round to.
 ********************************************************************************/
template<typename T>
void roundDecimal(std::vector<T>& data, int precision)
{
    for (auto& i : data)
    {
        i = std::round(i * std::pow(10, precision)) / std::pow(10, precision);
    }
}

#include <iostream>

template<typename T>
static T wrapAngle(const T rad)
{
    constexpr double two_pi = 2.0 * M_PI;
    double r = std::fmod(rad + M_PI, two_pi);
    if (r < 0.0)
    {
        r += two_pi;
    }
    return r - M_PI; // ∈ [-π, π)
}

// int main()
// {
//     std::cout << wrapAngle(2 * M_PI) << std::endl;
//     return 0;
// }

// int main()
// {
//     auto begin = std::chrono::system_clock::now();
//     casadi::Opti nlp; // Constructed NLP using CasADi
//     const casadi::Dict opts = {
//         {"ipopt.sb", "yes"},
//         {"ipopt.print_level", 0},
//         {"print_time", 0},
//         {"ipopt.max_iter", 99},
//         {"ipopt.acceptable_tol", 1e-3},
//         {"ipopt.acceptable_obj_change_tol", 1e-3}
//     };
//     nlp.solver("ipopt", opts); // Choose IPOPT as solver
//
//     casadi::MX x = nlp.variable(3, 1); // Define optimization variable
//     casadi::MX f = casadi::MX::mtimes(x.T(), x); // Define objective function
//     casadi::MX p = nlp.parameter(2, 1); // Define parameter
//
//     /* Set constraints */
//     nlp.subject_to(0 <= x);
//     nlp.subject_to(6 * x(0) + 3 * x(1) + 2 * x(2) - p(0) == 0);
//     nlp.subject_to(p(1) * x(0) + x(1) - x(2) - 1 == 0);
//
//     const std::vector<double> temp = {5.0, 1.0};
//     nlp.set_value(p, temp); // Set the value of parameter p
//     nlp.set_initial(x, {0.15, 0.15, 0.0}); // Set the initial value of x
//
//     nlp.minimize(f); // Set objective
//
//
//     casadi::OptiSol solution = nlp.solve(); // Solve the optimization problem
//     auto end = std::chrono::system_clock::now();
//
//
//     /* Get the optimal solution */
//     auto x_optimal = solution.value(x);
//     auto f_optimal = solution.value(f);
//
//     // roundDecimal(x_optimal, 3);
//     // roundDecimal(f_optimal, 3);
//
//     /* Print the optimal solution */
//     std::printf("-------------------------------------------------\n");
//     std::cout << "Optimal solution of x: " << x_optimal << std::endl;
//     std::cout << "Optimal solution of f: " << f_optimal << std::endl;
//
//     auto elapse = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
//     std::cout << "Elapse: " << elapse << std::endl;
//
//     // nlp.set_value(p, {2.0, 9.0}); // Set the value of parameter p
//     //
//     // begin = std::chrono::system_clock::now();
//     // solution = nlp.solve(); // Solve the optimization problem
//     // end = std::chrono::system_clock::now();
//     //
//     // elapse = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
//     // std::cout << "Elapse: " << elapse << std::endl;
//     //
//     // /* Get the optimal solution */
//     // x_optimal = solution.value(x).nonzeros();
//     // f_optimal = solution.value(f).nonzeros();
//     // p_optimal = solution.value(p).nonzeros();
//     //
//     // roundDecimal(x_optimal, 3);
//     // roundDecimal(f_optimal, 3);
//     // roundDecimal(p_optimal, 3);
//     //
//     // /* Print the optimal solution */
//     // std::printf("-------------------------------------------------\n");
//     // std::cout << "Optimal solution of x: " << x_optimal << std::endl;
//     // std::cout << "Optimal solution of f: " << f_optimal << std::endl;
//     // std::cout << "Optimal solution of p: " << p_optimal << std::endl;
//
//
//     return 0;
// }

#include <vector>

// int main(int argc, char** argv)
// {
//     auto begin = std::chrono::system_clock::now();
//     using namespace casadi;
//
//     // vars x=[x1,x2,x3], params p=[p1,p2]
//     SX x = SX::sym("x", 3);
//     SX p = SX::sym("p", 2);
//
//     // objective
//     SX f = dot(x, x);
//
//     // equality constraints
//     SX g = vertcat(6 * x(0) + 3 * x(1) + 2 * x(2) - p(0),
//                    p(1) * x(0) + x(1) - x(2) - 1);
//
//     // build NLP
//     SXDict nlp;
//     nlp["x"] = x;
//     nlp["f"] = f;
//     nlp["g"] = g;
//     nlp["p"] = p;
//
//     // Dict opts;
//     // opts["ipopt.print_level"] = 0;
//     // opts["print_time"] = 0;
//     const casadi::Dict opts = {
//         {"ipopt.sb", "yes"},
//         {"ipopt.print_level", 0},
//         {"print_time", 0},
//         {"ipopt.max_iter", 80},
//         {"ipopt.acceptable_tol", 1e-3},
//         {"ipopt.acceptable_obj_change_tol", 1e-3}
//     };
//     Function solver = nlpsol("solver", "ipopt", nlp, opts);
//
//     // params (可用命令行覆盖：./solve p1 p2)
//     double p1 = 5.0, p2 = 1.0;
//     if (argc >= 3)
//     {
//         p1 = std::stod(argv[1]);
//         p2 = std::stod(argv[2]);
//     }
//
//     // 显式构造容器，避免歧义
//     DM pval = DM::vertcat(DMVector{DM(p1), DM(p2)});
//
//     DMDict arg;
//     arg["x0"] = DM::zeros(3);
//     arg["lbx"] = DM::zeros(3); // 0 <= x
//     arg["lbg"] = DM::zeros(2); // g == 0
//     arg["ubg"] = DM::zeros(2);
//     arg["p"] = pval;
//
//
//     DMDict res = solver(arg);
//     DM x_opt = res.at("x");
//     DM f_opt = res.at("f");
//     auto end = std::chrono::system_clock::now();
//
//     std::cout << "x* = " << x_opt << "\n";
//     std::cout << "f* = " << f_opt << "\n";
//
//     // 验证约束：用位置参数形式，显式 vector
//     // Function G("G", {x, p}, {g});
//     // DMVector inG;
//     // inG.push_back(x_opt);
//     // inG.push_back(pval);
//     // DMVector outG = G(inG);
//     // std::cout << "g(x*, p) = " << outG.at(0) << "\n";
//
//     auto elapse = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
//     std::cout << "Elapse: " << elapse << std::endl;
//     return 0;
// }


// g++ -O2 -std=c++17 nmpc_opti.cpp -lcasadi -o nmpc_opti
#include <casadi/casadi.hpp>
using namespace casadi;

// ------------ 配置 ------------
static constexpr int NX = 5; // [X,Y,theta,v,delta]
static constexpr int NU = 2; // [a, w_delta]
static constexpr int N = 30; // 预测步数
static constexpr double DT = 0.05; // 步长
static constexpr double L = 1.2; // 轴距

// 物理约束
static constexpr double V_MIN = -1.5, V_MAX = 2.0;
static constexpr double DELTA_MAX = 0.5;
static constexpr double A_MIN = -1.5, A_MAX = 1.5;
static constexpr double W_MIN = -1.0, W_MAX = 1.0;

// 代价权重（示例，可自行调）
static constexpr double Qx = 25.0, Qy = 800.0, Qth = 550.0; // e_x,e_y,e_th
static constexpr double qv = 25.0; // (v - v*)^2
static constexpr double Ra = 1.0, Rw = 4.0; // u^T R u
static constexpr double Sa = 4.0, Sw = 11.0; // Δu^T S Δu
static constexpr double Pv = 10.0, Pdelta = 5.0; // 终端 v_N^2, delta_N^2

// v* = sat(kp*e_x)
static constexpr double KP_VREF = 0.8;
static constexpr double VREF_MIN = -1.5, VREF_MAX = 2.0;

MX f_cont(const MX& x, const MX& u)
{
    MX X = x(0), Y = x(1), th = x(2), v = x(3), del = x(4);
    MX a = u(0), w = u(1);
    return MX::vertcat({
        v * cos(th),
        v * sin(th),
        v / L * tan(del),
        a,
        w
    });
}

MX rk4_step(const MX& xk, const MX& uk, double h)
{
    MX k1 = f_cont(xk, uk);
    MX k2 = f_cont(xk + 0.5 * h * k1, uk);
    MX k3 = f_cont(xk + 0.5 * h * k2, uk);
    MX k4 = f_cont(xk + h * k3, uk);
    return xk + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4);
}

int main()
{
    using Sl = Slice;
    Opti opti;

    // -------- 参数 p = [x0(5); xg(3); u_prev(2)] --------
    MX p = opti.parameter(NX + 3 + NU);
    MX x0p = p(Sl(0, NX));
    MX xg = p(Sl(NX, NX + 3)); // [Xg,Yg,thg]
    MX u_prev = p(Sl(NX + 3, NX + 3 + NU)); // 上一拍控制

    // -------- 决策变量 --------
    MX X = opti.variable(NX, N + 1); // 状态轨迹
    MX U = opti.variable(NU, N); // 控制序列

    // -------- 边界（状态/控制）--------
    // v bounds
    opti.subject_to(X(3, Sl()) <= V_MAX);
    opti.subject_to(X(3, Sl()) >= V_MIN);
    // delta bounds
    opti.subject_to(X(4, Sl()) <= DELTA_MAX);
    opti.subject_to(X(4, Sl()) >= -DELTA_MAX);
    // control bounds
    opti.subject_to(U(0, Sl()) <= A_MAX);
    opti.subject_to(U(0, Sl()) >= A_MIN);
    opti.subject_to(U(1, Sl()) <= W_MAX);
    opti.subject_to(U(1, Sl()) >= W_MIN);

    // 初始状态约束
    opti.subject_to(X(Sl(), 0) == x0p);

    // -------- 目标函数 --------
    MX J = MX::zeros(1, 1);

    // 累计阶段代价 + 动力学约束
    for (int k = 0; k < N; ++k)
    {
        MX xk = X(Sl(), k);
        MX uk = U(Sl(), k);

        // 车体系误差（相对同一终点）
        MX dX = xg(0) - xk(0);
        MX dY = xg(1) - xk(1);
        MX th = xk(2);
        MX ex = cos(th) * dX + sin(th) * dY;
        MX ey = -sin(th) * dX + cos(th) * dY;
        MX eth = atan2(sin(xg(2) - th), cos(xg(2) - th));

        // 速度参考 v* = sat(kp*ex)
        MX vref = fmin(MX(VREF_MAX), fmax(MX(VREF_MIN), KP_VREF * ex));

        // Δu
        MX u_prev_k = (k == 0) ? u_prev : U(Sl(), k - 1);
        MX du = uk - u_prev_k;

        // 阶段代价
        J += Qx * ex * ex + Qy * ey * ey + Qth * eth * eth
                + qv * pow(xk(3) - vref, 2)
                + Ra * pow(uk(0), 2) + Rw * pow(uk(1), 2)
                + Sa * pow(du(0), 2) + Sw * pow(du(1), 2);

        // 动力学（RK4）
        MX x_next = rk4_step(xk, uk, DT);
        opti.subject_to(X(Sl(), k + 1) == x_next);
    }

    // 终端代价（到点 + 停车倾向）
    {
        MX xN = X(Sl(), N);
        MX dX = xg(0) - xN(0);
        MX dY = xg(1) - xN(1);
        MX th = xN(2);
        MX ex = cos(th) * dX + sin(th) * dY;
        MX ey = -sin(th) * dX + cos(th) * dY;
        MX eth = atan2(sin(xg(2) - th), cos(xg(2) - th));
        J += Qx * ex * ex + Qy * ey * ey + Qth * eth * eth
                + Pv * pow(xN(3), 2) + Pdelta * pow(xN(4), 2);

        // （可选）距离相关的 |v_N| 上限：误差小时逼停，远时放开
        // MX r = sqrt(ex*ex + ey*ey);
        // MX vmaxN = 2.0 * tanh(3.0*r);
        // opti.subject_to( -vmaxN <= xN(3) );
        // opti.subject_to(  xN(3) <= vmaxN );
    }

    opti.minimize(J);

    // -------- IPOPT 设置 --------
    Dict opts;
    opts["ipopt.print_level"] = 0;
    opts["ipopt.max_iter"] = 200;
    opts["ipopt.tol"] = 1e-4;
    opts["print_time"] = false;
    opti.solver("ipopt", opts);

    // -------- 演示一次求解 --------
    DM x0_meas = DM::zeros(NX, 1);
    DM goal = DM(std::vector<double>{2.0, 1.0, 0.0}); // 要到 (2,1,0)
    DM u_last = DM::zeros(NU, 1);
    DM par = vertcat(x0_meas, goal, u_last);
    opti.set_value(p, par);

    // 初值（warm start）：把 x0 平铺，u 置零
    opti.set_initial(X, repmat(x0_meas, 1, N + 1));
    opti.set_initial(U, DM::zeros(NU, N));

    // 求解
    auto sol = opti.solve();

    // 取第一个控制
    DM Uopt = sol.value(U);
    double a0 = double(Uopt(0, 0));
    double w0 = double(Uopt(1, 0));
    std::cout << "u0 = [a=" << a0 << ", w=" << w0 << "]\n";

    return 0;
}

