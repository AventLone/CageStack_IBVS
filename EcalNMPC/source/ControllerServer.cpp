#include "nmpc/ControllerServer.h"

ControllerServer::ControllerServer(const nmpc::Params& params) : mController(params),
                                                                 mGoalSubscriber("vehicle/status"),
                                                                 mCmdsPublisher("adjust_ctrl/cmd")
{
    if (!eCAL::IsInitialized())
    {
        std::cerr << "eCAL is not initialized!" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    const int interval = static_cast<int>(params.dt * 1000.0);
    const int sub_interval = interval / 10;
    double ddt = params.dt / 10.0;
    mTimer.Start(interval + 1, [this, sub_interval, ddt]() -> void
                     {
                         if (Cmd cmd{}; getCmd(cmd))
                         {
                             sim_data_flow::VehicleStateMsg msg;
                             msg.set_drive_velocity(cmd.drive_velocity);
                             msg.set_control_mode(3);
                             for (size_t i = 0; i < 10; ++i)
                             {
                                 cmd.steer_angle += cmd.steer_velocity * ddt;
                                 msg.set_steer_angle(cmd.steer_angle);
                                 mCmdsPublisher.Send(msg);
                                 std::this_thread::sleep_for(std::chrono::milliseconds(sub_interval));
                             }
                         }
                     });

    mControllerThread = std::thread(&ControllerServer::nmpcLoop, this);
    std::cout << "Controller server start." << std::endl;
}

void ControllerServer::nmpcLoop()
{
    const auto getGoalAndSteer = [](const sim_data_flow::VehicleStateMsg& goal_state_msg) -> std::pair<std::vector<double>, double>
        {
            std::vector<double> goal{goal_state_msg.goal_pose().x(), goal_state_msg.goal_pose().y(), goal_state_msg.goal_pose().yaw()};
            return std::make_pair(goal, goal_state_msg.steer_angle());
        };

    while (eCAL::Ok() && !mGoalSubscriber.IsCreated())
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    sim_data_flow::VehicleStateMsg goal_state_msg;
    while (eCAL::Ok())
    {
        if (!mGoalSubscriber.Receive(goal_state_msg))
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }

        std::cout << "--------------------- NMPC begin ---------------------" << std::endl;
        auto [goal, steer_angle] = getGoalAndSteer(goal_state_msg);

        static bool stage2 = false;
        if (std::abs(goal[1]) < 0.006 && std::abs(goal[2]) < 0.02)
        {
            if (!stage2)
            {
                stage2 = true;
                mController.setQ({666.6, 1.0, 1.0, 1.0});
            }
        }

        if (!stage2)
        {
            goal[0] -= 1.5;
        }
        else
        {
            goal[0] += 1.0;
        }

        std::cout << "Goal: x " << goal[0] << ", y " << goal[1] << ", yaw " << goal[2] << std::endl;
        mController.setGoalAndState(goal, {0.0, 0.0, 0.0, steer_angle});

        const auto begin = std::chrono::high_resolution_clock::now();
        std::pair<nmpc::Solution, nmpc::Solution> result;
        if (!mController.solve(result))
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }
        const auto end = std::chrono::high_resolution_clock::now();
        const long duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();

        std::queue<Cmd> cmds_queue;
        Cmd cmd{};
        cmd.steer_angle = steer_angle;

        const auto& us = result.first;
        for (size_t i = 0; i < 2; ++i)
        {
            const auto& u = us[i];
            cmd.drive_velocity = u[0];
            cmd.drive_velocity = u[1];
            cmds_queue.push(cmd);
        } //
        {
            std::lock_guard<std::mutex> lock(mCmdsMutex);
            mCmdsQueue.swap(cmds_queue);
        }

        std::cout << "First control: " << result.first.front() << std::endl;
        std::printf("----------------- NMPC end, elapse: %ld ms -----------------\n", duration);
        std::printf("##############################################################\n");

        std::this_thread::sleep_for(std::chrono::milliseconds(30));
    }
}
