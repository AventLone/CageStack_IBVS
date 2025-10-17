#include "colib/ControllerServer.h"

void ControllerServer::nmpcLoop()
{
    const auto getGoalAndState = [](const nmpc_test::Pose& goal_msg,
                                    const nmpc_test::State& state_msg) -> std::pair<std::vector<double>, std::vector<double>>
        {
            std::vector<double> goal{goal_msg.x(), goal_msg.y(), goal_msg.yaw()};
            std::vector<double> state{
                        state_msg.pose().x(), state_msg.pose().y(),
                        state_msg.pose().yaw(), state_msg.steer_angle()
                    };
            return std::make_pair(goal, state);
        };

    while (eCAL::Ok() && (!mGoalSubscriber.IsCreated() || !mStateSubscriber.IsCreated()))
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    nmpc_test::Pose goal_message{};
    nmpc_test::State state_message{};
    while (eCAL::Ok())
    {
        if (!mGoalSubscriber.Receive(goal_message))
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }
        if (!mStateSubscriber.Receive(state_message))
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }

        std::cout << "--------------------- NMPC begin ---------------------" << std::endl;
        const double error_x = goal_message.x() - state_message.pose().x();
        const double error_y = goal_message.y() - state_message.pose().y();
        const double error_yaw = goal_message.yaw() - state_message.pose().yaw();
        std::cout << "Error: x " << error_x << ", y " << error_y << ", yaw " << error_yaw << std::endl;
        // if (std::abs(error_x) <= 0.01 && std::abs(error_y) <= 0.01 && std::abs(error_yaw) <= 0.01)
        // {
        //     std::cout << "-----------------------------------------------------\n"
        //             "-----------------------NMPC finished!----------------" << std::endl;
        //     break;
        // }

        const auto [goal, state] = getGoalAndState(goal_message, state_message);
        mController.setGoalAndState(goal, state);

        const auto begin = std::chrono::high_resolution_clock::now();
        std::pair<nmpc::Solution, nmpc::Solution> result;
        if (!mController.solve(result))
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }
        const auto end = std::chrono::high_resolution_clock::now();
        const long duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();

        std::queue<nmpc_test::State> cmds_queue;
        nmpc_test::State cmd;

        cmd.set_drive_velocity(result.first.front()[0]);
        cmd.set_steer_velocity(result.first.front()[1]);
        mCmdsPublisher.Send(cmd);

        nmpc_test::Path path_msg;
        path_msg.mutable_poses()->Reserve(static_cast<int>(result.second.size()));
        for (const auto& x : result.second)
        {
            nmpc_test::Pose pose;
            pose.set_x(x[0]);
            pose.set_y(x[1]);
            pose.set_yaw(x[2]);
            path_msg.add_poses()->CopyFrom(pose);
        }
        mPathPublisher.Send(path_msg);

        // std::cout << "Goal: " << goal << std::endl;
        std::cout << "First control: " << result.first.front() << std::endl;
        // std::cout << "Last state: " << result.second.back() << std::endl;
        std::printf("----------------- NMPC end, elapse: %ld ms -----------------\n", duration);
        std::printf("##############################################################\n");
    }
}
