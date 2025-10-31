#include "nmpc/ControllerServer.h"

static void printPath(const nmpc::Solution& path)
{
    for (const auto& pose : path)
    {
        std::cout << pose << std::endl;
    }
}

ControllerServer::ControllerServer(const nmpc::Params& params) : mNmpcParams(params),
                                                                 mController(mNmpcParams),
                                                                 mServer("forklift/adaptive_control_trigger"),
                                                                 mGoalSubscriber("forklift/goal"),
                                                                 mCmdsPublisher("forklift/cmds")
{
    if (!eCAL::IsInitialized())
    {
        std::cerr << "eCAL is not initialized!" << std::endl;
        std::exit(EXIT_FAILURE);
    }
    mServer.AddMethodCallback("wake", "protobuf", "protobuf", [this](const std::string& method_name,
                                                                     const std::string& req_type,
                                                                     const std::string& resp_type,
                                                                     const std::string& request_bytes,
                                                                     std::string& response_bytes)-> int
                                  {
                                      adaptive_control_msg::ControlRequest request;
                                      request.ParseFromString(request_bytes);
                                      if (request.wake_up())
                                      {
                                          std::lock_guard<std::mutex> lock(mTriggerMutex);
                                          mIsTriggered = true;
                                          mTriggerEvent.notify_one();
                                      }
                                      else
                                      {
                                          std::lock_guard<std::mutex> lock(mTriggerMutex);
                                          mIsTriggered = false;
                                      }

                                      adaptive_control_msg::ControlResponse response;
                                      response.set_ok(true);
                                      response.set_run(request.wake_up());
                                      response.set_dt(mNmpcParams.dt);
                                      response_bytes = response.SerializeAsString();
                                      return 0;
                                  });

    // mTimer.Start(30, [this]() -> void
    //                  {
    //                      if (adaptive_control_msg::ForkliftState cmd; getCmd(cmd))
    //                      {
    //                          mCmdsPublisher.Send(cmd);
    //                      }
    //                  });

    mControllerThread = std::thread(&ControllerServer::nmpcLoop, this);
    std::cout << "Controller server start." << std::endl;
}

void ControllerServer::nmpcLoop()
{
    const auto getGoal = [](const adaptive_control_msg::ForkliftState& goal_state_msg) -> std::vector<double>
        {
            std::vector<double> goal{goal_state_msg.pose().x(), goal_state_msg.pose().y(), goal_state_msg.pose().yaw()};
            return goal;
        };

    while (eCAL::Ok() && !mGoalSubscriber.IsCreated())
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    bool stage2 = false;
    bool stage3 = false;
    adaptive_control_msg::ForkliftState goal_state_msg;
    while (eCAL::Ok())
    {
        {
            std::unique_lock<std::mutex> lock(mTriggerMutex);
            mTriggerEvent.wait(lock, [this]() -> bool { return mIsTriggered; });
        }

        if (!mGoalSubscriber.Receive(goal_state_msg))
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }

        auto goal = getGoal(goal_state_msg);
        const double steer_angle = goal_state_msg.steer_angle();
        const double fork_z = goal_state_msg.fork_pose().z();

        if (std::abs(goal[1]) < 0.005 && std::abs(goal[2]) < 0.006 && steer_angle < 0.006)
        {
            if (!stage2)
            {
                stage2 = true;
                mController.setWeightQ({1.0e3, 0.0, 0.0, 1.0e3, 0.0});
                mController.setWeightF({1.0e3, 0.0, 0.0, 1.0e3});
            }
        }

        if (!stage2)
        {
            goal[0] -= 1.5;
        }
        else
        {
            goal[0] += 0.3;
        }

        adaptive_control_msg::ControlCmd cmd;
        if (stage2 && goal[0] <= 0.01)
        {
            auto* fork_cmd = new adaptive_control_msg::ForkPose;
            fork_cmd->set_z(0.2);
            cmd.set_allocated_fork_pose(fork_cmd);
            mCmdsPublisher.Send(cmd);
            if (!stage3)
            {
                stage3 = true;
            }
        }

        if (stage3)
        {
            if (std::abs(fork_z - 0.2) < 0.01)
            {
                cmd.set_finished(true);
                mCmdsPublisher.Send(cmd);
                stage2 = false;
                stage3 = false;
                std::lock_guard<std::mutex> lock(mTriggerMutex);
                mIsTriggered = false;
            }
            std::cout << "Task done." << std::endl;
            continue;
        }
        std::cout << "--------------------- NMPC begin ---------------------" << std::endl;
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

        std::queue<adaptive_control_msg::ForkliftState> cmds_queue;


        const auto& us = result.first;
        const auto& u = us.front();

        cmd.set_drive_velocity(-u[0]);
        cmd.set_steer_velocity(u[1]);

        mCmdsPublisher.Send(cmd);

        const auto& xs = result.second;
        // for (size_t i = 0; i < 2; ++i)
        // {
        //     const auto& u = us[i];
        //     cmd.set_drive_velocity(u[0]);
        //     cmd.set_steer_velocity(u[1]);
        //     cmds_queue.push(cmd);
        // } //
        // {
        //     std::lock_guard<std::mutex> lock(mCmdsMutex);
        //     mCmdsQueue.swap(cmds_queue);
        // }

        std::cout << "First control: " << u << std::endl;
        // std::cout << "Path: " << std::endl;
        // printPath(xs);
        std::printf("----------------- NMPC end, elapse: %ld ms -----------------\n", duration);
        std::printf("##############################################################\n");
        // break;
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
}
