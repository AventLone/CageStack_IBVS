#include "nmpc/ControllerServer.h"
#include <Eigen/Geometry>
#include <utility>

static void printPath(const nmpc::Solution& path)
{
    for (const auto& pose : path)
    {
        std::cout << pose << std::endl;
    }
}

ControllerServer::ControllerServer(nmpc::Params params) : mNmpcParams(std::move(params)),
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

    mGoalSubscriber.AddReceiveCallback([this](const char* topic_name, const adaptive_control_msg::ForkliftState& state_msg,
                                              long long time_, long long clock_, long long id_)
        {
            std::lock_guard<std::mutex> lock(mGoalMutex);
            mForkliftStateQueue.push(state_msg);
            if (mForkliftStateQueue.size() > 1)
            {
                mForkliftStateQueue.pop();
            }
        });

    mServer.AddMethodCallback("wake", "protobuf", "protobuf", [this](const std::string& method_name,
                                                                     const std::string& req_type,
                                                                     const std::string& resp_type,
                                                                     const std::string& request_bytes,
                                                                     std::string& response_bytes)-> int
                                  {
                                      mController.resetWeights();
                                      mStageFlags.reset();

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
    // const auto getGoal = [](const adaptive_control_msg::ForkliftState& goal_state_msg) -> std::vector<double>
    //     {
    //         std::vector<double> goal{goal_state_msg.pose().x(), goal_state_msg.pose().y(), goal_state_msg.pose().yaw()};
    //         return goal;
    //     };

    while (eCAL::Ok() && !mGoalSubscriber.IsCreated())
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    uint16_t stable_count = 0;
    adaptive_control_msg::ForkliftState goal_state_msg;
    while (eCAL::Ok())
    {
        {
            std::unique_lock<std::mutex> lock(mTriggerMutex);
            mTriggerEvent.wait(lock, [this]() -> bool { return mIsTriggered; });
        }

        // if (!mGoalSubscriber.Receive(goal_state_msg))
        // {
        //     std::this_thread::sleep_for(std::chrono::milliseconds(10));
        //     continue;
        // }
        //
        // std::vector<double> goal = getGoal(goal_state_msg);
        // const double steer_angle = goal_state_msg.steer_angle();
        // const double fork_z = goal_state_msg.fork_pose().z();
        std::vector<double> goal;
        double steer_angle, fork_z;
        if (!getForkliftState(goal, steer_angle, fork_z))
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        adaptive_control_msg::ControlCmd cmd;
        if (std::abs(goal[1]) <= 0.03 && std::abs(goal[2]) <= 0.006)
        {
            ++stable_count;
            if (!mStageFlags.stage1 && stable_count >= 10)
            {
                mStageFlags.stage1 = true;
                std::cout << "Stage2!" << std::endl;
            }
            if (!mStageFlags.stage1)
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                continue;
            }
        }
        else
        {
            stable_count = 0;
        }

        if (mStageFlags.stage1 && !mStageFlags.stage2)
        {
            if (steer_angle <= 0.001)
            {
                mStageFlags.stage2 = true;
            }
            else
            {
                cmd.set_drive_velocity(0.0);
                cmd.set_steer_angle(0.0);
                mCmdsPublisher.Send(cmd);
                continue;
            }
        }

        // Eigen::Isometry2d transform_goal(Eigen::Isometry2d::Identity()), transform_goal_offset(Eigen::Isometry2d::Identity());
        // transform_goal.rotate(Eigen::Rotation2Dd(goal[2]));
        // transform_goal.translate(Eigen::Vector2d(goal[0], goal[1]));

        if (!mStageFlags.stage1)
        {
            goal[0] -= 1.5;
            // transform_goal_offset.translate(Eigen::Vector2d(-1.5, 0));
        }
        else
        {
            goal[0] += 0.4;
            // transform_goal_offset.translate(Eigen::Vector2d(0.4, 0));
        }
        // Eigen::Isometry2d transform_stage2_goal = transform_goal * transform_goal_offset;
        //
        // goal[0] = transform_stage2_goal.translation()[0];
        // goal[1] = transform_stage2_goal.translation()[1];


        if (mStageFlags.stage1 && goal[0] <= 0.02)
        {
            auto* fork_cmd = new adaptive_control_msg::ForkPose;
            fork_cmd->set_z(0.2);
            cmd.set_allocated_fork_pose(fork_cmd);
            mCmdsPublisher.Send(cmd);
            if (!mStageFlags.stage3)
            {
                mStageFlags.stage3 = true;
            }
        }

        if (mStageFlags.stage3)
        {
            if (std::abs(fork_z - 0.2) < 0.01)
            {
                cmd.set_finished(true);
                mCmdsPublisher.Send(cmd);
                mStageFlags.reset();
                stable_count = 0;
                mController.resetWeights();
                std::cout << "Task done." << std::endl;

                std::lock_guard<std::mutex> lock(mTriggerMutex);
                mIsTriggered = false;
            }
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
        if (!mStageFlags.stage2)
        {
            cmd.set_steer_angle(steer_angle);
            cmd.set_steer_velocity(u[1]);
        }
        else
        {
            cmd.set_steer_angle(0.0);
            cmd.set_steer_velocity(0.0);
        }

        mCmdsPublisher.Send(cmd);

        const auto& xs = result.second;

        std::cout << "First control: " << u << std::endl;
        // std::cout << "Path: " << std::endl;
        // printPath(xs);
        std::printf("----------------- NMPC end, elapse: %ld ms -----------------\n", duration);
        std::printf("##############################################################\n");
        // break;
    }
}
