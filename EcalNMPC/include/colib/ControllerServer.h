#pragma once
#include "./kinematics.hpp"
#include <ecal/msg/protobuf/publisher.h>
#include <ecal/msg/protobuf/subscriber.h>
#include "VehicleControl.pb.h"        // 依 proto 文件名而定
#include <thread>
#include <ecal/ecal_core.h>
#include <ecal/ecal_timer.h>
#include <ecal/ecal.h>
#include <queue>

class ControllerServer
{
public:
    explicit ControllerServer(const nmpc::Params& params) : mController(params),
                                                            mGoalSubscriber("goal"),
                                                            mStateSubscriber("vehicle/status"),
                                                            mCmdsPublisher("nmpc_cmd"),
                                                            mPathPublisher("nmpc_path")
    {
        if (!eCAL::IsInitialized())
        {
            std::cerr << "eCAL is not initialized!" << std::endl;
            std::exit(EXIT_FAILURE);
        }

        // mGoalSubscriber.AddReceiveCallback([this](const char*, const nmpc_test::Pose& goal_message, long long, long long, long long)
        //     {
        //         std::vector<double> temp_goal{goal_message.x(), goal_message.y(), goal_message.yaw()};
        //
        //         std::lock_guard<std::mutex> lock(this->mGoalMutex);
        //         this->mGoal.swap(temp_goal);
        //     });
        // mStateSubscriber.AddReceiveCallback([this](const char*, const nmpc_test::State& state_message, long long, long long, long long)
        //     {
        //         std::vector<double> temp_state{state_message.drive_velocity(), state_message.steer_angle()};
        //
        //         std::lock_guard<std::mutex> lock(this->mStateMutex);
        //         this->mState.swap(temp_state);
        //     });

        // mTimer.Start(20, [this]() -> void
        //                  {
        //                      if (nmpc_test::State cmd; getCmd(cmd))
        //                      {
        //                          mCmdsPublisher.Send(cmd);
        //                      }
        //                  });
        mControllerThread = std::thread(&ControllerServer::nmpcLoop, this);
        std::cout << "NMPC server start." << std::endl;
    }

    ~ControllerServer()
    {
        if (mControllerThread.joinable())
        {
            mControllerThread.join();
        }
        // mTimer.Stop();
        mCmdsPublisher.Destroy();
        mPathPublisher.Destroy();

        mGoalSubscriber.Destroy();
        mStateSubscriber.Destroy();
    }

    // void cmdsPubLoop();

private:
    ControllerE mController;

    std::vector<double> mGoal;
    std::vector<double> mState; // dirve velocity, steer_angle

    std::queue<nmpc_test::State> mCmdsQueue;

    eCAL::protobuf::CSubscriber<nmpc_test::Pose> mGoalSubscriber;
    eCAL::protobuf::CSubscriber<nmpc_test::State> mStateSubscriber;
    eCAL::protobuf::CPublisher<nmpc_test::State> mCmdsPublisher;
    eCAL::protobuf::CPublisher<nmpc_test::Path> mPathPublisher;
    eCAL::CTimer mTimer;

    std::thread mControllerThread;
    std::mutex mGoalMutex, mStateMutex, mCmdsMutex;

    void nmpcLoop();

    bool getCmd(nmpc_test::State& cmd)
    {
        std::lock_guard<std::mutex> lock(mCmdsMutex);
        if (mCmdsQueue.empty())
        {
            return false;
        }
        cmd = mCmdsQueue.front();
        mCmdsQueue.pop();
        return true;
    }
};

