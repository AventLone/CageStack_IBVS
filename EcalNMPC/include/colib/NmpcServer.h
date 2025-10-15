#pragma once
#include "./NMPC.h"
#include <ecal/msg/protobuf/publisher.h>
#include <ecal/msg/protobuf/subscriber.h>
#include <ecal/msg/string/publisher.h>
#include "VehicleControl.pb.h"        // 依 proto 文件名而定
#include <thread>
#include <ecal/ecal_core.h>
#include <ecal/ecal_timer.h>
#include <ecal/ecal.h>
#include <queue>

class NmpcServer
{
public:
    explicit NmpcServer(const NMPC::Params& params) : mControlLen(params.horizon),
                                                      mNMPC(4, 2, params),
                                                      mGoalSubscriber("goal"),
                                                      mStateSubscriber("vehicle/status"),
                                                      mCmdsPublisher("nmpc_cmd")
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
        mNmpcThread = std::thread(&NmpcServer::nmpcLoop, this);
        std::cout << "NMPC server start." << std::endl;
    }

    ~NmpcServer()
    {
        if (mNmpcThread.joinable())
        {
            mNmpcThread.join();
        }
        mTimer.Stop();
        mCmdsPublisher.Destroy();
        mGoalSubscriber.Destroy();
    }

    // void cmdsPubLoop();

private:
    // NMPC::Ptr mNMPC;
    uint16_t mControlLen;
    NMPC mNMPC;

    std::vector<double> mGoal;
    std::vector<double> mState; // dirve velocity, steer_angle

    std::queue<nmpc_test::State> mCmdsQueue;

    eCAL::protobuf::CSubscriber<nmpc_test::Pose> mGoalSubscriber;
    eCAL::protobuf::CSubscriber<nmpc_test::State> mStateSubscriber;
    eCAL::protobuf::CPublisher<nmpc_test::State> mCmdsPublisher;
    eCAL::string::CPublisher<std::string> mTestPub;
    eCAL::CTimer mTimer;

    std::thread mNmpcThread;
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

