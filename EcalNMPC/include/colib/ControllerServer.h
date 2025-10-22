#pragma once
#include "colib/kinematics.hpp"
#include <ecal/ecal.h>
#include <ecal/msg/protobuf/publisher.h>
#include <ecal/msg/protobuf/subscriber.h>
#include <ecal/ecal_server.h>
#include <condition_variable>
#include "adaptive_control_msg.pb.h"
#include <thread>

#include <queue>

class ControllerServer
{
public:
    explicit ControllerServer(const nmpc::Params& parmas);

    ~ControllerServer()
    {
        if (mControllerThread.joinable())
        {
            mControllerThread.join();
        }

        mCmdsPublisher.Destroy();
        mGoalSubscriber.Destroy();
    }

private:
    nmpc::Params mNmpcParams;
    ControllerBicycle mController;

    std::queue<adaptive_control_msg::ForkliftState> mCmdsQueue;

    eCAL::CTimer mTimer;
    eCAL::CServiceServer mServer;
    eCAL::protobuf::CSubscriber<adaptive_control_msg::ForkliftState> mGoalSubscriber;
    eCAL::protobuf::CPublisher<adaptive_control_msg::ForkliftState> mCmdsPublisher;

    std::thread mControllerThread;
    bool mIsTriggered{false};
    std::condition_variable mTriggerEvent;
    std::mutex mTriggerMutex;
    std::mutex mGoalMutex, mStateMutex, mCmdsMutex;

    void nmpcLoop();

    bool getCmd(adaptive_control_msg::ForkliftState& cmd)
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
