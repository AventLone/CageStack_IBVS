#pragma once
#include "nmpc/kinematics.hpp"
#include <ecal/ecal.h>
#include <ecal/msg/protobuf/publisher.h>
#include <ecal/msg/protobuf/subscriber.h>
#include "vehicle_state_msg.pb.h"
#include <thread>
#include <queue>

class ControllerServer
{
    struct Cmd
    {
        double steer_angle;
        double drive_velocity;
        double steer_velocity;
    };

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
    ControllerBicycle mController;

    std::queue<Cmd> mCmdsQueue;

    eCAL::CTimer mTimer;
    eCAL::protobuf::CSubscriber<sim_data_flow::VehicleStateMsg> mGoalSubscriber;
    eCAL::protobuf::CPublisher<sim_data_flow::VehicleStateMsg> mCmdsPublisher;

    std::thread mControllerThread;
    std::mutex mCmdsMutex;

    void nmpcLoop();

    bool getCmd(Cmd& cmd)
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
