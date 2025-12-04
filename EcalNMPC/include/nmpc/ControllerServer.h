#pragma once
#include "nmpc/kinematics.hpp"
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
    explicit ControllerServer(nmpc::Params params);

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
    struct
    {
        bool stage1, stage2, stage3;

        void reset() noexcept
        {
            stage1 = false;
            stage2 = false;
            stage3 = false;
        }
    } mStageFlags{};


    nmpc::Params mNmpcParams;
    ControllerBicycle mController;

    std::queue<adaptive_control_msg::ForkliftState> mForkliftStateQueue;
    std::queue<adaptive_control_msg::ForkliftState> mCmdsQueue;

    eCAL::CTimer mTimer;
    eCAL::CServiceServer mServer;
    eCAL::protobuf::CSubscriber<adaptive_control_msg::ForkliftState> mGoalSubscriber;
    eCAL::protobuf::CPublisher<adaptive_control_msg::ControlCmd> mCmdsPublisher;

    std::thread mControllerThread;
    bool mIsTriggered{false};
    std::condition_variable mTriggerEvent;
    std::mutex mTriggerMutex;
    std::mutex mGoalMutex, mStateMutex, mCmdsMutex;

    void nmpcLoop();

    bool getForkliftState(std::vector<double>& goal, double& steer_angle, double& fork_z)
    {
        adaptive_control_msg::ForkliftState forklift_state; //
        {
            std::lock_guard<std::mutex> lock(mGoalMutex);
            if (mForkliftStateQueue.empty())
            {
                return false;
            }
            forklift_state = std::move(mForkliftStateQueue.front());
            mForkliftStateQueue.pop();
        }

        goal = std::vector<double>{forklift_state.pose().x(), forklift_state.pose().y(), forklift_state.pose().yaw()};
        steer_angle = forklift_state.steer_angle();
        fork_z = forklift_state.fork_pose().z();

        return true;
    }

    bool getCmd(adaptive_control_msg::ForkliftState& cmd)
    {
        std::lock_guard<std::mutex> lock(mCmdsMutex);
        if (mCmdsQueue.empty())
        {
            return false;
        }
        cmd = std::move(mCmdsQueue.front());
        mCmdsQueue.pop();
        return true;
    }
};
