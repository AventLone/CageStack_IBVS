#include "control/node/ControlCmdPublisher.h"
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

void ControlCmdPublisher::initSolver()
{
    this->declare_parameter("NMPC.Horizon", 20);
    this->declare_parameter("NMPC.Dt", 0.1);

    this->declare_parameter("NMPC.InputLen", 2);
    this->declare_parameter("NMPC.StateLen", 4);
    this->declare_parameter("NMPC.OutputLen", 4);

    this->declare_parameter("NMPC.MaxAcc", 3.0);
    this->declare_parameter("NMPC.MaxSpeed", 3.5);

    this->declare_parameter("NMPC.MaxSteerSpeed", 0.5);
    this->declare_parameter("NMPC.MaxSteerAngle", 1.14);

    this->declare_parameter("NMPC.WheelBase", 1.20551);
    this->declare_parameter("NMPC.WheelRadius", 0.115);

    // this->declare_parameter("NMPC.WeightQ", std::vector<double>{1.0, 1.0, 2.0, 1.0});
    this->declare_parameter("NMPC.WeightQ", std::vector<double>{2.0, 2.0, 3.1, 0.1});
    this->declare_parameter("NMPC.WeightF", std::vector<double>{660.0, 660.0, 660.0, 660.0});
    this->declare_parameter("NMPC.WeightR", std::vector<double>{0.01, 0.001});

    this->get_parameter("NMPC.Horizon", mControllerParams.horizon);
    this->get_parameter("NMPC.Dt", mControllerParams.dt);

    this->get_parameter("NMPC.InputLen", mControllerParams.input_len);
    this->get_parameter("NMPC.StateLen", mControllerParams.state_len);
    this->get_parameter("NMPC.OutputLen", mControllerParams.output_len);

    this->get_parameter("NMPC.MaxAcc", mControllerParams.max_acc);
    this->get_parameter("NMPC.MaxSpeed", mControllerParams.max_speed);

    this->get_parameter("NMPC.MaxSteerSpeed", mControllerParams.max_steer_speed);
    this->get_parameter("NMPC.MaxSteerAngle", mControllerParams.max_steer_angle);

    this->get_parameter("NMPC.WheelBase", mControllerParams.wheel_base);
    this->get_parameter("NMPC.WheelRadius", mControllerParams.wheel_radius);

    this->get_parameter("NMPC.WeightQ", mControllerParams.weight_Q);
    this->get_parameter("NMPC.WeightF", mControllerParams.weight_F);
    this->get_parameter("NMPC.WeightR", mControllerParams.weight_R);

    mController = std::make_unique<BicycleController>(mControllerParams);
}

void ControlCmdPublisher::initSubscriptions()
{
    const std::string param_name = "TopicName.Perception.Target.Pose";
    this->declare_parameter(param_name, "/perception/slot_pose");
    const std::string target_pose_topic = this->get_parameter(param_name).as_string();

    mGoalPoseSub.subscribe(this, target_pose_topic, rclcpp::ServicesQoS().get_rmw_qos_profile());
    mSteerAngelSub.subscribe(this, "/lola/joint_states");
    mSynchronizer = std::make_unique<message_filters::Synchronizer<SyncPolicy>>(SyncPolicy(10),
                                                                                mGoalPoseSub, mSteerAngelSub);
    mSynchronizer->setMaxIntervalDuration(rclcpp::Duration(0, 20 * 100000)); // 设置时间容差（单位：秒）30ms 容差
    mSynchronizer->registerCallback(std::bind(&ControlCmdPublisher::dataHandler, this,
                                              std::placeholders::_1, std::placeholders::_2));

    // mGoalPoseSub = this->create_subscription<geometry_msgs::msg::Pose2D>(
    //     target_pose_topic, rclcpp::ServicesQoS(),
    //     [this](const geometry_msgs::msg::Pose2D::ConstSharedPtr& pose_msg) -> void
    //         {
    //             std::vector<double> goal(3, 0.0);
    //             goal[0] = pose_msg->x + 1.2f;
    //             goal[1] = pose_msg->y;
    //             goal[2] = pose_msg->theta;
    //
    //             while (!mGoalBuffer.empty())
    //             {
    //                 mGoalBuffer.pop();
    //             }
    //
    //             mGoalBuffer.push(std::move(goal));
    //             mTriggerEvent.notify_one();
    //         });
}

void ControlCmdPublisher::initPublishers()
{
    const std::string params_prefix = "TopicName.Control";
    this->declare_parameters<std::string>(params_prefix, {
                                              {"Command", "/lola/joint_command"},
                                              {"Path", "/control/path"}
                                          });
    std::map<std::string, std::string> control_topics;
    if (!this->get_parameters(params_prefix, control_topics))
    {
        RCLCPP_FATAL(this->get_logger(), "Failed to get parameters, control topic names!");
    }

    mCmdPub = this->create_publisher<sensor_msgs::msg::JointState>(control_topics["Command"], 10);
    mPathPub = this->create_publisher<nav_msgs::msg::Path>(control_topics["Path"], rclcpp::SensorDataQoS());
}

void ControlCmdPublisher::cmdPubLoop()
{
    // std::vector<double> goal;
    DataElement data;
    sensor_msgs::msg::JointState cmd_msg;
    cmd_msg.name = {"drive_joint", "steer_joint", "lift_z", "lift_y"};
    cmd_msg.position = std::vector(4, 0.0);
    cmd_msg.velocity = std::vector(4, 0.0);
    nav_msgs::msg::Path state_path;
    state_path.header.frame_id = "LOLA";

    bool stage_1{false}, stage_2{false};
    constexpr float fork_heel = 0.22f; // Distance between fork heel and truck base

    const rclcpp::Duration interval = rclcpp::Duration::from_seconds(mControllerParams.dt);
    while (rclcpp::ok())
    {
        cmd_msg.velocity = std::vector(4, 0.0);
        cmd_msg.position[2] = 0.5;
        // mCmdPub->publish(cmd_msg);
        state_path.poses.clear();
        //
        {
            std::unique_lock<std::mutex> lock(mBufferMutex);
            mTriggerEvent.wait(lock, [this]() -> bool { return !mDataBuffer.empty() || mIsShutdown; });
            if (mIsShutdown)
            {
                break;
            }
            data = std::move(mDataBuffer.front());
            mDataBuffer.pop();
        }
        // next_tick = std::chrono::steady_clock::now() + interval;
        rclcpp::Time next_tick = this->now() + interval;

        // if (std::abs(data.goal[1]) < 0.15 && std::abs(data.goal[2]) < 0.03)

        // if (std::abs(data.goal[1]) < 0.15 && std::abs(data.goal[2]) < 0.03)
        // {
        //     cmd_msg.velocity[0] = 0.0;
        //     cmd_msg.velocity[1] = 0.0;
        //     cmd_msg.header.stamp = this->now();
        // }

        // if (std::abs(data.goal[0]) < 0.01)
        // {
        //     cmd_msg.velocity[0] = 0.0;
        //     cmd_msg.velocity[1] = 0.0;
        // }
        // else
        if (const std::vector<double> stage1_goal{data.goal[0] + 1.6, 0.0, 0.0};
            std::abs(stage1_goal[0]) > 0.1)
        {
            mController->setGoal(stage1_goal, -data.steer_angle);
            std::pair<nmpc::Solution, nmpc::Solution> result;
            if (!mController->solve(result))
            {
                RCLCPP_WARN(get_logger(), "Controller failed to solve the problem!");
                continue;
            }
            const auto& [us, xs] = result;
            const std::vector<double>& cmd = us[0];
            cmd_msg.velocity[0] = -cmd[0];
            cmd_msg.velocity[1] = -cmd[1];

            cmd_msg.header.stamp = this->now();
            mCmdPub->publish(cmd_msg);
        }
        else
        {
            if (!stage_1)
            {
                cmd_msg.position[3] = data.goal[1];
                cmd_msg.velocity[0] = 0.0;
                cmd_msg.header.stamp = this->now();
                mCmdPub->publish(cmd_msg);
                RCLCPP_INFO(get_logger(), "First stage finished");
                stage_1 = true;
                continue;
            }
            const double dura = (data.goal[0] + fork_heel) / (M_PI * mControllerParams.wheel_radius);

            cmd_msg.velocity[0] = M_PI;
            cmd_msg.header.stamp = this->now();
            mCmdPub->publish(cmd_msg);
            this->get_clock()->sleep_for(rclcpp::Duration::from_seconds(dura));

            cmd_msg.velocity[0] = 0.0;
            cmd_msg.position[2] = 0.01;
            cmd_msg.header.stamp = this->now();
            mCmdPub->publish(cmd_msg);
            this->get_clock()->sleep_for(rclcpp::Duration::from_seconds(3.0));
            cmd_msg.velocity[0] = -3.0;
            cmd_msg.header.stamp = this->now();
            mCmdPub->publish(cmd_msg);
            this->get_clock()->sleep_for(rclcpp::Duration::from_seconds(3.0));

            cmd_msg.velocity[0] = 0.0;
            cmd_msg.header.stamp = this->now();
            mCmdPub->publish(cmd_msg);

            // cmd_msg.velocity[0] = M_PI;
            // cmd_msg.header.stamp = this->now();
            // const double dura = 1.6 / (M_PI * mControllerParams.wheel_radius);
            // mCmdPub->publish(cmd_msg);
            // this->get_clock()->sleep_for(rclcpp::Duration::from_seconds(dura));
            // cmd_msg.velocity[0] = 0.0;
            // cmd_msg.position[2] = 0.01;
            // cmd_msg.header.stamp = this->now();
            // mCmdPub->publish(cmd_msg);
            // this->get_clock()->sleep_for(rclcpp::Duration::from_seconds(2.0));
            // cmd_msg.velocity[0] = -5.0;
            // cmd_msg.header.stamp = this->now();
            // mCmdPub->publish(cmd_msg);
            // this->get_clock()->sleep_for(rclcpp::Duration::from_seconds(2.1));
            // cmd_msg.velocity[0] = 0.0;
            // mCmdPub->publish(cmd_msg);
            break;
        }


        /* Publish Path */
        // for (size_t i = 0; i < xs.size(); i += 3)
        // {
        //     geometry_msgs::msg::PoseStamped pose;
        //     pose.header.frame_id = "LOLA";
        //     pose.header.stamp = this->now();
        //     const auto& x = xs[i];
        //
        //     pose.pose.position.x = x[0];
        //     pose.pose.position.y = x[1];
        //     pose.pose.position.z = 0.1;
        //     tf2::Quaternion orien;
        //     orien.setRPY(0.0, 0.0, x[2]);
        //     orien.normalize();
        //     pose.pose.orientation = tf2::toMsg(orien);
        //     state_path.poses.push_back(pose);
        // }
        // mPathPub->publish(state_path);

        if (const auto this_timepoint = this->now(); this_timepoint > next_tick)
        {
            RCLCPP_WARN(get_logger(), "Solving NMPC has exceeded 100 ms!");
        }
        else
        {
            // const auto latency =
            //         std::chrono::duration_cast<std::chrono::milliseconds>(next_tick - this_timepoint);
            // const rclcpp::Duration latency = next_tick - this_timepoint;
            //
            // std::cout << "Latency: " << latency.seconds() * 1000.0 << " ms." << std::endl;
            // std::this_thread::sleep_until(next_tick);
            this->get_clock()->sleep_until(next_tick);
        }
    }
}
