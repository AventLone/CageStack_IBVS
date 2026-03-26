#include "control/node/ControlCmdPublisher.h"

void ControlCmdPublisher::initSolver()
{
    this->declare_parameter("NMPC.Horizon", 20);
    this->declare_parameter("NMPC.Dt", 0.1);

    this->declare_parameter("NMPC.InputLen", 2);
    this->declare_parameter("NMPC.StateLen", 4);
    this->declare_parameter("NMPC.OutputLen", 4);

    this->declare_parameter("NMPC.MaxAcc", 3.0);
    this->declare_parameter("NMPC.MaxSpeed", 1.2);

    this->declare_parameter("NMPC.MaxSteerSpeed", 0.5);
    this->declare_parameter("NMPC.MaxSteerAngle", 1.14);

    this->declare_parameter("NMPC.WheelBase", 1.37);
    this->declare_parameter("NMPC.WheelRadius", 0.11);

    this->declare_parameter("NMPC.WeightQ", std::vector<double>{360.0, 660.0, 370.0, 60.0});
    this->declare_parameter("NMPC.WeightF", std::vector<double>{4000.0, 4000.0, 4000.0});
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
    this->declare_parameter(param_name, "/perception/target/pose");
    const std::string target_pose_topic = this->get_parameter(param_name).as_string();

    mGoalPoseSub = this->create_subscription<geometry_msgs::msg::PoseStamped>(target_pose_topic, 10, 
        [this](const geometry_msgs::msg::PoseStamped::ConstSharedPtr &pose_msg) -> void
            {
                std::vector<double> goal(3, 0.0);
                tf2::Quaternion tf2_quat;
                tf2::convert(pose_msg->pose.orientation, tf2_quat);
                goal[0] = pose_msg->pose.position.x;
                goal[1] = pose_msg->pose.position.y;

                const tf2::Matrix3x3 mat(tf2_quat);
                double roll, pitch, yaw;
                mat.getRPY(roll, pitch, yaw);
                goal[2] = yaw;
                mGoalBuffer.push(std::move(goal));
                if (mGoalBuffer.size() > 1)
                {
                    mGoalBuffer.pop();
                }
            });
}

void ControlCmdPublisher::initPublishers()
{
    const std::string params_prefix = "TopicName.Control";
    this->declare_parameters<std::string>(params_prefix, {{"Command", "/control/cmds"},
                                                          {"Path", "/control/path"}});
    std::map<std::string, std::string> control_topics;
    if(!this->get_parameters(params_prefix, control_topics))
    {
        RCLCPP_FATAL(this->get_logger(), "Failed to get parameters, control topic names!");
    }

    mCmdPub = this->create_publisher<std_msgs::msg::Float64MultiArray>(control_topics["Command"], 10);
    mPathPub = this->create_publisher<nav_msgs::msg::Path>(control_topics["Path"], rclcpp::SensorDataQoS().best_effort());
}

void ControlCmdPublisher::cmdPubLoop() 
{
    if (mGoalBuffer.empty())
    {
        return;
    }

    const std::vector<double> goal = std::move(mGoalBuffer.front());
    mGoalBuffer.pop();

    mController->setGoal(goal);
    std::pair<nmpc::Solution, nmpc::Solution> result;
    if (!mController->solve(result))
    {
        return;
    }
    const auto& [us, xs] = result;
    const std::vector<double>& cmd = us[0];
    std_msgs::msg::Float64MultiArray cmd_msg;
    cmd_msg.data.push_back(cmd[0]);
    cmd_msg.data.push_back(cmd[1]);
    cmd_msg.data.push_back(0.0);
    mCmdPub->publish(cmd_msg);

    nav_msgs::msg::Path state_path;
    state_path.header.frame_id = "map";
    // for (const auto& x : xs)
    for (size_t i = 0; i < xs.size(); i += 3)
    {
        geometry_msgs::msg::PoseStamped pose;
        pose.header.frame_id = "map";
        pose.header.stamp = this->now();

        const auto& x = xs[i];

        pose.pose.position.x = x[0];
        pose.pose.position.y = x[1];
        pose.pose.position.z = 0.1;
        tf2::Quaternion orien;
        orien.setRPY(0.0, 0.0, x[2]);
        orien.normalize();
        pose.pose.orientation = tf2::toMsg(orien);
        state_path.poses.push_back(pose);
    }
    mPathPub->publish(state_path);
}
