#include "control/node/ControlCmdPublisher.h"

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
    for (const auto& x : xs)
    {
        geometry_msgs::msg::PoseStamped pose;
        pose.header.frame_id = "map";
        pose.header.stamp = this->now();

        pose.pose.position.x = x[0];
        pose.pose.position.y = x[1];
        tf2::Quaternion orien;
        orien.setRPY(0.0, 0.0, x[2]);
        orien.normalize();
        pose.pose.orientation = tf2::toMsg(orien);
        state_path.poses.push_back(pose);
    }
    mPathPub->publish(state_path);
}
