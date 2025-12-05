#include "perception/nodes/CloudPublisher.h"
#include "perception/nodes/TargetPosePublisher.h"

int main(const int argc, char** argv)
{
    rclcpp::init(argc, argv);
    const auto options = rclcpp::NodeOptions();
    const auto cloud_pub_node = std::make_shared<CloudPublisher>("cloud_publisher", options);
    const auto target_pose_pub_node = std::make_shared<TargetPosePublisher>("target_publisher", options);
    rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(), 2);
    executor.add_node(cloud_pub_node);
    executor.add_node(target_pose_pub_node);
    executor.spin();
    rclcpp::shutdown();
    return 0;
}
