#include "free_space_detection/nodes/FreeSpaceDetect.h"

int main(const int argc, char** argv)
{
    rclcpp::init(argc, argv);
    const auto options = rclcpp::NodeOptions();
    const auto target_pose_pub_node = std::make_shared<FreeSpaceDetect>("target_publisher", options);
    // rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(), 2);
    // executor.add_node(cloud_pub_node);
    // executor.add_node(target_pose_pub_node);
    // executor.spin();
    rclcpp::spin(target_pose_pub_node);
    rclcpp::shutdown();
    return 0;
}
