#include "ackermann_control/AckermannControl.h"

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    const auto node = std::make_shared<AckermannControl>();
    // 多线程执行器：4 个工作线程（按需改）
    // rclcpp::executors::MultiThreadedExecutor exec(
    //     rclcpp::ExecutorOptions(), /* number_of_threads = */ 4);
    //
    // exec.add_node(node);
    // exec.spin(); // 会用线程池并发调度回调
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
