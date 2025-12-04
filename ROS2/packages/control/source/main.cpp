#include "control/node/ControlCmdPublisher.h"

int main(const int argc, char** argv)
{
    rclcpp::init(argc, argv);
    const auto control_cmd_pub = std::make_shared<ControlCmdPublisher>();
    rclcpp::spin(control_cmd_pub);
    rclcpp::shutdown();
    return 0;
}
