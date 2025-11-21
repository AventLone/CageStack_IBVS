#include <perception/ImageProcess.h>

int main(const int argc, char** argv)
{
    rclcpp::init(argc, argv);
    const auto perception_node = std::make_shared<ImageProcess>();
    rclcpp::spin(perception_node);
    rclcpp::shutdown();
    return 0;
}
