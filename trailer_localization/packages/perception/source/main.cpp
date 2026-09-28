#include "perception/nodes/Localization_LIO.h"

int main(const int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<Localization_LIO>("localization"));
    rclcpp::shutdown();
    return 0;
}