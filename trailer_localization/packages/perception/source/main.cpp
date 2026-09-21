#include "perception/nodes/Localization_LO.h"

int main(const int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<Localization_LO>("localization"));
    rclcpp::shutdown();
    return 0;
}