#include "perception/nodes/Localization_LIO_tight.h"

int main(const int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<Localization_LIO_T>("localization"));
    rclcpp::shutdown();
    return 0;
}