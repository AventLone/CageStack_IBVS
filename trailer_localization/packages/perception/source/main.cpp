#include "perception/nodes/TrailerLocalization.h"

int main(const int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<TrailerLocalization>("trailer_localization"));
    rclcpp::shutdown();
    return 0;
}