#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <ackermann_msgs/msg/ackermann_drive_stamped.hpp>
#include <algorithm>
#include <cmath>

class TwistToAckermann final : public rclcpp::Node
{
public:
    TwistToAckermann() : Node("twist_to_ackermann")
    {
        mWheelBase = declare_parameter("wheelbase", 1.20); // 轴距 L (m)
        mMaxSteerDeg = declare_parameter("max_steer_deg", 45.0); // 最大转角(度)
        mMaxSpeed = declare_parameter("max_speed", 3.0); // 速度限幅 (m/s)
        v_eps_ = declare_parameter("v_eps", 0.05); // 低速阈值
        mFrameId = declare_parameter("frame_id", std::string("base_link"));
        const std::string out_topic = declare_parameter("out_topic", std::string("ackermann_cmd"));
        const std::string in_topic = declare_parameter("in_topic", std::string("cmd_vel"));

        const rclcpp::QoS qos = rclcpp::QoS(rclcpp::KeepLast(1)).reliable().durability_volatile();

        mAckermannPub = create_publisher<ackermann_msgs::msg::AckermannDriveStamped>(out_topic, qos);
        mTwistSubscriber = create_subscription<geometry_msgs::msg::Twist>(in_topic, qos,
                                                                          std::bind(&TwistToAckermann::twistCallback,
                                                                              this,
                                                                              std::placeholders::_1));
        RCLCPP_INFO(get_logger(), "This node has been activated.");
    }

    ~TwistToAckermann() override
    {
        RCLCPP_INFO(get_logger(), "This node has been shut down.");
    }

private:
    double mWheelBase, mMaxSteerDeg, mMaxSpeed, v_eps_;
    std::string mFrameId;

    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr mTwistSubscriber;
    rclcpp::Publisher<ackermann_msgs::msg::AckermannDriveStamped>::SharedPtr mAckermannPub;

    void twistCallback(const geometry_msgs::msg::Twist::ConstSharedPtr& msg) const
    {
        double v = std::clamp(msg->linear.x, -mMaxSpeed, mMaxSpeed);
        const double w = msg->angular.z;

        double delta = 0.0;
        if (std::abs(v) > v_eps_)
        {
            delta = std::atan((mWheelBase * w) / v);
        }
        else
        {
            // 原地转对 Ackermann 不可行：停住并给最大可转角（也可改为 0）
            delta = (std::abs(w) > 1e-6) ? std::copysign(deg2rad(mMaxSteerDeg), w) : 0.0;
            v = 0.0;
        }
        // 转角限幅
        const double max_steer = deg2rad(mMaxSteerDeg);
        delta = std::clamp(delta, -max_steer, max_steer);

        ackermann_msgs::msg::AckermannDriveStamped ackermann_drive_msg;
        ackermann_drive_msg.header.stamp = this->now();
        ackermann_drive_msg.header.frame_id = mFrameId;
        ackermann_drive_msg.drive.speed = static_cast<float>(v);
        ackermann_drive_msg.drive.steering_angle = static_cast<float>(delta);
        // 可选：out.drive.steering_angle_velocity / acceleration / jerk
        mAckermannPub->publish(ackermann_drive_msg);
    }

    static double deg2rad(const double d)
    {
        constexpr double deg2rad_factor = M_PI / 180.0;
        return d * deg2rad_factor;
    }
};

int main(const int argc, char** argv)
{
    rclcpp::init(argc, argv);
    const auto node = std::make_shared<TwistToAckermann>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
