#pragma once
#include <rclcpp/rclcpp.hpp>

static constexpr rmw_qos_profile_t gQoSProfile{
    RMW_QOS_POLICY_HISTORY_KEEP_LAST,
    1,
    RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT,
    RMW_QOS_POLICY_DURABILITY_VOLATILE,
    RMW_QOS_DEADLINE_DEFAULT,
    RMW_QOS_LIFESPAN_DEFAULT,
    RMW_QOS_POLICY_LIVELINESS_SYSTEM_DEFAULT,
    RMW_QOS_LIVELINESS_LEASE_DURATION_DEFAULT,
    false
};

static constexpr rmw_qos_profile_t gQoSProfile_ControlMsg{
    RMW_QOS_POLICY_HISTORY_KEEP_LAST,
    2,
    RMW_QOS_POLICY_RELIABILITY_RELIABLE,
    RMW_QOS_POLICY_DURABILITY_VOLATILE,
    RMW_QOS_DEADLINE_DEFAULT,
    RMW_QOS_LIFESPAN_DEFAULT,
    RMW_QOS_POLICY_LIVELINESS_SYSTEM_DEFAULT,
    RMW_QOS_LIVELINESS_LEASE_DURATION_DEFAULT,
    false
};


static constexpr rmw_qos_profile_t gQoSProfile_IMU{
    RMW_QOS_POLICY_HISTORY_KEEP_LAST,
    2000,
    RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT,
    RMW_QOS_POLICY_DURABILITY_VOLATILE,
    RMW_QOS_DEADLINE_DEFAULT,
    RMW_QOS_LIFESPAN_DEFAULT,
    RMW_QOS_POLICY_LIVELINESS_SYSTEM_DEFAULT,
    RMW_QOS_LIVELINESS_LEASE_DURATION_DEFAULT,
    false
};

static constexpr rmw_qos_profile_t gQoSProfile_Lidar{
    RMW_QOS_POLICY_HISTORY_KEEP_LAST,
    5,
    RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT,
    RMW_QOS_POLICY_DURABILITY_VOLATILE,
    RMW_QOS_DEADLINE_DEFAULT,
    RMW_QOS_LIFESPAN_DEFAULT,
    RMW_QOS_POLICY_LIVELINESS_SYSTEM_DEFAULT,
    RMW_QOS_LIVELINESS_LEASE_DURATION_DEFAULT,
    false
};

inline const auto gQoS = rclcpp::QoS(rclcpp::QoSInitialization(gQoSProfile.history, gQoSProfile.depth),
                                     gQoSProfile);

inline const rclcpp::QoS gRobStateQoS(rclcpp::QoSInitialization(gQoSProfile.history, gQoSProfile.depth),
                                      gQoSProfile);
inline const rclcpp::QoS gControlMsgQoS(
    rclcpp::QoSInitialization(gQoSProfile_ControlMsg.history, gQoSProfile_ControlMsg.depth),
    gQoSProfile_ControlMsg);

inline const auto gQoS_IMU = rclcpp::QoS(rclcpp::QoSInitialization(gQoSProfile_IMU.history, gQoSProfile_IMU.depth),
                                         gQoSProfile_IMU);

inline const auto gQoS_Lidar =
        rclcpp::QoS(rclcpp::QoSInitialization(gQoSProfile_Lidar.history, gQoSProfile_Lidar.depth), gQoSProfile_Lidar);
