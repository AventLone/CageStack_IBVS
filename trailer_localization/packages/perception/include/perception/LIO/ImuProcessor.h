#pragma once
#include "perception/LIO/ESKF.h"
#include "perception/types/stamped_cloud.hpp"

namespace lio
{
class ImuProcessor
{
public:
    // T_il maps LiDAR coordinates into IMU coordinates.
    explicit ImuProcessor(const Sophus::SE3d& T_il) : mT_il(T_il) {}

    // Ordered IMU must bracket [filter time, scan end]; filter time <= scan begin.
    // The filter itself supplies the nominal trajectory and covariance propagation.
    void process(const std::vector<ImuData>& imu, StampedCloud& cloud, ESKF& filter);
    static ImuData imuAt(const std::vector<ImuData>& imu, double time);

private:
    struct TimedPose
    {
        double timestamp;
        Sophus::SE3d pose;
    };
    Sophus::SE3d mT_il;
    std::vector<TimedPose> mTrajectory;
    Sophus::SE3d poseAt(double time) const;
};
} // namespace lio
