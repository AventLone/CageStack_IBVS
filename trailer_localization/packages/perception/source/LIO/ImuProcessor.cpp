#include "perception/LIO/ImuProcessor.h"
#include <algorithm>

namespace lio
{
ImuData ImuProcessor::imuAt(const std::vector<ImuData>& imu, const double time)
{
    const auto upper = std::lower_bound(imu.begin(), imu.end(), time,
        [](const ImuData& sample, double t) { return sample.timestamp < t; });
    if (upper == imu.begin() || upper->timestamp == time)
    {
        return *upper;
    }
    const auto& lower = *(upper - 1);
    const double alpha = (time - lower.timestamp) / (upper->timestamp - lower.timestamp);
    return {time, (1.0 - alpha) * lower.gyro + alpha * upper->gyro,
                  (1.0 - alpha) * lower.accel + alpha * upper->accel};
}

void ImuProcessor::process(const std::vector<ImuData>& imu, StampedCloud& cloud, ESKF& filter)
{
    mTrajectory.clear();
    mTrajectory.reserve(imu.size() + 1);
    mTrajectory.push_back({filter.state().timestamp, filter.pose()});
    for (const auto& sample : imu)
    {
        if (sample.timestamp > filter.state().timestamp && sample.timestamp < cloud.end_time)
        {
            filter.predict(sample);
            mTrajectory.push_back({sample.timestamp, filter.pose()});
        }
    }
    if (cloud.end_time > filter.state().timestamp)
    {
        filter.predict(imuAt(imu, cloud.end_time));
        mTrajectory.push_back({cloud.end_time, filter.pose()});
    }

    const Sophus::SE3d T_lend_w = (filter.pose() * mT_il).inverse();
    for (auto& point : cloud.points)
    {
        const Sophus::SE3d T_wl = poseAt(cloud.begin_time + point.time_offset) * mT_il;
        point.position = (T_lend_w * (T_wl * point.position.cast<double>())).cast<float>();
    }
}

Sophus::SE3d ImuProcessor::poseAt(const double time) const
{
    if (time >= mTrajectory.back().timestamp) return mTrajectory.back().pose;
    const auto upper = std::lower_bound(mTrajectory.begin(), mTrajectory.end(), time,
        [](const TimedPose& pose, double t) { return pose.timestamp < t; });
    if (upper == mTrajectory.begin() || upper->timestamp == time)
    {
        return upper->pose;
    }
    const auto& lower = *(upper - 1);
    const double alpha = (time - lower.timestamp) / (upper->timestamp - lower.timestamp);
    return {Sophus::SO3d(lower.pose.unit_quaternion().slerp(alpha, upper->pose.unit_quaternion())),
            (1.0 - alpha) * lower.pose.translation() + alpha * upper->pose.translation()};
}
} // namespace lio
