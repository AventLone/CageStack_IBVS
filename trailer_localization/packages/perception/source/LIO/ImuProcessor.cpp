#include "perception/LIO/ImuProcessor.h"

std::vector<lio::ImuData> lio::ImuProcessor::buildImuSequence(const std::vector<ImuData>& imu_data,
                                                              const double begin, const double end)
{
    std::vector<ImuData> result;
    result.reserve(imu_data.size() + 2);
    result.push_back(imuAt(imu_data, begin));

    for (const auto& imu : imu_data)
    {
        if (imu.timestamp > begin && imu.timestamp < end)
        {
            result.push_back(imu);
        }
    }

    result.push_back(imuAt(imu_data, end));
    return result;
}

void lio::ImuProcessor::propagate(const std::vector<ImuData>& imu_data, ImuState& state)
{
    if (imu_data.size() < 2)
    {
        return;
    }

    mTrajectory.clear();
    mTrajectory.reserve(imu_data.size());

    mTrajectory.push_back(
        PoseState{
            .timestamp = state.timestamp,
            .R_wi = state.R_wi,
            .p_wi = state.p_wi,
            .v_wi = state.v_wi,
        });

    for (std::size_t i = 0; i + 1 < imu_data.size(); ++i)
    {
        const ImuData& imu0 = imu_data[i];
        const ImuData& imu1 = imu_data[i + 1];

        const double dt = imu1.timestamp - imu0.timestamp;

        if (dt <= 0.0)
        {
            continue;
        }

        /*
         * Bias corrected angular velocity.
         * Midpoint integration: ω = (ω0 + ω1) / 2 - bg
         */
        const Eigen::Vector3d omega = 0.5 * (imu0.gyro + imu1.gyro) - state.gyro_bias;

        /*
         * Bias corrected specific force.
         * Accelerometer measures: f = R_IW (a_W - g_W)
         * therefore: a_W = R_WI * f + g_W
         */
        const Eigen::Vector3d specific_force = 0.5 * (imu0.accel + imu1.accel)- state.accel_bias;

        /*
         * Orientation at middle of the interval.
         *
         * This gives better acceleration integration
         * than simply using R_k.
         */
        const Sophus::SO3d R_mid = state.R_wi * Sophus::SO3d::exp(omega * (0.5 * dt));
        const Eigen::Vector3d accel_world = R_mid * specific_force + state.gravity;

        /*
         * Position / velocity integration.
         */
        state.p_wi += state.v_wi * dt + 0.5 * accel_world * dt * dt;
        state.v_wi += accel_world * dt;

        /*
         * Rotation integration.
         */
        state.R_wi = state.R_wi * Sophus::SO3d::exp(omega * dt);
        state.timestamp = imu1.timestamp;

        mTrajectory.push_back(
            PoseState{
                .timestamp = state.timestamp,
                .R_wi = state.R_wi,
                .p_wi = state.p_wi,
                .v_wi = state.v_wi,
            });
    }
}

void lio::ImuProcessor::deskew(StampedCloud& stamped_cloud) const
{
    if (mTrajectory.empty())
    {
        throw std::runtime_error("IMU trajectory is empty.");
    }

    const Sophus::SE3d T_WI_end = poseAt(stamped_cloud.end_time);
    const Sophus::SE3d T_WL_end = T_WI_end * mT_il;
    const Sophus::SE3d T_Lend_W = T_WL_end.inverse();

    for (auto& [position, _, offset] : stamped_cloud.points)
    {
        const double point_time = std::clamp(stamped_cloud.begin_time + offset,
                                             stamped_cloud.begin_time, stamped_cloud.end_time);
        const Sophus::SE3d T_WI_t = poseAt(point_time);
        const Sophus::SE3d T_WL_t = T_WI_t * mT_il;

        const Eigen::Vector3d p_L(position.cast<double>());
        const Eigen::Vector3d p_deskewed = T_Lend_W * (T_WL_t * p_L);

        position = p_deskewed.cast<float>();
    }
}

Sophus::SE3d lio::ImuProcessor::poseAt(const double timestamp) const
{
    if (mTrajectory.empty())
    {
        throw std::runtime_error("IMU trajectory is empty.");
    }

    if (timestamp <= mTrajectory.front().timestamp)
    {
        return toSE3(mTrajectory.front());
    }

    if (timestamp >= mTrajectory.back().timestamp)
    {
        return toSE3(mTrajectory.back());
    }

    const auto iter = std::lower_bound(mTrajectory.begin(), mTrajectory.end(), timestamp,
        [](const PoseState& state, const double time)
            {
                return state.timestamp < time;
            });

    const PoseState& s1 = *iter;
    const PoseState& s0 = *(iter - 1);

    const double dt = s1.timestamp - s0.timestamp;

    const double alpha = (timestamp - s0.timestamp) / dt;

    /* Translation interpolation */
    const Eigen::Vector3d position = (1.0 - alpha) * s0.p_wi + alpha * s1.p_wi;

    /*
     * SO(3) interpolation.
     * Instead of directly doing quaternion SLERP:
     *   R = R0 * Exp(alpha * Log(R0^-1 R1))
     * This fits naturally with Sophus.
     */
    const Sophus::SO3d delta_R = s0.R_wi.inverse() * s1.R_wi;
    const Sophus::SO3d rotation = s0.R_wi * Sophus::SO3d::exp(alpha * delta_R.log());

    return {rotation, position};
}