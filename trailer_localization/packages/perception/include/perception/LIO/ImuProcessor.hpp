#pragma once
#include <algorithm>
#include <cmath>
#include <functional>
#include <stdexcept>
#include <vector>
#include "perception/LIO/ImuTypes.hpp"

namespace lio
{
class ImuProcessor
{
public:
    // Called BEFORE each nominal integration step, using the same midpoint inputs.
    using StepCallback = std::function<void(const ImuState&, const ImuData&, const ImuData&)>;

    // T_IL maps LiDAR coordinates to IMU coordinates: p_I = T_IL * p_L.
    explicit ImuProcessor(const Sophus::SE3d& T_IL) : mTi2l(T_IL) {}

    void process(const std::vector<ImuData>& imu_data, std::vector<PointXYZT>& points, const double scan_begin,
        const double scan_end, ImuState& state, const StepCallback& before_step = {}, const double max_step = 0.01)
    {
        if (!std::isfinite(scan_begin) || !std::isfinite(scan_end) ||
            scan_end <= scan_begin || state.timestamp > scan_begin)
        {
            throw std::invalid_argument("Invalid or overlapping LiDAR scan interval.");
        }
        for (const auto& point : points)
        {
            if (!std::isfinite(point.relative_time) || point.relative_time < -1e-6 ||
                point.relative_time > scan_end - scan_begin + 1e-6)
            {
                throw std::invalid_argument("LiDAR point time is outside the scan interval.");
            }
        }
        // Include gaps between scans. The previous corrected state is the only prior.
        const auto samples = buildImuSequence(imu_data, state.timestamp, scan_end, max_step);
        propagate(samples, state, before_step);
        deskew(points, scan_begin, scan_end);
    }

    static bool finite(const ImuData& imu)
    {
        return std::isfinite(imu.timestamp) && imu.gyro.allFinite() && imu.accel.allFinite();
    }

    static void validate(const std::vector<ImuData>& imu_data)
    {
        if (imu_data.size() < 2)
        {
            throw std::invalid_argument("At least two IMU samples are required.");
        }
        for (std::size_t i = 0; i < imu_data.size(); ++i)
        {
            if (!finite(imu_data[i]) || (i > 0 && imu_data[i].timestamp <= imu_data[i - 1].timestamp))
            {
                throw std::invalid_argument("IMU samples must be finite and strictly time ordered.");
            }
        }
    }

    static ImuData interpolateImu(const ImuData& a, const ImuData& b, double timestamp)
    {
        const double alpha = (timestamp - a.timestamp) / (b.timestamp - a.timestamp);
        return {timestamp, (1.0 - alpha) * a.gyro + alpha * b.gyro,
                           (1.0 - alpha) * a.accel + alpha * b.accel};
    }

    // Exact boundary interpolation; never extrapolate beyond received measurements.
    static std::vector<ImuData> buildImuSequence(const std::vector<ImuData>& imu_data,
                                                 const double begin, const double end, const double max_step = 0.01)
    {
        validate(imu_data);
        if (!std::isfinite(begin) || !std::isfinite(end) || end < begin ||
            !std::isfinite(max_step) || max_step <= 0.0 ||
            imu_data.front().timestamp > begin || imu_data.back().timestamp < end)
        {
            throw std::invalid_argument("IMU samples do not cover the integration interval.");
        }
        const auto at = [&imu_data](const double time)
        {
            const auto it = std::lower_bound(imu_data.begin(), imu_data.end(), time, [](const ImuData& imu, const double t)
                {
                    return imu.timestamp < t;
                });
            if (it == imu_data.begin() || it->timestamp == time) return *it;
            return interpolateImu(*(it - 1), *it, time);
        };

        std::vector<ImuData> knots{at(begin)};
        for (const auto& imu : imu_data)
        {
            if (imu.timestamp > begin && imu.timestamp < end) knots.push_back(imu);
        }

        if (end > begin) knots.push_back(at(end));

        std::vector<ImuData> result{knots.front()};
        for (std::size_t i = 1; i < knots.size(); ++i)
        {
            const auto& a = knots[i - 1];
            const auto& b = knots[i];
            const int steps = static_cast<int>(std::ceil((b.timestamp - a.timestamp) / max_step));
            for (int k = 1; k < steps; ++k)
            {
                result.push_back(interpolateImu(a, b, a.timestamp + (b.timestamp - a.timestamp) * k / steps));
            }
            result.push_back(b);
        }
        return result;
    }

    static void integrate(const ImuData& a, const ImuData& b, ImuState& state)
    {
        const double dt = b.timestamp - a.timestamp;
        const Eigen::Vector3d omega = 0.5 * (a.gyro + b.gyro) - state.gyro_bias;
        const Eigen::Vector3d force = 0.5 * (a.accel + b.accel) - state.accel_bias;
        const Sophus::SO3d R_mid = state.R_WI * Sophus::SO3d::exp(omega * (0.5 * dt));
        const Eigen::Vector3d acceleration = R_mid * force + state.gravity;
        state.p_WI += state.v_WI * dt + 0.5 * acceleration * dt * dt;
        state.v_WI += acceleration * dt;
        state.R_WI *= Sophus::SO3d::exp(omega * dt);
        state.timestamp = b.timestamp;
    }

    void propagate(const std::vector<ImuData>& samples, ImuState& state,
                   const StepCallback& before_step = {})
    {
        validate(samples);
        if (std::abs(state.timestamp - samples.front().timestamp) > 1e-9)
        {
            throw std::invalid_argument("IMU sequence must start at the nominal state timestamp.");
        }
        mTrajectory.clear();
        mTrajectory.reserve(samples.size());
        mTrajectory.push_back(state);
        for (std::size_t i = 1; i < samples.size(); ++i)
        {
            if (before_step) before_step(state, samples[i - 1], samples[i]);
            integrate(samples[i - 1], samples[i], state);
            mTrajectory.push_back(state);
        }
    }

    Sophus::SE3d poseAt(const double timestamp) const
    {
        if (mTrajectory.empty() || timestamp < mTrajectory.front().timestamp - 1e-6 ||
            timestamp > mTrajectory.back().timestamp + 1e-6)
        {
            throw std::out_of_range("Requested pose is outside the IMU trajectory.");
        }
        if (timestamp <= mTrajectory.front().timestamp) return pose(mTrajectory.front());
        if (timestamp >= mTrajectory.back().timestamp) return pose(mTrajectory.back());
        const auto it = std::lower_bound(mTrajectory.begin(), mTrajectory.end(), timestamp, [](const ImuState& state, const double t)
            {
                return state.timestamp < t;
            });
        const auto& a = *(it - 1);
        const auto& b = *it;
        const double dt = b.timestamp - a.timestamp;
        const double t = timestamp - a.timestamp;
        const auto R = a.R_WI * Sophus::SO3d::exp((t / dt) * (a.R_WI.inverse() * b.R_WI).log());
        // Constant acceleration interpolation is consistent with midpoint propagation.
        const Eigen::Vector3d p = a.p_WI + a.v_WI * t + 0.5 * (b.v_WI - a.v_WI) * (t * t / dt);
        return Sophus::SE3d(R, p);
    }

    void deskew(std::vector<PointXYZT>& points, double scan_begin, double scan_end) const
    {
        const Sophus::SE3d T_Lend_W = (poseAt(scan_end) * mTi2l).inverse();
        for (auto& point : points)
        {
            const double t = std::clamp(scan_begin + point.relative_time, scan_begin, scan_end);
            const Eigen::Vector3d p = T_Lend_W * (poseAt(t) * (mTi2l * Eigen::Vector3d(point.x, point.y, point.z)));
            point.x = static_cast<float>(p.x());
            point.y = static_cast<float>(p.y());
            point.z = static_cast<float>(p.z());
        }
    }

private:
    static Sophus::SE3d pose(const ImuState& state)
    {
        return {state.R_WI, state.p_WI};
    }
    Sophus::SE3d mTi2l;
    std::vector<ImuState> mTrajectory;
};
}  // namespace lio
