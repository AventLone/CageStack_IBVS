#pragma once
#include <functional>
#include <limits>
#include <optional>
#include "perception/LIO/ImuProcessor.hpp"
// #include "perception/LIO/SparseVoxel.h"

namespace lio
{
// Right rotation perturbation: R_true = R * Exp(dtheta).
// Error order: [dp_W, dtheta_I, dv_W, dbg_I, dba_I]. Gravity is fixed in W.
using ErrorStateT = Eigen::Vector<double, 15>;
using StateCovariance = Eigen::Matrix<double, 15, 15>;
using PoseInformation = Eigen::Matrix<double, 6, 6>;
// using PoseGradient = Eigen::Vector<double, 6>;

class ESKF
{
public:
    struct Config
    {
        // Continuous-time noise densities: standard deviations, NOT variances.
        double gyro_noise{0.01};            // rad/s/sqrt(Hz)
        double accel_noise{0.1};            // m/s^2/sqrt(Hz)
        double gyro_bias_noise{0.0001};     // rad/s^2/sqrt(Hz)
        double accel_bias_noise{0.001};     // m/s^3/sqrt(Hz)
        double gravity_magnitude{9.81};
        double max_imu_gap{0.05};           // Reject missing data; never extrapolate.
        double max_integration_step{0.01};
        int max_iterations{6};
        std::size_t min_correspondences{30};
        double max_plane_rmse{0.15};        // m, evaluated at the final state.
        double convergence_translation{1e-4};
        double convergence_rotation{1e-4};
        double max_position_correction{1.0};
        double max_rotation_correction{0.5};
    };

    // Sums of raw geometric constraints H^T R^-1 H and H^T R^-1 r.
    // H has six nonzero columns [dp, dtheta]; the full prior couples v and biases.
    // Rebuild at EVERY iteration with a fixed map and fresh correspondences.
    struct Measurement
    {
        PoseInformation information{PoseInformation::Zero()};
        Sophus::SE3d::Tangent gradient{Sophus::SE3d::Tangent::Zero()};
        std::size_t count{0};
        double squared_error{0.0};
    };

    using MeasurementModel = std::function<Measurement(const ImuState&)>;

    struct Result
    {
        bool accepted{false};
        bool converged{false};
        int iterations{0};
        std::size_t num_correspondences{0};
        double plane_rmse{std::numeric_limits<double>::infinity()};
    };

    // struct Result
    // {
    //     bool converged{false};
    //     int iterations{0};
    //     std::size_t num_correspondences{0};
    //     double fitness_score{std::numeric_limits<double>::infinity()};
    // };

    ESKF() : ESKF(Config{})
    {
    }

    explicit ESKF(const Config& config, const Sophus::SE3d& T_IL = Sophus::SE3d());

    static StateCovariance initialCovariance();

    void clear() noexcept
    {
        mInitialized = false; mLastImu.reset();
    }

    void reset(const ImuState& state, const StateCovariance& covariance = initialCovariance());

    // Stationary initialization: roll/pitch and gyro bias; yaw/position set local W.
    // A single resting pose cannot identify all accelerometer bias components.
    bool initialize(const std::vector<ImuData>& stationary_imu);

    void predict(const ImuData& imu_data);

    void processScan(const std::vector<ImuData>& imu_data, std::vector<PointXYZT>& points, double scan_begin, double scan_end);

    Result update(const MeasurementModel& model);

    [[nodiscard]] bool initialized() const noexcept
    {
        return mInitialized;
    }

    [[nodiscard]] const ImuState& nominalState() const noexcept
    {
        return mState;
    }

    [[nodiscard]] const StateCovariance& covariance() const noexcept
    {
        return mCovariance;
    }

    [[nodiscard]] const Config& config() const noexcept
    {
        return mConfig;
    }

    [[nodiscard]] Eigen::Isometry3d state() const // T_WI
    {
        return Eigen::Isometry3d(Sophus::SE3d(mState.R_WI, mState.p_WI).matrix());
    }

    [[nodiscard]] Eigen::Isometry3d lidarPose() const // T_WL = T_WI * T_IL
    {
        return Eigen::Isometry3d((Sophus::SE3d(mState.R_WI, mState.p_WI) * mTi2l).matrix());
    }

private:
    Config mConfig;
    Sophus::SE3d mTi2l;
    ImuProcessor mImuProcessor;
    ImuState mState;
    StateCovariance mCovariance{initialCovariance()};
    std::optional<ImuData> mLastImu;
    bool mInitialized{false};

    // std::unique_ptr<SparseVoxel> mVoxelMap;

    void predictCovariance(const ImuState& state, const ImuData& a, const ImuData& b);

    static ImuState boxPlus(const ImuState& state, const ErrorStateT& increment)
    {
        ImuState result = state;
        result.p_WI += increment.segment<3>(0);
        result.R_WI *= Sophus::SO3d::exp(increment.segment<3>(3));
        result.v_WI += increment.segment<3>(6);
        result.gyro_bias += increment.segment<3>(9);
        result.accel_bias += increment.segment<3>(12);
        return result;
    }

    static ErrorStateT boxMinus(const ImuState& state, const ImuState& reference)
    {
        ErrorStateT result;
        result << state.p_WI - reference.p_WI, (reference.R_WI.inverse() * state.R_WI).log(),
                  state.v_WI - reference.v_WI, state.gyro_bias - reference.gyro_bias, state.accel_bias - reference.accel_bias;
        return result;
    }

    static Eigen::Matrix3d rightJacobianInverse(const Eigen::Vector3d& rotation);
};
}  // namespace lio
