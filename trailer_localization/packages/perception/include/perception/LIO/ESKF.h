#pragma once
#include <functional>
#include <limits>
#include <optional>
#include "perception/LIO/ImuProcessor.hpp"

namespace lio
{
// Right rotation perturbation: R_true = R * Exp(dtheta).
// Error order: [dp_W, dtheta_I, dv_W, dbg_I, dba_I]. Gravity is fixed in W.
using ErrorStateT = Eigen::Vector<double, 15>;
using StateCovariance = Eigen::Matrix<double, 15, 15>;
using PoseInformation = Eigen::Matrix<double, 6, 6>;
using PoseGradient = Eigen::Vector<double, 6>;

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
        double damping_factor{1e-3};        // Pose-block damping used while solving only.
        double max_fitness_score{0.01};     // Mean squared Euclidean GICP residual, m^2.
        double convergence_translation{1e-5};
        double convergence_rotation{1e-5};
        double max_position_correction{1.0};
        double max_rotation_correction{0.5};
    };

    // Sums of raw geometric constraints H^T R^-1 H and H^T R^-1 r.
    // H has six nonzero columns [dp, dtheta]; the full prior couples v and biases.
    // Rebuild at EVERY iteration with a fixed map and fresh correspondences.
    struct Measurement
    {
        PoseInformation information{PoseInformation::Zero()};
        PoseGradient gradient{PoseGradient::Zero()};
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
        double fitness_score{std::numeric_limits<double>::infinity()};
    };

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

    static ImuState boxPlus(const ImuState& state, const ErrorStateT& increment);
    static ErrorStateT boxMinus(const ImuState& state, const ImuState& reference);
    static Eigen::Matrix3d rightJacobianInverse(const Eigen::Vector3d& rotation);

private:
    Config mConfig;
    Sophus::SE3d mTi2l;
    ImuProcessor mImuProcessor;
    ImuState mState;
    StateCovariance mCovariance{initialCovariance()};
    std::optional<ImuData> mLastImu;
    bool mInitialized{false};

    void predictCovariance(const ImuState& state, const ImuData& a, const ImuData& b);
};
}  // namespace lio
