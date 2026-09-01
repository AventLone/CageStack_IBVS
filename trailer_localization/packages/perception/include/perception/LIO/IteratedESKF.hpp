#pragma once
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <cmath>
#include <limits>

/**
 * @brief Nominal state used by the inertial error-state filter.
 *
 * The error state is ordered as [dp, dv, dtheta, dbg, dba].
 */
struct ESKFState
{
    Eigen::Vector3f position{Eigen::Vector3f::Zero()};
    Eigen::Vector3f velocity{Eigen::Vector3f::Zero()};
    Eigen::Quaternionf orientation{Eigen::Quaternionf::Identity()};
    Eigen::Vector3f gyro_bias{Eigen::Vector3f::Zero()};
    Eigen::Vector3f accel_bias{Eigen::Vector3f::Zero()};
};

struct IMUSample
{
    Eigen::Vector3f gyro{Eigen::Vector3f::Zero()};
    Eigen::Vector3f acceleration{Eigen::Vector3f::Zero()};
};

/**
 * @brief 15-state inertial ESKF with a generic iterated measurement update.
 *
 * A measurement model returns h(state), while its Jacobian is with respect to
 * the local error state. This makes the class suitable for LiDAR residuals
 * such as point-to-plane or pose observations without imposing a LiDAR model.
 */
class IteratedESKF
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    static constexpr int ErrorStateSize = 15;
    static constexpr int NoiseStateSize = 12;
    using ErrorState = Eigen::Matrix<float, ErrorStateSize, 1>;
    using Covariance = Eigen::Matrix<float, ErrorStateSize, ErrorStateSize>;
    using NoiseCovariance = Eigen::Matrix<float, NoiseStateSize, NoiseStateSize>;
    using PoseMeasurement = Eigen::Matrix<float, 6, 1>;

    explicit IteratedESKF(const ESKFState& state = {}, const Covariance& covariance = Covariance::Identity(),
                          const NoiseCovariance& noise = NoiseCovariance::Identity(),
                          const Eigen::Vector3f& gravity = Eigen::Vector3f(0.0f, 0.0f, -9.81f))
        : mState(state), mCovariance(covariance), mNoise(noise), mGravity(gravity)
    {}

    void predict(const IMUSample& sample, const float dt)
    {
        if (!(dt > 0.0f) || !std::isfinite(dt))
        {
            return;
        }

        const Eigen::Vector3f unbiased_gyro = sample.gyro - mState.gyro_bias;
        const Eigen::Vector3f unbiasedAcceleration = sample.acceleration - mState.accel_bias;
        const Eigen::Matrix3f rotation = mState.orientation.toRotationMatrix();
        const Eigen::Vector3f worldAcceleration = rotation * unbiasedAcceleration + mGravity;

        mState.position += mState.velocity * dt + 0.5f * worldAcceleration * dt * dt;
        mState.velocity += worldAcceleration * dt;
        mState.orientation = (mState.orientation * deltaQuaternion(unbiased_gyro * dt)).normalized();

        Covariance F = Covariance::Identity();
        F.block<3, 3>(0, 3) = Eigen::Matrix3f::Identity() * dt;
        F.block<3, 3>(3, 6) = -rotation * skew(unbiasedAcceleration) * dt;
        F.block<3, 3>(3, 12) = -rotation * dt;
        F.block<3, 3>(6, 6) = Eigen::Matrix3f::Identity() - skew(unbiased_gyro) * dt;
        F.block<3, 3>(6, 9) = -Eigen::Matrix3f::Identity() * dt;

        Eigen::Matrix<float, ErrorStateSize, NoiseStateSize> G = Eigen::Matrix<float, ErrorStateSize, NoiseStateSize>::Zero();
        G.block<3, 3>(3, 3) = rotation * dt;
        G.block<3, 3>(6, 0) = -Eigen::Matrix3f::Identity() * dt;
        G.block<3, 3>(9, 6) = Eigen::Matrix3f::Identity() * dt;
        G.block<3, 3>(12, 9) = Eigen::Matrix3f::Identity() * dt;

        mCovariance = F * mCovariance * F.transpose() + G * mNoise * G.transpose();
        mCovariance = 0.5f * (mCovariance + mCovariance.transpose());
    }

    bool updatePose(const Eigen::Vector3f& position, const Eigen::Quaternionf& orientation,
                    const Eigen::Matrix<float, 6, 6>& measurement_noise,
                    const int iterations = 4, const float convergence_threshold = 1e-4f)
    {
        const Eigen::Quaternionf reference_orientation = mState.orientation;
        PoseMeasurement measurement = PoseMeasurement::Zero();
        measurement.head<3>() = position;
        measurement.tail<3>() = 2.0f *
            (reference_orientation.conjugate() * orientation.normalized()).vec();

        return updateIterated(
            measurement,
            [reference_orientation](const ESKFState& state) {
                PoseMeasurement prediction = PoseMeasurement::Zero();
                prediction.head<3>() = state.position;
                prediction.tail<3>() = 2.0f *
                    (reference_orientation.conjugate() * state.orientation).vec();
                return prediction;
            },
            [](const ESKFState&) {
                Eigen::Matrix<float, 6, ErrorStateSize> jacobian =
                    Eigen::Matrix<float, 6, ErrorStateSize>::Zero();
                jacobian.block<3, 3>(0, 0).setIdentity();
                jacobian.block<3, 3>(3, 6).setIdentity();
                return jacobian;
            },
            measurement_noise, iterations, convergence_threshold);
    }

    template<class MeasurementT, class MeasurementModel, class JacobianModel>
    bool updateIterated(const MeasurementT& measurement, MeasurementModel&& measurement_model, JacobianModel&& jacobian_model,
                        const Eigen::Matrix<float, MeasurementT::RowsAtCompileTime, MeasurementT::RowsAtCompileTime>& measurement_noise,
                        const int iterations = 4, const float convergence_threshold = 1e-4f)
    {
        static_assert(MeasurementT::ColsAtCompileTime == 1, "Measurement must be a vector");
        using MeasurementMatrix = Eigen::Matrix<float, MeasurementT::RowsAtCompileTime, MeasurementT::RowsAtCompileTime>;
        using MeasurementJacobian = Eigen::Matrix<float, MeasurementT::RowsAtCompileTime, ErrorStateSize>;

        if (iterations < 1)
        {
            return false;
        }

        const ESKFState prior = mState;
        const Covariance prior_covariance = mCovariance;
        MeasurementJacobian jacobian = MeasurementJacobian::Zero();
        Eigen::Matrix<float, ErrorStateSize, MeasurementT::RowsAtCompileTime> kalman_gain =
            Eigen::Matrix<float, ErrorStateSize, MeasurementT::RowsAtCompileTime>::Zero();
        float correction_norm = std::numeric_limits<float>::infinity();

        for (int iteration = 0; iteration < iterations && correction_norm > convergence_threshold; ++iteration)
        {
            const MeasurementT predicted = measurement_model(mState);
            jacobian = jacobian_model(mState);
            const MeasurementMatrix innovationCovariance = jacobian * prior_covariance * jacobian.transpose() + measurement_noise;
            kalman_gain = prior_covariance * jacobian.transpose();
            kalman_gain = innovationCovariance.ldlt().solve(kalman_gain.transpose()).transpose();

            const ErrorState accumulatedError = difference(prior, mState);
            const ErrorState correction = kalman_gain * (measurement - predicted + jacobian * accumulatedError);
            inject(correction);
            correction_norm = correction.norm();
        }

        const Eigen::Matrix<float, ErrorStateSize, ErrorStateSize> update = Covariance::Identity() - kalman_gain * jacobian;
        mCovariance = update * prior_covariance * update.transpose() + kalman_gain * measurement_noise * kalman_gain.transpose();
        mCovariance = 0.5f * (mCovariance + mCovariance.transpose());
        resetErrorCovariance(difference(prior, mState).segment<3>(6));
        return mCovariance.allFinite();
    }

    const ESKFState& state() const noexcept
    {
        return mState;
    }
    const Covariance& covariance() const noexcept
    {
        return mCovariance;
    }
    void setState(const ESKFState& state) noexcept
    {
        mState = state;
    }
    void setCovariance(const Covariance& covariance) noexcept
    {
        mCovariance = covariance;
    }

private:
    ESKFState mState;
    Covariance mCovariance;
    NoiseCovariance mNoise;
    Eigen::Vector3f mGravity;

    static Eigen::Matrix3f skew(const Eigen::Vector3f& vector)
    {
        Eigen::Matrix3f result;
        result << 0.0f, -vector.z(), vector.y(), vector.z(), 0.0f, -vector.x(), -vector.y(), vector.x(), 0.0f;
        return result;
    }

    static Eigen::Quaternionf deltaQuaternion(const Eigen::Vector3f& rotation)
    {
        const float angle = rotation.norm();
        if (angle < 1e-6f)
            return Eigen::Quaternionf(1.0f, 0.5f * rotation.x(), 0.5f * rotation.y(), 0.5f * rotation.z()).normalized();
        return Eigen::Quaternionf(Eigen::AngleAxisf(angle, rotation / angle));
    }

    static ErrorState difference(const ESKFState& reference, const ESKFState& state)
    {
        ErrorState result = ErrorState::Zero();
        result.segment<3>(0) = state.position - reference.position;
        result.segment<3>(3) = state.velocity - reference.velocity;
        result.segment<3>(6) = 2.0f * (reference.orientation.conjugate() * state.orientation).vec();
        result.segment<3>(9) = state.gyro_bias - reference.gyro_bias;
        result.segment<3>(12) = state.accel_bias - reference.accel_bias;
        return result;
    }

    void inject(const ErrorState& error)
    {
        mState.position += error.segment<3>(0);
        mState.velocity += error.segment<3>(3);
        mState.orientation = (mState.orientation * deltaQuaternion(error.segment<3>(6))).normalized();
        mState.gyro_bias += error.segment<3>(9);
        mState.accel_bias += error.segment<3>(12);
    }

    void resetErrorCovariance(const Eigen::Vector3f& attitudeError)
    {
        Covariance reset = Covariance::Identity();
        reset.block<3, 3>(6, 6) -= 0.5f * skew(attitudeError);
        mCovariance = reset * mCovariance * reset.transpose();
    }
};