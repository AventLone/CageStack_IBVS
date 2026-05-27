#pragma once
#include <Eigen/Dense>
#include <type_traits>
#include <memory>

/**
 * @brief Template type for covariance matrices
 * @tparam Type The vector type for which to generate a covariance (usually a state or measurement type)
 */
template<class Type>
using CovarianceMatrix = Eigen::Matrix<typename Type::Scalar, Type::RowsAtCompileTime, Type::RowsAtCompileTime>;

template<class State>
using StateMatrix = Eigen::Matrix<typename State::Scalar, State::RowsAtCompileTime, State::RowsAtCompileTime>;

template<class State, class Control>
using ControlMatrix = Eigen::Matrix<typename State::Scalar, State::RowsAtCompileTime, Control::RowsAtCompileTime>;

template<class State, class Measure>
using MeasureMatrix = Eigen::Matrix<typename State::Scalar, Measure::RowsAtCompileTime, State::RowsAtCompileTime>;

/**
 * @brief This class is used to describ the commonest linear system model and measurement model.
 */
template<class StateT, class ControlT, class MeasurementT>
class KalmanFilter
{
    static_assert(StateT::RowsAtCompileTime > 0, "State vector must contain at least 1 element");
    static_assert(MeasurementT::RowsAtCompileTime > 0, "State vector must contain at least 1 element");
    static_assert(ControlT::RowsAtCompileTime >= 0, "Control vector must contain at least 0 elements");
    static_assert(std::is_same_v<typename StateT::Scalar, typename ControlT::Scalar>,
                  "State and Control scalar types must be identical");
    static_assert(std::is_same_v<typename StateT::Scalar, typename MeasurementT::Scalar>,
                  "State and Measurement scalar types must be identical");

public:
    using Ptr = std::unique_ptr<KalmanFilter>;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    explicit KalmanFilter(const StateMatrix<StateT>& matA, const ControlMatrix<StateT, ControlT>& matB,
                          const CovarianceMatrix<StateT>& matQ, const MeasureMatrix<StateT, MeasurementT>& matH,
                          const CovarianceMatrix<MeasurementT>& matR,
                          const CovarianceMatrix<StateT>& matP0 = CovarianceMatrix<StateT>::Identity())
        : mMatA(matA), mMatB(matB), mMatQ(matQ), mMatH(matH), mMatV(matR), mMatP(matP0)
    {}

    KalmanFilter(const KalmanFilter&) = default;

    KalmanFilter operator=(const KalmanFilter&) = delete;

    ~KalmanFilter() = default;

    void setInitialX(StateT x0)
    {
        mVecX = x0;
    }

    void setInitialMatP(CovarianceMatrix<StateT> matP0)
    {
        mMatP = matP0;
    }

    static Ptr create(const StateMatrix<StateT>& matA, const ControlMatrix<StateT, ControlT>& matB,
                      const CovarianceMatrix<StateT>& matQ, const MeasureMatrix<StateT, MeasurementT>& matH,
                      const CovarianceMatrix<MeasurementT>& matR) noexcept
    {
        return std::make_unique<KalmanFilter>(matA, matB, matQ, matH, matR);
    }

    /* Predict the state vector x */
    void predict(const ControlT& vecU)
    {
        mVecX = mMatA * mVecX + mMatB * vecU;
        mMatP = mMatA * mMatP * mMatA.transpose() + mMatQ;
    }

    void correct(const MeasurementT& vecZ)
    {
        /* Kalman Gain */
        const auto matK = mMatP * mMatH.transpose() * (mMatH * mMatP * mMatH.transpose() + mMatV).inverse();

        /* Correct */
        mVecX += matK * (vecZ - mMatH * mVecX);
        mMatP -= matK * mMatH * mMatP;
    }

    /* Simultaneous prediction and correction  */
    void update(const ControlT& vecU, const MeasurementT& vecZ)
    {
        /* Predict */
        mVecX = mMatA * mVecX + mMatB * vecU;
        mMatP = mMatA * mMatP * mMatA.transpose() + mMatQ;

        /* Correction */
        auto matK = mMatP * mMatH.transpose() * (mMatH * mMatP * mMatH.transpose() + mMatV).inverse(); // Kalman Gain
        mVecX += matK * (vecZ - mMatH * mVecX);
        mMatP -= matK * mMatH * mMatP;
    }

    const StateT& getState()
    {
        return mVecX;
    }

private:
    StateMatrix<StateT> mMatA; // System Matrix
    ControlMatrix<StateT, ControlT> mMatB; // Control Matrix
    CovarianceMatrix<StateT> mMatQ; // System Noise Covariance Matrix

    MeasureMatrix<StateT, MeasurementT> mMatH; // Measurement Matrix
    CovarianceMatrix<MeasurementT> mMatV; // Measurement Noise Covariance Matrix

    StateT mVecX; // Estimated state
    CovarianceMatrix<StateT> mMatP; // Estimated state covariance
};
