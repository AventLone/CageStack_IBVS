#pragma once
#include "perception/types/common.hpp"
#include <type_traits>

template<class StateT, class ControlT, class MeasurementT, class SystemModel, class MeasureModel>
class BaseEKF
{
    static_assert(StateT::RowsAtCompileTime > 0, "State vector must contain at least 1 element");
    static_assert(MeasurementT::RowsAtCompileTime > 0, "State vector must contain at least 1 element");
    static_assert(ControlT::RowsAtCompileTime >= 0, "Control vector must contain at least 0 elements");
    static_assert(std::is_same_v<typename StateT::Scalar, typename ControlT::Scalar>,
                  "State and Control scalar types must be identical");
    static_assert(std::is_same_v<typename StateT::Scalar, typename MeasurementT::Scalar>,
                  "State and Measurement scalar types must be identical");

protected:
    explicit BaseEKF(const CovarianceMatrix<StateT>& matQ, const CovarianceMatrix<MeasurementT>& matR,
                     const StateT& vecX0, const CovarianceMatrix<StateT>& matP0)
        : mVecX(vecX0), mMatQ(matQ), mMatV(matR), mMatP(matP0)
    {
        mJacobianA.setIdentity();
        mJacobianW.setIdentity();
        mJacobianH.setIdentity();
        mJacobianV.setIdentity();
    }

    BaseEKF(const BaseEKF&) = default;

    BaseEKF operator=(const BaseEKF&) = delete;

    virtual ~BaseEKF() = default;

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    void predict(const ControlT& vecU)
    {}

    void correct(const MeasurementT& vecZ)
    {}

    /* 在const成员函数中调用非const成员函数会报错 */
    /* Simultaneous prediction and correction */
    void update(const ControlT& vecU, const MeasurementT& vecZ) noexcept
    {
        updateJacobians(vecU);

        /* prediction */
        mVecX = fSystemModel(mVecX, vecU);
        mMatP = mJacobianA * mMatP * mJacobianA.transpose() + mJacobianW * mMatQ * mJacobianW.transpose();

        /* Correction */
        // Compute innovation covariance
        auto matS = mJacobianH * mMatP * mJacobianH.transpose() + mJacobianV * mMatV * mJacobianV.transpose();
        auto matK = mMatP * mJacobianH.transpose() * matS.inverse(); // Compute Kalman Gain

        mVecX += matK * (vecZ - fMeasurementModel(mVecX)); // Update state estimate
        mMatP -= matK * mJacobianH * mMatP; // Update covariance matrix
    }

    const StateT& getState()
    {
        return mVecX;
    }

protected:
    StateT mVecX; // Estimated state

    CovarianceMatrix<StateT> mMatQ; // System Noise
    CovarianceMatrix<MeasurementT> mMatV; // Measurement Noise

    Jacobian<StateT, StateT> mJacobianA; // System model Jacobian
    Jacobian<StateT, StateT> mJacobianW; // System model noise jacobian

    Jacobian<MeasurementT, StateT> mJacobianH; // Measurement model Jacobian
    Jacobian<MeasurementT, MeasurementT> mJacobianV; // Measurement model noise Jacobian

private:
    CovarianceMatrix<StateT> mMatP; // Estimated state covariance

    /* Function object or Functor */
    SystemModel fSystemModel;
    MeasureModel fMeasurementModel;

    virtual void updateJacobians(const ControlT& vecU) = 0;
};
