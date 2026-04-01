#pragma once
#include "perception/types/common.hpp"
#include <type_traits>

/**
 * @brief Unscented Kalman Filter
 */
template<class SystemModel, class MeasureModel, class StateT>
class UKF
{
    using MeasurementT = typename MeasureModel::MeasurementT;
    using ControlT = typename SystemModel::ControlT;

    static_assert(StateT::RowsAtCompileTime > 0, "State vector must contain at least 1 element");
    static_assert(MeasurementT::RowsAtCompileTime > 0, "State vector must contain at least 1 element");
    static_assert(ControlT::RowsAtCompileTime >= 0, "Control vector must contain at least 0 elements");
    static_assert(std::is_same_v<typename StateT::Scalar, typename ControlT::Scalar>,
                  "State and Control scalar types must be identical");
    static_assert(std::is_same_v<typename StateT::Scalar, typename MeasurementT::Scalar>,
                  "State and Measurement scalar types must be identical");

    using Type = typename StateT::Scalar;
    static constexpr auto N = StateT::RowsAtCompileTime;
    /* The number of sigma points (depending on state dimensionality) */
    static constexpr int SigmaPointsCount = 2 * N + 1;

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    explicit UKF(const StateT& vecX0 = StateT::Zero(),
                 const CovarianceMatrix<StateT>& matP0 = CovarianceMatrix<StateT>::Identity(),
                 Type a = Type(1.0), Type b = Type(2.0), Type k = Type(0.0))
        : mVecX(vecX0), alpha(a), beta(b), kappa(k), mMatP(matP0),
          mMatQ(SystemModel::MatQ), mMatV(MeasureModel::MatR)
    {
        computeWeights(); // Pre-compute all weights
        mStateSigmaPoints.setZero();
        mMeasurementSigmaPoints.setZero();
    }

    UKF(const UKF&) = default;

    UKF operator=(const UKF&) = delete;

    ~UKF() = default;

    void predict(const ControlT& vecU);

    void correct(const MeasurementT& vecZ);

    /* 在const成员函数中调用非const成员函数会报错 */
    /* Simultaneous prediction and correction  */
    void update(const ControlT& vecU, const MeasurementT& vecZ) noexcept
    {
        predict(vecU);
        correct(vecZ);
    }

    const StateT& getState()
    {
        return mVecX;
    }

private:
    StateT mVecX; // Estimated state

    /* Vector containg the sigma scaling weights */
    using SigmaWeights = Eigen::Vector<Type, SigmaPointsCount>;

    /* Matrix type containing the state or measurement sigma points */
    template<class Vec>
    using SigmaPoints = Eigen::Matrix<Type, Vec::RowsAtCompileTime, SigmaPointsCount>;

    const Type alpha, beta, kappa;
    Type gamma{}, lambda{}; // Weight parameters

    SigmaWeights mWeightsM, mWeightsC;

    SigmaPoints<StateT> mStateSigmaPoints; // State sigma points
    SigmaPoints<MeasurementT> mMeasurementSigmaPoints; // Measurement sigma points

    CovarianceMatrix<StateT> mMatP; // Estimated state covariance

    CovarianceMatrix<StateT> mMatQ; // System Noise
    CovarianceMatrix<MeasurementT> mMatV; // Measurement Noise

    SystemModel fSystemModel;
    MeasureModel fMeasurementModel;

    void computeWeights(); // Compute sigma weights

    void sampleStateSigmaPoints();
};

template<class SystemModel, class MeasureModel, class StateT>
void UKF<SystemModel, MeasureModel, StateT>::predict(const ControlT& vecU)
{
    sampleStateSigmaPoints(); // 后验采样 Posterior Sample

    /* Compute the transition of sigma points */
    for (int i = 0; i <= 2 * N; ++i)
    {
        mStateSigmaPoints.col(i) = fSystemModel(mStateSigmaPoints.col(i), vecU);
    }

    /* Prediction */
    mVecX.setZero();
    for (int i = 0; i <= 2 * N; ++i)
    {
        mVecX += mWeightsM[i] * mStateSigmaPoints.col(i);
    }
    mMatP.setZero();
    for (int i = 0; i <= 2 * N; ++i)
    {
        auto temp_vec = mStateSigmaPoints.col(i) - mVecX;
        mMatP += mWeightsC[i] * temp_vec * temp_vec.transpose();
    }
    mMatP += mMatQ;
}

template<class SystemModel, class MeasureModel, class StateT>
void UKF<SystemModel, MeasureModel, StateT>::correct(const MeasurementT& vecZ)
{
    sampleStateSigmaPoints(); // 先验采样

    /* Compute the transition of sigma points */
    for (int i = 0; i <= 2 * N; ++i)
    {
        mMeasurementSigmaPoints.col(i) = fMeasurementModel(mStateSigmaPoints.col(i));
    }

    /* Compute the distribution of measurement */
    MeasurementT meanZ = MeasurementT::Zero();
    CovarianceMatrix<MeasurementT> CovZ = CovarianceMatrix<MeasurementT>::Zero();
    for (int i = 0; i <= 2 * N; ++i)
    {
        meanZ += mWeightsM[i] * mMeasurementSigmaPoints.col(i);
    }
    for (int i = 0; i <= 2 * N; ++i)
    {
        auto temp_vec = mMeasurementSigmaPoints.col(i) - meanZ;
        CovZ += mWeightsC[i] * temp_vec * temp_vec.transpose();
    }
    CovZ += mMatV;

    Eigen::Matrix<typename StateT::Scalar, N, MeasurementT::RowsAtCompileTime> CovXZ;
    CovXZ.setZero();
    for (int i = 0; i <= 2 * N; ++i)
    {
        CovXZ +=
                mWeightsC[i] * (mStateSigmaPoints.col(i) - mVecX) * (mMeasurementSigmaPoints.col(i) - meanZ).transpose();
    }

    auto matK = CovXZ * CovZ.inverse();

    /* Correction */
    mVecX += matK * (vecZ - meanZ);
    mMatP -= matK * CovZ * matK.transpose();
}


template<class SystemModel, class MeasureModel, class StateT>
void UKF<SystemModel, MeasureModel, StateT>::computeWeights()
{
    lambda = std::pow(alpha, 2) * (N + kappa) - N;
    gamma = std::sqrt(static_cast<Type>(N) + lambda);

    assert(std::abs(static_cast<Type>(N) + lambda) > 1e-6); // Make sure L != -lambda to avoid division by zero
    assert(std::abs(static_cast<Type>(N) + kappa) > 1e-6); // Make sure L != -kappa to avoid division by zero

    Type weightM_0 = lambda / (static_cast<Type>(N) + lambda);
    Type weightC_0 = weightM_0 + (static_cast<Type>(1) - alpha * alpha + beta);
    Type weight_i = static_cast<Type>(1) / (static_cast<Type>(2) * alpha * alpha * (N + kappa));

    assert(weight_i > static_cast<Type>(0)); // Make sure W_i > 0 to avoid square-root of negative number

    mWeightsM[0] = weightM_0;
    mWeightsC[0] = weightC_0;

    for (int i = 1; i < SigmaPointsCount; ++i)
    {
        mWeightsM[i] = weight_i;
        mWeightsC[i] = weight_i;
    }
}

template<class SystemModel, class MeasureModel, class StateT>
void UKF<SystemModel, MeasureModel, StateT>::sampleStateSigmaPoints()
{
    CovarianceSquareRoot<StateT> llt;
    llt.compute(mMatP);
    if (llt.info() != Eigen::Success)
    {
        throw std::runtime_error("Fail to compute the Square root of mMatP!");
    }

    auto matL = llt.matrixL().toDenseMatrix();

    /**
     * template leftCols<1>():
     * 这是一个Eigen库中的模板函数调用。leftCols<n>()是一个模板函数，用于获取矩阵的左侧n列的引用。这里的<1>指明了模板参数为1，
     * 意味着函数返回矩阵的最左侧的一列。使用.template语法是因为leftCols是一个依赖于模板参数的成员函数，
     * 当在一个模板类或者模板函数内部调用依赖于模板参数的成员模板函数时，需要使用.template来明确地告诉编译器，接下来的符号是一个模板。
     */
    mStateSigmaPoints.template leftCols<1>() = mVecX;

    /**
     * template block<Rows, Cols>(i, j)：.block<Rows, Cols>(i, j)是Eigen中的一个方法，用于获取矩阵的一个块（或子矩阵）。
     * 这个块以(i, j)为起点，具有Rows行和Cols列。Rows和Cols是模板参数，它们在编译时定义了块的大小。在这个特定的例子中，
     * StateT::RowsAtCompileTime被用作行和列的数量，这意味着所选块是正方形的，
     * 并且它的大小是由StateT类型在编译时定义的行数决定的。i和j指定了块的起始位置，这里是从第0行和第1列开始。
     */
    mStateSigmaPoints.template block<StateT::RowsAtCompileTime, StateT::RowsAtCompileTime>(0, 1) =
            (gamma * matL).colwise() + mVecX;

    mStateSigmaPoints.template rightCols<StateT::RowsAtCompileTime>() = -((gamma * matL).colwise() - mVecX);
}
