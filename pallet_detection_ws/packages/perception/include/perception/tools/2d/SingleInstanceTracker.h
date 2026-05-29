#pragma once
#include <optional>
// #include <opencv2/opencv.hpp>
#include "perception/tools/math/KalmanFilter.hpp"


class SingleInstanceTracker
{
    using StateT = Eigen::Matrix<float, 8, 1>; // state: [cx, cy, w, h, vx, vy, vw, vh]
    using ControlT = Eigen::Matrix<float, 0, 1>; // There's no control input
    using MeasurementT = Eigen::Matrix<float, 4, 1>; // meas : [cx, cy, w, h]

    const int MAX_LOST_FRAMES; // 最大允许漏检帧数
    static constexpr float dt = 1.0f;

    static constexpr float PROCESS_NOISE_SIGMA = 1.0f;
    static constexpr float MEASUREMENT_POS_SIGMA = 3.0f;
    static constexpr float MEASUREMENT_SIZE_SIGMA = 3.0f;

    static constexpr float IoU_THREASHOLD = 0.3f;
    static constexpr float CENTER_DISTANCE_THRESHOLD = 1.5f;
    static constexpr float AREA_RATIO_THRESHOLD = 2.0f;

public:
    explicit SingleInstanceTracker(const int max_lost_frames = 6) : MAX_LOST_FRAMES(max_lost_frames)
    {
        /* Setup Kalman filter */
        StateMatrix<StateT> matA = StateMatrix<StateT>::Identity(); // transition_matrix
        matA(0, 4) = dt;
        matA(1, 5) = dt;
        matA(2, 6) = dt;
        matA(3, 7) = dt;

        MeasureMatrix<StateT, MeasurementT> matH = MeasureMatrix<StateT, MeasurementT>::Zero(); //measurement_matrix
        matH(0, 0) = 1.0f;
        matH(1, 1) = 1.0f;
        matH(2, 2) = 1.0f;
        matH(3, 3) = 1.0f;

        CovarianceMatrix<StateT> matQ = CovarianceMatrix<StateT>::Zero(); // process noise
        constexpr float dt2 = dt * dt;
        constexpr float dt3 = dt2 * dt;
        constexpr float dt4 = dt2 * dt2;
        constexpr float sigma2 = PROCESS_NOISE_SIGMA * PROCESS_NOISE_SIGMA;
        constexpr float qPos = 0.25f * dt4 * sigma2;
        constexpr float qCross = 0.5f * dt3 * sigma2;
        constexpr float qVel = dt2 * sigma2;
        for (int i = 0; i < 4; ++i)
        {
            const int vi = i + 4;
            matQ(i, i) = qPos;
            matQ(i, vi) = qCross;
            matQ(vi, i) = qCross;
            matQ(vi, vi) = qVel;
        }

        // measurement noise
        const CovarianceMatrix<MeasurementT> matR = MeasurementT(MEASUREMENT_POS_SIGMA * MEASUREMENT_POS_SIGMA,
                                                                 MEASUREMENT_POS_SIGMA * MEASUREMENT_POS_SIGMA,
                                                                 MEASUREMENT_SIZE_SIGMA * MEASUREMENT_SIZE_SIGMA,
                                                                 MEASUREMENT_SIZE_SIGMA * MEASUREMENT_SIZE_SIGMA).asDiagonal();

        mKF = KalmanFilter<StateT, ControlT, MeasurementT>::create(matA, ControlMatrix<StateT, ControlT>::Zero(), matQ, matH, matR);
    }

    void reset(const cv::Rect2f& init_bbox)
    {
        mInitialized = true;
        mLostCount = 0;

        const auto vecZ = bboxToMeasurement(init_bbox);

        StateT vecX0 = StateT::Zero();
        vecX0(0) = vecZ(0);
        vecX0(1) = vecZ(1);
        vecX0(2) = vecZ(2);
        vecX0(3) = vecZ(3);

        mKF->setInitialX(vecX0);
        mKF->setInitialMatP(CovarianceMatrix<StateT>::Identity() * 10.f);
    }

    std::optional<std::pair<cv::Rect, cv::Mat>> update(const std::optional<cv::Rect>& det_bbox, const std::optional<cv::Mat>& mask)
    {
        if (!mInitialized)
        {
            if (det_bbox.has_value())
            {
                mLastMask = mask.value();
                mLastBbox = det_bbox.value();
                reset(det_bbox.value());
                return std::make_pair(det_bbox.value(), mLastMask);
            }
            return std::nullopt;
        }

        const ControlT vecU = ControlT::Zero();
        mKF->predict(vecU);

        const cv::Rect2f predicted_bbox = stateToBbox(mKF->getState());

        if (!det_bbox.has_value())
        {
            return handleMiss(predicted_bbox);
        }

        const cv::Rect2f& det = det_bbox.value();
        if (!passGating(predicted_bbox, det))
        {
            return handleMiss(predicted_bbox);
        }

        mLastMask = mask.value().clone();
        mLastBbox = det_bbox.value();

        const MeasurementT vecZ = bboxToMeasurement(det);
        mKF->correct(vecZ);
        mLostCount = 0;

        return std::make_pair(stateToBbox(mKF->getState()), mask.value());
    }

    [[nodiscard]] std::optional<cv::Rect> getPredictedBbox() const
    {
        if (!mInitialized)
        {
            return std::nullopt;
        }

        return static_cast<cv::Rect>(stateToBbox(mKF->getState()));
    }

    static float IoU(const cv::Rect2f& a, const cv::Rect2f& b)
    {
        const float inter_x1 = std::max(a.x, b.x);
        const float inter_y1 = std::max(a.y, b.y);
        const float inter_x2 = std::min(a.x + a.width, b.x + b.width);
        const float inter_y2 = std::min(a.y + a.height, b.y + b.height);

        const float inter_w = std::max(0.0f, inter_x2 - inter_x1);
        const float inter_h = std::max(0.0f, inter_y2 - inter_y1);
        const float inter_area = inter_w * inter_h;

        const float area_a = std::max(0.0f, a.width) * std::max(0.0f, a.height);
        const float area_b = std::max(0.0f, b.width) * std::max(0.0f, b.height);
        const float union_area = area_a + area_b - inter_area;

        if (union_area <= 0.0f)
        {
            return 0.0f;
        }

        return inter_area / union_area;
    }

private:
    bool mInitialized{false};
    int mLostCount{0};
    cv::Rect mLastBbox;
    cv::Mat mLastMask;
    KalmanFilter<StateT, ControlT, MeasurementT>::Ptr mKF;

    static cv::Rect2f stateToBbox(const StateT& vecX)
    {
        const float cx = vecX(0);
        const float cy = vecX(1);
        const float w = std::max(1.0f, vecX(2));
        const float h = std::max(1.0f, vecX(3));

        return {cx - 0.5f * w, cy - 0.5f * h, w, h};
    }

    static MeasurementT bboxToMeasurement(const cv::Rect2f& bbox)
    {
        const float cx = bbox.x + bbox.width * 0.5f;
        const float cy = bbox.y + bbox.height * 0.5f;
        return {cx, cy, std::max(1.0f, bbox.width), std::max(1.0f, bbox.height)};
    }


    static bool passGating(const cv::Rect2f& predicted_bbox, const cv::Rect2f& det_bbox)
    {
        // --------------------------------------------------
        // 1. center distance gating
        // --------------------------------------------------
        const float pred_cx = predicted_bbox.x + predicted_bbox.width * 0.5f;
        const float pred_cy = predicted_bbox.y + predicted_bbox.height * 0.5f;

        const float det_cx = det_bbox.x + det_bbox.width * 0.5f;
        const float det_cy = det_bbox.y + det_bbox.height * 0.5f;

        const float dx = pred_cx - det_cx;
        const float dy = pred_cy - det_cy;

        const float center_distance = std::sqrt(dx * dx + dy * dy);

        /* 使用 bbox 对角线做归一化，更鲁棒 */
        if (const float diag = std::hypot(predicted_bbox.width, predicted_bbox.height);
            center_distance > diag * CENTER_DISTANCE_THRESHOLD)
        {
            return false;
        }

        // --------------------------------------------------
        // 2. area ratio gating
        // --------------------------------------------------
        const float pred_area = predicted_bbox.width * predicted_bbox.height;
        const float det_area = det_bbox.width * det_bbox.height;
        if (pred_area <= 1e-3f || det_area <= 1e-3f)
        {
            return false;
        }

        if (const float area_ratio = std::max(pred_area, det_area) / std::min(pred_area, det_area);
            area_ratio > AREA_RATIO_THRESHOLD)
        {
            return false;
        }

        // --------------------------------------------------
        // 3. IoU gating
        // --------------------------------------------------
        if (IoU(predicted_bbox, det_bbox) < IoU_THREASHOLD)
        {
            return false;
        }

        return true;
    }

    std::optional<std::pair<cv::Rect, cv::Mat>> handleMiss(const cv::Rect2f& predicted_bbox)
    {
        ++mLostCount;

        if (mLostCount > MAX_LOST_FRAMES)
        {
            mInitialized = false;
            return std::nullopt;
        }

        cv::Mat predict_mask;
        cv::resize(mLastMask(mLastBbox), predict_mask, static_cast<cv::Rect>(predicted_bbox).size(), 0, 0, cv::INTER_NEAREST);
        mLastMask.setTo(0);
        predict_mask.copyTo(mLastMask(predicted_bbox));
        mLastBbox = predicted_bbox;
        return std::make_pair(predicted_bbox, mLastMask);
    }
};
