#pragma once
#include <iostream>
#include <opencv2/opencv.hpp>

class SingleObjectTracker
{
    cv::KalmanFilter mKF;
    cv::Rect last_rect;
    int lost_frames = 0;
    static constexpr int MAX_LOST_FRAMES = 5; // 最大允许漏检帧数

public:
    SingleObjectTracker()
    {
        mKF.init(4, 2, 0); // 状态转移矩阵维数: 4 [x, y, vx, vy]; 观测矩阵维数: 2 [x, y]

        // 状态转移矩阵 (F) - 假设 dt = 1
        mKF.transitionMatrix = (cv::Mat_<float>(4, 4) <<
                                1, 0, 1, 0,
                                0, 1, 0, 1,
                                0, 0, 1, 0,
                                0, 0, 0, 1);

        // 观测矩阵 (H) - 我们只能直接观测到位置 x 和 y
        mKF.measurementMatrix = (cv::Mat_<float>(2, 4)(0) = init_rect.x + init_rect.width / 2.0f;
        mKF.statePost.at<float>(1) = init_rect.y + init_rect.height / 2.0f;
        mKF.statePost.at<float>(2) = 0.0f; // 初始速度 x
        mKF.statePost.at<float>(3) = 0.0f; // 初始速度 y
    }

    // 核心更新逻辑：每一帧调用一次
    cv::Rect update(const cv::Rect& detected_rect, bool is_detected)
    {
        // 阶段 1：预测 (Predict)
        cv::Mat prediction = mKF.predict();
        float pred_cx = prediction.at<float>(0);
        float pred_cy = prediction.at<float>(1);

        if (is_detected)
        {
            // 阶段 2：更新 (Update) - 只有检测到目标时才执行
            lost_frames = 0;
            last_rect = detected_rect;

            // 获取当前帧检测到的中心点
            cv::Mat measurement = (cv::Mat_<float>(2, 1) <<
                                   detected_rect.x + detected_rect.width / 2.0f,
                                   detected_rect.y + detected_rect.height / 2.0f);

            // 修正卡尔曼滤波内部状态
            cv::Mat estimated = mKF.correct(measurement);
            float est_cx = estimated.at<float>(0);
            float est_cy = estimated.at<float>(1);

            // 使用修正后的中心点更新最终输出框
            cv::Rect updated_rect;
            updated_rect.width = detected_rect.width;
            updated_rect.height = detected_rect.height;
            updated_rect.x = est_cx - updated_rect.width / 2.0f;
            updated_rect.y = est_cy - updated_rect.height / 2.0f;
            return updated_rect;
        }
        else
        {
            // 检测丢失处理：完全依赖卡尔曼滤波的预测阶段
            lost_frames++;
            if (lost_frames > MAX_LOST_FRAMES)
            {
                std::cout << "Target lost completely." << std::endl;
                return cv::Rect(); // 返回空框，表示追踪彻底丢失
            }

            // 使用上一帧的宽高，结合预测出来的中心点 [pred_cx, pred_cy] 拼出新框
            cv::Rect predicted_rect;
            predicted_rect.width = last_rect.width;
            predicted_rect.height = last_rect.height;
            predicted_rect.x = pred_cx - predicted_rect.width / 2.0f;
            predicted_rect.y = pred_cy - predicted_rect.height / 2.0f;

            last_rect = predicted_rect; // 滚动更新
            return predicted_rect;
        }
    }
};
