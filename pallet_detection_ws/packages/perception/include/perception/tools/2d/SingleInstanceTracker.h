#pragma once
#include <iostream>
#include <opencv2/opencv.hpp>


class SingleInstanceTracker
{
    cv::KalmanFilter mKF;
    cv::Rect last_rect;
    int lost_frames = 0;
    static constexpr int MAX_LOST_FRAMES = 6; // 最大允许漏检帧数

public:
};
