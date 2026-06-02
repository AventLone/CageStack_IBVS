#pragma once
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <optional>
#include <vector>

/* instance tracking based on optical flow */
class OpticalFlowTracking
{
    static constexpr int MAX_CORNERS = 250;
    static constexpr double QUALITY_LEVEL = 0.01;
    static constexpr double MIN_DISTANCE = 8.0;
    static constexpr int BLOCK_SIZE = 3;
    static constexpr int WIN_SIZE = 21;
    static constexpr int PYRAMID_LEVEL = 3;
    static constexpr int MIN_TRACKED_POINTS = 12;
    static constexpr float FUSION_ALPHA = 0.65f;

    const uint16_t MAX_LOST_FRAMES;

public:
    explicit OpticalFlowTracking(const uint16_t max_lost_frames = 360) : MAX_LOST_FRAMES(max_lost_frames)
    {
        mCriteria = cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01);
    }

    std::optional<std::pair<cv::Rect, cv::Mat>> update(const cv::Mat& rgb_frame, const cv::Rect* seg_bbox, const cv::Mat* seg_mask,
                                                       cv::Mat* flow_vis = nullptr)
    {
        CV_Assert(!rgb_frame.empty());

        if (not mInitialized)
        {
            if (seg_mask == nullptr or seg_bbox == nullptr)
            {
                return std::nullopt;
            }
            updateStates(toGray(rgb_frame), *seg_bbox, *seg_mask);
            mInitialized = true;
            if (flow_vis != nullptr)
            {
                cv::rectangle(*flow_vis, *seg_bbox, cv::Scalar(0, 255, 0), 2);
                cv::putText(*flow_vis, "Initialized by segmentation", cv::Point(20, 30), cv::FONT_HERSHEY_SIMPLEX,
                            0.8, cv::Scalar(0, 255, 0), 2);
            }
            return std::make_pair(*seg_bbox, *seg_mask);
        }

        if (seg_mask != nullptr and seg_bbox != nullptr)
        {
            mLostFrames = 0;
            updateStates(toGray(rgb_frame), *seg_bbox, *seg_mask);
            return std::make_pair(*seg_bbox, *seg_mask);
        }

        if (++mLostFrames > MAX_LOST_FRAMES)
        {
            return std::nullopt;
        }

        if (mLastFeaturePoints.size() < static_cast<size_t>(MIN_TRACKED_POINTS))
        {
            return std::nullopt;
        }
        cv::Mat curr_gray = toGray(rgb_frame);
        std::vector<cv::Point2f> next_points;
        std::vector<uchar> status;
        std::vector<float> err;
        cv::calcOpticalFlowPyrLK(mLastFrame, curr_gray, mLastFeaturePoints, next_points, status, err,
                                 cv::Size(WIN_SIZE, WIN_SIZE), PYRAMID_LEVEL, mCriteria, 0, 1e-4);

        std::vector<cv::Point2f> prev_good;
        std::vector<cv::Point2f> curr_good;
        prev_good.reserve(mLastFeaturePoints.size());
        curr_good.reserve(mLastFeaturePoints.size());

        for (size_t i = 0; i < status.size(); ++i)
        {
            if (status[i] > 0 and err[i] < 30.0f)
            {
                prev_good.push_back(mLastFeaturePoints[i]);
                curr_good.push_back(next_points[i]);
            }
        }

        if (curr_good.size() < 6)
        {
            return std::nullopt;
        }

        cv::Mat inlier_mask;
        const cv::Mat affine = cv::estimateAffinePartial2D(prev_good, curr_good, inlier_mask, cv::RANSAC, 3.0, 2000, 0.99, 10);
        if (affine.empty())
        {
            return std::nullopt;
        }

        cv::Mat warped_mask;
        cv::warpAffine(mLastMask, warped_mask, affine, curr_gray.size(), cv::INTER_NEAREST, cv::BORDER_CONSTANT, cv::Scalar(0));

        cv::Rect bbox = mask2bbox(warped_mask);
        if (bbox.area() <= 0)
        {
            return std::nullopt;
        }

        // 光流可视化
        if (flow_vis != nullptr)
        {
            *flow_vis = rgb_frame.clone();

            for (size_t i = 0; i < prev_good.size(); ++i)
            {
                const cv::Point2f& p0 = prev_good[i];
                const cv::Point2f& p1 = curr_good[i];

                bool is_inlier = true;

                if (!inlier_mask.empty())
                {
                    is_inlier = inlier_mask.at<uchar>(static_cast<int>(i), 0) > 0;
                }

                if (is_inlier)
                {
                    // RANSAC inlier: green arrow
                    cv::arrowedLine(*flow_vis, p0, p1, cv::Scalar(0, 255, 0), 1, cv::LINE_AA, 0, 0.1);
                    cv::circle(*flow_vis, p1, 2, cv::Scalar(0, 0, 255), -1);
                }
                else
                {
                    // RANSAC outlier: gray arrow
                    cv::arrowedLine(*flow_vis, p0, p1, cv::Scalar(128, 128, 128), 1, cv::LINE_AA, 0, 0.1);
                }
            }

            // tracked bbox
            cv::rectangle(*flow_vis, bbox, cv::Scalar(255, 0, 0), 2);

            // previous bbox
            cv::rectangle(*flow_vis, mLastBBox, cv::Scalar(0, 255, 255), 1);
            std::string text = "Optical flow tracking | good: " + std::to_string(curr_good.size()) +
                               " | lost: " + std::to_string(mLostFrames);
            cv::putText(*flow_vis, text, cv::Point(20, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
        }

        // if (seg_mask != nullptr and seg_bbox != nullptr)
        // {
        //     mLostFrames = 0;
        //     updateStates(toGray(rgb_frame), *seg_bbox, *seg_mask);
        //     return std::make_pair(*seg_bbox, *seg_mask);
        // }

        updateStates(curr_gray, bbox, warped_mask); // 更新内部状态，供下一帧继续用
        return std::make_pair(bbox, warped_mask);
    }

    [[nodiscard]] const cv::Rect& getLastBBox() const
    {
        return mLastBBox;
    }

private:
    bool mInitialized{false};
    uint16_t mLostFrames{0};
    cv::Rect mLastBBox;
    cv::Mat mLastFrame, mLastMask;
    std::vector<cv::Point2f> mLastFeaturePoints;
    cv::TermCriteria mCriteria;

    void updateStates(const cv::Mat& frame, const cv::Rect& bbox, const cv::Mat& mask)
    {
        extractPointsFromBBox(frame, bbox, mLastFeaturePoints, 1.2);
        mLastFrame = frame;
        mLastMask = mask;
        mLastBBox = bbox;
    }

    static cv::Mat toGray(const cv::Mat& frame)
    {
        cv::Mat gray;

        if (frame.empty())
        {
            return gray;
        }

        if (frame.channels() == 1)
        {
            gray = frame.clone();
        }
        else
        {
            cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        }

        return gray;
    }

    static cv::Rect mask2bbox(const cv::Mat& mask)
    {
        std::vector<cv::Point> points;
        cv::findNonZero(mask, points);

        if (points.empty())
        {
            return {};
        }

        return cv::boundingRect(points);
    }

    void extractPointsFromBBox(const cv::Mat& gray, const cv::Rect& bbox, std::vector<cv::Point2f>& points,
                               const double scale_bbox = 1.1) const
    {
        points.clear();

        if (gray.empty())
        {
            return;
        }

        const cv::Rect valid_bbox = scaleBBox(bbox, scale_bbox) & cv::Rect(0, 0, gray.cols, gray.rows); // 防止 bbox 越界

        if (valid_bbox.empty())
        {
            return;
        }

        const cv::Mat gray_roi = gray(valid_bbox); // 只在 bbox 区域内提取角点
        std::vector<cv::Point2f> corners_roi;
        cv::goodFeaturesToTrack(gray(valid_bbox), corners_roi, MAX_CORNERS, QUALITY_LEVEL, MIN_DISTANCE);
        if (corners_roi.empty())
        {
            return;
        }

        if (const cv::Size subpix_win(3, 3); gray_roi.cols >= subpix_win.width * 2 + 5 and gray_roi.rows >= subpix_win.height * 2 + 5)
        {
            cv::cornerSubPix(gray_roi, corners_roi, subpix_win, cv::Size(-1, -1), mCriteria);
        }

        points.reserve(corners_roi.size());
        // ROI 坐标转成原图坐标
        for (const auto& p : corners_roi)
        {
            points.emplace_back(p.x + static_cast<float>(valid_bbox.x), p.y + static_cast<float>(valid_bbox.y));
        }
    }

    static cv::Rect scaleBBox(const cv::Rect& bbox, const double scale)
    {
        // 1. Calculate the new width and height
        const int new_w = static_cast<int>(bbox.width * scale);
        const int new_h = static_cast<int>(bbox.height * scale);

        // 2. Adjust X and Y to keep the center stable
        const int new_x = bbox.x - (new_w - bbox.width) / 2;
        const int new_y = bbox.y - (new_h - bbox.height) / 2;

        return {new_x, new_y, new_w, new_h};
    }
};
