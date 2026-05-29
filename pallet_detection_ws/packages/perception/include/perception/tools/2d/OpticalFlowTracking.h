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

public:
    OpticalFlowTracking()
    {
        mCriteria = cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01);
        mMorphKernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    }

    bool init(const cv::Mat& frame, const cv::Mat& init_mask)
    {
        mPrevFrame = to_gray(frame);
        mStableMask = to_binary_mask(init_mask);

        if (mPrevFrame.empty() or mStableMask.empty())
        {
            return false;
        }

        mStableMask = postprocessMask(mStableMask);
        extractPointsFromMask(mPrevFrame, mStableMask, mPervFeaturePoints);

        return !mPervFeaturePoints.empty();
    }

    std::pair<cv::Rect, cv::Mat> update(const cv::Mat& rgb_frame, const cv::Rect* seg_bbox, const cv::Mat* seg_mask)
    {
        CV_Assert(!rgb_frame.empty());

        if (not mInitialized)
        {
            if (seg_mask == nullptr or seg_bbox == nullptr)
            {
                throw std::runtime_error("Failed to initialize with nullptr!");
            }
            mPrevBBox = *seg_bbox;
            // const auto dilated_bbox = scaleBBox(mPrevBBox, 1.2, rgb_frame.size());
            mPrevFrame = to_gray(rgb_frame);
            mStableMask = to_binary_mask(*seg_mask);
            mStableMask = postprocessMask(mStableMask);
            extractPointsFromMask(mPrevFrame, mStableMask, mPervFeaturePoints);

            mInitialized = true;

            return {*seg_bbox, *seg_mask};
        }

        // const auto dilated_bbox = scaleBBox(seg_bbox != nullptr ? *seg_bbox : mPrevBBox, 1.2, rgb_frame.size());
        cv::Mat curr_gray = to_gray(rgb_frame);

        // 如果点太少，先从当前稳定 mask 重新采样
        if (mPervFeaturePoints.size() < static_cast<size_t>(MIN_TRACKED_POINTS))
        {
            extractPointsFromMask(mPrevFrame, mStableMask, mPervFeaturePoints);
        }

        if (mPervFeaturePoints.size() < static_cast<size_t>(MIN_TRACKED_POINTS))
        {
            return {};
        }

        std::vector<cv::Point2f> next_points;
        std::vector<uchar> status;
        std::vector<float> err;

        cv::calcOpticalFlowPyrLK(mPrevFrame, curr_gray, mPervFeaturePoints, next_points, status, err,
                                 cv::Size(WIN_SIZE, WIN_SIZE), PYRAMID_LEVEL, mCriteria, 0, 1e-4);

        std::vector<cv::Point2f> prev_good;
        std::vector<cv::Point2f> curr_good;
        prev_good.reserve(mPervFeaturePoints.size());
        curr_good.reserve(mPervFeaturePoints.size());

        for (size_t i = 0; i < status.size(); ++i)
        {
            if (status[i] and err[i] < 30.0f)
            {
                prev_good.push_back(mPervFeaturePoints[i]);
                curr_good.push_back(next_points[i]);
            }
        }

        if (curr_good.size() < 6)
        {
            return {};
        }

        cv::Mat inlier_mask;
        const cv::Mat affine = cv::estimateAffinePartial2D(prev_good, curr_good, inlier_mask, cv::RANSAC, 3.0, 2000, 0.99, 10);
        if (affine.empty())
        {
            return {};
        }

        cv::Mat warped_mask;
        cv::warpAffine(mStableMask, warped_mask, affine, curr_gray.size(), cv::INTER_NEAREST, cv::BORDER_CONSTANT, cv::Scalar(0));
        warped_mask = postprocessMask(warped_mask);

        cv::Mat fused_mask;
        if (seg_mask != nullptr)
        {
            if (cv::Mat curr_seg_mask = to_binary_mask(*seg_mask); !curr_seg_mask.empty())
            {
                curr_seg_mask = postprocessMask(curr_seg_mask);

                // 这里用 SDF 融合：
                // stable_mask_ 提供历史形状先验
                // curr_seg_mask 提供当前帧观测
                // fusion_alpha_ 越大，越相信历史
                fused_mask = fuseMasksBySDF(warped_mask, curr_seg_mask, FUSION_ALPHA);
                fused_mask = postprocessMask(fused_mask);
                fused_mask = keep_largest_component(fused_mask);
            }
            else
            {
                fused_mask = warped_mask;
            }
        }
        else
        {
            fused_mask = warped_mask;
        }

        cv::Rect bbox = mask2bbox(fused_mask);
        if (bbox.area() <= 0)
        {
            return {};
        }

        // 更新内部状态，供下一帧继续用
        mPrevFrame = curr_gray;
        mStableMask = fused_mask.clone();
        extractPointsFromMask(mPrevFrame, mStableMask, mPervFeaturePoints);

        return {bbox, fused_mask};
    }

    cv::Rect mPrevBBox;

private:
    bool mInitialized{false};

    cv::Mat mPrevFrame;
    cv::Mat mStableMask;
    std::vector<cv::Point2f> mPervFeaturePoints;
    cv::TermCriteria mCriteria;
    cv::Mat mMorphKernel;

    static cv::Mat to_gray(const cv::Mat& frame)
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

    static cv::Mat to_binary_mask(const cv::Mat& mask)
    {
        cv::Mat gray;

        if (mask.empty())
        {
            return gray;
        }

        if (mask.channels() == 1)
        {
            gray = mask.clone();
        }
        else
        {
            cv::cvtColor(mask, gray, cv::COLOR_BGR2GRAY);
        }

        cv::Mat binary_mask;

        if (gray.depth() == CV_32F || gray.depth() == CV_64F)
        {
            cv::threshold(gray, binary_mask, 0.5, 255.0, cv::THRESH_BINARY);
            binary_mask.convertTo(binary_mask, CV_8U);
        }
        else
        {
            cv::threshold(gray, binary_mask, 0, 255, cv::THRESH_BINARY);
            binary_mask.convertTo(binary_mask, CV_8U);
        }

        return binary_mask;
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

    [[nodiscard]] cv::Mat postprocessMask(const cv::Mat& mask) const
    {
        if (mask.empty())
        {
            return mask;
        }

        cv::Mat out = mask.clone();
        cv::morphologyEx(out, out, cv::MORPH_OPEN, mMorphKernel);
        cv::morphologyEx(out, out, cv::MORPH_CLOSE, mMorphKernel);
        return out;
    }

    static cv::Mat keep_largest_component(const cv::Mat& mask)
    {
        if (mask.empty())
        {
            return mask;
        }

        cv::Mat labels, stats, centroids;
        const int num_labels = cv::connectedComponentsWithStats(mask, labels, stats, centroids, 8, CV_32S);

        if (num_labels <= 1)
        {
            return mask;
        }

        int best_label = -1;
        int best_area = 0;

        for (int i = 1; i < num_labels; ++i)
        {
            if (const int area = stats.at<int>(i, cv::CC_STAT_AREA); area > best_area)
            {
                best_area = area;
                best_label = i;
            }
        }

        if (best_label < 0)
        {
            return mask;
        }

        cv::Mat out = cv::Mat::zeros(mask.size(), CV_8UC1);
        out.setTo(255, labels == best_label);

        return out;
    }

    static cv::Mat mask_to_signed_distance(const cv::Mat& mask)
    {
        CV_Assert(mask.type() == CV_8UC1);

        cv::Mat mask_8u;
        mask.copyTo(mask_8u);

        cv::Mat inside_mask;
        cv::threshold(mask_8u, inside_mask, 0, 255, cv::THRESH_BINARY);

        cv::Mat outside_mask;
        cv::bitwise_not(inside_mask, outside_mask);

        cv::Mat dist_inside;
        cv::Mat dist_outside;

        cv::distanceTransform(inside_mask, dist_inside, cv::DIST_L2, 3);
        cv::distanceTransform(outside_mask, dist_outside, cv::DIST_L2, 3);

        return dist_inside - dist_outside;
    }

    static cv::Mat fuseMasksBySDF(const cv::Mat& pred_mask, const cv::Mat& seg_mask, float pred_weight)
    {
        CV_Assert(!pred_mask.empty());
        CV_Assert(!seg_mask.empty());
        CV_Assert(pred_mask.size() == seg_mask.size());

        float seg_weight = 1.0f - pred_weight;
        pred_weight = std::clamp(pred_weight, 0.0f, 1.0f);
        seg_weight = std::clamp(seg_weight, 0.0f, 1.0f);

        const cv::Mat pred_sdf = mask_to_signed_distance(pred_mask);
        const cv::Mat seg_sdf = mask_to_signed_distance(seg_mask);

        cv::Mat fused_sdf = pred_weight * pred_sdf + seg_weight * seg_sdf;

        // 稍微平滑一下边界
        cv::GaussianBlur(fused_sdf, fused_sdf, cv::Size(5, 5), 0.0);

        cv::Mat fused_mask;
        cv::threshold(fused_sdf, fused_mask, 0.0, 255.0, cv::THRESH_BINARY);
        fused_mask.convertTo(fused_mask, CV_8U);

        return fused_mask;
    }

    void extractPointsFromMask(const cv::Mat& gray, const cv::Mat& mask, std::vector<cv::Point2f>& points) const
    {
        points.clear();

        if (gray.empty() || mask.empty())
        {
            return;
        }

        cv::Mat inner_mask = mask.clone();

        // 尽量只在 mask 内部取点，避免边界点太容易漂
        // cv::erode(inner_mask, inner_mask, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)));
        cv::dilate(inner_mask, inner_mask, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)));

        std::vector<cv::Point2f> corners;
        cv::goodFeaturesToTrack(gray, corners, MAX_CORNERS, QUALITY_LEVEL, MIN_DISTANCE, inner_mask, BLOCK_SIZE);

        if (!corners.empty())
        {
            cv::cornerSubPix(gray, corners, cv::Size(5, 5), cv::Size(-1, -1), mCriteria);
            points.insert(points.end(), corners.begin(), corners.end());
        }

        // 如果纹理太少，补网格点，保证至少有点可跟
        if (points.size() < static_cast<size_t>(MIN_TRACKED_POINTS))
        {
            constexpr int step = 8;
            for (int y = step / 2; y < mask.rows; y += step)
            {
                const uchar* mask_row = inner_mask.ptr<uchar>(y);
                for (int x = step / 2; x < mask.cols; x += step)
                {
                    if (mask_row[x] > 0)
                    {
                        points.emplace_back(static_cast<float>(x), static_cast<float>(y));
                    }
                }
            }
        }

        if (points.size() > static_cast<size_t>(MAX_CORNERS))
        {
            points.resize(MAX_CORNERS);
        }
    }

    static cv::Rect scaleBBox(const cv::Rect& bbox, const double scale, const cv::Size& img_size)
    {
        // 1. Calculate the new width and height
        const int new_w = static_cast<int>(bbox.width * scale);
        const int new_h = static_cast<int>(bbox.height * scale);

        // 2. Adjust X and Y to keep the center stable
        const int new_x = bbox.x - (new_w - bbox.width) / 2;
        const int new_y = bbox.y - (new_h - bbox.height) / 2;

        // 3. Create the enlarged box
        const cv::Rect enlarged_box(new_x, new_y, new_w, new_h);

        // 4. 🛡️ Crucial Boundary Protection: Clip it to the image bounds
        return enlarged_box & cv::Rect(0, 0, img_size.width, img_size.height);
    }
};
