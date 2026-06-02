#pragma once
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <optional>
#include <vector>

/* instance tracking based on ORB feature matching + keyframe */
class FeatureMatchingTracking
{
    static constexpr int ORB_MAX_FEATURES = 600;
    static constexpr float ORB_SCALE_FACTOR = 1.2f;
    static constexpr int ORB_N_LEVELS = 8;

    static constexpr int MIN_MATCHES = 16;
    static constexpr int MIN_INLIERS = 8;

    static constexpr float RATIO_TEST = 0.75f;
    static constexpr int MAX_GOOD_MATCHES = 120;

    static constexpr double RANSAC_REPROJ_THRESHOLD = 3.0;
    static constexpr int RANSAC_MAX_ITERS = 2000;
    static constexpr double RANSAC_CONFIDENCE = 0.99;

    static constexpr double KEYFRAME_BBOX_SCALE = 1.25;
    static constexpr double SEARCH_BBOX_BASE_SCALE = 1.8;
    static constexpr double SEARCH_BBOX_LOST_GROWTH = 0.04;
    static constexpr double SEARCH_BBOX_MAX_SCALE = 1.3;

    const uint16_t MAX_LOST_FRAMES;

public:
    explicit FeatureMatchingTracking(const uint16_t max_lost_frames = 360)
        : MAX_LOST_FRAMES(max_lost_frames)
    {
        mOrb = cv::ORB::create(ORB_MAX_FEATURES, ORB_SCALE_FACTOR, ORB_N_LEVELS, 20, 0, 2, cv::ORB::HARRIS_SCORE, 15, 5);
    }

    std::optional<std::pair<cv::Rect, cv::Mat>> update(const cv::Mat& rgb_frame, const cv::Rect* seg_bbox,
                                                       const cv::Mat* seg_mask, cv::Mat* match_vis = nullptr)
    {
        CV_Assert(!rgb_frame.empty());

        if (match_vis != nullptr)
        {
            *match_vis = rgb_frame.clone();
        }

        const cv::Mat curr_gray = toGray(rgb_frame);

        // 1. 初始化：必须依赖 segmentation
        if (!mInitialized)
        {
            if (seg_bbox == nullptr || seg_mask == nullptr)
            {
                return std::nullopt;
            }

            updateKeyFrame(curr_gray, *seg_bbox, *seg_mask);
            mLastBBox = *seg_bbox;
            mLastMask = seg_mask->clone();
            mLostFrames = 0;
            mInitialized = true;

            if (match_vis != nullptr)
            {
                cv::rectangle(*match_vis, *seg_bbox, cv::Scalar(0, 255, 0), 2);
                cv::putText(*match_vis, "Initialized by segmentation",
                            cv::Point(20, 30), cv::FONT_HERSHEY_SIMPLEX,
                            0.8, cv::Scalar(0, 255, 0), 2);
            }

            return std::make_pair(*seg_bbox, seg_mask->clone());
        }

        // 2. 有 segmentation：认为是强观测，更新 keyframe
        if (seg_bbox != nullptr && seg_mask != nullptr)
        {
            updateKeyFrame(curr_gray, *seg_bbox, *seg_mask);
            mLastBBox = *seg_bbox;
            mLastMask = seg_mask->clone();
            mLostFrames = 0;

            if (match_vis != nullptr)
            {
                cv::rectangle(*match_vis, *seg_bbox, cv::Scalar(0, 255, 0), 2);
                cv::putText(*match_vis, "Updated by segmentation / keyframe refreshed",
                            cv::Point(20, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
            }

            return std::make_pair(*seg_bbox, seg_mask->clone());
        }

        // 3. 没有 segmentation：用当前帧匹配 keyframe
        if (++mLostFrames > MAX_LOST_FRAMES)
        {
            return std::nullopt;
        }

        if (mKeyDescriptors.empty() || mKeyKeypoints.size() < MIN_MATCHES)
        {
            return std::nullopt;
        }

        const double search_scale = std::min(SEARCH_BBOX_MAX_SCALE,
                                             SEARCH_BBOX_BASE_SCALE + SEARCH_BBOX_LOST_GROWTH * static_cast<double>(mLostFrames));

        std::vector<cv::KeyPoint> curr_keypoints;
        cv::Mat curr_descriptors;

        extractORBFromBBox(curr_gray, mLastBBox, curr_keypoints, curr_descriptors, search_scale);

        if (curr_descriptors.empty() || curr_keypoints.size() < MIN_MATCHES)
        {
            return std::nullopt;
        }

        std::vector<cv::DMatch> good_matches = matchDescriptors(mKeyDescriptors, curr_descriptors);

        if (good_matches.size() < MIN_MATCHES)
        {
            return std::nullopt;
        }

        std::vector<cv::Point2f> key_pts;
        std::vector<cv::Point2f> curr_pts;

        key_pts.reserve(good_matches.size());
        curr_pts.reserve(good_matches.size());

        for (const auto& m : good_matches)
        {
            key_pts.push_back(mKeyKeypoints[m.queryIdx].pt);
            curr_pts.push_back(curr_keypoints[m.trainIdx].pt);
        }

        cv::Mat inlier_mask;

        const cv::Mat affine = cv::estimateAffinePartial2D(key_pts, curr_pts, inlier_mask, cv::RANSAC,
                                                           RANSAC_REPROJ_THRESHOLD,
                                                           RANSAC_MAX_ITERS, RANSAC_CONFIDENCE, 10);
        if (affine.empty())
        {
            return std::nullopt;
        }
        const int inlier_count = cv::countNonZero(inlier_mask);

        if (inlier_count < MIN_INLIERS)
        {
            return std::nullopt;
        }

        cv::Mat warped_mask;

        cv::warpAffine(mKeyMask, warped_mask, affine, curr_gray.size(), cv::INTER_NEAREST, cv::BORDER_CONSTANT, cv::Scalar(0));
        cv::Rect bbox = mask2bbox(warped_mask);

        if (bbox.area() <= 0)
        {
            return std::nullopt;
        }

        if (!isAffineReasonable(affine, mKeyBBox, bbox, curr_gray.size()))
        {
            return std::nullopt;
        }

        if (match_vis != nullptr)
        {
            drawMatchVisualization(*match_vis, mKeyBBox, bbox, key_pts, curr_pts, inlier_mask, good_matches.size(), inlier_count);
        }

        // 注意：这里不更新 keyframe，只更新 last bbox/mask
        // 这样 search region 会跟着走，但不会把漂移结果变成新的真值。
        mLastBBox = bbox;
        mLastMask = warped_mask.clone();

        return std::make_pair(bbox, warped_mask);
    }

    [[nodiscard]] const cv::Rect& getLastBBox() const
    {
        return mLastBBox;
    }

private:
    bool mInitialized{false};
    uint16_t mLostFrames{0};

    cv::Ptr<cv::ORB> mOrb;

    cv::Rect mLastBBox;
    cv::Mat mLastMask;

    cv::Mat mKeyFrame;
    cv::Mat mKeyMask;
    cv::Rect mKeyBBox;
    std::vector<cv::KeyPoint> mKeyKeypoints;
    cv::Mat mKeyDescriptors;

    void updateKeyFrame(const cv::Mat& gray, const cv::Rect& bbox, const cv::Mat& mask)
    {
        mKeyFrame = gray.clone();
        mKeyMask = mask.clone();
        mKeyBBox = bbox;

        extractORBFromBBox(mKeyFrame, mKeyBBox, mKeyKeypoints, mKeyDescriptors, KEYFRAME_BBOX_SCALE);
    }

    void extractORBFromBBox(const cv::Mat& gray, const cv::Rect& bbox,
                            std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors,
                            const double scale_bbox) const
    {
        keypoints.clear();
        descriptors.release();

        if (gray.empty() || bbox.area() <= 0)
        {
            return;
        }

        const cv::Rect image_rect(0, 0, gray.cols, gray.rows);
        const cv::Rect valid_bbox = scaleBBox(bbox, scale_bbox) & image_rect;

        if (valid_bbox.empty())
        {
            return;
        }

        const cv::Mat gray_roi = gray(valid_bbox);

        std::vector<cv::KeyPoint> roi_keypoints;
        cv::Mat roi_descriptors;
        mOrb->detectAndCompute(gray_roi, cv::noArray(), roi_keypoints, roi_descriptors);

        if (roi_keypoints.empty() || roi_descriptors.empty())
        {
            return;
        }

        keypoints.reserve(roi_keypoints.size());

        for (auto kp : roi_keypoints)
        {
            kp.pt.x += static_cast<float>(valid_bbox.x);
            kp.pt.y += static_cast<float>(valid_bbox.y);
            keypoints.push_back(kp);
        }

        descriptors = roi_descriptors.clone();
    }

    static std::vector<cv::DMatch> matchDescriptors(const cv::Mat& query_desc, const cv::Mat& train_desc)
    {
        std::vector<cv::DMatch> good_matches;

        if (query_desc.empty() || train_desc.empty())
        {
            return good_matches;
        }

        if (query_desc.rows < 2 || train_desc.rows < 2)
        {
            return good_matches;
        }

        const cv::BFMatcher matcher(cv::NORM_HAMMING, false);

        std::vector<std::vector<cv::DMatch>> knn_matches;
        matcher.knnMatch(query_desc, train_desc, knn_matches, 2);
        for (const auto& knn : knn_matches)
        {
            if (knn.size() < 2)
            {
                continue;
            }

            if (knn[0].distance < RATIO_TEST * knn[1].distance)
            {
                good_matches.push_back(knn[0]);
            }
        }

        std::sort(good_matches.begin(), good_matches.end(), [](const cv::DMatch& a, const cv::DMatch& b)
                      {
                          return a.distance < b.distance;
                      });

        if (good_matches.size() > MAX_GOOD_MATCHES)
        {
            good_matches.resize(MAX_GOOD_MATCHES);
        }

        return good_matches;
    }

    static bool isAffineReasonable(const cv::Mat& affine, const cv::Rect& key_bbox,
                                   const cv::Rect& curr_bbox, const cv::Size& image_size)
    {
        if (affine.empty() || affine.rows != 2 || affine.cols != 3)
        {
            return false;
        }

        if (key_bbox.area() <= 0 || curr_bbox.area() <= 0)
        {
            return false;
        }

        if ((curr_bbox & cv::Rect(0, 0, image_size.width, image_size.height)) != curr_bbox)
        {
            return false;
        }

        const double a = affine.at<double>(0, 0);
        const double b = affine.at<double>(0, 1);
        const double c = affine.at<double>(1, 0);
        const double d = affine.at<double>(1, 1);

        const double scale_x = std::sqrt(a * a + c * c);
        const double scale_y = std::sqrt(b * b + d * d);

        if (scale_x < 0.5 || scale_x > 2.0)
        {
            return false;
        }

        if (scale_y < 0.5 || scale_y > 2.0)
        {
            return false;
        }

        if (const double area_ratio = static_cast<double>(curr_bbox.area()) / static_cast<double>(key_bbox.area());
            area_ratio < 0.35 || area_ratio > 3.0)
        {
            return false;
        }

        return true;
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
            // 如果你的输入真的是 RGB，不是 OpenCV 默认 BGR，
            // 这里改成 cv::COLOR_RGB2GRAY。
            cv::cvtColor(frame, gray, cv::COLOR_RGB2GRAY);
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

    static cv::Rect scaleBBox(const cv::Rect& bbox, const double scale)
    {
        const int new_w = static_cast<int>(bbox.width * scale);
        const int new_h = static_cast<int>(bbox.height * scale);

        const int new_x = bbox.x - (new_w - bbox.width) / 2;
        const int new_y = bbox.y - (new_h - bbox.height) / 2;

        return {new_x, new_y, new_w, new_h};
    }

    static void drawMatchVisualization(cv::Mat& vis, const cv::Rect& key_bbox, const cv::Rect& curr_bbox,
                                       const std::vector<cv::Point2f>& key_pts, const std::vector<cv::Point2f>& curr_pts,
                                       const cv::Mat& inlier_mask, const size_t match_count, const int inlier_count)
    {
        for (size_t i = 0; i < key_pts.size(); ++i)
        {
            bool is_inlier = true;

            if (!inlier_mask.empty())
            {
                is_inlier = inlier_mask.at<uchar>(static_cast<int>(i), 0) > 0;
            }

            if (is_inlier)
            {
                cv::arrowedLine(vis, key_pts[i], curr_pts[i], cv::Scalar(0, 255, 0), 1, cv::LINE_AA, 0, 0.12);
                cv::circle(vis, curr_pts[i], 2, cv::Scalar(0, 0, 255), -1);
            }
            // else
            // {
            //     cv::arrowedLine(vis, key_pts[i], curr_pts[i], cv::Scalar(128, 128, 128), 1, cv::LINE_AA, 0, 0.08);
            //     cv::circle(vis, curr_pts[i], 2, cv::Scalar(0, 255, 0), -1);
            // }
        }

        // 当前匹配推出来的 bbox
        cv::rectangle(vis, curr_bbox, cv::Scalar(255, 0, 0), 2);

        // keyframe bbox，只是参考位置，不代表当前目标位置
        cv::rectangle(vis, key_bbox, cv::Scalar(0, 255, 255), 1);

        const std::string text = "ORB keyframe tracking | matches: " + std::to_string(match_count) +
                                 " | inliers: " + std::to_string(inlier_count);

        cv::putText(vis, text, cv::Point(20, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
    }
};
