#pragma once
#include <opencv2/core.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/common.h>

using PointIndicesInPixels = std::vector<std::vector<std::vector<std::size_t>>>;

/**
 * @brief Orthographic Views, decide which direction you are looking from
 */
enum class View
{
    FRONT, // y-z
    LEFT, // x-z
    TOP // x-y
};

/**
 * @brief Apply dimensionality reduction to the 3D point cloud to obtain a 2D representation.
 *        Get a orthograph from a point cloud.
 */
template<class PointT>
class OrthographicProjector final
{
    using Cloud = pcl::PointCloud<PointT>;
    using CloudPtr = typename Cloud::Ptr;

public:
    explicit OrthographicProjector(const View view_direction, const float resolution = 0.005f);

    void setResolution(const float resolution) noexcept
    {
        mResolution = resolution;
        mResolutionInverse = 1.0f / resolution;
    }

    void setViewDirection(View view_direction);

    void setThicknessTarget(const float thickness) noexcept
    {
        mThickness = thickness;
        mThicknessInverse = 1.0f / thickness;
    }

    cv::Mat project(const Cloud& cloud);

    void setCloud(const CloudPtr& cloud) noexcept
    {
        mCloud = cloud;
    }

    cv::Mat projection();

    /**
     * @brief Get world coordinate from a pixel
     */
    cv::Point2f getCoordinate(const cv::Point& pixel) const
    {
        return static_cast<cv::Point2f>(pixel) * mResolution + mMinBound;
    }

    float getCoordinate0(const uint32_t pixel_x) const
    {
        return static_cast<float>(pixel_x) * mResolution + mMinBound.x;
    }

    float getCoordinate1(const uint32_t pixel_y) const
    {
        return static_cast<float>(pixel_y) * mResolution + mMinBound.y;
    }

    float getResolution() const
    {
        return mResolution;
    }

    pcl::PointCloud<PointT> extractCloud(const cv::Mat& mask);

    pcl::PointCloud<PointT> extractCloud(const std::vector<cv::Point>& pixels);

private:
    uint16_t mAxes[3];
    float mResolution, mResolutionInverse;
    float mThickness{0.01f}, mThicknessInverse{100.0f};

    cv::Point2f mMinBound;

    PointIndicesInPixels mIndicesInPixels;
    CloudPtr mCloud;
};

template<class PointT>
OrthographicProjector<PointT>::OrthographicProjector(const View view_direction, const float resolution)
    : mResolution(resolution), mResolutionInverse(1.0f / resolution)
{
    switch (view_direction)
    {
        case View::TOP:
            mAxes[0] = 0;
            mAxes[1] = 1;
            mAxes[2] = 2;
            break;
        case View::FRONT:
            mAxes[0] = 1;
            mAxes[1] = 2;
            mAxes[2] = 0;
            break;
        case View::LEFT:
            mAxes[0] = 0;
            mAxes[1] = 2;
            mAxes[2] = 1;
            break;
        default:
            break;
    }
}

template<class PointT>
void OrthographicProjector<PointT>::setViewDirection(const View view_direction)
{
    switch (view_direction)
    {
        case View::TOP:
            mAxes[0] = 0;
            mAxes[1] = 1;
            mAxes[2] = 2;
            break;
        case View::FRONT:
            mAxes[0] = 1;
            mAxes[1] = 2;
            mAxes[2] = 0;
            break;
        case View::LEFT:
            mAxes[0] = 0;
            mAxes[1] = 2;
            mAxes[2] = 1;
            break;
        default:
            break;
    }
}

template<class PointT>
cv::Mat OrthographicProjector<PointT>::project(const pcl::PointCloud<PointT>& cloud)
{
    Eigen::Vector4f min_point, max_point;
    pcl::getMinMax3D(cloud, min_point, max_point);
    const float cloud_width = max_point[mAxes[0]] - min_point[mAxes[0]];
    const float cloud_height = max_point[mAxes[1]] - min_point[mAxes[1]];
    const float width_offset = 0.1f * cloud_width;
    const float height_offset = 0.1f * cloud_height;

    Eigen::Vector3f space_bound_min = min_point.head<3>();
    Eigen::Vector3f space_bound_max = max_point.head<3>();
    space_bound_max[mAxes[0]] += width_offset;
    space_bound_min[mAxes[0]] -= width_offset;
    space_bound_max[mAxes[1]] += height_offset;
    space_bound_min[mAxes[1]] -= height_offset;
    mMinBound.x = space_bound_min[mAxes[0]];
    mMinBound.y = space_bound_min[mAxes[1]];

    /* The range of the cloud's projection on a certain plane (such as x-y plane) */
    const cv::Size2f cloud_size(space_bound_max[mAxes[0]] - space_bound_min[mAxes[0]],
                                space_bound_max[mAxes[1]] - space_bound_min[mAxes[1]]);

    const cv::Size img_size(std::ceil(cloud_size.width * mResolutionInverse), std::ceil(cloud_size.height * mResolutionInverse));

    cv::Mat image = cv::Mat::zeros(img_size, CV_8UC1);
    for (const auto& point : cloud.points)
    {
        const uint32_t pixel_col = (reinterpret_cast<const float*>(&point)[mAxes[0]] - space_bound_min[mAxes[0]]) * mResolutionInverse;
        const uint32_t pixel_row = (space_bound_max[mAxes[1]] - reinterpret_cast<const float*>(&point)[mAxes[1]]) * mResolutionInverse;

        image.ptr<uchar>(pixel_row)[pixel_col] = 255;
    }

    return image;
}

template<class PointT>
cv::Mat OrthographicProjector<PointT>::projection()
{
    Eigen::Vector4f min_point, max_point;
    pcl::getMinMax3D(*mCloud, min_point, max_point);
    const float cloud_width = max_point[mAxes[0]] - min_point[mAxes[0]];
    const float cloud_height = max_point[mAxes[1]] - min_point[mAxes[1]];
    const float width_offset = 0.1f * cloud_width;
    const float height_offset = 0.1f * cloud_height;

    Eigen::Vector3f space_bound_min = min_point.head<3>();
    Eigen::Vector3f space_bound_max = max_point.head<3>();
    space_bound_max[mAxes[0]] += width_offset;
    space_bound_min[mAxes[0]] -= width_offset;
    space_bound_max[mAxes[1]] += height_offset;
    space_bound_min[mAxes[1]] -= height_offset;
    mMinBound.x = space_bound_min[mAxes[0]];
    mMinBound.y = space_bound_min[mAxes[1]];

    /* The range of the cloud's projection on a certain plane (such as x-y plane) */
    const cv::Size2f cloud_size(space_bound_max[mAxes[0]] - space_bound_min[mAxes[0]],
                                space_bound_max[mAxes[1]] - space_bound_min[mAxes[1]]);
    const cv::Size img_size(std::ceil(cloud_size.width * mResolutionInverse), std::ceil(cloud_size.height * mResolutionInverse));

    mIndicesInPixels = std::vector<std::vector<std::vector<std::size_t>>>(
        img_size.height, std::vector<std::vector<std::size_t>>(img_size.width));

    std::size_t point_index = 0;
    cv::Mat image = cv::Mat::zeros(img_size, CV_8UC1);
    for (const auto& point : mCloud->points)
    {
        const uint32_t pixel_row = (space_bound_max[mAxes[1]] - reinterpret_cast<const float*>(&point)[mAxes[1]]) * mResolutionInverse;
        const uint32_t pixel_col = (reinterpret_cast<const float*>(&point)[mAxes[0]] - space_bound_min[mAxes[0]]) * mResolutionInverse;

        image.ptr<uchar>(pixel_row)[pixel_col] = 255;

        /* Record point indices in pixels */
        mIndicesInPixels[pixel_row][pixel_col].push_back(point_index);
        ++point_index;
    }
    return image;
}

template<class PointT>
pcl::PointCloud<PointT> OrthographicProjector<PointT>::extractCloud(const cv::Mat& mask)
{
    pcl::PointCloud<PointT> cloud;

    for (int row = 0; row < mask.rows; ++row)
    {
        const auto* mask_ptr = mask.ptr<uchar>(row);
        for (int col = 0; col < mask.cols; ++col)
        {
            if (mask_ptr[col] > 0)
            {
                for (std::size_t idx : mIndicesInPixels[row][col])
                {
                    cloud.push_back(mCloud->points[idx]);
                }
            }
        }
    }
    return cloud;
}

template<class PointT>
pcl::PointCloud<PointT> OrthographicProjector<PointT>::extractCloud(const std::vector<cv::Point>& pixels)
{
    pcl::PointCloud<PointT> cloud;

    for (const auto& pixel : pixels)
    {
        for (std::size_t idx : mIndicesInPixels[pixel.y][pixel.x])
        {
            cloud.push_back(mCloud->points[idx]);
        }
    }

    return cloud;
}
