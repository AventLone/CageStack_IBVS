#pragma once
#include <pcl/filters/radius_outlier_removal.h>
#include "./OrthographicProjector.hpp"
// #include "./filter_3d.h"
#include "./feature_detect_2d.h"

template<class PointT>
class PoseEstimate
{
    using CloudPtr = typename pcl::PointCloud<PointT>::Ptr;

public:
    PoseEstimate()
    {
        mOutlierRemover.setRadiusSearch(0.05);
        mOutlierRemover.setMinNeighborsInRadius(3);
        // mOutlierRemover.setSearchMethod();
    }

    /**
     *
     * @param cloud
     * @param pose
     * @return The dimensions of the target
     */
    Eigen::Vector3f compute(const CloudPtr cloud, Eigen::Vector3f& pose)
    {
        mOutlierRemover.setInputCloud(cloud);
        const auto denoiesd_cloud = std::make_shared<pcl::PointCloud<PointT>>();
        mOutlierRemover.filter(*denoiesd_cloud);

        mProjector.setCloud(denoiesd_cloud);
        const cv::Mat projection = mProjector.projection();
        const std::vector<cv::Point2f> rect_corners = feature2d::detectMinRect(projection);

        // Front edge points
        std::sort(rect_corners.begin(), rect_corners.end(), [](const cv::Point2f& a, const cv::Point2f& b) -> bool
                      {
                          return a.x < b.x;
                      });
        const auto front_edge_point_1 = rect_corners[2];
        const auto front_edge_point_2 = rect_corners[3];
        const float dimensions_y = cv::norm(front_edge_point_1 - front_edge_point_2) * mProjector.getResolution();

        /* Calculate the pose of the target */
        const auto mid_point = 0.5f * (front_edge_point_1 + front_edge_point_2);
        const auto world_coordinate = mProjector.getCoordinate(mid_point);
        const auto yaw = std::atan2(front_edge_point_1.x - front_edge_point_2.x, front_edge_point_1.y - front_edge_point_2.y);
        pose << world_coordinate.x << world_coordinate.y << yaw;
        
        // Right edge points
        std::sort(rect_corners.begin(), rect_corners.end(), [](const cv::Point2f& a, const cv::Point2f& b) -> bool
                      {
                          return a.y < b.y;
                      });
        const auto right_edge_point_1 = rect_corners[0];
        const auto right_edge_point_2 = rect_corners[1];
        const float dimensions_x = cv::norm(right_edge_point_1 - right_edge_point_2) * mProjector.getResolution();

        float dimensions_z = std::numeric_limits<float>::min();
        for (const auto& point : cloud->points)
        {
            dimensions_z = std::max(dimensions_z, point.z);
        }

        // return Eigen::Vector3f(dimensions_x, dimensions_y, dimensions_z);
        return {dimensions_x, dimensions_y, dimensions_z};
    }

private:
    CloudPtr mCloud;
    pcl::RadiusOutlierRemoval<PointT> mOutlierRemover;
    OrthographicProjector<PointT> mProjector;
};
