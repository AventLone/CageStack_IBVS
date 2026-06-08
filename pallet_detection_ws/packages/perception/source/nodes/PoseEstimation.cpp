#include "perception/nodes/PoseEstimation.h"
#include "perception/nodes/CloudPublisher.h"
#include <cv_bridge/cv_bridge.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <tf2_eigen/tf2_eigen.hpp> // ROS 2 header
#include <execution>
#include "perception/tools/2d/filter.h"
#include "perception/tools/3d/filter.h"
#include "perception/tools/2d/feature_detect.h"
#include "perception/tools/3d/feature_detect.hpp"


void PoseEstimation::initSubscriptions()
{
    const std::string param_name = "TopicName.Perception.SemanticCloud";
    this->declare_parameter(param_name, "/perception/instance_cloud");
    const std::string semantic_cloud_topic = this->get_parameter(param_name).as_string();
    mCloudSub = create_subscription<sensor_msgs::msg::PointCloud2>(semantic_cloud_topic,
                                                                   rclcpp::SensorDataQoS(),
                                                                   [this](const sensor_msgs::msg::PointCloud2::ConstSharedPtr& msg)
                                                                       {
                                                                           auto temp = std::make_unique<InstanceCloud>();
                                                                           pcl::fromROSMsg(*msg, *temp);
                                                                           this->pushInBuffer(std::move(temp));
                                                                           this->mTriggerEvent.notify_one();
                                                                       });
    mGoalSub = create_subscription<geometry_msgs::msg::PoseStamped>(
        "/goal_pose", rclcpp::ServicesQoS(), [this](const geometry_msgs::msg::PoseStamped::ConstSharedPtr& goal_msg) -> void
            {
                RCLCPP_INFO(get_logger(), "Received a goal...");
                this->mGoalMsg = *goal_msg;
                //
                {
                    std::lock_guard<std::mutex> lock(mBufferMutex);
                    this->mHasGoal = true;
                }
                this->mTriggerEvent.notify_one();
            });
}

void PoseEstimation::initPublishers()
{
    const std::string params_prefix = "TopicName.Perception.Target";
    this->declare_parameters<std::string>(params_prefix, {
                                              {"Pose", "/perception/target/pose"},
                                              {"BBox", "/perception/visualization/cubes"}
                                          });
    std::map<std::string, std::string> target_topics;
    if (!this->get_parameters<std::string>(params_prefix, target_topics))
    {
        RCLCPP_ERROR(this->get_logger(), "Failed to get parameters, target topic names!");
    }
    mPosesPub = create_publisher<geometry_msgs::msg::PoseArray>("visualization/poses", rclcpp::SensorDataQoS());
    mVisualizationPub = create_publisher<visualization_msgs::msg::MarkerArray>(target_topics["BBox"], rclcpp::SensorDataQoS());
    mTargetPosePub = create_publisher<geometry_msgs::msg::PoseStamped>(target_topics["Pose"], rclcpp::ServicesQoS());
    mLoadPosePub = create_publisher<geometry_msgs::msg::PoseStamped>("load_pose", rclcpp::ServicesQoS());
    mSlotPosePub = create_publisher<geometry_msgs::msg::PoseStamped>("slot_pose", rclcpp::ServicesQoS());
    mRoiCloudPub = create_publisher<sensor_msgs::msg::PointCloud2>("visualization/cloud_in_roi", rclcpp::SensorDataQoS());
}

void PoseEstimation::workerLoop()
{
    while (rclcpp::ok())
    {
        // sensor_msgs::msg::PointCloud2 cloud_msg;
        InstanceCloudPtr instance_cloud;

        geometry_msgs::msg::PoseArray pose_array_msg;
        // visualization_msgs::msg::Marker delete_all_marker;
        // delete_all_marker.action = visualization_msgs::msg::Marker::DELETEALL;
        // visualization_msg.markers.push_back(delete_all_marker);
        // mVisualizationPub->publish(visualization_msg);
        // visualization_msg.markers.clear();
        //
        {
            std::unique_lock<std::mutex> lock(mBufferMutex);
            // mTriggerEvent.wait(lock, [this]() -> bool { return (mHasGoal && !mCloudBuffer.empty()) || mIsShutdown; });
            mTriggerEvent.wait(lock, [this]() -> bool { return !mCloudBuffer.empty() || mIsShutdown; });
            if (mIsShutdown)
            {
                break;
            }
            instance_cloud = std::move(mCloudBuffer.front());
            mCloudBuffer.pop();
        }

        /* Stage 1. Detect the pose of the load on the forks */
        /* Step 1. Get the pose of the forks */
        // Eigen::Isometry3f T_body2fork;
        // try
        // {
        //     // This returns the pose of 'fork' in 'body' coordinates
        //     const geometry_msgs::msg::TransformStamped tf_body2fork =
        //             mTfBuffer->lookupTransform("LOLA", "fork", tf2::TimePointZero);
        //     T_body2fork = tf2::transformToEigen(tf_body2fork).cast<float>();
        // }
        // catch (const tf2::TransformException& e)
        // {
        //     RCLCPP_ERROR(this->get_logger(), "Could not transform fork to body: %s", e.what());
        //     std::this_thread::sleep_for(std::chrono::milliseconds(50));
        //     continue;
        // }

        const auto instance_clusters = getInstanceClusters(*instance_cloud, 0);

        pose_array_msg.poses.reserve(instance_clusters.size());
        for (const auto& cluster : instance_clusters)
        {
            // Eigen::Vector3f dimensions;

            // if (!estimateDimensionsAndPose(cluster, pose, dimensions))

            // poses.push_back(std::move(pose));
            // dimensions_list.push_back(std::move(dimensions));

            // if (dimensions[1] > 0.5f && dimensions[2] > 0.1f)
            if (feature3d::measureDimensionsY(*cluster) > 0.5f)
            {
                Eigen::Vector3f pose;
                if (!this->estimatePose(cluster, pose))
                {
                    continue;
                }

                if (std::abs(pose[2]) > M_PIf / 6.0f)
                {
                    continue;
                }

                geometry_msgs::msg::Pose pose_msg;
                pose_msg.position.x = pose[0];
                pose_msg.position.y = pose[1];
                pose_msg.position.z = 0.06;
                pose_msg.orientation = toQuaternionMsg(pose[2]);
                pose_array_msg.poses.push_back(pose_msg);
            }
        }


        // Eigen::Isometry3f goal_pose{};
        // //
        // {
        //     Eigen::Isometry3d temp{};
        //     tf2::fromMsg(mGoalMsg.pose, temp);
        //     goal_pose = temp.cast<float>();
        // }
        pose_array_msg.header.frame_id = "LOLA";
        pose_array_msg.header.stamp = this->now();
        mPosesPub->publish(pose_array_msg);
        // mVisualizationPub->publish(visualization_msg);
    }
}

bool PoseEstimation::estimatePose(const RawCloud::Ptr& cloud, Eigen::Vector3f& load_pose) const
{
    if (cloud->size() < 10)
    {
        return false;
    }
    static OrthographicProjector<pcl::PointXYZ> projector(View::TOP, 0.01f);
    projector.setCloud(cloud);
    const cv::Mat projection = projector.projection();

    cv::Mat src_img;
    //
    {
        cv::Mat closed_img;
        feature2d::close(projection, closed_img);
        feature2d::open(closed_img, src_img);
    }
    if (src_img.empty() || cv::countNonZero(src_img) < 10)
    {
        return false;
    }
    std::vector<cv::Point> src_points, hull_points;
    cv::findNonZero(src_img, src_points);
    cv::convexHull(src_points, hull_points);

    cv::Mat debug_img;
    cv::cvtColor(projection, debug_img, cv::COLOR_GRAY2BGR);
    for (const auto& point : hull_points)
    {
        cv::circle(debug_img, point, 2, cv::Scalar(0, 0, 255), -1);
    }

    const auto convex_lines_result = feature2d::detectConvexHullEdge(src_img, feature2d::EdgeType::RIGHT);
    if (!convex_lines_result.has_value())
    {
        return false;
    }
    const auto& convex_lines = convex_lines_result.value();
    const auto line_with_max_length = std::max_element(convex_lines.begin(), convex_lines.end(),
                                                       [](const feature2d::Line& line1, const feature2d::Line& line2) -> bool
                                                           {
                                                               return line1.length() < line2.length();
                                                           });
    std::vector<feature2d::Line> filtered_lines;
    for (const auto line : convex_lines)
    {
        if (line.length() > line_with_max_length->length() * 0.8)
        {
            filtered_lines.push_back(line);
            cv::line(debug_img, line.p1, line.p2, cv::Scalar(255, 0, 0), 1);
        }
    }
    const auto most_right_line = std::max_element(filtered_lines.begin(), filtered_lines.end(),
                                                  [](const feature2d::Line& line1, const feature2d::Line& line2) -> bool
                                                      {
                                                          return line1.center().x < line2.center().x;
                                                      });
    cv::line(debug_img, most_right_line->p1, most_right_line->p2, cv::Scalar(255, 255, 0), 1);

    // const feature2d::Line line = feature2d::detectRectEdge(hull_points, feature2d::EdgeType::RIGHT, &debug_img);
    const feature2d::Line& line = *most_right_line;

    cv::Mat mask = cv::Mat::zeros(src_img.size(), CV_8UC1);
    try
    {
        cv::line(mask, line.p1, line.p2, cv::Scalar(255), 12);
    }
    catch (const std::runtime_error& e)
    {
        RCLCPP_ERROR(get_logger(), "%s", e.what());
    }
    cv::Mat front_edge;
    src_img.copyTo(front_edge, mask);
    RawCloud front_edge_cloud = projector.extractCloud(front_edge);
    if (front_edge_cloud.size() < 10)
    {
        RCLCPP_ERROR(get_logger(), "front_edge_cloud is too small!");
        return false;
    }
    RawCloud front_edge_inlier_cloud;
    feature3d::findInliers(front_edge_cloud, front_edge_inlier_cloud, 0.05f);

    Eigen::Vector4f centroid;
    pcl::compute3DCentroid(front_edge_inlier_cloud, centroid);
    const float yaw = feature3d::calculateLineAngle(front_edge_inlier_cloud);

    load_pose.head<2>() = centroid.head<2>();
    load_pose[2] = yaw;

    return true;
}

bool PoseEstimation::estimateDimensionsAndPose(const RawCloud::Ptr& input_cloud, Eigen::Vector3f& pose, Eigen::Vector3f& dimensions)
{
    if (input_cloud->size() < 10)
    {
        return false;
    }

    // static pcl::RadiusOutlierRemoval<pcl::PointXYZ> outlier_removal;
    // outlier_removal.setRadiusSearch(0.05);
    // outlier_removal.setMinNeighborsInRadius(3);
    // outlier_removal.setInputCloud(input_cloud);
    // const auto denoiesd_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    // outlier_removal.filter(*denoiesd_cloud);

    // if (denoiesd_cloud->size() < 10)
    // {
    //     return false;
    // }
    OrthographicProjector<pcl::PointXYZ> mProjector{View::TOP};
    mProjector.setCloud(input_cloud);
    const cv::Mat projection = mProjector.projection();
    cv::Mat closed_img, denoised_img;
    feature2d::close(projection, closed_img);
    feature2d::removeIsolatedPoints(closed_img, denoised_img);
    if (denoised_img.empty() || cv::countNonZero(denoised_img) < 6)
    {
        return false;
    }
    std::vector<cv::Point2f> rect_corners = feature2d::detectMinRect(denoised_img);

    // Front edge points
    std::sort(rect_corners.begin(), rect_corners.end(), [](const cv::Point2f& a, const cv::Point2f& b) -> bool
                  {
                      return a.x < b.x;
                  });
    const auto front_edge_point_1 = rect_corners[2];
    const auto front_edge_point_2 = rect_corners[3];
    const float dimensions_y = cv::norm(front_edge_point_1 - front_edge_point_2) * mProjector.getResolution();
    if (dimensions_y < 0.5f)
    {
        return false;
    }

    /* Calculate the pose of the target */
    const auto mid_point = 0.5f * (front_edge_point_1 + front_edge_point_2);
    const auto world_coordinate = mProjector.getCoordinate(static_cast<cv::Point>(mid_point));
    const auto yaw = std::atan2(front_edge_point_1.x - front_edge_point_2.x, front_edge_point_1.y - front_edge_point_2.y);
    pose << world_coordinate.x, world_coordinate.y, yaw;

    // Right edge points
    std::sort(rect_corners.begin(), rect_corners.end(), [](const cv::Point2f& a, const cv::Point2f& b) -> bool
                  {
                      return a.y < b.y;
                  });
    const auto right_edge_point_1 = rect_corners[0];
    const auto right_edge_point_2 = rect_corners[1];
    const float dimensions_x = cv::norm(right_edge_point_1 - right_edge_point_2) * mProjector.getResolution();

    float dimensions_z = std::numeric_limits<float>::min();
    for (const auto& point : input_cloud->points)
    {
        dimensions_z = std::max(dimensions_z, point.z);
    }

    dimensions << dimensions_x, dimensions_y, dimensions_z;

    // return Eigen::Vector3f(dimensions_x, dimensions_y, dimensions_z);
    return true;
}
