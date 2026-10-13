#include "free_space_detection/nodes/FreeSpaceDetect.h"
#include "free_space_detection/nodes/CloudPublisher.h"
#include <cv_bridge/cv_bridge.hpp>
#include <memory>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <Eigen/Geometry>
#include <pcl_conversions/pcl_conversions.h>
#include <tf2_eigen/tf2_eigen.hpp> // ROS 2 header
#include <execution>
#include <random>
#include "free_space_detection/tools/filter_2d.h"
#include "free_space_detection/tools/filter_3d.h"
#include "free_space_detection/tools/feature_detect_2d.h"
#include "free_space_detection/tools/feature_detect_3d.h"


void FreeSpaceDetect::initSubscriptions()
{
    const std::string param_name = "TopicName.Perception.SemanticCloud";
    this->declare_parameter(param_name, "/perception/semantic_cloud");
    const std::string semantic_cloud_topic = this->get_parameter(param_name).as_string();
    mGoalSub = create_subscription<geometry_msgs::msg::PoseStamped>("/goal_pose", rclcpp::ServicesQoS(),
        [this](const geometry_msgs::msg::PoseStamped::ConstSharedPtr& goal_msg) -> void
            {
                RCLCPP_INFO(get_logger(), "Received a goal...");
                this->mGoalMsg = *goal_msg;
                {
                    std::lock_guard lock(mBufferMutex);
                    this->mHasGoal = true;
                }
                reset();
                this->mTriggerEvent.notify_one();
            });

    mCloudSub = create_subscription<sensor_msgs::msg::PointCloud2>("/cloud_map", rclcpp::SensorDataQoS(),
        [this](sensor_msgs::msg::PointCloud2::ConstSharedPtr msg)
            {
                std::lock_guard lock(mBufferMutex);
                while (!mCloudBuffer.empty())
                {
                    mCloudBuffer.pop();
                }
                mCloudBuffer.push(std::move(msg));
                this->mTriggerEvent.notify_one();
            });
}

void FreeSpaceDetect::initPublishers()
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
    mVisualizationPub = create_publisher<visualization_msgs::msg::MarkerArray>(target_topics["BBox"], rclcpp::SensorDataQoS());
    mSlotPosePub = create_publisher<geometry_msgs::msg::PoseStamped>("slot_pose", rclcpp::ServicesQoS());
    mRoiCloudPub = create_publisher<sensor_msgs::msg::PointCloud2>("visualization/cloud_in_roi", rclcpp::SensorDataQoS());
}

void FreeSpaceDetect::workerLoop()
{
    constexpr std::string_view ns = "visualization";
    constexpr std::string_view FRAME_ID = "map";
    constexpr float load_size_x = 1.2f;
    constexpr float load_size_y = 1.0f;
    constexpr float load_size_z = 1.5f;

    // rclcpp::Rate publish_rate(5.0, this->get_clock());
    constexpr auto worker_period = std::chrono::milliseconds(500);
    while (rclcpp::ok())
    {
        sensor_msgs::msg::PointCloud2::ConstSharedPtr cloud_msg;
        //
        {
            std::unique_lock lock(mBufferMutex);
            mTriggerEvent.wait(lock, [this]() -> bool
                                   {
                                       return (mHasGoal && !mCloudBuffer.empty()) || mIsShutdown;
                                   });
            if (mIsShutdown)
            {
                break;
            }
            cloud_msg = std::move(mCloudBuffer.front());
            mCloudBuffer.pop();
        }

        visualization_msgs::msg::MarkerArray visualization_msg;

        auto cloud = std::make_shared<RawCloud>();
        pcl::fromROSMsg(*cloud_msg, *cloud);

        constexpr ROI trailer_roi{-20.0f, 3.0f, -3.0f, 3.0f, -1.0f, 5.0f};

        /* Visualize slot_space_roi */
        tf2::Quaternion tf_q;
        tf2::fromMsg(mGoalMsg.pose.orientation, tf_q); // Convert ROS msg → tf2 quaternion

        // 3. Convert quaternion to roll/pitch/yaw (Euler angles)
        double roll, pitch, yaw;
        tf2::Matrix3x3(tf_q).getRPY(roll, pitch, yaw); // Order: X(roll), Y(pitch), Z(yaw)
        Eigen::Isometry2f T_1(Eigen::Isometry2f::Identity()), T_2(Eigen::Isometry2f::Identity());
        T_1.rotate(yaw);
        T_1.pretranslate(Eigen::Vector2f(mGoalMsg.pose.position.x, mGoalMsg.pose.position.y));
        T_2.translate(Eigen::Vector2f(-0.5f * 17.0f, 0.0f));
        Eigen::Isometry2f T_3 = T_1 * T_2;
        visualization_msgs::msg::Marker trailer_roi_marker;
        trailer_roi_marker.header.frame_id = FRAME_ID;
        trailer_roi_marker.header.stamp = this->now();
        trailer_roi_marker.ns = ns;
        trailer_roi_marker.id = 1;
        trailer_roi_marker.type = visualization_msgs::msg::Marker::CUBE;
        trailer_roi_marker.action = visualization_msgs::msg::Marker::ADD;
        trailer_roi_marker.scale.x = static_cast<double>(trailer_roi.max_x - trailer_roi.min_x);
        trailer_roi_marker.scale.y = static_cast<double>(trailer_roi.max_y - trailer_roi.min_y);
        trailer_roi_marker.scale.z = static_cast<double>(trailer_roi.max_z - trailer_roi.min_z);
        trailer_roi_marker.color.r = 0.8;
        trailer_roi_marker.color.g = 1.0;
        trailer_roi_marker.color.b = 0.8;
        trailer_roi_marker.color.a = 0.2;
        trailer_roi_marker.pose.orientation = mGoalMsg.pose.orientation;
        trailer_roi_marker.pose.position.x = T_3.translation()[0];
        trailer_roi_marker.pose.position.y = T_3.translation()[1];
        trailer_roi_marker.pose.position.z = 0.5f * (trailer_roi.min_z + trailer_roi.max_z);
        visualization_msg.markers.push_back(trailer_roi_marker);

        Eigen::Isometry3f goal_pose;
        {
            Eigen::Isometry3d temp{};
            tf2::fromMsg(mGoalMsg.pose, temp);
            goal_pose = temp.cast<float>();
        }

        /*------ Stage 2. Slot pose estimate ------*/
        auto rough_trailer_cloud = std::make_shared<RawCloud>();
        filter3d::getCloud(*cloud, rough_trailer_cloud, nullptr, goal_pose, trailer_roi);

        if (rough_trailer_cloud->size() < 10)
        {
            RCLCPP_ERROR(get_logger(), "slot_space_cloud has too few points!");
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            continue;
        }

        const auto refined_trailer_cloud = std::make_shared<RawCloud>();
        Eigen::Vector3f trailer_pose;
        if (!locateTrailer(rough_trailer_cloud, refined_trailer_cloud, trailer_pose))
        {
            RCLCPP_ERROR(get_logger(), "Failed to locate the trailer or the ramp!");
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }

        const ROI trailer_bbox = ROI::getBBox(*refined_trailer_cloud);
        const auto trailer_dimensions = Eigen::Vector3f(trailer_bbox.max_x - trailer_bbox.min_x,
                                                        trailer_bbox.max_y - trailer_bbox.min_y,
                                                        trailer_bbox.max_z - trailer_bbox.min_z);

        visualization_msgs::msg::Marker trailer_marker;
        trailer_marker.header.frame_id = FRAME_ID;
        trailer_marker.header.stamp = this->now();
        trailer_marker.ns = ns;
        trailer_marker.id = 2;
        trailer_marker.type = visualization_msgs::msg::Marker::CUBE;
        trailer_marker.action = visualization_msgs::msg::Marker::ADD;
        trailer_marker.scale.x = trailer_dimensions.x();
        trailer_marker.scale.y = trailer_dimensions.y();
        trailer_marker.scale.z = trailer_dimensions.z();
        trailer_marker.color.r = 0.95f;
        trailer_marker.color.g = 0.45f;
        trailer_marker.color.b = 0.1f;
        trailer_marker.color.a = 0.3f;

        const Eigen::Vector3f center_rotated(0.5f * (trailer_bbox.min_x + trailer_bbox.max_x),
                                             0.5f * (trailer_bbox.min_y + trailer_bbox.max_y),
                                             0.5f * (trailer_bbox.min_z + trailer_bbox.max_z));
        const Eigen::Vector2f center_in_frame = Eigen::Rotation2Df(trailer_pose.z()) * center_rotated.head<2>();
        trailer_marker.pose.position.x = center_in_frame.x();
        trailer_marker.pose.position.y = center_in_frame.y();
        trailer_marker.pose.position.z = center_rotated.z();
        trailer_marker.pose.orientation = toQuaternionMsg(trailer_pose.z());
        visualization_msg.markers.push_back(trailer_marker);

        const auto rough_ramp_cloud = std::make_shared<RawCloud>();
        ROI ramp_roi{};
        ramp_roi.min_x = trailer_pose.x();
        ramp_roi.min_x = trailer_pose.x() + 1.0f;
        ramp_roi.min_y = -0.5f * trailer_dimensions.y();
        ramp_roi.max_y = 0.5f * trailer_dimensions.y();
        ramp_roi.min_z = -1.0f;
        ramp_roi.max_z = 1.0f;
        filter3d::getCloud(*rough_trailer_cloud, rough_ramp_cloud, nullptr, ramp_roi);
        Eigen::Vector3f ramp_position;
        float ramp_tilt;
        Eigen::Vector2f ramp_size;
        if (!estimateRampPose(rough_ramp_cloud, ramp_position, ramp_tilt, ramp_size))
        {
            RCLCPP_ERROR(get_logger(), "Failed to estimate pose of the ramp!");
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }

        visualization_msgs::msg::Marker ramp_marker;
        ramp_marker.header.frame_id = FRAME_ID;
        ramp_marker.header.stamp = this->now();
        ramp_marker.ns = ns;
        ramp_marker.id = 3;
        ramp_marker.type = visualization_msgs::msg::Marker::CUBE;
        ramp_marker.action = visualization_msgs::msg::Marker::ADD;
        ramp_marker.scale.x = ramp_size.x();
        ramp_marker.scale.y = ramp_size.y();
        ramp_marker.scale.z = 0.04;
        ramp_marker.color.r = 0.1f;
        ramp_marker.color.g = 0.75f;
        ramp_marker.color.b = 1.0f;
        ramp_marker.color.a = 0.8f;
        ramp_marker.pose.position.x = ramp_position.x();
        ramp_marker.pose.position.y = ramp_position.y();
        ramp_marker.pose.position.z = ramp_position.z();
        tf2::Quaternion ramp_orientation;
        ramp_orientation.setRPY(0.0, -static_cast<double>(ramp_tilt), 0.0);
        ramp_marker.pose.orientation = tf2::toMsg(ramp_orientation);
        visualization_msg.markers.push_back(ramp_marker);

        if (mPlacementZones.empty())
        {
            if (!planPlacementZone(trailer_pose, trailer_dimensions.head<2>() - Eigen::Vector2f(0.01f, 0.01f),
                                   {load_size_x + 0.01f, load_size_y + 0.01f}))
            {
                RCLCPP_ERROR(get_logger(), "Failed to plan placement zones!");
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                continue;
            }
        }

        for (std::size_t index = 0; index < mPlacementZones.size(); ++index)
        {
            constexpr float placement_marker_height = 0.03f;
            visualization_msgs::msg::Marker placement_marker;
            placement_marker.header.frame_id = FRAME_ID;
            placement_marker.header.stamp = this->now();
            placement_marker.ns = ns;
            placement_marker.id = static_cast<int>(index) + 10;
            placement_marker.type = visualization_msgs::msg::Marker::CUBE;
            placement_marker.action = visualization_msgs::msg::Marker::ADD;
            placement_marker.scale.x = load_size_x;
            placement_marker.scale.y = load_size_y;
            placement_marker.scale.z = placement_marker_height;
            std::mt19937 color_generator(index);
            std::uniform_real_distribution color_distribution(0.2f, 1.0f);
            placement_marker.color.r = color_distribution(color_generator);
            placement_marker.color.g = color_distribution(color_generator);
            placement_marker.color.b = color_distribution(color_generator);
            placement_marker.color.a = 0.55f;

            const Eigen::Vector2f position_in_map = Eigen::Rotation2Df(trailer_pose.z()) * mPlacementZones[index];
            placement_marker.pose.position.x = position_in_map.x();
            placement_marker.pose.position.y = position_in_map.y();
            placement_marker.pose.position.z = trailer_bbox.min_z + 0.2f;
            placement_marker.pose.orientation = toQuaternionMsg(trailer_pose.z());
            visualization_msg.markers.push_back(placement_marker);

            visualization_msgs::msg::Marker center_marker = placement_marker;
            center_marker.ns = "placement_centers";
            center_marker.type = visualization_msgs::msg::Marker::SPHERE;
            center_marker.scale.x = 0.08;
            center_marker.scale.y = 0.08;
            center_marker.scale.z = 0.08;
            center_marker.color.r = 1.0f;
            center_marker.color.g = 1.0f;
            center_marker.color.b = 1.0f;
            center_marker.color.a = 1.0f;
            visualization_msg.markers.push_back(center_marker);

            visualization_msgs::msg::Marker label_marker = center_marker;
            label_marker.ns = "placement_labels";
            label_marker.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
            label_marker.text = std::to_string(index + 1);
            label_marker.scale.x = 0.0;
            label_marker.scale.y = 0.0;
            label_marker.scale.z = 0.3;
            label_marker.pose.position.z += 0.15;
            label_marker.pose.orientation = toQuaternionMsg(0.0);
            visualization_msg.markers.push_back(label_marker);
        }


        mVisualizationPub->publish(visualization_msg);

        // publish_rate.sleep();
        std::this_thread::sleep_for(worker_period);
    }
}

bool FreeSpaceDetect::locateTrailer(const RawCloud::Ptr& cloud, const RawCloud::Ptr& trailer_cloud, Eigen::Vector3f& trailer_pose) const
{
    if (cloud->size() < 100)
    {
        RCLCPP_WARN(get_logger(), "There's too few points to locate the trailer!");
        return false;
    }

    /* Detect and remove floor */
    // const auto floor_cloud = std::make_shared<RawCloud>();
    // feature3d::detectFloor(*cloud, floor_cloud);
    //
    // if (floor_cloud->size() < 100)
    // {
    //     RCLCPP_WARN(get_logger(), "Failed to detect the floor!");
    //     return false;
    // }
    //
    // float floor_z = std::numeric_limits<float>::lowest();
    // for (const auto& point : floor_cloud->points)
    // {
    //     floor_z
    // }
    const auto cloud_without_floor = std::make_shared<RawCloud>();
    cloud_without_floor->reserve(cloud->size());
    for (const auto& point : cloud->points)
    {
        if (point.z > 0.2)
        {
            cloud_without_floor->emplace_back(point.x, point.y, point.z);
        }
    }

    if (cloud_without_floor->size() < 100)
    {
        RCLCPP_WARN(get_logger(), "cloud_without_floor has too few points!");
        return false;
    }

    /* Get the top-down view image */
    constexpr float resolution = 0.05f;
    // Trailer side walls must be 1-5 m apart; convert the limits to image pixels.
    constexpr float min_side_separation = 1.0f / resolution;
    constexpr float max_side_separation = 5.0f / resolution;
    OrthographicProjector<pcl::PointXYZ> projector(View::TOP, resolution);
    projector.setCloud(cloud_without_floor);

    cv::Mat top_down_img = projector.projection();
    cv::Mat debug_img;
    cv::cvtColor(top_down_img, debug_img, cv::COLOR_GRAY2BGR);

    std::vector<cv::Vec4i> lines;
    cv::HoughLinesP(top_down_img, lines, 1.0, CV_PI / 180.0, 60, 40.0, 30.0);

    std::vector<int> candidate_indices;
    for (int index = 0; index < static_cast<int>(lines.size()); ++index)
    {
        const cv::Vec4i& line = lines[index];

        if (const double angle = std::atan2(static_cast<double>(line[3] - line[1]), static_cast<double>(line[2] - line[0])) * 180.0 / CV_PI;
            std::abs(angle) <= 30.0)
        {
            candidate_indices.push_back(index);
        }
    }

    int best_first = -1;
    int best_second = -1;
    double best_score = -1.0;

    for (std::size_t i = 0; i < candidate_indices.size(); ++i)
    {
        const cv::Vec4i& first_line = lines[candidate_indices[i]];
        const cv::Point2f first_start(static_cast<float>(first_line[0]), static_cast<float>(first_line[1]));
        const cv::Point2f first_end(static_cast<float>(first_line[2]), static_cast<float>(first_line[3]));
        const cv::Point2f first_direction = first_end - first_start;
        const double first_length = cv::norm(first_direction);
        const cv::Point2f first_unit = first_direction / static_cast<float>(first_length);
        const cv::Point2f first_midpoint = (first_start + first_end) * 0.5f;

        for (std::size_t j = i + 1; j < candidate_indices.size(); ++j)
        {
            const cv::Vec4i& second_line = lines[candidate_indices[j]];
            const cv::Point2f second_start(static_cast<float>(second_line[0]), static_cast<float>(second_line[1]));
            const cv::Point2f second_end(static_cast<float>(second_line[2]), static_cast<float>(second_line[3]));
            const cv::Point2f second_direction = second_end - second_start;
            const double second_length = cv::norm(second_direction);

            if (const cv::Point2f second_unit = second_direction / static_cast<float>(second_length);
                std::abs(first_unit.dot(second_unit)) < std::cos(10.0 * CV_PI / 180.0))
            {
                continue;
            }

            const cv::Point2f second_midpoint = (second_start + second_end) * 0.5f;
            const cv::Point2f midpoint_delta = second_midpoint - first_midpoint;

            if (const double separation = std::abs(first_unit.x * midpoint_delta.y - first_unit.y * midpoint_delta.x);
                separation < min_side_separation || separation > max_side_separation)
            {
                continue;
            }

            const double first_min = first_start.dot(first_unit);
            const double first_max = first_end.dot(first_unit);
            const double second_min = second_start.dot(first_unit);
            const double second_max = second_end.dot(first_unit);
            const double overlap = std::min(first_max, second_max) - std::max(first_min, second_min);

            if (overlap < 0.5 * std::min(first_length, second_length))
            {
                continue;
            }

            if (const double score = std::min(first_length, second_length) + 0.5 * overlap; score > best_score)
            {
                best_score = score;
                best_first = candidate_indices[i];
                best_second = candidate_indices[j];
            }
        }
    }

    for (const auto candidate : candidate_indices)
    {
        const auto& line = lines[candidate];
        cv::line(debug_img, cv::Point(line[0], line[1]), cv::Point(line[2], line[3]), cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
    }

    if (best_first < 0)
    {
        RCLCPP_WARN(get_logger(), "Failed to find the trailer!");
        return false;
    }

    std::vector<cv::Mat> side_masks(2);
    const auto line_indices = std::vector{best_first, best_second};
    for (std::size_t i = 0; i < 2; ++i)
    {
        side_masks[i] = cv::Mat::zeros(top_down_img.size(), CV_8UC1);
        const cv::Vec4i& line = lines[line_indices[i]];
        cv::line(debug_img, cv::Point(line[0], line[1]), cv::Point(line[2], line[3]), cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
        cv::line(side_masks[i], cv::Point(line[0], line[1]), cv::Point(line[2], line[3]), cv::Scalar(255), 5, cv::LINE_AA);
    }

    const cv::Vec4i& first_line = lines[best_first];
    cv::Point2f first_start(static_cast<float>(first_line[0]), static_cast<float>(first_line[1]));
    cv::Point2f first_end(static_cast<float>(first_line[2]), static_cast<float>(first_line[3]));

    const cv::Vec4i& second_line = lines[best_second];
    cv::Point2f second_start(static_cast<float>(second_line[0]), static_cast<float>(second_line[1]));
    cv::Point2f second_end(static_cast<float>(second_line[2]), static_cast<float>(second_line[3]));

    // Orient both side lines in the same direction before comparing their ends.
    if (const cv::Point2f first_direction = first_end - first_start;
        first_direction.dot(second_end - second_start) < 0.0f)
    {
        std::swap(second_start, second_end);
    }

    // const cv::Point2f rear_start = first_end;
    // const cv::Point2f rear_end = second_end;

    cv::line(debug_img, first_start, second_start, cv::Scalar(255, 0, 0), 2, cv::LINE_AA);

    /* Get 2 side wall point cloud */
    const auto side_wall_1 = projector.extractCloud(side_masks[0]);
    const auto side_wall_2 = projector.extractCloud(side_masks[1]);

    /* Get 2 refined side wall point cloud */
    pcl::PointCloud<pcl::PointXYZ> refined_side_wall_1, refined_side_wall_2;
    feature3d::findInliers(side_wall_1, refined_side_wall_1, 0.04f);
    feature3d::findInliers(side_wall_2, refined_side_wall_2, 0.04f);

    const auto side_wall_1_plane = feature3d::fitPlaneByPCA(refined_side_wall_1, 2);
    auto side_wall_2_plane = feature3d::fitPlaneByPCA(refined_side_wall_2, 2);
    if (side_wall_1_plane.head<3>().dot(side_wall_2_plane.head<3>()) < 0.0f)
    {
        side_wall_2_plane = -side_wall_2_plane;
    }
    Eigen::Vector4f mid_plane = 0.5f * (side_wall_1_plane + side_wall_2_plane);
    if (mid_plane[1] < 0.0f)
    {
        mid_plane = -mid_plane;
    }
    const float angle = std::atan2(-mid_plane[0], mid_plane[1]);

    pcl::PointCloud<pcl::PointXYZ> side_walls = refined_side_wall_1 + refined_side_wall_2;

    // 1. Rotation transformation from truck frame to rotated frame
    Eigen::Isometry3f T_rot = Eigen::Isometry3f::Identity();
    T_rot.rotate(Eigen::AngleAxisf(-angle, Eigen::Vector3f::UnitZ()));
    pcl::transformPointCloud(side_walls, side_walls, T_rot);

    Eigen::Vector4f rotated_mid_plane;
    rotated_mid_plane.head<3>() = T_rot.rotation() * mid_plane.head<3>();
    rotated_mid_plane[3] = mid_plane[3];

    /* Find out the max x of `side_walls` */
    float max_x = std::numeric_limits<float>::lowest();
    for (const auto& point : side_walls)
    {
        max_x = std::max(max_x, point.x);
    }
    const float y_offset = -rotated_mid_plane[3] / rotated_mid_plane[1];

    /* Get the pose of the trailer */
    trailer_pose = Eigen::Vector3f(max_x, y_offset, angle);

    /* Get the dimensions of the trailer */
    ROI trailer_roi = ROI::getBBox(side_walls);
    trailer_roi.min_x -= 9.9f;
    trailer_roi.min_z = -1.1f;
    trailer_roi.max_z += 3.1f;

    RawCloud non_angular_cloud;
    pcl::transformPointCloud(*cloud, non_angular_cloud, T_rot);
    filter3d::getCloud(non_angular_cloud, trailer_cloud, nullptr, trailer_roi);

    if (trailer_cloud->size() < 100)
    {
        RCLCPP_WARN(get_logger(), "Failed to find the trailer!");
        return false;
    }

    return true;
}

bool FreeSpaceDetect::estimateRampPose(const RawCloud::Ptr& cloud, Eigen::Vector3f& ramp_position,
                                       float& ramp_tilt, Eigen::Vector2f& ramp_size) const
{
    if (cloud->size() < 100)
    {
        RCLCPP_WARN(get_logger(), "There are too few points for ramp detection!");
        return false;
    }
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PARALLEL_PLANE); // ramp 与 y 轴平行
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setMaxIterations(100);
    seg.setDistanceThreshold(0.03);
    seg.setAxis(Eigen::Vector3f(0.0f, 1.0f, 0.0f));  // Y 轴
    seg.setEpsAngle(2.0 * M_PI / 180.0);            // 允许偏差 2°
    seg.setInputCloud(cloud);

    pcl::PointIndices inliers;
    pcl::ModelCoefficients coefficients;
    seg.segment(inliers, coefficients);

    if (inliers.indices.size() < 30 || coefficients.values.size() < 4)
    {
        RCLCPP_WARN(get_logger(), "Failed to fit the ramp plane!");
        return false;
    }

    Eigen::Vector3f normal(coefficients.values[0], coefficients.values[1], coefficients.values[2]);
    const float normal_length = normal.norm();
    if (normal_length < 1.0e-6f)
    {
        RCLCPP_WARN(get_logger(), "Ramp plane has an invalid normal!");
        return false;
    }
    normal /= normal_length;
    const float plane_offset = coefficients.values[3] / normal_length;

    Eigen::Vector3f ramp_direction = Eigen::Vector3f::UnitY().cross(normal);
    const float direction_length = ramp_direction.norm();
    if (direction_length < 1.0e-6f)
    {
        RCLCPP_WARN(get_logger(), "Ramp direction is degenerate!");
        return false;
    }
    ramp_direction /= direction_length;
    if (ramp_direction.x() < 0.0f)
    {
        ramp_direction = -ramp_direction;
    }

    float min_ramp = std::numeric_limits<float>::max();
    float max_ramp = std::numeric_limits<float>::lowest();
    float min_width = std::numeric_limits<float>::max();
    float max_width = std::numeric_limits<float>::lowest();
    for (const int index : inliers.indices)
    {
        const auto& point = cloud->points[static_cast<std::size_t>(index)];
        const Eigen::Vector3f position(point.x, point.y, point.z);
        const float ramp_coordinate = ramp_direction.dot(position);
        min_ramp = std::min(min_ramp, ramp_coordinate);
        max_ramp = std::max(max_ramp, ramp_coordinate);
        min_width = std::min(min_width, point.y);
        max_width = std::max(max_width, point.y);
    }

    const float ramp_length = max_ramp - min_ramp;
    const float ramp_width = max_width - min_width;
    if (ramp_length <= 0.0f || ramp_width <= 0.0f)
    {
        RCLCPP_WARN(get_logger(), "Ramp plane has invalid dimensions!");
        return false;
    }

    const float ramp_coordinate = 0.5f * (min_ramp + max_ramp);
    const float width_coordinate = 0.5f * (min_width + max_width);
    ramp_position = ramp_direction * ramp_coordinate + Eigen::Vector3f::UnitY() * width_coordinate -
                    normal * plane_offset;
    ramp_tilt = std::atan2(-normal.x(), normal.z());
    ramp_size = Eigen::Vector2f(ramp_length, ramp_width);
    return true;
}

bool FreeSpaceDetect::planPlacementZone(const Eigen::Vector3f& trailer_pose, const Eigen::Vector2f& trailer_size,
                                        const Eigen::Vector2f& load_size)
{
    mPlacementZones.clear();
    if (!trailer_pose.allFinite() || !trailer_size.allFinite() || !load_size.allFinite() ||
        (trailer_size.array() <= 0.0f).any() || (load_size.array() <= 0.0f).any())
    {
        RCLCPP_WARN(get_logger(), "Invalid trailer pose or dimensions for placement planning!");
        return false;
    }

    if ((load_size.array() > trailer_size.array()).any())
    {
        RCLCPP_WARN(get_logger(), "Load footprint does not fit within the trailer!");
        return false;
    }

    constexpr float count_tolerance = 1.0e-6f;
    const int longitudinal_count = std::floor(trailer_size.x() / load_size.x() + count_tolerance);
    const int lateral_count = std::floor(trailer_size.y() / load_size.y() + count_tolerance);
    if (longitudinal_count > std::numeric_limits<int>::max() || lateral_count > std::numeric_limits<int>::max())
    {
        RCLCPP_WARN(get_logger(), "Requested placement grid is too large!");
        return false;
    }

    const int rows = longitudinal_count;
    const int columns = lateral_count;
    const std::size_t zone_count = static_cast<std::size_t>(rows) * static_cast<std::size_t>(columns);
    if (zone_count > mPlacementZones.max_size())
    {
        RCLCPP_WARN(get_logger(), "Requested placement grid is too large!");
        return false;
    }
    mPlacementZones.reserve(zone_count);

    const float first_x = trailer_pose.x() - trailer_size.x() + 0.5f * load_size.x();
    const float first_y = trailer_pose.y() - 0.5f * static_cast<float>(columns) * load_size.y() + 0.5f * load_size.y();
    for (int row = 0; row < rows; ++row)
    {
        const float center_x = first_x + static_cast<float>(row) * load_size.x();
        for (int column = 0; column < columns; ++column)
        {
            const float center_y = first_y + static_cast<float>(column) * load_size.y();
            mPlacementZones.emplace_back(center_x, center_y);
        }
    }

    RCLCPP_INFO(get_logger(), "Planned %zu placement zones (%d rows, %d columns)!", mPlacementZones.size(), rows, columns);

    return true;
}


// bool FreeSpaceDetect::estimateSlotPose(const RawCloud::Ptr& cloud, Eigen::Vector3f& slot_pose) const
// {
//     if (cloud->size() < 10)
//     {
//         return false;
//     }
//
//     static OrthographicProjector<pcl::PointXYZ> projector(View::TOP, 0.02f);
//     projector.setCloud(cloud);
//     const cv::Mat projection = projector.projection();
//
//     cv::Mat closed_img;
//
//     {
//         cv::Mat temp_img;
//         filter2d::close(projection, temp_img);
//         filter2d::removeIsolatedPoints(temp_img, closed_img);
//     }
//
//
//     bool has_left_side{true}, has_right_side{true}; // Flags to check if there is left side or right side boundary
//     /* The coefficients of 3 planes, assume they are all perpendicular to the ground,
//      * so Vector3 instead of Vector4 */
//     Eigen::Vector3f rear_plane_coefficients{}, left_plane_coefficients{}, right_plane_coefficients{};
//     /* Step 1. Locate the boundary on x direction */
//     cv::Mat right_edge_img;
//     feature2d::detectEdge(closed_img(cv::Range::all(), cv::Range(0, closed_img.cols / 2)), right_edge_img,
//                           feature2d::EdgeType::RIGHT);
//     std::vector<cv::Point> edge_inliers;
//     if (!feature2d::findInliers(right_edge_img, edge_inliers, 5.0f))
//     {
//         RCLCPP_ERROR(get_logger(), "Can't find right edge inliers!");
//         return false;
//     }
//     auto edge_end_points = std::minmax_element(edge_inliers.begin(), edge_inliers.end(),
//                                                [](const cv::Point& a, const cv::Point& b) -> bool
//                                                    {
//                                                        return a.y < b.y;
//                                                    });
//     cv::Mat right_edge_mask = cv::Mat::zeros(closed_img.size(), CV_8UC1);
//     cv::line(right_edge_mask, *edge_end_points.first, *edge_end_points.second, cv::Scalar(255), 16);
//     const auto raw_rear_edge_cloud = projector.extractCloud(right_edge_mask);
//     RawCloud refined_rear_edge_cloud;
//     feature3d::findInliers(raw_rear_edge_cloud, refined_rear_edge_cloud, 0.06f);
//     if (!feature3d::planeCoefficients(refined_rear_edge_cloud, rear_plane_coefficients))
//     {
//         RCLCPP_ERROR(get_logger(), "Failed to compute the coefficients of rear plane!");
//         return false;
//     }
//     if (rear_plane_coefficients[0] < 0.0f)
//     {
//         rear_plane_coefficients *= -1.0f;
//     }
//     /* Locate the outer surface of the point cloud */
//     const auto [dist_front, _] = feature3d::farthestDistanceFromLine(refined_rear_edge_cloud, rear_plane_coefficients);
//     rear_plane_coefficients[2] += dist_front - mLoadDimensions[0] - 0.1f; // Move the plane from rear to the front
//
//     cv::line(right_edge_mask, *edge_end_points.first, *edge_end_points.second, cv::Scalar(255), 26);
//     closed_img.setTo(0, right_edge_mask);
//     /* Step 2. Locate the boundary on y direction, right side */
//     cv::Mat upper_edge_img;
//     feature2d::detectEdge(closed_img(cv::Range(0, closed_img.rows / 2),
//                                      cv::Range(0, static_cast<int>(closed_img.cols * 0.8))),
//                           upper_edge_img, feature2d::EdgeType::LOWER);
//     if (!feature2d::findInliers(upper_edge_img, edge_inliers, 3.0f))
//     {
//         RCLCPP_WARN(get_logger(), "Can't locate rigth side boundary!");
//         has_right_side = false;
//     }
//     else
//     {
//         edge_end_points = std::minmax_element(edge_inliers.begin(), edge_inliers.end(),
//                                               [](const cv::Point& a, const cv::Point& b) -> bool
//                                                   {
//                                                       return a.x < b.x;
//                                                   });
//         cv::Mat upper_edge_mask = cv::Mat::zeros(closed_img.size(), CV_8UC1);
//         cv::line(upper_edge_mask, *edge_end_points.first, *edge_end_points.second, cv::Scalar(255), 10);
//
//         const auto raw_right_edge_cloud = projector.extractCloud(upper_edge_mask);
//         RawCloud refined_right_edge_cloud;
//         feature3d::findInliers(raw_right_edge_cloud, refined_right_edge_cloud, 0.05f);
//         if (!feature3d::planeCoefficients(refined_right_edge_cloud, right_plane_coefficients))
//         {
//             RCLCPP_WARN(get_logger(), "Failed to compute the coefficients of right plane!");
//             has_right_side = false;
//         }
//         if (right_plane_coefficients[1] < 0.0f)
//         {
//             right_plane_coefficients *= -1.0f;
//         }
//     }
//
//     /* Step 3. Locate the boundary on y direction, left side */
//     cv::Mat lower_edge_img;
//     feature2d::detectEdge(closed_img(cv::Range(closed_img.rows / 2, closed_img.rows),
//                                      cv::Range(0, static_cast<int>(closed_img.cols * 0.8))),
//                           lower_edge_img, feature2d::EdgeType::LOWER);
//     if (!feature2d::findInliers(lower_edge_img, edge_inliers, 3.0f))
//     {
//         RCLCPP_WARN(get_logger(), "Can't locate left side boundary!");
//         has_left_side = false;
//     }
//     else
//     {
//         edge_end_points = std::minmax_element(edge_inliers.begin(), edge_inliers.end(),
//                                               [](const cv::Point& a, const cv::Point& b) -> bool
//                                                   {
//                                                       return a.x < b.x;
//                                                   });
//         cv::Mat lower_edge_mask = cv::Mat::zeros(closed_img.size(), CV_8UC1);
//         cv::Point lower_edge_endpoint_1(edge_end_points.first->x, edge_end_points.first->y + closed_img.rows / 2);
//         cv::Point lower_edge_endpoint_2(edge_end_points.second->x, edge_end_points.second->y + closed_img.rows / 2);
//         cv::line(lower_edge_mask, lower_edge_endpoint_1, lower_edge_endpoint_2, cv::Scalar(255), 10);
//
//         const auto raw_left_edge_cloud = projector.extractCloud(lower_edge_mask);
//         RawCloud refined_left_edge_cloud;
//         feature3d::findInliers(raw_left_edge_cloud, refined_left_edge_cloud, 0.05f);
//         if (!feature3d::planeCoefficients(refined_left_edge_cloud, left_plane_coefficients))
//         {
//             RCLCPP_WARN(get_logger(), "Failed to compute the coefficients of left plane!");
//             has_left_side = false;
//         }
//         if (left_plane_coefficients[1] < 0.0f)
//         {
//             left_plane_coefficients *= -1.0f;
//         }
//     }
//
//     /* Deal with it on different situations */
//     Eigen::Vector2f slot_position{};
//
//     /* Situation 1. There's only left side boundary */
//     if (has_left_side)
//     {
//         left_plane_coefficients[2] -= 0.5f * mLoadDimensions[1] + 0.08f; //Move the side plane to center
//         if (!feature3d::intersectionPoint(rear_plane_coefficients, left_plane_coefficients, slot_position))
//         {
//             RCLCPP_ERROR(get_logger(), "Can't locate intersection of left and rear planes!");
//             return false;
//         }
//         slot_pose[2] = std::atan2(-left_plane_coefficients[0], left_plane_coefficients[1]);
//     }
//     else if (has_right_side)
//     {
//         right_plane_coefficients[2] += 0.5f * mLoadDimensions[1] + 0.08f; //Move the side plane to center
//         if (!feature3d::intersectionPoint(rear_plane_coefficients, right_plane_coefficients, slot_position))
//         {
//             RCLCPP_ERROR(get_logger(), "Can't locate intersection of right and rear planes!");
//             return false;
//         }
//         slot_pose[2] = std::atan2(-right_plane_coefficients[0], right_plane_coefficients[1]);
//     }
//     else
//     {
//         RCLCPP_ERROR(get_logger(), "Can't locate side boundary!");
//         return false;
//     }
//
//     slot_pose.x() = slot_position[0];
//     slot_pose.y() = slot_position[1];
//
//     return true;
// }

bool FreeSpaceDetect::validateSlot(const RawCloud::Ptr& cloud, const Eigen::Vector3d& slot_pose, const ROI& load_dimensions) const
{}
