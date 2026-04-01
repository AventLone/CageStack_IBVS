#include "perception/nodes/PoseEstimation.h"
#include "perception/nodes/CloudPublisher.h"
#include <cv_bridge/cv_bridge.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <Eigen/Geometry>
#include <pcl_conversions/pcl_conversions.h>
#include <tf2_eigen/tf2_eigen.hpp> // ROS 2 header
#include <execution>
#include "perception/tools/filter_2d.h"
#include "perception/tools/filter_3d.h"
#include "perception/tools/feature_detect_2d.h"
#include "perception/tools/feature_detect_3d.hpp"


void PoseEstimation::initSubscriptions()
{
    const std::string param_name = "TopicName.Perception.SemanticCloud";
    this->declare_parameter(param_name, "/perception/colored_cloud");
    const std::string semantic_cloud_topic = this->get_parameter(param_name).as_string();
    mCloudSub = create_subscription<sensor_msgs::msg::PointCloud2>(semantic_cloud_topic,
                                                                   rclcpp::SensorDataQoS(),
                                                                   [this](const sensor_msgs::msg::PointCloud2::ConstSharedPtr& msg)
                                                                       {
                                                                           this->pushInBuffer(*msg);
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
    mVisualizationPub = create_publisher<visualization_msgs::msg::MarkerArray>(target_topics["BBox"], rclcpp::SensorDataQoS());
    mTargetPosePub = create_publisher<geometry_msgs::msg::PoseStamped>(target_topics["Pose"], rclcpp::ServicesQoS());
    mLoadPosePub = create_publisher<geometry_msgs::msg::PoseStamped>("load_pose", rclcpp::ServicesQoS());
    mSlotPosePub = create_publisher<geometry_msgs::msg::PoseStamped>("slot_pose", rclcpp::ServicesQoS());
    mRoiCloudPub = create_publisher<sensor_msgs::msg::PointCloud2>("visualization/cloud_in_roi", rclcpp::SensorDataQoS());
}

void PoseEstimation::workerLoop()
{
    static constexpr std::string_view ns = "visualization";
    static constexpr std::string_view frame_id = "map";
    static constexpr float load_size_x = 1.2f;
    static constexpr float load_size_y = 1.0f;
    static constexpr float load_size_z = 1.5f;

    static constexpr int pallet_label = 1;
    while (rclcpp::ok())
    {
        sensor_msgs::msg::PointCloud2 cloud_msg;
        visualization_msgs::msg::MarkerArray visualization_msg;
        //
        {
            std::unique_lock<std::mutex> lock(mBufferMutex);
            mTriggerEvent.wait(lock, [this]() -> bool { return (mHasGoal && !mCloudBuffer.empty()) || mIsShutdown; });
            if (mIsShutdown)
            {
                break;
            }
            cloud_msg = mCloudBuffer.front();
            mCloudBuffer.pop();
        }

        const auto start = std::chrono::high_resolution_clock::now();
        ColoredCloud::Ptr temp = std::make_shared<ColoredCloud>();
        pcl::fromROSMsg(cloud_msg, *temp);

        RawCloud::Ptr cloud = std::make_shared<RawCloud>();
        pcl::copyPointCloud(*temp, *cloud);

        /* Stage 1. Detect the pose of the load on the forks */
        /* Step 1. Get the pose of the forks */
        Eigen::Isometry3f T_body2fork;
        try
        {
            // This returns the pose of 'fork' in 'body' coordinates
            const geometry_msgs::msg::TransformStamped tf_body2fork =
                    mTfBuffer->lookupTransform("LOLA", "fork", tf2::TimePointZero);
            T_body2fork = tf2::transformToEigen(tf_body2fork).cast<float>();
        }
        catch (const tf2::TransformException& e)
        {
            RCLCPP_ERROR(this->get_logger(), "Could not transform fork to body: %s", e.what());
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            continue;
        }

        constexpr ROI load_bbox{-1.5f, 0.1f, -0.66f, 0.66f, -0.3f, 1.5f};
        constexpr ROI load_roi{-0.3, 0.3f, -0.5f, 0.5f, -0.1f, 0.3f};

        /* Visualize slot_space_roi */
        tf2::Quaternion tf_q;
        tf2::fromMsg(mGoalMsg.pose.orientation, tf_q); // Convert ROS msg → tf2 quaternion

        // 3. Convert quaternion to roll/pitch/yaw (Euler angles)
        double roll, pitch, yaw;
        // tf2::Matrix3x3(tf_q).getRPY(roll, pitch, yaw); // Order: X(roll), Y(pitch), Z(yaw)
        // Eigen::Isometry2f T_1(Eigen::Isometry2f::Identity()), T_2(Eigen::Isometry2f::Identity());
        // T_1.rotate(yaw);
        // T_1.pretranslate(Eigen::Vector2f(mGoalMsg.pose.position.x, mGoalMsg.pose.position.y));
        // T_2.translate(Eigen::Vector2f(-0.5f * mLoadDimensions[0], 0.0f));
        // Eigen::Isometry2f T_3 = T_1 * T_2;
        visualization_msgs::msg::Marker load_roi_msg;
        load_roi_msg.header.frame_id = "LOLA";
        load_roi_msg.header.stamp = this->now();
        load_roi_msg.ns = ns;
        load_roi_msg.id = 0;
        load_roi_msg.type = visualization_msgs::msg::Marker::CUBE;
        load_roi_msg.action = visualization_msgs::msg::Marker::ADD;
        load_roi_msg.scale.x = static_cast<double>(load_roi.max_x - load_roi.min_x);
        load_roi_msg.scale.y = static_cast<double>(load_roi.max_y - load_roi.min_y);
        load_roi_msg.scale.z = static_cast<double>(load_roi.max_z - load_roi.min_z);
        load_roi_msg.color.r = 0.2;
        load_roi_msg.color.g = 0.9;
        load_roi_msg.color.b = 0.2;
        load_roi_msg.color.a = 0.4;
        load_roi_msg.pose = mGoalMsg.pose;
        // load_roi_msg.pose.orientation = mGoalMsg.pose.orientation;
        // load_roi_msg.pose.position.x = T_3.translation()[0];
        // load_roi_msg.pose.position.y = T_3.translation()[1];
        load_roi_msg.pose.position.z = 0.5 * load_roi_msg.scale.z;
        visualization_msg.markers.push_back(load_roi_msg);

        Eigen::Isometry3f goal_pose{};
        //
        {
            Eigen::Isometry3d temp{};
            tf2::fromMsg(mGoalMsg.pose, temp);
            goal_pose = temp.cast<float>();
        }

        /*------ Stage 2. Slot pose estimate ------*/
        RawCloud::Ptr load_roi_cloud = std::make_shared<RawCloud>();
        getCloud(*cloud, load_roi_cloud, nullptr, goal_pose, load_roi);
        if (load_roi_cloud->size() < 10)
        {
            RCLCPP_ERROR(get_logger(), "slot_space_cloud has too few points!");
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            continue;
        }
        RawCloud::Ptr load_roi_clouod_without_ground = std::make_shared<RawCloud>();
        removeGround(*load_roi_cloud, *load_roi_clouod_without_ground, 0.05f);
        Eigen::Vector3f load_pose;
        if (!estimateLoadPose(load_roi_clouod_without_ground, load_pose))
        {
            RCLCPP_ERROR(get_logger(), "Failed to estimate pose of the slot!");
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            continue;
        }
        geometry_msgs::msg::PoseStamped slot_pose_msg;
        slot_pose_msg.pose.position.x = load_pose.x();
        slot_pose_msg.pose.position.y = load_pose.y();
        slot_pose_msg.pose.orientation = toQuaternionMsg(load_pose[2]);
        slot_pose_msg.header.frame_id = "LOLA";
        slot_pose_msg.header.stamp = now();
        mSlotPosePub->publish(slot_pose_msg);

        ColoredCloud slot_space_cloud_colored;
        pcl::copyPointCloud(*load_roi_clouod_without_ground, slot_space_cloud_colored);
        std::for_each(std::execution::par_unseq,
                      slot_space_cloud_colored.begin(), slot_space_cloud_colored.end(),
                      [](pcl::PointXYZRGB& point)
                          {
                              point.r = 200;
                              point.g = 250;
                              point.b = 200;
                          });

        ColoredCloud cloud_in_roi = slot_space_cloud_colored;
        cloud_in_roi.width = cloud_in_roi.size();
        cloud_in_roi.height = 1;
        sensor_msgs::msg::PointCloud2 cloud_in_roi_msg;
        pcl::toROSMsg(cloud_in_roi, cloud_in_roi_msg);
        cloud_in_roi_msg.header.frame_id = "LOLA";
        cloud_in_roi_msg.header.stamp = this->now();
        mRoiCloudPub->publish(cloud_in_roi_msg);

        // mGoalMsg.pose = slot_pose_msg.pose;

        /*------ Visualize the slot pose Cube ------*/
        // visualization_msgs::msg::Marker slot_cube_msg = getCubeMarker("LOLA", ns.data(),
        //                                                               3, mLoadDimensions[0],
        //                                                               mLoadDimensions[1], mLoadDimensions[2],
        //                                                               0.2, 0.9, 0.2, 0.86, load_pose);
        // slot_cube_msg.pose.position.z = 0.5 * load_size_z;
        // visualization_msg.markers.push_back(slot_cube_msg);
        mVisualizationPub->publish(visualization_msg);

        if (const auto delta_translation = goal_pose.translation().head<2>() - load_pose.head<2>();
            delta_translation.norm() < 0.2f)
        {
            mGoalMsg.pose.position.x = load_pose.x();
            mGoalMsg.pose.position.y = load_pose.y();
        }

        /* Record the average elapsed time */
        const auto end = std::chrono::high_resolution_clock::now();
        const auto elapse = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        mTotalElapseTime += static_cast<double>(elapse);
        ++mLoopCount;

        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

bool PoseEstimation::estimateLoadPose(const RawCloud::Ptr& cloud, Eigen::Vector3f& load_pose) const
{
    if (cloud->size() < 10)
    {
        return false;
    }
    static OrthographicProjector<pcl::PointXYZ> projector(View::TOP, 0.02f);
    projector.setCloud(cloud);
    const cv::Mat projection = projector.projection();

    cv::Mat closed_img;
    //
    {
        cv::Mat temp_img;
        filter2d::close(projection, temp_img);
        filter2d::removeIsolatedPoints(temp_img, closed_img);
    }

    /* Step 1. Locate the boundary on x direction */
    cv::Mat right_edge_img;
    feature2d::detectEdge(closed_img, right_edge_img, feature2d::EdgeType::RIGHT);
    auto line = feature2d::detectRectEdge(right_edge_img, feature2d::EdgeType::RIGHT);

    cv::Mat debug_img;
    cv::cvtColor(projection, debug_img, cv::COLOR_GRAY2BGR);
    cv::line(debug_img, line.p1, line.p2, cv::Scalar(0, 255, 0), 2);

    cv::Mat mask = cv::Mat::zeros(closed_img.size(), CV_8UC1);
    try
    {
        cv::line(mask, line.p1, line.p2, cv::Scalar(255), 50);
    }
    catch (const std::runtime_error& e)
    {
        RCLCPP_ERROR(get_logger(), "%s", e.what());
    }
    cv::Mat front_edge;
    closed_img.copyTo(front_edge, mask);
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
