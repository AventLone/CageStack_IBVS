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
    this->declare_parameter(param_name, "/perception/semantic_cloud");
    const std::string semantic_cloud_topic = this->get_parameter(param_name).as_string();
    mCloudSub = create_subscription<sensor_msgs::msg::PointCloud2>(semantic_cloud_topic, rclcpp::SensorDataQoS(),
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
    mVisualizationPub = this->create_publisher<visualization_msgs::msg::MarkerArray>(target_topics["BBox"], rclcpp::SensorDataQoS());
    mTargetPosePub = this->create_publisher<geometry_msgs::msg::PoseStamped>(target_topics["Pose"], rclcpp::ServicesQoS());
    mLoadPosePub = this->create_publisher<geometry_msgs::msg::PoseStamped>("load_pose", rclcpp::ServicesQoS());
    mSlotPosePub = this->create_publisher<geometry_msgs::msg::PoseStamped>("slot_pose", rclcpp::ServicesQoS());
    mRoiCloudPub = this->create_publisher<sensor_msgs::msg::PointCloud2>("visualization/cloud_in_roi", rclcpp::SensorDataQoS());
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
        SemanticCloud::Ptr cloud = std::make_shared<SemanticCloud>();
        pcl::fromROSMsg(cloud_msg, *cloud);

        /* Stage 1. Detect the pose of the load on the forks */
        /* Step 1. Get the pose of the forks */
        Eigen::Isometry3f T_body2fork;
        try
        {
            // This returns the pose of 'fork' in 'body' coordinates
            const geometry_msgs::msg::TransformStamped tf_body2fork = mTfBuffer->lookupTransform("LOLA", "fork", tf2::TimePointZero);
            T_body2fork = tf2::transformToEigen(tf_body2fork).cast<float>();
        }
        catch (const tf2::TransformException& ex)
        {
            RCLCPP_ERROR(this->get_logger(), "Could not transform fork to body: %s", ex.what());
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            continue;
        }

        constexpr ROI fork_roi{-1.5f, 0.1f, -0.66f, 0.66f, -0.3f, 1.5f};
        constexpr ROI slot_space_roi{-1.2f - 0.6f, 0.3f, -0.5f - 0.5f, 0.5f + 0.5f, 0.0f, 1.5f};
        // constexpr ROI slot_space_roi{-3.2f - 0.3f, 0.3f, -0.5f - 1.4f, 0.5f + 1.4f, 0.0f, 1.5f};

        /* Visualize fork_roi */
        visualization_msgs::msg::Marker fork_roi_msg;
        fork_roi_msg.header.frame_id = "LOLA";
        fork_roi_msg.header.stamp = this->now();
        fork_roi_msg.ns = ns;
        fork_roi_msg.id = 0;
        fork_roi_msg.type = visualization_msgs::msg::Marker::CUBE;
        fork_roi_msg.action = visualization_msgs::msg::Marker::ADD;
        fork_roi_msg.scale.x = static_cast<double>(fork_roi.max_x - fork_roi.min_x);
        fork_roi_msg.scale.y = static_cast<double>(fork_roi.max_y - fork_roi.min_y);
        fork_roi_msg.scale.z = static_cast<double>(fork_roi.max_z - fork_roi.min_z);
        fork_roi_msg.color.r = 0.8;
        fork_roi_msg.color.g = 0.8;
        fork_roi_msg.color.b = 1.0;
        fork_roi_msg.color.a = 0.2;
        fork_roi_msg.pose.position.x = T_body2fork.translation()[0] - 0.5 * fork_roi_msg.scale.x;
        fork_roi_msg.pose.position.y = T_body2fork.translation()[1];
        fork_roi_msg.pose.position.z = T_body2fork.translation()[2] + 0.5 * fork_roi_msg.scale.z - 0.3;
        visualization_msg.markers.push_back(fork_roi_msg);

        /* Visualize slot_space_roi */
        tf2::Quaternion tf_q;
        tf2::fromMsg(mGoalMsg.pose.orientation, tf_q); // Convert ROS msg → tf2 quaternion

        // 3. Convert quaternion to roll/pitch/yaw (Euler angles)
        double roll, pitch, yaw;
        tf2::Matrix3x3(tf_q).getRPY(roll, pitch, yaw); // Order: X(roll), Y(pitch), Z(yaw)
        Eigen::Isometry2f T_1(Eigen::Isometry2f::Identity()), T_2(Eigen::Isometry2f::Identity());
        T_1.rotate(yaw);
        T_1.pretranslate(Eigen::Vector2f(mGoalMsg.pose.position.x, mGoalMsg.pose.position.y));
        T_2.translate(Eigen::Vector2f(-0.5f * mLoadDimensions[0], 0.0f));
        Eigen::Isometry2f T_3 = T_1 * T_2;
        visualization_msgs::msg::Marker slot_roi_msg;
        slot_roi_msg.header.frame_id = "LOLA";
        slot_roi_msg.header.stamp = this->now();
        slot_roi_msg.ns = ns;
        slot_roi_msg.id = 1;
        slot_roi_msg.type = visualization_msgs::msg::Marker::CUBE;
        slot_roi_msg.action = visualization_msgs::msg::Marker::ADD;
        slot_roi_msg.scale.x = static_cast<double>(slot_space_roi.max_x - slot_space_roi.min_x);
        slot_roi_msg.scale.y = static_cast<double>(slot_space_roi.max_y - slot_space_roi.min_y);
        slot_roi_msg.scale.z = static_cast<double>(slot_space_roi.max_z - slot_space_roi.min_z);
        slot_roi_msg.color.r = 0.8;
        slot_roi_msg.color.g = 1.0;
        slot_roi_msg.color.b = 0.8;
        slot_roi_msg.color.a = 0.2;
        slot_roi_msg.pose.orientation = mGoalMsg.pose.orientation;
        slot_roi_msg.pose.position.x = T_3.translation()[0];
        slot_roi_msg.pose.position.y = T_3.translation()[1];
        slot_roi_msg.pose.position.z = 0.5 * slot_roi_msg.scale.z;
        visualization_msg.markers.push_back(slot_roi_msg);

        SemanticCloud::Ptr cloud_on_forks = std::make_shared<SemanticCloud>();
        SemanticCloud::Ptr cloud_off_forks = std::make_shared<SemanticCloud>();
        getCloud(*cloud, cloud_on_forks, cloud_off_forks, T_body2fork.translation(), fork_roi);
        if (cloud_on_forks->size() < 10 || cloud_off_forks->size() < 10)
        {
            RCLCPP_ERROR(get_logger(), "cloud_on_forks or cloud_off_forks has too less points!");
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            continue;
        }
        ColoredCloud cloud_on_forks_colored;
        pcl::copyPointCloud(*cloud_on_forks, cloud_on_forks_colored);
        std::for_each(std::execution::par_unseq, cloud_on_forks_colored.begin(), cloud_on_forks_colored.end(),
                      [](pcl::PointXYZRGB& point)
                          {
                              point.r = 200;
                              point.g = 200;
                              point.b = 250;
                          });
        RawCloud::Ptr pallet_cloud = std::make_shared<RawCloud>();
        getCloud(*cloud_on_forks, pallet_label, *pallet_cloud);
        geometry_msgs::msg::Pose2D load_pose;
        // geometry_msgs::msg::Pose2D load_pose_2d;

        /*------ Stage 1. Pose estimate for load on the forks ------*/
        if (!estimateLoadPose(pallet_cloud, load_pose))
        {
            RCLCPP_ERROR(get_logger(), "Failed to estimate pose of the load!");
            continue;
        }
        geometry_msgs::msg::PoseStamped load_pose_msg;
        load_pose_msg.pose.position.x = load_pose.x;
        load_pose_msg.pose.position.y = load_pose.y;
        load_pose_msg.pose.position.z = T_body2fork.translation()[2] - 0.1;
        load_pose_msg.pose.orientation = toQuaternionMsg(load_pose.theta);
        load_pose_msg.header.frame_id = "LOLA";
        load_pose_msg.header.stamp = now();
        mLoadPosePub->publish(load_pose_msg);

        visualization_msgs::msg::Marker load_cube_msg = getCubeMarker("LOLA", ns.data(),
                                                                      2, mLoadDimensions[0], mLoadDimensions[1], mLoadDimensions[2],
                                                                      0.6, 0.6, 0.8, 0.5, load_pose);
        load_cube_msg.pose.position.z = T_body2fork.translation()[2] + 0.5 * load_size_z - 0.1;
        visualization_msg.markers.push_back(load_cube_msg);
        mVisualizationPub->publish(visualization_msg);

        Eigen::Isometry3f goal_pose{};
        //
        {
            Eigen::Isometry3d temp{};
            tf2::fromMsg(mGoalMsg.pose, temp);
            goal_pose = temp.cast<float>();
        }

        /*------ Stage 2. Slot pose estimate ------*/
        SemanticCloud::Ptr slot_space_cloud = std::make_shared<SemanticCloud>();
        getCloud(*cloud_off_forks, slot_space_cloud, nullptr, goal_pose, slot_space_roi);
        if (slot_space_cloud->size() < 10)
        {
            RCLCPP_ERROR(get_logger(), "slot_space_cloud has too less points!");
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            continue;
        }
        SemanticCloud::Ptr slot_space_cloud_without_ground = std::make_shared<SemanticCloud>();
        removeGround(*slot_space_cloud, *slot_space_cloud_without_ground);
        if (!estimateSlotPose(slot_space_cloud_without_ground))
        {
            RCLCPP_ERROR(get_logger(), "Failed to estimate pose of the slot!");
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            continue;
        }

        ColoredCloud slot_space_cloud_colored;
        pcl::copyPointCloud(*slot_space_cloud_without_ground, slot_space_cloud_colored);
        std::for_each(std::execution::par_unseq, slot_space_cloud_colored.begin(), slot_space_cloud_colored.end(),
                      [](pcl::PointXYZRGB& point)
                          {
                              point.r = 200;
                              point.g = 250;
                              point.b = 200;
                          });

        ColoredCloud cloud_in_roi = slot_space_cloud_colored + cloud_on_forks_colored;
        cloud_in_roi.width = cloud_in_roi.size();
        cloud_in_roi.height = 1;
        sensor_msgs::msg::PointCloud2 cloud_in_roi_msg;
        pcl::toROSMsg(cloud_in_roi, cloud_in_roi_msg);
        cloud_in_roi_msg.header.frame_id = "LOLA";
        cloud_in_roi_msg.header.stamp = this->now();
        mRoiCloudPub->publish(cloud_in_roi_msg);

        /* Record the average elapsed time */
        const auto end = std::chrono::high_resolution_clock::now();
        const auto elapse = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        mTotalElapseTime += static_cast<double>(elapse);
        ++mLoopCount;

        constexpr ROI load_roi{-1.2f, 0.0f, -0.5f, 0.5f, 0.0f, 1.5f};
        // if (checkSpace(slot_space_cloud_without_ground, goal_pose, load_roi))
        // {
        //     RCLCPP_WARN(get_logger(), "Didn't detect anything!");
        //     continue;
        // }

        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

bool PoseEstimation::estimateLoadPose(const RawCloud::Ptr& pallet_cloud, geometry_msgs::msg::Pose2D& load_pose) const
{
    RawCloud::Ptr mid_block_cloud = std::make_shared<RawCloud>();
    mid_block_cloud->reserve(pallet_cloud->size());
    for (const auto& point : pallet_cloud->points)
    {
        if (point.y > -0.15f && point.y < 0.15f)
        {
            mid_block_cloud->emplace_back(point.x, point.y, point.z);
        }
    }

    static const auto narrow_down = [](const float angle) -> float
        {
            if (angle > M_PI_2f)
            {
                return angle - M_PIf;
            }

            if (angle < -M_PI_2f)
            {
                return angle + M_PIf;
            }
            return angle;
        };

    const float yaw = narrow_down(feature3d::calculateLineAngle(*mid_block_cloud) - M_PI_2f);

    /* Get the rear end of the mid block */
    pcl::PointXYZ min_point, max_point;
    pcl::getMinMax3D(*mid_block_cloud, min_point, max_point);
    RawCloud::Ptr rear_mid_block_cloud = std::make_shared<RawCloud>();
    rear_mid_block_cloud->reserve(mid_block_cloud->size() / 2);
    const float rear_thresh = min_point.x + 0.02f;
    for (const auto& point : mid_block_cloud->points)
    {
        if (point.x < rear_thresh)
        {
            rear_mid_block_cloud->emplace_back(point.x, point.y, point.z);
        }
    }

    Eigen::Vector4f centroid;
    pcl::compute3DCentroid(*rear_mid_block_cloud, centroid);
    Eigen::Isometry2f T_1(Eigen::Isometry2f::Identity()), T_2(Eigen::Isometry2f::Identity());
    T_1.rotate(yaw);
    T_1.pretranslate(centroid.head<2>());
    T_2.translate(Eigen::Vector2f(mLoadDimensions[0], 0.0f));
    const Eigen::Isometry2f T_3 = T_1 * T_2;

    load_pose.x = T_3.translation()[0];
    load_pose.y = T_3.translation()[1];
    load_pose.theta = yaw;

    // load_pose.position.x = T_3.translation()[0];
    // load_pose.position.y = T_3.translation()[1];
    // load_pose.position.z = centroid[2];
    //
    // load_pose.orientation.x = 0.0;
    // load_pose.orientation.y = 0.0;
    // load_pose.orientation.z = std::sin(0.5f * yaw);
    // load_pose.orientation.w = std::cos(0.5f * yaw);


    // static OrthographicProjector<pcl::PointXYZ> projector(View::TOP, 0.01f);
    // projector.setCloud(mid_block_cloud);
    // const cv::Mat projection = projector.projection();
    // if (projection.empty())
    // {
    //     RCLCPP_ERROR(get_logger(), "Cloud projection is empty!");
    //     return false;
    // }
    //
    // cv::Mat denoised_img;
    // filter2d::denoise(projection, denoised_img);
    //
    // cv::Mat closed_img;
    // filter2d::close(projection, closed_img);
    // filter2d::removeIsolatedPoints(closed_img, closed_img);
    //
    // cv::Mat edge_img;
    // feature2d::detectEdge(closed_img, edge_img, feature2d::EdgeType::LEFT);

    return true;
}

bool PoseEstimation::estimateSlotPose(const SemanticCloud::Ptr& cloud) const
{
    if (cloud->size() < 10)
    {
        return false;
    }
    static OrthographicProjector<SemanticPoint> projector(View::TOP, 0.01f);
    projector.setCloud(cloud);
    const cv::Mat projection = projector.projection();

    cv::Mat closed_img;
    //
    {
        cv::Mat temp_img;
        filter2d::close(projection, temp_img);
        filter2d::removeIsolatedPoints(temp_img, closed_img);
    }

    bool has_left_side{true}, has_right_side{true}; // Flags to check if there is left side or right side boundary

    /* Step 1. Locate the boundary on x direction */
    cv::Mat right_edge_img;
    feature2d::detectEdge(closed_img(cv::Range::all(), cv::Range(0, closed_img.cols / 2)), right_edge_img,
                          feature2d::EdgeType::RIGHT);
    std::vector<cv::Point> edge_inliers;
    if (!feature2d::findInliers(right_edge_img, edge_inliers, 3.0f))
    {
        RCLCPP_ERROR(get_logger(), "Can't find right edge inliers!");
        return false;
    }
    auto edge_end_points = std::minmax_element(edge_inliers.begin(), edge_inliers.end(),
                                               [](const cv::Point& a, const cv::Point& b) -> bool
                                                   {
                                                       return a.y < b.y;
                                                   });
    cv::Mat right_edge_mask = cv::Mat::zeros(closed_img.size(), CV_8UC1);
    cv::line(right_edge_mask, *edge_end_points.first, *edge_end_points.second, cv::Scalar(255), 16);

    cv::line(right_edge_mask, *edge_end_points.first, *edge_end_points.second, cv::Scalar(255), 26);
    closed_img.setTo(0, right_edge_mask);
    /* Step 2. Locate the boundary on y direction, left side */
    cv::Mat upper_edge_img;
    feature2d::detectEdge(closed_img(cv::Range(0, closed_img.rows / 2), cv::Range::all()), upper_edge_img,
                          feature2d::EdgeType::LOWER);
    if (!feature2d::findInliers(upper_edge_img, edge_inliers, 3.0f))
    {
        RCLCPP_WARN(get_logger(), "Can't locate rigth side boundary!");
        has_right_side = false;
    }
    else
    {
        edge_end_points = std::minmax_element(edge_inliers.begin(), edge_inliers.end(),
                                              [](const cv::Point& a, const cv::Point& b) -> bool
                                                  {
                                                      return a.x < b.x;
                                                  });
        cv::Mat upper_edge_mask = cv::Mat::zeros(closed_img.size(), CV_8UC1);
        cv::line(upper_edge_mask, *edge_end_points.first, *edge_end_points.second, cv::Scalar(255), 16);
    }
    /* Step 3. Locate the boundary on y direction, rights side */
    cv::Mat lower_edge_img;
    feature2d::detectEdge(closed_img(cv::Range(closed_img.rows / 2, closed_img.rows), cv::Range::all()), lower_edge_img,
                          feature2d::EdgeType::LOWER);
    if (!feature2d::findInliers(lower_edge_img, edge_inliers, 3.0f))
    {
        RCLCPP_WARN(get_logger(), "Can't locate left side boundary!");
        has_left_side = false;
    }
    else
    {
        edge_end_points = std::minmax_element(edge_inliers.begin(), edge_inliers.end(),
                                              [](const cv::Point& a, const cv::Point& b) -> bool
                                                  {
                                                      return a.x < b.x;
                                                  });
        cv::Mat lower_edge_mask = cv::Mat::zeros(closed_img.size(), CV_8UC1);
        cv::line(lower_edge_mask, *edge_end_points.first, *edge_end_points.second, cv::Scalar(255), 16);
    }

    /* Deal with it on different situations */
    /* Situation 1. There's only left side boundary */
    if (has_left_side && !has_right_side)
    {}
    else if (!has_left_side && has_right_side)
    {}
    else
    {
        RCLCPP_ERROR(get_logger(), "Can't locate side boundary!");
        return false;
    }
    return true;
}
