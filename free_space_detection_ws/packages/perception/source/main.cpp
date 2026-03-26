#include "perception/nodes/CloudPublisher.h"
#include "perception/nodes/PoseEstimation.h"
#include "perception/tools/feature_detect_3d.hpp"

int main(const int argc, char** argv)
{
    rclcpp::init(argc, argv);
    const auto options = rclcpp::NodeOptions();
    const auto cloud_pub_node = std::make_shared<CloudBuild>("cloud_publisher", options);
    const auto target_pose_pub_node = std::make_shared<PoseEstimation>("target_publisher", options);
    rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(), 2);
    executor.add_node(cloud_pub_node);
    executor.add_node(target_pose_pub_node);
    executor.spin();
    rclcpp::shutdown();
    return 0;
}

// #include <pcl_conversions/pcl_conversions.h>
// #include "perception/tools/OrthographicProjector.hpp"
// #include "perception/tools/features_detect_2d.h"
// #include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
//
// class Test : public rclcpp::Node
// {
// public:
//     explicit Test() : rclcpp::Node("test"), mProjector(View::TOP, 0.01f)
//     {
//         const auto parallel = create_callback_group(rclcpp::CallbackGroupType::Reentrant);
//         auto sub_options = rclcpp::SubscriptionOptions();
//         sub_options.callback_group = parallel;
//         mCloudSub = create_subscription<sensor_msgs::msg::PointCloud2>("test_cloud", rclcpp::SensorDataQoS(),
//                                                                        [this](const sensor_msgs::msg::PointCloud2::ConstSharedPtr& msg)
//                                                                            {
//                                                                                pushInBuffer(*msg);
//                                                                                mTriggerEvent.notify_one();
//                                                                            }, sub_options);
//
//         mCloudPub = create_publisher<sensor_msgs::msg::PointCloud2>("test_cloud", rclcpp::SensorDataQoS());
//         mFrontFacePub = create_publisher<sensor_msgs::msg::PointCloud2>("front_face", rclcpp::SensorDataQoS());
//         mMarkerPub = create_publisher<visualization_msgs::msg::Marker>("marks", rclcpp::SensorDataQoS());
//         mTargetPosePub = create_publisher<geometry_msgs::msg::PoseStamped>("target_pose", rclcpp::ServicesQoS());
//
//         mTimer = create_wall_timer(std::chrono::milliseconds(100), std::bind(&Test::pubLoop, this), parallel);
//         mCloudProcessThread = std::thread(&Test::cloudProcessLoop, this);
//         RCLCPP_INFO(get_logger(), "The node has been activated.");
//     }
//
//     ~Test() override
//     {
//         //
//         {
//             std::unique_lock<std::mutex> lock(mBufferMutex);
//             mIsShutDown = true;
//         }
//         mTriggerEvent.notify_one();
//         if (mCloudProcessThread.joinable())
//         {
//             mCloudProcessThread.join();
//         }
//         RCLCPP_INFO(get_logger(), "The node has been shutdown.");
//     }
//
// private:
//     rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr mCloudSub;
//
//     rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr mCloudPub, mFrontFacePub;
//     rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr mMarkerPub;
//     rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr mTargetPosePub;
//
//     rclcpp::TimerBase::SharedPtr mTimer;
//
//     /* Data Buffer */
//     std::queue<sensor_msgs::msg::PointCloud2> mCloudBuffer;
//     std::mutex mBufferMutex;
//
//     OrthographicProjector<pcl::PointXYZ> mProjector;
//     // bool mIsTriggered{false};
//     bool mIsShutDown{false};
//     std::condition_variable mTriggerEvent;
//     std::thread mCloudProcessThread;
//
//     void pushInBuffer(const sensor_msgs::msg::PointCloud2& msg)
//     {
//         std::lock_guard<std::mutex> lock(mBufferMutex);
//         while (!mCloudBuffer.empty())
//         {
//             mCloudBuffer.pop();
//         }
//         mCloudBuffer.push(msg);
//     }
//
//     void pubLoop() const
//     {
//         static std::mt19937 rng(std::random_device{}());
//         static std::uniform_real_distribution<float> uni(-M_PIf / 3.0f, M_PIf / 3.0f);
//         const RawCloud test_cloud = feature3d::createPalletCloud(uni(rng));
//         sensor_msgs::msg::PointCloud2 cloud_msg;
//         pcl::toROSMsg(test_cloud, cloud_msg);
//         cloud_msg.header.frame_id = "map";
//         mCloudPub->publish(cloud_msg);
//     }
//
//     void cloudProcessLoop()
//     {
//         while (rclcpp::ok())
//         {
//             sensor_msgs::msg::PointCloud2 cloud_msg;
//             //
//             {
//                 std::unique_lock<std::mutex> lock(mBufferMutex);
//                 mTriggerEvent.wait(lock, [this]() -> bool { return !mCloudBuffer.empty() || mIsShutDown; });
//                 if (mIsShutDown)
//                 {
//                     break;
//                 }
//                 cloud_msg = mCloudBuffer.front();
//                 mCloudBuffer.pop();
//             }
//
//             RawCloud::Ptr cloud = std::make_shared<RawCloud>();
//             pcl::fromROSMsg(cloud_msg, *cloud);
//
//             if (cloud->size() < 10)
//             {
//                 std::this_thread::sleep_for(std::chrono::milliseconds(100));
//                 continue;
//             }
//
//             mProjector.setCloud(cloud);
//             const cv::Mat projection = mProjector.projection();
//             if (projection.empty())
//             {
//                 RCLCPP_ERROR(get_logger(), "Cloud projection is empty!");
//                 continue;
//             }
//             cv::Mat opened_img, deisolated_img, denoised_img;
//             filter2d::open(projection, opened_img);
//             filter2d::removeIsolatedPoints(projection, deisolated_img);
//             filter2d::denoise(projection, denoised_img);
//
//             cv::Mat closed_img;
//             filter2d::close(denoised_img, closed_img);
//             cv::Mat edge_img;
//             feature2d::detectEdge(closed_img, edge_img);
//
//             auto line = feature2d::detectRectEdge(denoised_img, feature2d::EdgeType::RIGHT);
//
//             cv::Mat debug_img;
//             cv::cvtColor(projection, debug_img, cv::COLOR_GRAY2BGR);
//             cv::line(debug_img, line.p1, line.p2, cv::Scalar(0, 255, 0), 2);
//
//             cv::Mat mask = cv::Mat::zeros(denoised_img.size(), CV_8UC1);
//             try
//             {
//                 cv::line(mask, line.p1, line.p2, cv::Scalar(255), 50);
//             }
//             catch (const std::runtime_error& e)
//             {
//                 RCLCPP_ERROR(get_logger(), "%s", e.what());
//             }
//             cv::Mat front_edge;
//             denoised_img.copyTo(front_edge, mask);
//             RawCloud front_edge_cloud = mProjector.extractCloud(front_edge);
//             if (front_edge_cloud.size() < 10)
//             {
//                 RCLCPP_ERROR(get_logger(), "front_edge_cloud is too small!");
//                 return;
//             }
//             RawCloud front_edge_inlier_cloud;
//             feature3d::findInliers(front_edge_cloud, front_edge_inlier_cloud, 0.04f);
//
//             Eigen::Vector4f centroid;
//             pcl::compute3DCentroid(front_edge_inlier_cloud, centroid);
//             const float yaw = feature3d::calculateLineAngle(front_edge_inlier_cloud);
//             // RCLCPP_INFO(get_logger(), "The angle is %f degrees", yaw);
//             // RCLCPP_INFO(get_logger(), "The position is (%f, %f)", centroid[0], centroid[1]);
//             tf2::Quaternion quat_tf;
//             quat_tf.setRPY(0.0, 0.0, yaw); // RPY顺序：roll(x), pitch(y), yaw(z)
//
//             // 2. 将tf2四元数转换为ROS 2的Quaternion消息类型
//             // geometry_msgs::msg::Quaternion quat_msg;
//             // quat_msg = tf2::toMsg<tf2::Quaternion, geometry_msgs::msg::Quaternion>(quat_tf);
//             geometry_msgs::msg::PoseStamped target_pose_msg;
//             target_pose_msg.header.frame_id = "map";
//             geometry_msgs::msg::Quaternion target_orientation;
//             tf2::convert(quat_tf, target_orientation);
//             target_pose_msg.pose.orientation = target_orientation;
//             target_pose_msg.pose.position.x = centroid[0];
//             target_pose_msg.pose.position.y = centroid[1];
//             target_pose_msg.pose.position.z = centroid[2];
//             target_pose_msg.header.stamp = now();
//             mTargetPosePub->publish(target_pose_msg);
//
//             sensor_msgs::msg::PointCloud2 front_edge_msg;
//             pcl::toROSMsg(front_edge_inlier_cloud, front_edge_msg);
//             front_edge_msg.header.frame_id = "map";
//             mFrontFacePub->publish(front_edge_msg);
//
//             /* Publish marker */
//             static const Eigen::Vector3f target_size(1.2f, 1.0f, 1.5f);
//
//             visualization_msgs::msg::Marker target_marker_msg;
//             target_marker_msg.header.frame_id = "map";
//             target_marker_msg.header.stamp = this->now();
//             target_marker_msg.ns = "bbox";
//             target_marker_msg.id = 0;
//             target_marker_msg.type = visualization_msgs::msg::Marker::CUBE;
//             target_marker_msg.action = visualization_msgs::msg::Marker::ADD;
//
//             Eigen::Isometry2f T_1(Eigen::Isometry2f::Identity()), T_2(Eigen::Isometry2f::Identity());
//             T_1.rotate(yaw);
//             T_1.pretranslate(centroid.head<2>());
//             T_2.translate(-Eigen::Vector2f(0.5f * target_size[0], 0.0f));
//             Eigen::Isometry2f T_3 = T_1 * T_2;
//             const Eigen::Vector2f mark_position = T_3.translation();
//
//             /* The dimensions of the cube mark, uint: meter */
//             target_marker_msg.scale.x = target_size[0];
//             target_marker_msg.scale.y = target_size[1];
//             target_marker_msg.scale.z = target_size[2];
//
//             /* Semi-transparent color */
//             target_marker_msg.color.r = 0.0f;
//             target_marker_msg.color.g = 1.0f;
//             target_marker_msg.color.b = 0.0f;
//             target_marker_msg.color.a = 0.8f; // "alpha < 1" indicates semi-transparency
//             target_marker_msg.pose.orientation = target_orientation;
//             target_marker_msg.pose.position.x = mark_position[0];
//             target_marker_msg.pose.position.y = mark_position[1];
//             target_marker_msg.pose.position.z = centroid[2] + 0.5 * target_size[2];
//             mMarkerPub->publish(target_marker_msg);
//
//             std::this_thread::sleep_for(std::chrono::milliseconds(10));
//         }
//     }
// };
//
// int main(const int argc, char** argv)
// {
//     rclcpp::init(argc, argv);
//     const auto node = std::make_shared<Test>();
//     // rclcpp::spin(node);
//     rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(), 2);
//     executor.add_node(node);
//     executor.spin();
//     rclcpp::shutdown();
//
//     return 0;
// }
