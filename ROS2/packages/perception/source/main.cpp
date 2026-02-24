#include "perception/nodes/CloudPublisher.h"
#include "perception/nodes/PoseEstimation.h"
#include "perception/tools/feature_detect_3d.hpp"

// int main(const int argc, char** argv)
// {
//     rclcpp::init(argc, argv);
//     const auto options = rclcpp::NodeOptions();
//     const auto cloud_pub_node = std::make_shared<CloudBuild>("cloud_publisher", options);
//     const auto target_pose_pub_node = std::make_shared<PoseEstimation>("target_publisher", options);
//     rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(), 2);
//     executor.add_node(cloud_pub_node);
//     executor.add_node(target_pose_pub_node);
//     executor.spin();
//     rclcpp::shutdown();
//     return 0;
// }

#include <pcl_conversions/pcl_conversions.h>
#include "perception/tools/OrthographicProjector.hpp"
#include "perception/tools/features_detect_2d.h"

class Test : public rclcpp::Node
{
public:
    explicit Test() : rclcpp::Node("test"), mProjector(View::TOP, 0.01f)
    {
        mCloudPub = create_publisher<sensor_msgs::msg::PointCloud2>("test_cloud", rclcpp::SensorDataQoS());
        mMarkerPub = create_publisher<visualization_msgs::msg::Marker>("marks", rclcpp::SensorDataQoS());

        mTimer = create_wall_timer(std::chrono::milliseconds(200), std::bind(&Test::pubLoop, this));
    }

private:
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr mCloudPub;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr mMarkerPub;

    rclcpp::TimerBase::SharedPtr mTimer;
    OrthographicProjector<pcl::PointXYZ> mProjector;


    void pubLoop()
    {
        const RawCloud test_cloud = createPalletCloud(M_PIf / 6.0f);
        sensor_msgs::msg::PointCloud2 cloud_msg;
        pcl::toROSMsg(test_cloud, cloud_msg);
        cloud_msg.header.frame_id = "map";
        mCloudPub->publish(cloud_msg);

        mProjector.setCloud(test_cloud.makeShared());
        const cv::Mat projection = mProjector.projection();
        if (projection.empty())
        {
            RCLCPP_ERROR(get_logger(), "Cloud projection is empty!");
            return;
        }
        cv::Mat opened_img, deisolated_img, denoised_img;
        filter2d::open(projection, opened_img);

        const auto min_rect = feature2d::detectMinRect(opened_img);
        filter2d::removeIsolatedPoints(projection, deisolated_img);
        filter2d::denoise(projection, denoised_img);

        cv::Mat closed_img;
        filter2d::close(denoised_img, closed_img);
        cv::Mat edge_img;
        feature2d::detectEdge(closed_img, edge_img);

        auto line = feature2d::detectRectEdge(opened_img, feature2d::EdgeType::RIGHT);
        cv::Mat debug_img;
        cv::cvtColor(projection, debug_img, cv::COLOR_GRAY2BGR);
        cv::line(debug_img, line.p1, line.p2, cv::Scalar(0, 255, 0), 2);

        RCLCPP_INFO(get_logger(), "Hello");
    }
};

int main(const int argc, char** argv)
{
    rclcpp::init(argc, argv);
    const auto node = std::make_shared<Test>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
