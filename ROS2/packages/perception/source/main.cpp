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
        // mCloud = std::make_shared<RawCloud>();
        const auto parallel = create_callback_group(rclcpp::CallbackGroupType::Reentrant);
        auto sub_options = rclcpp::SubscriptionOptions();
        sub_options.callback_group = parallel;
        mCloudSub = create_subscription<sensor_msgs::msg::PointCloud2>("test_cloud", rclcpp::SensorDataQoS(),
                                                                       [this](const sensor_msgs::msg::PointCloud2::ConstSharedPtr& msg)
                                                                           {
                                                                               // RawCloud cloud;

                                                                               // 步骤1：循环出队，清空所有旧数据（关键！不依赖size_approx()）
                                                                               // while (mClouds.try_dequeue(cloud))
                                                                               // {}

                                                                               // 步骤2：入队新数据
                                                                               {
                                                                                   std::lock_guard<std::mutex> lock(mCloudMutex);
                                                                                   pcl::fromROSMsg(*msg, mCloud);
                                                                               }
                                                                               // pcl::fromROSMsg(*msg, cloud);
                                                                               // if (!mClouds.enqueue(std::move(cloud)))
                                                                               // {
                                                                               //     return;
                                                                               // }

                                                                               // std::unique_lock<std::mutex> lock(mTriggerMutex);
                                                                               // mIsTriggered = true;
                                                                               // mTriggerEvent.notify_one();
                                                                           }, sub_options);

        mCloudPub = create_publisher<sensor_msgs::msg::PointCloud2>("test_cloud", rclcpp::SensorDataQoS());
        mFrontFacePub = create_publisher<sensor_msgs::msg::PointCloud2>("front_face", rclcpp::SensorDataQoS());
        mMarkerPub = create_publisher<visualization_msgs::msg::Marker>("marks", rclcpp::SensorDataQoS());
        mTargetPosePub = create_publisher<geometry_msgs::msg::PoseStamped>("target_pose", rclcpp::ServicesQoS());

        mTimer = create_wall_timer(std::chrono::milliseconds(100), std::bind(&Test::pubLoop, this), parallel);
        mCloudProcessThread = std::thread(&Test::cloudProcessLoop, this);
        RCLCPP_INFO(get_logger(), "The node has been activated.");
    }

    ~Test() override
    {
        //
        // {
        //     std::unique_lock<std::mutex> lock(mTriggerMutex);
        //     mIsTriggered = true;
        //     mTriggerEvent.notify_one();
        // }
        if (mCloudProcessThread.joinable())
        {
            mCloudProcessThread.join();
        }
        RCLCPP_INFO(get_logger(), "The node has been shutdown.");
    }

private:
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr mCloudSub;

    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr mCloudPub, mFrontFacePub;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr mMarkerPub;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr mTargetPosePub;

    rclcpp::TimerBase::SharedPtr mTimer;
    OrthographicProjector<pcl::PointXYZ> mProjector;
    lockfree_queue<RawCloud> mClouds;
    RawCloud mCloud;
    bool mIsTriggered{false};
    std::mutex mCloudMutex, mTriggerMutex;
    std::condition_variable mTriggerEvent;
    std::thread mCloudProcessThread;

    void pubLoop() const
    {
        const RawCloud test_cloud = createPalletCloud(-M_PIf / 6.0f);
        sensor_msgs::msg::PointCloud2 cloud_msg;
        pcl::toROSMsg(test_cloud, cloud_msg);
        cloud_msg.header.frame_id = "map";
        mCloudPub->publish(cloud_msg);
    }

    void cloudProcessLoop()
    {
        while (rclcpp::ok())
        {
            // {
            //     std::unique_lock<std::mutex> lock(mTriggerMutex);
            //     mTriggerEvent.wait(lock, [this]() -> bool { return mIsTriggered; });
            // }

            RawCloud::Ptr cloud = std::make_shared<RawCloud>();
            //
            {
                std::lock_guard<std::mutex> lock(mCloudMutex);
                *cloud = mCloud;
            }
            if (cloud->size() < 10)
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                continue;
            }

            mProjector.setCloud(cloud);
            const cv::Mat projection = mProjector.projection();
            if (projection.empty())
            {
                RCLCPP_ERROR(get_logger(), "Cloud projection is empty!");
                continue;
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

            auto line = feature2d::detectRectEdge(denoised_img, feature2d::EdgeType::RIGHT);

            cv::Mat debug_img;
            cv::cvtColor(projection, debug_img, cv::COLOR_GRAY2BGR);
            cv::line(debug_img, line.p1, line.p2, cv::Scalar(0, 255, 0), 2);

            cv::Mat mask = cv::Mat::zeros(denoised_img.size(), CV_8UC1);
            cv::line(mask, line.p1, line.p2, cv::Scalar(255), 40);
            cv::Mat front_edge;
            denoised_img.copyTo(front_edge, mask);
            RawCloud front_edge_cloud = mProjector.extractCloud(front_edge);
            if (front_edge_cloud.size() < 10)
            {
                RCLCPP_ERROR(get_logger(), "front_edge_cloud is too small!");
                return;
            }
            RawCloud front_edge_inlier_cloud;
            findInliers(front_edge_cloud, front_edge_inlier_cloud, 0.04f);

            Eigen::Vector4f centroid;
            pcl::compute3DCentroid(front_edge_inlier_cloud, centroid);
            const float angle = calculateLineAngle(front_edge_inlier_cloud);
            RCLCPP_INFO(get_logger(), "The angle is %f degrees", angle);
            RCLCPP_INFO(get_logger(), "The position is (%f, %f)", centroid[0], centroid[1]);

            sensor_msgs::msg::PointCloud2 front_edge_msg;
            pcl::toROSMsg(front_edge_inlier_cloud, front_edge_msg);
            front_edge_msg.header.frame_id = "map";
            mFrontFacePub->publish(front_edge_msg);

            std::this_thread::sleep_for(std::chrono::milliseconds(60));
        }
    }
};

int main(const int argc, char** argv)
{
    rclcpp::init(argc, argv);
    const auto node = std::make_shared<Test>();
    // rclcpp::spin(node);
    rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(), 2);
    executor.add_node(node);
    executor.spin();
    rclcpp::shutdown();

    return 0;
}
