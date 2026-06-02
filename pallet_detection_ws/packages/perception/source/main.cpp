#include "perception/nodes/CloudPublisher.h"
#include "perception/nodes/PoseEstimation.h"

int main(const int argc, char** argv)
{
    rclcpp::init(argc, argv);
    const auto options = rclcpp::NodeOptions().use_intra_process_comms(true);
    const auto cloud_pub_node = std::make_shared<CloudBuild>("cloud_publisher", options);
    const auto target_pose_pub_node = std::make_shared<PoseEstimation>("target_publisher", options);
    rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(), 2);
    executor.add_node(cloud_pub_node);
    executor.add_node(target_pose_pub_node);
    executor.spin();
    rclcpp::shutdown();
    return 0;
}

// int main()
// {
//     // const auto segmentor = std::make_unique<RfDetrSeg>("/home/avent/Desktop/TensorRTInfer/TrtNet/rf_detr.engine");
//     RfDetrSeg segmentor("/home/avent/Desktop/CageStack_IBVS/pallet_detection_ws/rfdetr.plan");
//     const cv::Mat img_1 = cv::imread("/home/avent/Desktop/generated_data/rf/valid/Replicator/0007.png");
//     const cv::Mat img_2 = cv::imread("/media/avent/CC5D-B805/real_data/image_2.png");
//
//     const auto result_1 = segmentor.seg(img_1, 0.9f);
//     const auto result_2 = segmentor.seg(img_2, 0.9f);
//     if (result_1.empty())
//     {
//         std::cerr << "Failed to segment image 1!" << std::endl;
//     }
//     else
//     {
//         cv::Mat visualization;
//         visualizeInstanceSeg(img_1, result_1, visualization);
//         cv::imwrite("/home/avent/Desktop/CageStack_IBVS/pallet_detection_ws/packages/perception/seg_result_1.png", visualization);
//     }
//
//     if (result_2.empty())
//     {
//         std::cerr << "Failed to segment image 2!" << std::endl;
//     }
//     else
//     {
//         cv::Mat visualization;
//         visualizeInstanceSeg(img_2, result_2, visualization);
//         cv::imwrite("/home/avent/Desktop/CageStack_IBVS/pallet_detection_ws/packages/perception/seg_result_2.png", visualization);
//     }
//
//     return 0;
// }
