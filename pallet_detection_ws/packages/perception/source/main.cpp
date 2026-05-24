#include "perception/nodes/CloudPublisher.h"
#include "perception/nodes/PoseEstimation.h"
// #include "perception/tools/rfdetr_obj_detector.h"

int main(const int argc, char** argv)
{
    rclcpp::init(argc, argv);
    const auto options = rclcpp::NodeOptions();
    if (!options.use_intra_process_comms())
    {
        std::cerr << "Failed to use intra_process_comms!" << std::endl;
    }
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
//     // RfDetrSeg segmentor("/home/avent/Downloads/rfdetr-medium.plan");
//     RfDetrDetector detector("/home/avent/Downloads/rfdetr-medium.plan");
//     const cv::Mat img_1 = cv::imread("/home/avent/Pictures/Screenshots/Screenshot from 2026-05-24 12-01-16.png");
//     const cv::Mat img_2 = cv::imread("/home/avent/Pictures/images_4.jpeg");
//
//     const auto result_1 = detector.detect(img_1, 0.7f);
//     const auto result_2 = detector.detect(img_2, 0.7f);
//
//     const std::unordered_map<int, std::string> label_dict{{0, "pallet"}, {1, "storage_cage"}, {2, "goods"}};
//
//     if (result_1.empty())
//     {
//         std::cerr << "Failed to segment image 1!" << std::endl;
//     }
//     else
//     {
//         cv::Mat visualization;
//         visualizeInstanceSeg(img_1, visualization, result_1, label_dict);
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
//         visualizeInstanceSeg(img_2, visualization, result_2, label_dict);
//         cv::imwrite("/home/avent/Desktop/CageStack_IBVS/pallet_detection_ws/packages/perception/seg_result_2.png", visualization);
//     }
//
//     return 0;
// }
