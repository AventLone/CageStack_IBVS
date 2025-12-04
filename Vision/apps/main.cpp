#include <opencv2/opencv.hpp>


int main()
{
    const cv::Mat img = cv::imread("/home/vn/Pictures/Screenshots/Screenshot from 2025-09-11 19-36-36.png");
    cv::imshow("Hello", img);
    cv::waitKey(0);
    return 0;
}