#pragma once
#include <sophus/se3.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <perception/LIO/ImuProcessor.hpp>

namespace lio
{
using ErrorStateT = Eigen::Vector<double, 17>;


class ESKF
{
public:
    void predict(const ImuData& imu_data);

    void update(const pcl::PointCloud<pcl::PointXYZ>& cloud, double dt);

    Eigen::Isometry3d state();

private:
    ErrorStateT mErrorState;

};
}
