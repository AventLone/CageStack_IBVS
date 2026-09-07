#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <sophus/se3.hpp>

namespace perception::lio
{
Eigen::Isometry3f sophusExpUpdate(const Eigen::Matrix<float, 6, 1>& delta)
{
    return Eigen::Isometry3f(Sophus::SE3f::exp(delta).matrix());
}
} // namespace perception::lio