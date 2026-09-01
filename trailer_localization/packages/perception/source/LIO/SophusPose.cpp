#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <sophus/se3.hpp>

namespace perception::lio
{
Eigen::Isometry3f sophusExpUpdate(Eigen::Matrix<float, 6, 1> delta, const bool constrain_to_se2)
{
    if (constrain_to_se2)
    {
        delta.y() = 0.0f;
        delta.z() = 0.0f;
        delta(3) = 0.0f;
        delta(4) = 0.0f;
    }
    return Eigen::Isometry3f(Sophus::SE3f::exp(delta).matrix());
}
} // namespace perception::lio