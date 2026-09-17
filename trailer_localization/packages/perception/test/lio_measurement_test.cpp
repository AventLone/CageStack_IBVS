#include "perception/LIO/ESKF.h"
#include "perception/LIO/CloudTiming.hpp"
#include <iostream>

namespace
{
void require(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}
template<class Function> void rejects(Function&& function)
{
    bool threw = false;
    try { function(); } catch (const std::exception&) { threw = true; }
    require(threw, "Invalid cloud was not rejected");
}

sensor_msgs::msg::PointCloud2 makeCloud(const std::string& time_name, bool big_endian)
{
    using Field = sensor_msgs::msg::PointField;
    sensor_msgs::msg::PointCloud2 msg;
    msg.header.stamp.sec = 100;
    msg.width = 2; msg.height = 2; msg.point_step = 20; msg.row_step = 48;
    msg.is_bigendian = big_endian;
    msg.data.resize(96); // Includes organized-cloud row padding.
    const auto field = [&](const std::string& name, std::uint32_t offset, std::uint8_t datatype)
    {
        Field f;
        f.name = name; f.offset = offset; f.datatype = datatype; f.count = 1;
        msg.fields.push_back(f);
    };
    field("x", 0, Field::FLOAT32); field("y", 4, Field::FLOAT32); field("z", 8, Field::FLOAT32);
    if (!time_name.empty()) field(time_name, 12, time_name == "t" ? Field::UINT32 : Field::FLOAT64);
    const auto put = [&msg, big_endian]<class T>(std::size_t offset, T value)
    {
        unsigned char bytes[sizeof(T)];
        std::memcpy(bytes, &value, sizeof(T));
        const std::uint16_t probe = 1;
        if (big_endian != (*reinterpret_cast<const unsigned char*>(&probe) == 0))
            std::reverse(bytes, bytes + sizeof(T));
        std::memcpy(msg.data.data() + offset, bytes, sizeof(T));
    };
    for (std::size_t i = 0; i < 4; ++i)
    {
        const auto offset = (i / 2) * msg.row_step + (i % 2) * msg.point_step;
        put(offset, static_cast<float>(i)); put(offset + 4, 2.0f); put(offset + 8, 3.0f);
        if (time_name == "t") put(offset + 12, static_cast<std::uint32_t>(i * 10000000));
        else put(offset + 12, 100.0 + 0.01 * i);
    }
    return msg;
}

void cloudTimes()
{
    for (bool big : {false, true})
    {
        const auto cloud = lio::readTimedCloud(makeCloud("timestamp", big), {});
        require(cloud.points.size() == 4 && cloud.has_point_time, "Failed organized/endian cloud read");
        require(std::abs(cloud.begin - 100.0) < 1e-8 && std::abs(cloud.end - 100.03) < 1e-8, "Absolute seconds decoded incorrectly");
        require(cloud.points[3].x == 3.0f && std::abs(cloud.points[3].relative_time - 0.03) < 1e-8, "Point values/time incorrect");
        lio::CloudTimingConfig relative;
        relative.stamp_is_end = true;
        const auto nanoseconds = lio::readTimedCloud(makeCloud("t", big), relative);
        require(std::abs(nanoseconds.begin - 99.97) < 1e-8 && std::abs(nanoseconds.end - 100.0) < 1e-8,
                "Relative nanoseconds or end stamp decoded incorrectly");
    }
    auto no_time = makeCloud("", false);
    rejects([&] { lio::readTimedCloud(no_time, {}); });
    lio::CloudTimingConfig allowed;
    allowed.allow_untimed_cloud = true;
    const auto cloud = lio::readTimedCloud(no_time, allowed);
    require(!cloud.has_point_time && cloud.points[0].relative_time == 0.1, "Untimed explicit mode failed");
    auto malformed = makeCloud("timestamp", false);
    malformed.fields[0].offset = 20;
    rejects([&] { lio::readTimedCloud(malformed, {}); });
    malformed = makeCloud("timestamp", false); malformed.data.resize(20);
    rejects([&] { lio::readTimedCloud(malformed, {}); });
    lio::CloudTimingConfig wrong_scale;
    wrong_scale.time_scale = 1e9;
    rejects([&] { lio::readTimedCloud(makeCloud("timestamp", false), wrong_scale); });
}

void scanToMap()
{
    pcl::PointCloud<pcl::PointXYZ> world;
    for (int i = -10; i <= 10; ++i)
    {
        for (int j = -5; j <= 5; ++j)
        {
            world.emplace_back(3.0f, i * 0.1f, j * 0.1f);
            world.emplace_back(i * 0.1f, 4.0f, j * 0.1f);
            world.emplace_back(i * 0.1f, j * 0.1f, -2.0f);
        }
    }
    const Sophus::SE3d T_IL(Sophus::SO3d::exp(Eigen::Vector3d(0.05, 0.1, -0.1)), Eigen::Vector3d(0.2, -0.1, 0.05));
    const auto makeScan = [&](const lio::ImuState& state)
    {
        const Sophus::SE3d T_LW = (Sophus::SE3d(state.R_WI, state.p_WI) * T_IL).inverse();
        pcl::PointCloud<pcl::PointXYZ> scan;
        scan.reserve(world.size());
        for (const auto& point : world)
        {
            const Eigen::Vector3d p_L = T_LW * Eigen::Vector3d(point.x, point.y, point.z);
            scan.emplace_back(static_cast<float>(p_L.x()), static_cast<float>(p_L.y()), static_cast<float>(p_L.z()));
        }
        return scan;
    };

    lio::ESKF::Config config;
    config.map_voxel_size = 0.5f;
    config.max_iterations = 10;
    lio::ESKF filter(config, T_IL);
    lio::ImuState prior;
    lio::StateCovariance covariance = lio::StateCovariance::Identity() * 0.1;
    covariance(5, 11) = covariance(11, 5) = -0.02;
    filter.reset(prior, covariance);

    const auto first = filter.update(makeScan(prior));
    require(first.accepted && first.map_initialized && !filter.map().empty(),
            "Initial scan did not create the ESKF-owned map");

    lio::ImuState truth = prior;
    truth.p_WI = Eigen::Vector3d(0.04, -0.03, 0.02);
    truth.R_WI = Sophus::SO3d::exp(Eigen::Vector3d(0.01, -0.015, 0.02));
    const auto result = filter.update(makeScan(truth));
    require(result.accepted, "Internal right-perturbation ESKF update was rejected");
    require(result.num_correspondences > 500, "Internal ESKF update found too few correspondences");
    require(result.fitness_score < 1e-3, "Internal ESKF update left an excessive residual");
    require(lio::ESKF::boxMinus(filter.nominalState(), truth).head<6>().norm() < 0.002,
            "Failed to recover the IMU pose with non-identity LiDAR extrinsics");
    require(filter.nominalState().gyro_bias.norm() > 0.001, "Geometric constraints did not correct gyro bias");

    const auto saved_state = filter.nominalState();
    const auto saved_covariance = filter.covariance();
    const auto saved_map_size = filter.map().pointCount();
    require(!filter.update(pcl::PointCloud<pcl::PointXYZ>{}).accepted, "Empty cloud accepted");
    lio::ImuState far_state = truth;
    far_state.p_WI.x() += 10.0;
    require(!filter.update(makeScan(far_state)).accepted, "GICP correspondence distance gate failed");
    require(lio::ESKF::boxMinus(filter.nominalState(), saved_state).norm() == 0.0 &&
            filter.covariance() == saved_covariance && filter.map().pointCount() == saved_map_size,
            "Rejected update changed the prior or internal map");
}

void rightEskfJacobian()
{
    const Sophus::SE3d T_IL(Sophus::SO3d::exp(Eigen::Vector3d(-0.1, 0.2, 0.05)),
                            Eigen::Vector3d(0.2, -0.1, 0.05));
    lio::ImuState state;
    state.p_WI = Eigen::Vector3d(0.1, -0.05, 0.02);
    state.R_WI = Sophus::SO3d::exp(Eigen::Vector3d(0.2, -0.1, 0.15));
    const Eigen::Vector3d p_I = T_IL * Eigen::Vector3d(0.4, -0.2, 0.3);
    const Eigen::Vector3d target(0.9, -0.1, 0.4);
    Eigen::Matrix<double, 3, 6> jacobian;
    jacobian.leftCols<3>().setIdentity();
    jacobian.rightCols<3>() = -state.R_WI.matrix() * Sophus::SO3d::hat(p_I);
    for (int i = 0; i < 6; ++i)
    {
        lio::ErrorStateT delta = lio::ErrorStateT::Zero();
        delta[i] = 1e-6;
        const auto residual = [&](const lio::ImuState& value) -> Eigen::Vector3d
            { return value.R_WI * p_I + value.p_WI - target; };
        const Eigen::Vector3d numerical =
            (residual(lio::ESKF::boxPlus(state, delta)) -
             residual(lio::ESKF::boxPlus(state, -delta))) / 2e-6;
        require((numerical - jacobian.col(i)).norm() < 1e-8,
                "ESKF GICP right-error Jacobian is inconsistent");
    }
}
}

int main()
{
    try { cloudTimes(); scanToMap(); rightEskfJacobian(); std::cout << "PASS: cloud timing/layout, ESKF-owned map, right-perturbation update, pose/bias correction and Jacobian\n"; return 0; }
    catch (const std::exception& error) { std::cerr << "FAIL: " << error.what() << '\n'; return 1; }
}
