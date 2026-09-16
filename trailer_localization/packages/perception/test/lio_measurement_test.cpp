#include "perception/LIO/GicpMeasurement.h"
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
    SparseVoxel::Config map_config;
    map_config.voxel_size = 0.5f;
    map_config.max_points_per_voxel = 100;
    map_config.estimate_covariances = true;
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
    SparseVoxel map(map_config);
    map.initialize(world);
    const Sophus::SE3d T_IL(Sophus::SO3d::exp(Eigen::Vector3d(0.05, 0.1, -0.1)), Eigen::Vector3d(0.2, -0.1, 0.05));
    pcl::PointCloud<pcl::PointXYZ> scan;
    for (const auto& p : world)
    {
        const Eigen::Vector3d p_L = T_IL.inverse() * Eigen::Vector3d(p.x, p.y, p.z);
        scan.emplace_back(static_cast<float>(p_L.x()), static_cast<float>(p_L.y()), static_cast<float>(p_L.z()));
    }
    lio::ESKF filter(lio::ESKF::Config{}, T_IL);
    lio::ImuState prior;
    prior.p_WI = Eigen::Vector3d(0.04, -0.03, 0.02);
    prior.R_WI = Sophus::SO3d::exp(Eigen::Vector3d(0.01, -0.015, 0.02));
    lio::StateCovariance covariance = lio::StateCovariance::Identity() * 0.1;
    covariance(5, 11) = covariance(11, 5) = -0.02;
    filter.reset(prior, covariance);
    lio::GicpMeasurement::Config measurement_config;
    measurement_config.voxel_size = map_config.voxel_size;
    const lio::GicpMeasurement builder(measurement_config, T_IL);
    const auto source = builder.prepareScan(scan);
    const auto result = filter.update([&](const auto& state) { return builder.build(state, *source, map); });
    require(result.accepted && result.num_correspondences > 500 && result.fitness_score < 1e-6,
            "Three-plane tightly coupled GICP update failed");
    require(filter.nominalState().p_WI.norm() < 0.001 && filter.nominalState().R_WI.log().norm() < 0.001,
            "Failed to recover IMU pose with non-identity LiDAR extrinsics");
    require(filter.nominalState().gyro_bias.norm() > 0.001, "Geometric constraints did not correct gyro bias");
    pcl::PointCloud<pcl::PointXYZ> empty;
    const auto empty_source = builder.prepareScan(empty);
    require(builder.build(prior, *empty_source, map).count == 0, "Empty geometry accepted");

    auto far_state = prior;
    far_state.p_WI = Eigen::Vector3d(10.0, 0.0, 0.0);
    require(builder.build(far_state, *source, map).count == 0, "GICP correspondence distance gate failed");
}

void gicpEskfJacobian()
{
    lio::GicpMeasurement::Config config;
    config.voxel_size = 1.0f;
    config.max_correspondence_distance = 2.0f;
    config.cauchy_kernel_scale = 0.0f;
    config.lidar_noise = 1.0;
    const Sophus::SE3d T_IL(Sophus::SO3d::exp(Eigen::Vector3d(-0.1, 0.2, 0.05)),
                            Eigen::Vector3d(0.2, -0.1, 0.05));
    const lio::GicpMeasurement builder(config, T_IL);
    pcl::PointCloud<pcl::PointXYZ> scan_cloud;
    scan_cloud.emplace_back(0.4f, -0.2f, 0.3f);
    pcl::PointCloud<pcl::PointXYZ> map_cloud;
    map_cloud.emplace_back(0.9f, -0.1f, 0.4f);
    const auto scan = builder.prepareScan(scan_cloud);
    SparseVoxel::Config map_config;
    map_config.voxel_size = config.voxel_size;
    SparseVoxel map(map_config);
    map.initialize(map_cloud);
    lio::ImuState state;
    state.p_WI = Eigen::Vector3d(0.1, -0.05, 0.02);
    state.R_WI = Sophus::SO3d::exp(Eigen::Vector3d(0.2, -0.1, 0.15));
    const auto measurement = builder.build(state, *scan, map);
    require(measurement.count == 1, "Synthetic GICP correspondence missing");

    const Eigen::Vector3d p_I = T_IL * Eigen::Vector3d(0.4, -0.2, 0.3);
    const Eigen::Vector3d residual = state.R_WI * p_I + state.p_WI -
                                     Eigen::Vector3d(0.9, -0.1, 0.4);
    Eigen::Matrix<double, 3, 6> jacobian;
    jacobian.leftCols<3>().setIdentity();
    jacobian.rightCols<3>() = -state.R_WI.matrix() * Sophus::SO3d::hat(p_I);
    require((measurement.gradient - jacobian.transpose() * residual).norm() < 1e-5,
            "GICP right increment was mapped incorrectly into the ESKF right-error state");
}
}

int main()
{
    try { cloudTimes(); scanToMap(); gicpEskfJacobian(); std::cout << "PASS: cloud timing/layout, shared right-perturbation GICP update, pose/bias correction and frame Jacobian\n"; return 0; }
    catch (const std::exception& error) { std::cerr << "FAIL: " << error.what() << '\n'; return 1; }
}
