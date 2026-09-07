#include "perception/nodes/TrailerLocalization.h"

int main(const int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<TrailerLocalization>("trailer_localization"));
    rclcpp::shutdown();
    return 0;
}

// #include <chrono>
// #include <iomanip>
// #include <iostream>

// #include <pcl/filters/voxel_grid.h>
// #include <pcl/io/pcd_io.h>
// #include <pcl/io/ply_io.h>

// #include "perception/tools/filter_3d.h"

// namespace
// {
// constexpr char kCloudPath[] = "/home/avent/Downloads/happy_stand/happyStandRight_0.ply";
// constexpr char kVoxelGridOutputPath[] = "/home/avent/Downloads/happy_stand/happyStandRight_0_voxel_grid.pcd";
// constexpr char kCustomOutputPath[] = "/home/avent/Downloads/happy_stand/happyStandRight_0_custom.pcd";
// constexpr float kResolution = 0.005f;
// constexpr int kWarmupIterations = 5;
// constexpr int kBenchmarkIterations = 100;

// template<typename Filter>
// double benchmark(Filter&& filter)
// {
// 	for (int iteration = 0; iteration < kWarmupIterations; ++iteration)
// 	{
// 		filter();
// 	}

// 	const auto start = std::chrono::steady_clock::now();
// 	for (int iteration = 0; iteration < kBenchmarkIterations; ++iteration)
// 	{
// 		filter();
// 	}
// 	const auto elapsed = std::chrono::steady_clock::now() - start;

// 	return std::chrono::duration<double, std::milli>(elapsed).count() / kBenchmarkIterations;
// }
// }

// int main()
// {
// 	pcl::PointCloud<pcl::PointXYZ> input_cloud;
// 	if (pcl::io::loadPLYFile(kCloudPath, input_cloud) != 0)
// 	{
// 		std::cerr << "Failed to load " << kCloudPath << '\n';
// 		return 1;
// 	}

// 	pcl::VoxelGrid<pcl::PointXYZ> voxel_grid;
// 	voxel_grid.setInputCloud(input_cloud.makeShared());
// 	voxel_grid.setLeafSize(kResolution, kResolution, kResolution);

// 	pcl::PointCloud<pcl::PointXYZ> voxel_grid_cloud;
// 	const double voxel_grid_time_ms = benchmark([&]
// 	{
// 		voxel_grid.filter(voxel_grid_cloud);
// 	});

// 	pcl::PointCloud<pcl::PointXYZ> custom_cloud;
// 	const double custom_time_ms = benchmark([&]
// 	{
// 		filter3d::downsampleCloud(input_cloud, kResolution, custom_cloud);
// 	});

// 	std::cout << std::fixed << std::setprecision(3);
// 	std::cout << "Input points: " << input_cloud.size() << '\n';
// 	std::cout << "Resolution: " << kResolution << " m\n";
// 	std::cout << "Iterations: " << kBenchmarkIterations << " (after " << kWarmupIterations << " warmup iterations)\n";
// 	std::cout << "pcl::VoxelGrid: " << voxel_grid_time_ms << " ms, " << voxel_grid_cloud.size() << " output points\n";
// 	std::cout << "filter3d::downsampleCloud: " << custom_time_ms << " ms, " << custom_cloud.size() << " output points\n";

// 	if (pcl::io::savePCDFileBinary(kVoxelGridOutputPath, voxel_grid_cloud) != 0 ||
// 	    pcl::io::savePCDFileBinary(kCustomOutputPath, custom_cloud) != 0)
// 	{
// 		std::cerr << "Failed to save downsampled point clouds\n";
// 		return 1;
// 	}
// 	std::cout << "Saved pcl::VoxelGrid output to " << kVoxelGridOutputPath << '\n';
// 	std::cout << "Saved filter3d::downsampleCloud output to " << kCustomOutputPath << '\n';

// 	return 0;
// }