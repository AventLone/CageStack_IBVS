#pragma once
#include <pcl/point_types.h>
#include <pcl/register_point_struct.h>
#include <pcl/point_cloud.h>

// custom point type with semantic label
struct SemanticPoint
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW // ensure proper alignment
    PCL_ADD_POINT4D; // adds float x,y,z and padding (w)
    int label; // semantic label (choose uint8_t / uint32_t as needed)
    SemanticPoint() = default;

    explicit SemanticPoint(const float x, const float y, const float z, const int label)
    {
        this->x = x;
        this->y = y;
        this->z = z;
        this->label = label;
    }
} EIGEN_ALIGN16; // align to 16 bytes

// Register the point struct so PCL can handle it in I/O, cloud operations, etc.
POINT_CLOUD_REGISTER_POINT_STRUCT(SemanticPoint,
                                  (float, x, x)
                                  (float, y, y)
                                  (float, z, z)
                                  (int, label, label))
