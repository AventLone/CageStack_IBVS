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

// custom point type with semantic label
struct InstancePoint
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW // ensure proper alignment
    PCL_ADD_POINT4D; // adds float x,y,z and padding (w)
    uint16_t label; // semantic label (choose uint8_t / uint32_t as needed)
    uint16_t instance;

    InstancePoint() = default;

    explicit InstancePoint(const float x, const float y, const float z,
                           const uint16_t label, const uint16_t instance)
    {
        this->x = x;
        this->y = y;
        this->z = z;
        this->label = label;
        this->instance = instance;
    }
} EIGEN_ALIGN16; // align to 16 bytes

// Register the point struct so PCL can handle it in I/O, cloud operations, etc.
POINT_CLOUD_REGISTER_POINT_STRUCT(InstancePoint,
                                  (float, x, x)
                                  (float, y, y)
                                  (float, z, z)
                                  (uint16_t, label, label)
                                  (uint16_t, instance, instance))
