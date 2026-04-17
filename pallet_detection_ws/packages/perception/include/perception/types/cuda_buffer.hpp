#pragma once
#include <string>
#include <vector>
#include <memory>
#include <numeric>
#include <unordered_map>
#include <NvInfer.h>
#include <stdexcept>

constexpr uint32_t getElementSize(const nvinfer1::DataType data_type)
{
    switch (data_type)
    {
        case nvinfer1::DataType::kINT64:
            return sizeof(int64_t);
        case nvinfer1::DataType::kINT32:
        case nvinfer1::DataType::kFLOAT:
            return sizeof(float);
        case nvinfer1::DataType::kBF16:
        case nvinfer1::DataType::kHALF:
            return sizeof(int16_t);
        case nvinfer1::DataType::kBOOL:
        case nvinfer1::DataType::kUINT8:
        case nvinfer1::DataType::kINT8:
        case nvinfer1::DataType::kFP8:
            return sizeof(int8_t);
        default:
            throw std::invalid_argument("Data type of the tensor doesn't surport!");
    }
}

class CudaBuffer
{
public:
    /**
     * @brief Construct an empty buffer.
     */
    explicit CudaBuffer(const nvinfer1::DataType type = nvinfer1::DataType::kFLOAT) : mType(type), mSize(0), mBytes(0)
    {}

    /**
     * @brief Construct a buffer with the specified allocation size in bytes.
     */
    explicit CudaBuffer(const nvinfer1::DataType type, const size_t size) : mType(type), mSize(size)
    {
        mBytes = size * getElementSize(type);
        if (cudaMalloc(&mBuffer, mBytes) != cudaSuccess)
        {
            throw std::bad_alloc();
        }
    }

    explicit CudaBuffer(const nvinfer1::DataType type, const nvinfer1::Dims& dimention) : mType(type)
    {
        // Get the product of all elements in the array using `std::accumulate`
        mSize = std::accumulate(dimention.d, dimention.d + dimention.nbDims, static_cast<int64_t>(1), std::multiplies<>());
        mBytes = mSize * getElementSize(mType);
        if (cudaMalloc(&mBuffer, bytes()) != cudaSuccess)
        {
            throw std::bad_alloc();
        }
    }

    CudaBuffer(CudaBuffer&& buf) noexcept : mType(buf.mType), mSize(buf.mSize), mBytes(buf.mBytes), mBuffer(buf.mBuffer)
    {
        buf.mType = nvinfer1::DataType::kFLOAT;
        buf.mSize = 0;
        buf.mBytes = 0;
        buf.mBuffer = nullptr;
    }

    CudaBuffer& operator=(CudaBuffer&& buf) noexcept
    {
        if (this != &buf)
        {
            cudaFree(mBuffer);
            mType = buf.mType;
            mSize = buf.mSize;
            mBytes = buf.mBytes;
            mBuffer = buf.mBuffer;

            // Reset `buf`
            buf.mSize = 0;
            buf.mBytes = 0;
            buf.mBuffer = nullptr;
        }
        return *this;
    }

    ~CudaBuffer()
    {
        if (mBuffer != nullptr)
        {
            cudaFree(mBuffer);
        }
    }

    /**
     * @brief Returns pointer to underlying array.
     */
    void* data()
    {
        return mBuffer;
    }

    /**
     * @brief Returns pointer to underlying array.
     */
    const void* data() const
    {
        return mBuffer;
    }

    /**
     * @brief Returns the size (in number of elements) of the buffer.
     */
    size_t size() const
    {
        return mSize;
    }

    /**
     * @brief Returns the size (in bytes) of the buffer.
     */
    size_t bytes() const
    {
        return mBytes;
    }

private:
    nvinfer1::DataType mType;
    size_t mSize;
    size_t mBytes;
    void* mBuffer{nullptr};
};
