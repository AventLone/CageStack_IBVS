#pragma once
#include "perception/types/cuda_buffer.hpp"
#include <iostream>
#include <opencv2/opencv.hpp>

struct InstanceResult
{
    int query_idx{-1};
    int class_id{-1};
    float score{0.0f};
    cv::Rect bbox;
    cv::Mat mask;
};

class RfDetrSeg
{
    class Logger final : public nvinfer1::ILogger
    {
    public:
        void log(const Severity severity, const char* message) noexcept override
        {
            if (severity <= Severity::kWARNING)
            {
                std::cerr << "[TensorRT] " << message << std::endl;
            }
        }
    };

public:
    explicit RfDetrSeg(const std::string& model_path)
    {
        Logger logger;
        const std::unique_ptr<nvinfer1::IRuntime> runtime{nvinfer1::createInferRuntime(logger)};

        const std::vector<char> model_data = readModel(model_path);
        mCudaEngine = std::unique_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(model_data.data(), model_data.size()));
        if (mCudaEngine == nullptr)
        {
            throw std::runtime_error("Failed to deserialize tensorrt model plan!");
        }

        mContext = std::unique_ptr<nvinfer1::IExecutionContext>(mCudaEngine->createExecutionContext());

        /* Setup cuda buffers */
        for (int i = 0; i < mCudaEngine->getNbIOTensors(); ++i)
        {
            const char* name = mCudaEngine->getIOTensorName(i);
            mCudaBuffers[name] = CudaBuffer(mCudaEngine->getTensorDataType(name), getTensorSize(name));
            mContext->setTensorAddress(name, mCudaBuffers[name].data());
        }
        //
        {
            const auto [_, d] = mCudaEngine->getTensorShape("input");
            mInputSize = cv::Size(static_cast<int>(d[3]), static_cast<int>(d[2]));
        }
        //
        {
            const auto [_, d] = mCudaEngine->getTensorShape("labels");
            mQueriesNum = d[1];
            mClassesNum = d[2];
        }
        //
        {
            const auto [_, d] = mCudaEngine->getTensorShape("masks");
            mMaskSize = cv::Size(static_cast<int>(d[3]), static_cast<int>(d[2]));
        }


        mTensorBBox.reserve(mCudaBuffers["dets"].size());
        mTensorLabels.reserve(mCudaBuffers["labels"].size());
        mTensorInstanceMask.reserve(mCudaBuffers["masks"].size());

        mContext->setInputShape("input", nvinfer1::Dims4{1, 3, 432, 432}); // Set shape of the input if it is dynamic shape
        cudaStreamCreate(&mCudaStream);
    }

    std::vector<InstanceResult> seg(const cv::Mat& img, const float confidence_thresh = 0.5f, const float mask_thresh = 0.5f)
    {
        if (!infer(img))
        {
            return {};
        }

        return postprocess(img, confidence_thresh, mask_thresh);
    }

private:
    cudaStream_t mCudaStream{};
    std::unique_ptr<nvinfer1::ICudaEngine> mCudaEngine;
    std::unique_ptr<nvinfer1::IExecutionContext> mContext;

    /* I/O shapes */
    cv::Size mInputSize, mMaskSize;
    size_t mQueriesNum, mClassesNum;

    std::unordered_map<std::string, CudaBuffer> mCudaBuffers;
    std::vector<float> mTensorBBox, mTensorLabels, mTensorInstanceMask; // Outputs of the model

    static std::vector<char> readModel(const std::string& file_path);

    size_t getTensorSize(const char* name) const
    {
        const auto [nbDims, d] = mCudaEngine->getTensorShape(name);
        size_t size = 1;
        for (int i = 0; i < nbDims; ++i)
        {
            size *= d[i];
        }
        return size;
    }

    bool infer(const cv::Mat& input_image);

    std::vector<InstanceResult> postprocess(const cv::Mat& original_image,
                                            float score_thresh = 0.5f,
                                            float mask_thresh = 0.5f) const;
};

inline void visualizeInstanceSeg(const cv::Mat& original_image, const std::vector<InstanceResult>& results, cv::Mat& output_image)
{
    static cv::RNG rng(123456);

    output_image = original_image.clone();

    for (const auto& result : results)
    {
        cv::Scalar color(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));

        // mask overlay
        for (int y = 0; y < output_image.rows; ++y)
        {
            const auto* mask_row = result.mask.ptr<uchar>(y);
            auto* img_row = output_image.ptr<cv::Vec3b>(y);

            for (int x = 0; x < output_image.cols; ++x)
            {
                if (mask_row[x])
                {
                    for (int c = 0; c < 3; ++c)
                    {
                        img_row[x][c] = static_cast<uchar>(img_row[x][c] * 0.5 + color[c] * 0.5);
                    }
                }
            }
        }

        cv::rectangle(output_image, result.bbox, color, 2);

        char text[128];
        std::snprintf(text, sizeof(text), "cls:%d score:%.2f", result.class_id, result.score);

        int base_line = 0;
        const cv::Size text_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &base_line);

        const int tx = std::max(0, result.bbox.x);
        const int ty = std::max(text_size.height + 2, result.bbox.y);

        cv::rectangle(output_image, cv::Rect(tx, ty - text_size.height - 2, text_size.width + 4, text_size.height + 4),
                      color, cv::FILLED);
        cv::putText(output_image, text, cv::Point(tx + 2, ty), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    }
}

