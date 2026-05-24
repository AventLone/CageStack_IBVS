#pragma once
#include "perception/types/cuda_buffer.hpp"
#include <iostream>
#include <opencv2/opencv.hpp>

struct Instance
{
    int class_id{-1};
    float score{0.0f};
    cv::Rect bbox;

    explicit Instance(const int _class_id, const float _score, const cv::Rect _bbox)
        : class_id(_class_id), score(_score), bbox(_bbox)
    {}
};

class RfDetrDetector
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
    explicit RfDetrDetector(const std::string& model_path)
    {
        cudaSetDeviceFlags(cudaDeviceScheduleYield);

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

        mTensorBBox.reserve(mCudaBuffers["dets"].size());
        mTensorLabels.reserve(mCudaBuffers["labels"].size());

        cudaStreamCreate(&mCudaStream);
    }

    std::vector<Instance> detect(const cv::Mat& img,
                                 const float confidence_thresh = 0.5f,
                                 const int choose_the_best = -1)
    {
        if (!infer(img))
        {
            return {};
        }

        return postprocess(img, confidence_thresh, choose_the_best);
    }

private:
    cudaStream_t mCudaStream{};
    std::unique_ptr<nvinfer1::ICudaEngine> mCudaEngine;
    std::unique_ptr<nvinfer1::IExecutionContext> mContext;

    /* I/O shapes */
    cv::Size mInputSize, mMaskSize;
    size_t mQueriesNum, mClassesNum;

    std::unordered_map<std::string, CudaBuffer> mCudaBuffers;
    std::vector<float> mTensorBBox, mTensorLabels; // Outputs of the model

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

    static std::vector<float> preprocess(const cv::Mat& image, const cv::Size& input_size,
                                         const cv::Scalar& mean, const cv::Scalar& std_,
                                         bool swap_rb = false);

    std::vector<Instance> postprocess(const cv::Mat& original_image,
                                      float score_thresh = 0.5f,
                                      int choose_the_bset_count = -1) const;
};

inline void visualizeInstanceSeg(const cv::Mat& original_image, cv::Mat& output_image,
                                 const std::vector<Instance>& results,
                                 const std::unordered_map<int, std::string>& label_dict)
{
    // static cv::RNG rng(123456);
    const static auto get_random_color = []() -> cv::Scalar_<uint8_t>
        {
            static cv::RNG rng(66);
            return cv::Scalar(rng.uniform(66, 256), rng.uniform(66, 256), rng.uniform(66, 256));
        };

    const static auto get_color = [](int id) -> cv::Scalar_<uint8_t>
        {
            static std::unordered_map<int, cv::Scalar_<uint8_t>> color_map; // map label -> (r,g,b)
            cv::Scalar_<uint8_t> color;
            if (const auto it = color_map.find(id); it == color_map.end())
            {
                color = get_random_color();

                color_map.emplace(id, color);
            }
            else
            {
                color = it->second;
            }
            return color;
        };

    output_image = original_image.clone();

    for (const auto& [class_id, score, bbox] : results)
    {
        cv::Scalar color = get_color(class_id);

        cv::rectangle(output_image, bbox, color, 2);

        std::ostringstream oss;
        oss << label_dict.at(class_id) << ": " << std::fixed << std::setprecision(2) << score;

        int base_line = 0;
        const cv::Size text_size = cv::getTextSize(oss.str(), cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &base_line);

        const int tx = std::max(0, bbox.x);
        const int ty = std::max(text_size.height + 2, bbox.y);

        cv::rectangle(output_image, cv::Rect(tx, ty - text_size.height - 2, text_size.width + 4, text_size.height + 4),
                      color, cv::FILLED);
        cv::putText(output_image, oss.str(), cv::Point(tx + 2, ty), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    }
}
