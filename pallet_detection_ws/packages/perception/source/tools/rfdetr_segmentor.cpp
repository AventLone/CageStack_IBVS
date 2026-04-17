#include "perception/tools/rfdetr_segmentor.h"
#include <fstream>
#include "perception/tools/math.hpp"

static cv::Rect clamp_rect(const cv::Rect& r, const int width, const int height)
{
    const cv::Rect img_rect(0, 0, width, height);
    return r & img_rect;
}

std::vector<char> RfDetrSeg::readModel(const std::string& file_path)
{
    std::ifstream file(file_path, std::ios::binary | std::ios::ate);
    if (!file.is_open())
    {
        throw std::runtime_error("Failed to open file: " + file_path);
    }
    const std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg); // 将读指针从文件末尾开始移动0个字节
    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), size))
    {
        throw std::runtime_error("Failed to read file: " + file_path);
    }
    return buffer;
}

bool RfDetrSeg::infer(const cv::Mat& input_image)
{
    /* Preprocess of the input. */
    cv::Mat input_tensor;
    cv::dnn::blobFromImage(input_image, input_tensor, 1.0 / 255.0, mInputSize, {0.0, 0.0, 0.0}, true, false);
    cudaMemcpyAsync(mCudaBuffers["input"].data(), input_tensor.ptr<float>(),
                    mCudaBuffers["input"].bytes(), cudaMemcpyHostToDevice, mCudaStream);

    if (!mContext->enqueueV3(mCudaStream))
    {
        return false;
    }
    cudaMemcpyAsync(mTensorBBox.data(), mCudaBuffers["dets"].data(),
                    mCudaBuffers["dets"].bytes(), cudaMemcpyDeviceToHost, mCudaStream);
    cudaMemcpyAsync(mTensorLabels.data(), mCudaBuffers["labels"].data(),
                    mCudaBuffers["labels"].bytes(), cudaMemcpyDeviceToHost, mCudaStream);
    cudaMemcpyAsync(mTensorInstanceMask.data(), mCudaBuffers["masks"].data(),
                    mCudaBuffers["masks"].bytes(), cudaMemcpyDeviceToHost, mCudaStream);
    if (cudaStreamSynchronize(mCudaStream) != cudaSuccess)
    {
        return false;
    }
    return true;
}

std::vector<InstanceResult> RfDetrSeg::postprocess(const cv::Mat& original_image,
                                                   const float score_thresh,
                                                   const float mask_thresh) const
{
    const int img_width = original_image.cols;
    const int img_height = original_image.rows;

    std::vector<InstanceResult> results;
    results.reserve(mQueriesNum);

    for (size_t q = 0; q < mQueriesNum; ++q)
    {
        const float* cls_logits = mTensorLabels.data() + q * mClassesNum;
        std::vector<float> cls_probs = softmax(cls_logits, mClassesNum);

        const auto it = std::max_element(cls_probs.begin(), cls_probs.end());
        int class_id = static_cast<int>(std::distance(cls_probs.begin(), it));
        const float score = *it;

        if (score < score_thresh)
        {
            continue;
        }

        const float cx = mTensorBBox[q * 4 + 0];
        const float cy = mTensorBBox[q * 4 + 1];
        const float w = mTensorBBox[q * 4 + 2];
        const float h = mTensorBBox[q * 4 + 3];

        // dets 是归一化坐标
        int x1 = static_cast<int>((cx - w * 0.5f) * static_cast<float>(img_width));
        int y1 = static_cast<int>((cy - h * 0.5f) * static_cast<float>(img_height));
        int x2 = static_cast<int>((cx + w * 0.5f) * static_cast<float>(img_width));
        int y2 = static_cast<int>((cy + h * 0.5f) * static_cast<float>(img_height));

        cv::Rect box = clamp_rect(cv::Rect(x1, y1, x2 - x1, y2 - y1), img_width, img_height);
        if (box.width <= 0 || box.height <= 0)
        {
            continue;
        }

        // 取出 108 x 108 mask logits
        cv::Mat mask_small(mMaskSize, CV_32F);
        const float* mask_ptr = mTensorInstanceMask.data() + q * mMaskSize.width * mMaskSize.height;

        for (int y = 0; y < mMaskSize.height; ++y)
        {
            for (int x = 0; x < mMaskSize.width; ++x)
            {
                mask_small.at<float>(y, x) = sigmoid(mask_ptr[y * mMaskSize.height + x]);
            }
        }

        // resize 回原图大小
        cv::Mat mask_big;
        cv::resize(mask_small, mask_big, cv::Size(img_width, img_height), 0, 0, cv::INTER_LINEAR);

        // 二值化
        cv::Mat mask_bin;
        cv::threshold(mask_big, mask_bin, mask_thresh, 255, cv::THRESH_BINARY);
        mask_bin.convertTo(mask_bin, CV_8U);

        // 可选：只保留 bbox 内部
        cv::Mat box_mask = cv::Mat::zeros(mask_bin.size(), CV_8U);
        mask_bin(box).copyTo(box_mask(box));
        mask_bin = box_mask;

        InstanceResult result;
        result.query_idx = static_cast<int>(q);
        result.class_id = class_id;
        result.score = score;
        result.bbox = box;
        result.mask = mask_bin;
        results.push_back(std::move(result));
    }

    // std::sort(results.begin(), results.end(),
    //           [](const InstanceResult& a, const InstanceResult& b)
    //               {
    //                   return a.score > b.score;
    //               });

    return results;
}
