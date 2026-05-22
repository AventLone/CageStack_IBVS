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
    cv::dnn::blobFromImage(input_image, input_tensor, 1.0 / 255.0, mInputSize);
    // const std::vector<float> input_tensor = preprocess(input_image, mInputSize, {0.485, 0.456, 0.406}, {0.229, 0.224, 0.225});
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

std::vector<float> RfDetrSeg::preprocess(const cv::Mat& image, const cv::Size& input_size,
                                         const cv::Scalar& mean, const cv::Scalar& std_,
                                         const bool swap_rb)
{
    cv::Mat resized_img;
    cv::resize(image, resized_img, input_size);

    // 2. 将 BGR 转换为 RGB 并转为 float 格式 (一步到位，且缩放到 [0, 1])
    cv::Mat rgb_fp32;
    if (swap_rb)
    {
        cv::cvtColor(resized_img, rgb_fp32, cv::COLOR_BGR2RGB);
    }
    rgb_fp32.convertTo(rgb_fp32, CV_32FC3, 1.0 / 255.0);

    // 3. 核心：分通道高效处理不同 Mean 和 Std
    const cv::Scalar alpha(1.0 / std_[0], 1.0 / std_[1], 1.0 / std_[2]); // 对应 1/std
    const cv::Scalar beta(-mean[0] / std_[1], -mean[1] / std_[2], -mean[2] / std_[3]); // 对应 -mean/std

    // 一行矩阵乘加操作，底层直接触发 CPU 的 AVX2 / NEON 向量化加速
    cv::multiply(rgb_fp32, alpha, rgb_fp32);
    cv::add(rgb_fp32, beta, rgb_fp32);

    // 4. 将 HWC 转换为 NCHW
    // 我们必须将排布好的 RRR...GGG...BBB... 存入一个连续的临时 Host 内存中
    const int plane_size = input_size.area();
    std::vector<cv::Mat> chw_planes(3);

    // 巧妙利用 cv::Mat 构造函数，让 chw_planes 直接指向一片连续内存（无二次拷贝！）
    std::vector<float> host_nchw_buffer(plane_size * 3);
    chw_planes[0] = cv::Mat(input_size, CV_32FC1, host_nchw_buffer.data());
    chw_planes[1] = cv::Mat(input_size, CV_32FC1, host_nchw_buffer.data() + plane_size);
    chw_planes[2] = cv::Mat(input_size, CV_32FC1, host_nchw_buffer.data() + plane_size * 2);

    // split 会把 rgb_fp32 的 HWC 数据打散，并极其高效地写入到我们指定的连续 NCHW 内存中
    cv::split(rgb_fp32, chw_planes);
    return host_nchw_buffer;
}

std::vector<Instance> RfDetrSeg::postprocess(const cv::Mat& original_image,
                                             const float score_thresh,
                                             const float mask_thresh,
                                             const int choose_the_bset_count) const
{
    const int img_width = original_image.cols;
    const int img_height = original_image.rows;

    std::vector<Instance> results;
    results.reserve(mQueriesNum / 2);

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
        if (box.width <= 6 || box.height <= 6 || box.area() < 100)
        {
            continue;
        }

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

        Instance result;
        result.class_id = class_id;
        result.score = score;
        result.bbox = box;
        result.mask = mask_bin;
        results.push_back(std::move(result));
    }

    std::sort(results.begin(), results.end(), [](const Instance& a, const Instance& b)
                  {
                      return a.score > b.score;
                  });

    if (choose_the_bset_count == -1 or choose_the_bset_count > results.size())
    {
        return results;
    }

    return {std::make_move_iterator(results.begin()), std::make_move_iterator(results.begin() + choose_the_bset_count)};
}
