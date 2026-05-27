#pragma once
#include <cmath>
#include <vector>

inline float sigmoid(const float x)
{
    return 1.0f / (1.0f + std::exp(-x));
}

inline std::vector<float> softmax(const float* logits, const size_t n)
{
    float max_v = logits[0];
    for (size_t i = 1; i < n; ++i)
    {
        max_v = std::max(max_v, logits[i]);
    }

    std::vector<float> probs(n);
    float sum = 0.0f;
    for (size_t i = 0; i < n; ++i)
    {
        probs[i] = std::exp(logits[i] - max_v);
        sum += probs[i];
    }

    if (sum > 0.0f)
    {
        for (size_t i = 0; i < n; ++i)
        {
            probs[i] /= sum;
        }
    }

    return probs;
}
