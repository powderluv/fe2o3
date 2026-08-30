#include <hip/hip_runtime.h>

#include <cstdint>

namespace {

constexpr int kHidden = 2880;
constexpr int kExperts = 128;
constexpr int kRows = 16;
constexpr int kContext = 16;
constexpr int kHeadDim = 64;
constexpr int kValue = 16;
constexpr int kBlocks = 4;
constexpr int kExpertK = 128;
constexpr int kExpertN = 16;
constexpr int kBlock = 32;
constexpr float kAttentionScale = 0.125f;
constexpr float kRouterFloor = -1.0e30f;

__device__ __forceinline__ float decode_bf16(std::uint16_t bits) {
    return __uint_as_float(static_cast<unsigned>(bits) << 16);
}

__device__ __forceinline__ float decode_fp4(std::uint8_t bits) {
    float magnitude;
    switch (bits & 7) {
    case 0: magnitude = 0.0f; break;
    case 1: magnitude = 0.5f; break;
    case 2: magnitude = 1.0f; break;
    case 3: magnitude = 1.5f; break;
    case 4: magnitude = 2.0f; break;
    case 5: magnitude = 3.0f; break;
    case 6: magnitude = 4.0f; break;
    default: magnitude = 6.0f; break;
    }
    return (bits & 8) == 0 ? magnitude : -magnitude;
}

__device__ __forceinline__ void insert_candidate(
    float score,
    unsigned id,
    float (&best)[4],
    unsigned (&ids)[4]) {
    for (int rank = 0; rank < 4; ++rank) {
        const bool take = score > best[rank] || (score == best[rank] && id < ids[rank]);
        if (take) {
            const float old_score = best[rank];
            const unsigned old_id = ids[rank];
            best[rank] = score;
            ids[rank] = id;
            score = old_score;
            id = old_id;
        }
    }
}

} // namespace

extern "C" __global__ __launch_bounds__(64)
void gpt_oss_unfused_router(
    const float* hidden,
    const float* router,
    unsigned* packed_top4) {
    const unsigned lane = threadIdx.x;
    const unsigned expert0 = lane * 2;
    const unsigned expert1 = expert0 + 1;
    float score0 = 0.0f;
    float score1 = 0.0f;
    for (int depth = 0; depth < kHidden; ++depth) {
        const float activation = hidden[depth];
        score0 += activation * router[expert0 * kHidden + depth];
        score1 += activation * router[expert1 * kHidden + depth];
    }
    float best[4] = {kRouterFloor, kRouterFloor, kRouterFloor, kRouterFloor};
    unsigned ids[4] = {~0u, ~0u, ~0u, ~0u};
    for (unsigned source = 0; source < 64; ++source) {
        insert_candidate(__shfl(score0, source, 64), source * 2, best, ids);
        insert_candidate(__shfl(score1, source, 64), source * 2 + 1, best, ids);
    }
    if (lane == 0) {
        packed_top4[0] = ids[0] | (ids[1] << 7) | (ids[2] << 14) | (ids[3] << 21);
    }
}

extern "C" __global__ __launch_bounds__(64)
void gpt_oss_unfused_attention(
    const std::uint16_t* query,
    const std::uint16_t* key_transposed,
    const float* value,
    const float* sinks,
    float* output) {
    const unsigned lane = threadIdx.x;
    const int column = lane & 15;
    const int row_group = lane >> 4;
    for (int component = 0; component < 4; ++component) {
        const int row = row_group * 4 + component;
        float scores[kContext];
        float maximum = sinks[row];
        for (int token = 0; token < kContext; ++token) {
            float score = 0.0f;
            for (int depth = 0; depth < kHeadDim; ++depth) {
                score += decode_bf16(query[row * kHeadDim + depth])
                    * decode_bf16(key_transposed[depth * kContext + token]);
            }
            score *= kAttentionScale;
            scores[token] = score;
            maximum = fmaxf(maximum, score);
        }
        float denominator = expf(sinks[row] - maximum);
        for (int token = 0; token < kContext; ++token) {
            scores[token] = expf(scores[token] - maximum);
            denominator += scores[token];
        }
        float result = 0.0f;
        for (int token = 0; token < kContext; ++token) {
            result += scores[token] / denominator * value[token * kValue + column];
        }
        output[component * 64 + lane] = result;
    }
}

extern "C" __global__ __launch_bounds__(64)
void gpt_oss_unfused_expert(
    const std::uint8_t* activation,
    const std::uint8_t* weights,
    const float* activation_scales,
    const float* weight_scales,
    const unsigned* packed_top4,
    float* output) {
    const unsigned lane = threadIdx.x;
    const int column = lane & 15;
    const int row_group = lane >> 4;
    const int selected = packed_top4[0] & (kExperts - 1);
    constexpr int expert_stride = kBlocks * kExpertK * kExpertN;
    constexpr int block_stride = kExpertK * kExpertN;
    for (int component = 0; component < 4; ++component) {
        const int row = row_group * 4 + component;
        float result = 0.0f;
        if (row == 0) {
            for (int block = 0; block < kBlocks; ++block) {
                const int activation_base = block * kRows * kExpertK;
                const int weight_base =
                    selected * expert_stride + block * block_stride;
                float dot = 0.0f;
                for (int depth = block * kBlock; depth < (block + 1) * kBlock; ++depth) {
                    dot += decode_fp4(activation[activation_base + depth])
                        * decode_fp4(weights[weight_base + depth * kExpertN + column]);
                }
                result += dot * activation_scales[block]
                    * weight_scales[(selected * kBlocks + block) * kExpertN + column];
            }
        }
        output[component * 64 + lane] = result;
    }
}
