// src/kernels/cuda/route_prior.cu - residency-biased routing, adapted from Strata's route_prior.cu (MIT, Niko1221 and
// the Strata contributors). When a token's 10 experts are picked, an expert already in the card's cache gets a
// `lambda` bonus on its logit; the weights stay the router's softmax over the 512 unbiased logits, renormalised over
// the 10 picked. Fewer missed experts for the CPU tier or the link, at the cost of a different expert set.
#include "basalt/kernels/route_prior.hpp"

#include <cuda_runtime.h>

#include <cfloat>
#include <cmath>
#include <cstdlib>

namespace basalt::kernels {
namespace {

__device__ __forceinline__ float wmax(float v) {
#pragma unroll
    for (int m = 16; m; m >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, m, 32));
    return v;
}

// one warp per token; 512 experts = 16 values per lane
__global__ void route_prior_k(const float* __restrict__ logits, const int32_t* __restrict__ res, float lambda,
                              int32_t* __restrict__ ids, float* __restrict__ weights) {
    const float* lg = logits + (size_t) blockIdx.x * 512;
    int32_t* id = ids + (size_t) blockIdx.x * 10;
    float* w = weights + (size_t) blockIdx.x * 10;
    const int lane = threadIdx.x;
    float p[16], s[16];
    float mx = -INFINITY;
#pragma unroll
    for (int i = 0; i < 16; ++i) { p[i] = lg[lane + 32 * i]; mx = fmaxf(mx, p[i]); }
    mx = wmax(mx);
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        const int e = lane + 32 * i;
        s[i] = p[i] + (res[e] >= 0 ? lambda : 0.0f);   // the selection score
        p[i] = expf(p[i] - mx);                         // the router's probability, unnormalised
    }
    float sel_p = 0.0f, sel_sum = 0.0f;
    for (int rank = 0; rank < 10; ++rank) {
        float best = -FLT_MAX, bp = 0.0f;
        int be = 1 << 30;
#pragma unroll
        for (int i = 0; i < 16; ++i)
            if (s[i] > best) { best = s[i]; be = lane + 32 * i; bp = p[i]; }
#pragma unroll
        for (int m = 16; m; m >>= 1) {
            const float ob = __shfl_xor_sync(0xffffffffu, best, m, 32);
            const int oe = __shfl_xor_sync(0xffffffffu, be, m, 32);
            const float op = __shfl_xor_sync(0xffffffffu, bp, m, 32);
            if (ob > best || (ob == best && oe < be)) { best = ob; be = oe; bp = op; }
        }
        if ((be & 31) == lane) s[be >> 5] = -FLT_MAX;
        if (lane == rank) { id[rank] = be; sel_p = bp; }
        sel_sum += bp;   // the same on every lane
    }
    if (lane < 10) w[lane] = sel_p / fmaxf(sel_sum, 1e-30f);
}

}  // namespace

void route_prior_top10(const float* logits, const int32_t* res_layer, float lambda, int32_t* ids, float* weights,
                       int n_tok, void* stream) {
    if (n_tok <= 0) return;
    route_prior_k<<<n_tok, 32, 0, (cudaStream_t) stream>>>(logits, res_layer, lambda, ids, weights);
}

float route_prior_lambda() {
    static const float v = [] {
        const char* s = std::getenv("BASALT_ROUTE_PRIOR");
        return s != nullptr ? (float) std::atof(s) : 0.0f;
    }();
    return v;
}

}  // namespace basalt::kernels
