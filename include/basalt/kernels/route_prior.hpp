// include/basalt/kernels/route_prior.hpp - residency-biased routing (local experiment, changes the output).
#pragma once

#include <cstdint>

namespace basalt::kernels {

/// Top-10 per token over `logits` (n_tok x 512) where an expert resident in this card's cache (`res_layer[e] >= 0`)
/// gets `lambda` added to its logit for the selection only; the weights are the router's probabilities (softmax over
/// the 512 unbiased logits) renormalised over the 10 picked.
void route_prior_top10(const float* logits, const int32_t* res_layer, float lambda, int32_t* ids, float* weights,
                       int n_tok, void* stream);

/// BASALT_ROUTE_PRIOR=lambda; 0 (off) when unset.
float route_prior_lambda();

}  // namespace basalt::kernels
