#pragma once

#include <cstdint>
#include <string>

// DenseX: the llama-side DENSEX_* switches. Read once, here, by llama_densex_env_get() (implemented in
// llama-densex-env.cpp) and reported once on the first call. The CUDA backend has its own equivalent
// struct (ggml/src/ggml-cuda/densex-env.cuh); llama and ggml environment layers are independent.
struct llama_densex_env {
    int         gdn_deferred;      // DENSEX_GDN_DEFERRED:      unset -> -1 (on for the validated arch with a single
                                   //                           sequence); =1 -> also the other GDN archs; =0 -> off
    std::string draft_vocab_order; // DENSEX_DRAFT_VOCAB_ORDER: <tsv path>; empty means unset or an empty value
    int64_t     draft_vocab;       // DENSEX_DRAFT_VOCAB:       unset -> 0; otherwise atoll(env)
};

const llama_densex_env & llama_densex_env_get();
