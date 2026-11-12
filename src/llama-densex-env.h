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
    int         kv_vmm;            // DENSEX_KV_VMM:            unset -> 0 (off); =1 -> the offloaded KV cache is
                                   //                           allocated on persistent CUDA VMM buffers (issue #7)
    int         kv_lazy;           // DENSEX_KV_LAZY:           unset -> 0 (off); =1 -> those buffers keep only their
                                   //                           address space and map physical pages as the KV uses
                                   //                           them (issue #8); needs DENSEX_KV_VMM=1
    int64_t     kv_lazy_cells;     // DENSEX_KV_LAZY_CELLS:     unset -> 8192; the cells mapped at once, both for the
                                   //                           first map and for every later expansion (issue #8)
};

const llama_densex_env & llama_densex_env_get();
