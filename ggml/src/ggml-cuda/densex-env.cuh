#pragma once

// DenseX: every DENSEX_* CUDA backend environment switch is read once, here, and reported once on first use.
// The implementation lives in ggml-cuda.cu. This header must stay free of llama.cpp includes: ggml does not
// depend on llama and the two environment layers are independent (llama layer: src/llama-densex-env.h).
struct ggml_cuda_densex_env {
    bool fa_decode_mma; // DENSEX_FA_DECODE_MMA: unset -> true, otherwise atoi(env) != 0
    bool mmq_slots;     // DENSEX_MMQ_SLOTS:     unset -> true, otherwise atoi(env) != 0
    int  mmvq_max;      // DENSEX_MMVQ_MAX:      unset -> -1, otherwise atoi(env)
    bool graph_verify;  // DENSEX_GRAPH_VERIFY:  unset -> false, otherwise atoi(env) != 0
};

const ggml_cuda_densex_env & ggml_cuda_densex_env_get();
