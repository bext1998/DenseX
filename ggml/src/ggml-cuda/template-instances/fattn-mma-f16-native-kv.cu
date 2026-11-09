// DenseX: MMA FlashAttention with K/V read in place from a q8_0 / q4_0 cache (dim 256, GQA > 4).

#include "../fattn-mma-f16.cuh"

#define DECL_NATIVE_KV(type_K, type_V)                                      \
    template void ggml_cuda_flash_attn_ext_mma_f16_case<256, 256, 1, 8, type_K, type_V>(ggml_backend_cuda_context & ctx, ggml_tensor * dst); \
    template void ggml_cuda_flash_attn_ext_mma_f16_case<256, 256, 2, 8, type_K, type_V>(ggml_backend_cuda_context & ctx, ggml_tensor * dst); \
    template void ggml_cuda_flash_attn_ext_mma_f16_case<256, 256, 4, 8, type_K, type_V>(ggml_backend_cuda_context & ctx, ggml_tensor * dst); \
    template void ggml_cuda_flash_attn_ext_mma_f16_case<256, 256, 8, 8, type_K, type_V>(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

DECL_NATIVE_KV(GGML_TYPE_Q8_0, GGML_TYPE_Q8_0)
DECL_NATIVE_KV(GGML_TYPE_Q8_0, GGML_TYPE_Q4_0)
DECL_NATIVE_KV(GGML_TYPE_Q4_0, GGML_TYPE_Q4_0)
