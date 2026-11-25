// DenseX (#36): 依顯存與 ctx 自動決定 CPU FFN 層數 N 與 KV 格式
// 輸入: common_params（含模型路徑、ctx、草稿設定、使用者覆寫旗標）
// 輸出: 成功時更新 params（N、KV 型別、tensor_buft_overrides、串流環境變數）並回傳 true
// 開關與預設值: --densex-auto-place（預設關閉）、--densex-auto-place-dry（預設關閉）
// 依賴模組: gguf、ggml-backend、common_params、log

#include "densex-auto-place.h"

#include "common.h"
#include "ggml-backend.h"
#include "gguf.h"
#include "log.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr double MIB = 1024.0 * 1024.0;

// 安全餘裕 300 MiB，來源: AGY_36_VRAM_PLANNER_CLAUDE_VERIFY.md 與 QUANT_TIERS_VRAM.md:150
constexpr double DENSEX_M_SAFE_MIB = 300.0;

// 較高 KV 格式若需移出超過 24 層 FFN，降級為 K4V4 以免 PCIe 串流過慢（AGY_36_VRAM_PLANNER_REPORT.md §4.2）
constexpr int DENSEX_N_KV_SWITCH = 24;

// 讀取 uint32_t 類型的 GGUF metadata（支援 u32 / i32）
int32_t get_gguf_i32(const struct gguf_context * ctx, const char * key, int32_t fallback) {
    const int64_t idx = gguf_find_key(ctx, key);
    if (idx < 0) {
        return fallback;
    }
    const enum gguf_type t = gguf_get_kv_type(ctx, idx);
    if (t == GGUF_TYPE_UINT32) {
        return (int32_t) gguf_get_val_u32(ctx, idx);
    }
    if (t == GGUF_TYPE_INT32) {
        return gguf_get_val_i32(ctx, idx);
    }
    return fallback;
}

// 解析 blk.<idx>.ffn_(gate|up|down).weight 的層編號；非 FFN 權重回傳 -1
int parse_ffn_layer_idx(const char * name) {
    if (std::strncmp(name, "blk.", 4) != 0) {
        return -1;
    }
    const char * p = name + 4;
    if (!std::isdigit((unsigned char) *p)) {
        return -1;
    }
    int idx = 0;
    while (std::isdigit((unsigned char) *p)) {
        idx = idx * 10 + (*p - '0');
        ++p;
    }
    if (*p != '.') {
        return -1;
    }
    ++p;
    if (std::strcmp(p, "ffn_gate.weight") == 0 ||
        std::strcmp(p, "ffn_up.weight") == 0 ||
        std::strcmp(p, "ffn_down.weight") == 0) {
        return idx;
    }
    return -1;
}

// 解析 blk.<idx>. 的層編號；非 blk.<idx>. 回傳 -1
int parse_blk_idx(const char * name) {
    if (std::strncmp(name, "blk.", 4) != 0) {
        return -1;
    }
    const char * p = name + 4;
    if (!std::isdigit((unsigned char) *p)) {
        return -1;
    }
    int idx = 0;
    while (std::isdigit((unsigned char) *p)) {
        idx = idx * 10 + (*p - '0');
        ++p;
    }
    return (*p == '.') ? idx : -1;
}

double bytes_per_elem(enum ggml_type t) {
    const size_t blck = ggml_blck_size(t);
    if (blck == 0) {
        return 0.0;
    }
    return (double) ggml_type_size(t) / (double) blck;
}

// 計算 spread(N) 層集合，照 DenseX/bench/dx_slot.py:37-39
std::vector<int> compute_spread_layers(int n_offload, int n_trunk) {
    std::set<int> uniq;
    for (int k = 0; k < n_offload; ++k) {
        int layer = ((k + 1) * n_trunk) / n_offload - 1;
        if (layer >= n_trunk) {
            layer = n_trunk - 1;
        }
        if (layer < 0) {
            layer = 0;
        }
        uniq.insert(layer);
    }
    return std::vector<int>(uniq.begin(), uniq.end());
}

std::string build_spread_pattern(const std::vector<int> & layers) {
    std::string pat = "blk\\.(";
    for (size_t i = 0; i < layers.size(); ++i) {
        if (i > 0) {
            pat += "|";
        }
        pat += std::to_string(layers[i]);
    }
    pat += ")\\.ffn_(gate|up|down)\\.weight";
    return pat;
}

void set_env_if_unset(const char * name, const char * value) {
    const char * cur = std::getenv(name);
    if (cur != nullptr && cur[0] != '\0') {
        return;
    }
#ifdef _WIN32
    _putenv_s(name, value);
#else
    setenv(name, value, 0);
#endif
}

struct auto_place_estimate {
    int n = 0;
    enum ggml_type ctk = GGML_TYPE_Q4_0;
    enum ggml_type ctv = GGML_TYPE_Q4_0;
    double w_trunk_mib = 0.0;
    double w_draft_mib = 0.0;
    double ffn_offload_mib = 0.0;
    double kv_main_mib = 0.0;
    double kv_draft_mib = 0.0;
    double prefetch_mib = 0.0;
    double oh_mib = 0.0;
    double total_est_mib = 0.0;
};

} // namespace

bool densex_auto_place_apply(common_params & params) {
    if (params.model.path.empty()) {
        LOG_ERR("densex-auto-place: error: model path is empty\n");
        return false;
    }

    // 1. 讀取主模型 GGUF metadata 與張量大小
    struct gguf_init_params gparams = { /*.no_alloc =*/ true, /*.ctx =*/ nullptr };
    struct gguf_context * ctx_main = gguf_init_from_file(params.model.path.c_str(), gparams);
    if (ctx_main == nullptr) {
        LOG_ERR("densex-auto-place: error: failed to open GGUF '%s'\n", params.model.path.c_str());
        return false;
    }

    const int32_t block_count = get_gguf_i32(ctx_main, "qwen35.block_count", 64);
    const int32_t nextn_layers = get_gguf_i32(ctx_main, "qwen35.nextn_predict_layers", 0);
    const int32_t n_trunk = std::max(1, block_count - nextn_layers);
    const int32_t full_attn_interval = std::max(1, get_gguf_i32(ctx_main, "qwen35.full_attention_interval", 4));
    const int32_t n_full_attn = std::max(1, n_trunk / full_attn_interval);
    const int32_t n_head_kv = std::max(1, get_gguf_i32(ctx_main, "qwen35.attention.head_count_kv", 4));
    const int32_t head_dim = std::max(1, get_gguf_i32(ctx_main, "qwen35.attention.key_length", 256));

    double w_trunk_bytes = 0.0;
    double w_mtp_bytes = 0.0;
    double max_ffn_tensor_bytes = 0.0;
    std::vector<double> ffn_layer_bytes((size_t) n_trunk, 0.0);

    const int64_t n_tensors = gguf_get_n_tensors(ctx_main);
    for (int64_t i = 0; i < n_tensors; ++i) {
        const char * tname = gguf_get_tensor_name(ctx_main, i);
        const double tsize = (double) gguf_get_tensor_size(ctx_main, i);
        const int bidx = parse_blk_idx(tname);
        if (bidx >= n_trunk) {
            w_mtp_bytes += tsize;
        } else {
            w_trunk_bytes += tsize;
            const int fidx = parse_ffn_layer_idx(tname);
            if (fidx >= 0 && fidx < n_trunk) {
                ffn_layer_bytes[(size_t) fidx] += tsize;
                if (tsize > max_ffn_tensor_bytes) {
                    max_ffn_tensor_bytes = tsize;
                }
            }
        }
    }
    gguf_free(ctx_main);

    // 2. 計算草稿模型權重與草稿 KV
    const uint32_t n_ctx = params.n_ctx > 0 ? (uint32_t) params.n_ctx : 131072u;
    const bool has_dft_model = params.speculative.has_dft();
    const bool has_mtp = std::find(params.speculative.types.begin(), params.speculative.types.end(),
        COMMON_SPECULATIVE_TYPE_DRAFT_MTP) != params.speculative.types.end();

    double w_draft_bytes = 0.0;
    double kv_draft_bytes = 0.0;

    if (has_dft_model) {
        // DFlash2 獨立草稿模型：主模型 blk.64 不載入（TENSOR_SKIP）
        struct gguf_context * ctx_dft = gguf_init_from_file(params.speculative.draft.mparams.path.c_str(), gparams);
        if (ctx_dft == nullptr) {
            LOG_ERR("densex-auto-place: error: failed to open draft GGUF '%s'\n", params.speculative.draft.mparams.path.c_str());
            return false;
        }
        const int64_t n_dft_tensors = gguf_get_n_tensors(ctx_dft);
        for (int64_t i = 0; i < n_dft_tensors; ++i) {
            w_draft_bytes += (double) gguf_get_tensor_size(ctx_dft, i);
        }
        gguf_free(ctx_dft);
    } else if (has_mtp && nextn_layers > 0) {
        w_draft_bytes = w_mtp_bytes;
        uint32_t draft_cells = n_ctx;
        if (params.speculative.draft.kv_window > 0) {
            const uint32_t raw = std::min(n_ctx,
                (uint32_t) params.speculative.draft.kv_window +
                (uint32_t) std::max(0, params.speculative.draft.n_max) +
                (uint32_t) std::max(1, params.n_ubatch));
            draft_cells = ((raw + 255u) / 256u) * 256u;
        }
        const enum ggml_type dft_ctk = params.speculative.draft.cache_type_k;
        const enum ggml_type dft_ctv = params.speculative.draft.cache_type_v;
        kv_draft_bytes = (double) nextn_layers * (double) n_head_kv * (double) head_dim *
                         (bytes_per_elem(dft_ctk) + bytes_per_elem(dft_ctv)) * (double) draft_cells;
    }

    // 3. 取得目標 GPU 總顯存
    size_t vram_free_bytes = 0;
    size_t vram_total_bytes = 0;
    int gpu_idx = 0;
    bool found_gpu = false;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            if (gpu_idx == params.main_gpu) {
                ggml_backend_dev_memory(dev, &vram_free_bytes, &vram_total_bytes);
                found_gpu = true;
                break;
            }
            ++gpu_idx;
        }
    }
    if (!found_gpu || vram_total_bytes == 0) {
        LOG_ERR("densex-auto-place: error: no usable GPU device found (main_gpu=%d)\n", params.main_gpu);
        return false;
    }

    const double vram_total_mib = (double) vram_total_bytes / MIB;
    const double budget_mib = vram_total_mib - DENSEX_M_SAFE_MIB;

    // 生成期計算與驅動開銷校準常數，來源: AGY_36_VRAM_PLANNER_REPORT.md §2-3（16K=800 MiB, 131K=848 MiB）
    // 注意: 依 AGY_36_VRAM_PLANNER_CLAUDE_VERIFY.md，此校準值在正式設定點可能高估約 74 MiB
    const double ctx_frac = std::clamp(((double) n_ctx - 16384.0) / (131072.0 - 16384.0), 0.0, 1.0);
    const double oh_mib = 800.0 + 48.0 * ctx_frac;
    const double prefetch_per_slot_mib = max_ffn_tensor_bytes / MIB;

    auto eval_config = [&](int n_offload, enum ggml_type ctk, enum ggml_type ctv) -> auto_place_estimate {
        auto_place_estimate e;
        e.n = n_offload;
        e.ctk = ctk;
        e.ctv = ctv;
        e.w_trunk_mib = w_trunk_bytes / MIB;
        e.w_draft_mib = w_draft_bytes / MIB;
        if (n_offload > 0) {
            const std::vector<int> layers = compute_spread_layers(n_offload, n_trunk);
            double sum_bytes = 0.0;
            for (int l : layers) {
                sum_bytes += ffn_layer_bytes[(size_t) l];
            }
            e.ffn_offload_mib = sum_bytes / MIB;
            e.prefetch_mib = 4.0 * prefetch_per_slot_mib;
        }
        e.kv_main_mib = (double) n_full_attn * (double) n_head_kv * (double) head_dim *
                        (bytes_per_elem(ctk) + bytes_per_elem(ctv)) * (double) n_ctx / MIB;
        e.kv_draft_mib = kv_draft_bytes / MIB;
        e.oh_mib = oh_mib;
        e.total_est_mib = (e.w_trunk_mib + e.w_draft_mib - e.ffn_offload_mib) +
                          e.kv_main_mib + e.kv_draft_mib + e.prefetch_mib + e.oh_mib;
        return e;
    };

    // 4. 檢查使用者覆寫
    const bool kv_locked = params.densex_user_set_ctk || params.densex_user_set_ctv;
    const bool placement_locked = params.densex_user_set_ot || params.densex_user_set_ngl;

    if (kv_locked) {
        LOG_INF("densex-auto-place: keeping user KV types (%s/%s)\n",
                ggml_type_name(params.cache_type_k), ggml_type_name(params.cache_type_v));
    }
    if (placement_locked) {
        LOG_INF("densex-auto-place: keeping user placement (-ot/-ngl); skipping auto placement\n");
        const auto_place_estimate est = eval_config(0, params.cache_type_k, params.cache_type_v);
        LOG_INF("densex-auto-place: selected N=%d kv=%s/%s | W_trunk=%.1f W_draft=%.1f FFN_offload=%.1f KV_main=%.1f KV_draft=%.1f Prefetch=%.1f OH=%.1f -> est=%.1f MiB / budget=%.1f MiB (total=%.1f, M_safe=%.1f)\n",
                est.n, ggml_type_name(est.ctk), ggml_type_name(est.ctv),
                est.w_trunk_mib, est.w_draft_mib, est.ffn_offload_mib,
                est.kv_main_mib, est.kv_draft_mib, est.prefetch_mib, est.oh_mib,
                est.total_est_mib, budget_mib, vram_total_mib, DENSEX_M_SAFE_MIB);
        return true;
    }

    std::vector<std::pair<enum ggml_type, enum ggml_type>> kv_candidates;
    if (kv_locked) {
        kv_candidates.push_back({ params.cache_type_k, params.cache_type_v });
    } else {
        kv_candidates.push_back({ GGML_TYPE_Q8_0, GGML_TYPE_Q8_0 });
        kv_candidates.push_back({ GGML_TYPE_Q8_0, GGML_TYPE_Q4_0 });
        kv_candidates.push_back({ GGML_TYPE_Q4_0, GGML_TYPE_Q4_0 });
    }

    bool found = false;
    auto_place_estimate best;

    // Step 1: 優先嘗試全 GPU（N = 0）
    for (const auto & kv : kv_candidates) {
        const auto_place_estimate cand = eval_config(0, kv.first, kv.second);
        if (cand.total_est_mib <= budget_mib) {
            best = cand;
            found = true;
            break;
        }
    }

    // Step 2: 無法全 GPU 時，從 N = 1..n_trunk 找最小 N
    if (!found) {
        for (size_t ki = 0; ki < kv_candidates.size(); ++ki) {
            const auto & kv = kv_candidates[ki];
            int min_n = -1;
            auto_place_estimate cand_best;
            for (int n = 1; n <= n_trunk; ++n) {
                const auto_place_estimate cand = eval_config(n, kv.first, kv.second);
                if (cand.total_est_mib <= budget_mib) {
                    min_n = n;
                    cand_best = cand;
                    break;
                }
            }
            if (min_n > 0) {
                const bool has_next_kv = (ki + 1 < kv_candidates.size());
                if (!kv_locked && has_next_kv && min_n > DENSEX_N_KV_SWITCH) {
                    continue;
                }
                best = cand_best;
                found = true;
                break;
            }
        }
    }

    if (!found) {
        const auto_place_estimate worst = eval_config(n_trunk, kv_candidates.back().first, kv_candidates.back().second);
        LOG_ERR("densex-auto-place: error: insufficient VRAM even with N=%d kv=%s/%s (est=%.1f MiB > budget=%.1f MiB, total=%.1f MiB)\n",
                n_trunk, ggml_type_name(worst.ctk), ggml_type_name(worst.ctv),
                worst.total_est_mib, budget_mib, vram_total_mib);
        return false;
    }

    LOG_INF("densex-auto-place: selected N=%d kv=%s/%s | W_trunk=%.1f W_draft=%.1f FFN_offload=%.1f KV_main=%.1f KV_draft=%.1f Prefetch=%.1f OH=%.1f -> est=%.1f MiB / budget=%.1f MiB (total=%.1f, M_safe=%.1f)\n",
            best.n, ggml_type_name(best.ctk), ggml_type_name(best.ctv),
            best.w_trunk_mib, best.w_draft_mib, best.ffn_offload_mib,
            best.kv_main_mib, best.kv_draft_mib, best.prefetch_mib, best.oh_mib,
            best.total_est_mib, budget_mib, vram_total_mib, DENSEX_M_SAFE_MIB);

    if (params.densex_auto_place_dry) {
        return true;
    }

    // 5. 套用配置至 params
    params.fit_params = false;
    params.n_gpu_layers = 999;
    params.cache_type_k = best.ctk;
    params.cache_type_v = best.ctv;
    if ((best.ctk != GGML_TYPE_F16 || best.ctv != GGML_TYPE_F16) &&
        params.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_AUTO) {
        params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    }

    if (best.n > 0) {
        const std::vector<int> layers = compute_spread_layers(best.n, n_trunk);
        params.densex_auto_place_ot_pattern = build_spread_pattern(layers);
        params.tensor_buft_overrides.clear();
        params.tensor_buft_overrides.push_back({ params.densex_auto_place_ot_pattern.c_str(), ggml_backend_cpu_buffer_type() });
        params.tensor_buft_overrides.push_back({ nullptr, nullptr });
        params.load_mode = LLAMA_LOAD_MODE_NONE;
        set_env_if_unset("GGML_OP_OFFLOAD_MIN_BATCH", "1");
        set_env_if_unset("GGML_SCHED_PREFETCH", "4");
    }

    return true;
}
