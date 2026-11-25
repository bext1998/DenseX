// DenseX (#75): 依主模型 GGUF 標頭自動決定草稿設定（首版只有一組已實測 profile）
// 輸入: common_params（主模型路徑、草稿相關參數與 densex_draft_explicit 遮罩）
// 輸出: 命中精確已測 profile 時更新 params.speculative；其他情況只寫一行決策 log、params 不動
// 開關與預設值: --densex-auto-draft（預設關閉）、--densex-auto-draft-dry（預設關閉）
// 依賴模組: gguf、common_params、log
//
// 辨識策略：只用 GGUF 標頭的精確值，不靠檔名或路徑。
//   主模型 = ISTA GSQ-RCO IQ3_XXS-mtp（已實測）：arch=qwen35、name/basename/size_label、
//   quantized_by/repo_url、block_count=65 且 nextn_predict_layers=1、15 個 blk.64 nextn 張量齊全、
//   以及整個張量型別指紋（IQ2_XS 檔會差在 type 19/21/29 的張量數，因此不會命中）。
//   general.file_type 不可靠（IQ3_XXS 檔寫 26=IQ3_S），只記錄不用於比對。
// 草稿檔：只在本機既有相對位置找（主模型目錄與其上層最多 4 層：
//   <dir>/Qwen3.8-27B-DFlash2-Q4_K_M.gguf、<dir>/DFlash2-Qwen3.8-27B/Qwen3.8-27B-DFlash2-Q4_K_M.gguf），
//   並再讀一次草稿檔標頭確認為 dflash 架構；找不到或不符就 WARN，參數不動，絕不下載。

#include "densex-auto-draft.h"

#include "common.h"
#include "gguf.h"
#include "log.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace {

namespace fs = std::filesystem;

// 已實測 profile（報告：docs/tasks/CODEX_75_DRAFT_PROFILE_REPORT.md 設定表第一列）
constexpr const char * DENSEX_AUTO_DRAFT_PROFILE = "ista-iq3xxs-mtp-df2-v1";

// 已實測草稿資產（DenseX/models/DFlash2-Qwen3.8-27B/）
constexpr const char * DENSEX_DFLASH2_FILE = "Qwen3.8-27B-DFlash2-Q4_K_M.gguf";
constexpr const char * DENSEX_DFLASH2_DIR  = "DFlash2-Qwen3.8-27B";
constexpr const char * DENSEX_DFLASH2_ARCH = "dflash";
constexpr const char * DENSEX_DFLASH2_NAME = "Qwen3.8-27B-DFlash2";

// 已實測主模型（DenseX/models/ISTA-Qwen3.8-27B-GSQ-RCO/Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf）的張量型別指紋：
// ggml type id -> 張量數，2026-11-25 用 `.tmp/cx75_gguf_headers_v2.json` 實讀。
struct type_count {
    int type;
    int count;
};

const type_count DENSEX_ISTA_IQ3XXS_TYPES[] = {
    { 0, 360 }, { 10, 28 }, { 12, 19 }, { 14, 8 },  { 16, 30 }, { 17, 32 }, { 18, 83 },
    { 19, 4 },  { 21, 97 }, { 22, 72 }, { 23, 33 }, { 29, 4 },  { 30, 96 },
};

// MTP 張量集合必須完整（blk.64 共 15 個）
const char * DENSEX_ISTA_MTP_TENSORS[] = {
    "blk.64.nextn.eh_proj.weight",
    "blk.64.nextn.shared_head_norm.weight",
    "blk.64.nextn.enorm.weight",
    "blk.64.nextn.hnorm.weight",
    "blk.64.attn_q.weight",
    "blk.64.attn_k.weight",
    "blk.64.attn_v.weight",
    "blk.64.attn_output.weight",
    "blk.64.attn_norm.weight",
    "blk.64.attn_q_norm.weight",
    "blk.64.attn_k_norm.weight",
    "blk.64.post_attention_norm.weight",
    "blk.64.ffn_gate.weight",
    "blk.64.ffn_up.weight",
    "blk.64.ffn_down.weight",
};

struct gguf_header {
    bool ok = false;
    std::string arch;
    std::string general_type;
    std::string name;
    std::string basename;
    std::string size_label;
    std::string quantized_by;
    std::string repo_url;
    int32_t block_count  = 0;
    int32_t nextn_layers = 0;
    int32_t file_type    = 0;
    int32_t split_count  = 1;
    std::map<int, int> tensor_types;
    std::set<std::string> tensor_names;
};

bool read_string(const struct gguf_context * ctx, const char * key, std::string & out) {
    const int64_t idx = gguf_find_key(ctx, key);
    if (idx < 0 || gguf_get_kv_type(ctx, idx) != GGUF_TYPE_STRING) {
        return false;
    }
    out = gguf_get_val_str(ctx, idx);
    return true;
}

int32_t read_i32(const struct gguf_context * ctx, const char * key, int32_t fallback) {
    const int64_t idx = gguf_find_key(ctx, key);
    if (idx < 0) {
        return fallback;
    }
    switch (gguf_get_kv_type(ctx, idx)) {
        case GGUF_TYPE_UINT16: return (int32_t) gguf_get_val_u16(ctx, idx);
        case GGUF_TYPE_UINT32: return (int32_t) gguf_get_val_u32(ctx, idx);
        case GGUF_TYPE_INT32:  return gguf_get_val_i32(ctx, idx);
        default:               return fallback;
    }
}

// 只讀 metadata 與 tensor directory（no_alloc，不載入權重）
bool read_header(const std::string & path, gguf_header & h) {
    struct gguf_init_params gparams = { /*.no_alloc =*/ true, /*.ctx =*/ nullptr };
    struct gguf_context * ctx = gguf_init_from_file(path.c_str(), gparams);
    if (ctx == nullptr) {
        return false;
    }
    h.ok = true;
    read_string(ctx, "general.architecture", h.arch);
    read_string(ctx, "general.type",         h.general_type);
    read_string(ctx, "general.name",         h.name);
    read_string(ctx, "general.basename",     h.basename);
    read_string(ctx, "general.size_label",   h.size_label);
    read_string(ctx, "general.quantized_by", h.quantized_by);
    read_string(ctx, "general.repo_url",     h.repo_url);
    h.block_count  = read_i32(ctx, "qwen35.block_count", 0);
    h.nextn_layers = read_i32(ctx, "qwen35.nextn_predict_layers", 0);
    h.file_type    = read_i32(ctx, "general.file_type", 0);
    h.split_count  = read_i32(ctx, "split.count", 1);

    const int64_t n_tensors = gguf_get_n_tensors(ctx);
    for (int64_t i = 0; i < n_tensors; ++i) {
        h.tensor_names.emplace(gguf_get_tensor_name(ctx, i));
        h.tensor_types[(int) gguf_get_tensor_type(ctx, i)] += 1;
    }
    gguf_free(ctx);
    return true;
}

bool is_ista_iq3xxs_mtp(const gguf_header & h) {
    if (h.arch != "qwen35" || h.general_type != "model") {
        return false;
    }
    if (h.name != "Qwen3.8-27B GSQ-RCO" || h.basename != "Qwen3.8-27B" || h.size_label != "27B") {
        return false;
    }
    if (h.quantized_by != "ISTA DASLab" || h.repo_url != "https://huggingface.co/ISTA-DASLab") {
        return false;
    }
    if (h.block_count != 65 || h.nextn_layers != 1 || h.split_count != 1) {
        return false;
    }
    for (const char * t : DENSEX_ISTA_MTP_TENSORS) {
        if (h.tensor_names.count(t) == 0) {
            return false;
        }
    }
    // 張量型別指紋：數量與每一項都要完全一致（IQ2_XS 檔在此被排除）
    if ((int) h.tensor_types.size() != (int) (sizeof(DENSEX_ISTA_IQ3XXS_TYPES) / sizeof(DENSEX_ISTA_IQ3XXS_TYPES[0]))) {
        return false;
    }
    for (const type_count & e : DENSEX_ISTA_IQ3XXS_TYPES) {
        const auto it = h.tensor_types.find(e.type);
        if (it == h.tensor_types.end() || it->second != e.count) {
            return false;
        }
    }
    return true;
}

bool is_dflash_sidecar(const std::string & path) {
    gguf_header h;
    if (!read_header(path, h)) {
        return false;
    }
    return h.arch == DENSEX_DFLASH2_ARCH && h.name == DENSEX_DFLASH2_NAME && h.split_count == 1;
}

// 主模型目錄與其上層最多 4 層，只找兩個既有相對位置；不下載、不接受 GGUF 內的字串當路徑
bool find_sidecar(const std::string & main_path, std::string & out) {
    std::error_code ec;
    fs::path dir = fs::path(main_path).parent_path();
    std::string searched;
    for (int depth = 0; depth < 4; ++depth) {
        if (dir.empty()) {
            break;
        }
        const fs::path candidates[2] = { dir / DENSEX_DFLASH2_FILE, dir / DENSEX_DFLASH2_DIR / DENSEX_DFLASH2_FILE };
        for (const fs::path & cand : candidates) {
            if (!searched.empty()) {
                searched += ",";
            }
            searched += cand.string();
            if (fs::is_regular_file(cand, ec) && !ec && is_dflash_sidecar(cand.string())) {
                out = cand.string();
                return true;
            }
        }
        const fs::path parent = dir.parent_path();
        if (parent == dir) {
            break;
        }
        dir = parent;
    }
    LOG_WRN("densex-auto-draft: warning: profile=%s source=auto reason=sidecar-missing searched=%s\n",
            DENSEX_AUTO_DRAFT_PROFILE, searched.c_str());
    return false;
}

uint32_t env_explicit_flags() {
    uint32_t flags = 0;
    const char * vocab = std::getenv("DENSEX_DRAFT_VOCAB");
    const char * order = std::getenv("DENSEX_DRAFT_VOCAB_ORDER");
    if ((vocab != nullptr && vocab[0] != '\0') || (order != nullptr && order[0] != '\0')) {
        flags |= DENSEX_DRAFT_EXPLICIT_VOCAB;
    }
    return flags;
}

std::string describe_explicit(uint32_t mask) {
    struct bit_name {
        uint32_t bit;
        const char * name;
    };
    static const bit_name names[] = {
        { DENSEX_DRAFT_EXPLICIT_TYPE,      "spec-type" },
        { DENSEX_DRAFT_EXPLICIT_MODEL,     "spec-draft-model" },
        { DENSEX_DRAFT_EXPLICIT_N_MAX,     "spec-draft-n-max" },
        { DENSEX_DRAFT_EXPLICIT_N_MIN,     "spec-draft-n-min" },
        { DENSEX_DRAFT_EXPLICIT_TYPE_K,    "spec-draft-type-k" },
        { DENSEX_DRAFT_EXPLICIT_TYPE_V,    "spec-draft-type-v" },
        { DENSEX_DRAFT_EXPLICIT_KV_WINDOW, "spec-draft-kv-window" },
        { DENSEX_DRAFT_EXPLICIT_VOCAB,     "densex-draft-vocab-env" },
    };
    std::string out;
    for (const bit_name & b : names) {
        if ((mask & b.bit) != 0) {
            if (!out.empty()) {
                out += ",";
            }
            out += b.name;
        }
    }
    return out;
}

} // namespace

bool densex_auto_draft_apply(common_params & params) {
    // 1. 明確指定一律優先：CLI／環境變數／設定檔任一草稿參數就整套跳過
    const uint32_t explicit_mask = params.densex_draft_explicit | env_explicit_flags();
    if (explicit_mask != 0) {
        LOG_INF("densex-auto-draft: profile=none source=explicit reason=%s mask=0x%x auto_applied=false\n",
                describe_explicit(explicit_mask).c_str(), explicit_mask);
        return true;
    }

    // 2. 只支援本機主模型檔（遠端 selector 不在這版範圍）
    if (params.model.path.empty() || !params.model.hf_repo.empty() || !params.model.url.empty() ||
        !params.model.docker_repo.empty() || !params.models_dir.empty() || !params.models_preset.empty() ||
        !params.models_preset_hf.empty()) {
        LOG_INF("densex-auto-draft: profile=none source=auto reason=no-local-main-model\n");
        return true;
    }

    // 3. 讀主模型標頭
    gguf_header main_h;
    if (!read_header(params.model.path, main_h)) {
        LOG_WRN("densex-auto-draft: warning: profile=none source=auto reason=main-header-unreadable path=%s\n",
                params.model.path.c_str());
        return true;
    }

    // 4. 精確比對已測 profile；其他任何模型都不動參數
    if (!is_ista_iq3xxs_mtp(main_h)) {
        LOG_INF("densex-auto-draft: profile=none source=auto reason=not-tested arch=%s name=%s basename=%s size=%s file_type=%d\n",
                main_h.arch.c_str(), main_h.name.c_str(), main_h.basename.c_str(),
                main_h.size_label.c_str(), main_h.file_type);
        return true;
    }

    // 5. 找本機草稿檔並確認身分
    std::string draft_path;
    if (!find_sidecar(params.model.path, draft_path)) {
        return true; // 已在 find_sidecar 內 WARN
    }

    // 6. 套用已實測草稿設定（不碰 N／placement；不設 DENSEX_DRAFT_VOCAB*）
    if (std::find(params.speculative.types.begin(), params.speculative.types.end(),
                  COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH) == params.speculative.types.end()) {
        params.speculative.types.push_back(COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH);
    }
    params.speculative.draft.mparams.path    = draft_path;
    params.speculative.draft.mparams.hf_file = draft_path;
    params.speculative.draft.n_max           = 7;
    params.speculative.draft.n_min           = 0;
    params.speculative.draft.cache_type_k    = GGML_TYPE_Q8_0;
    params.speculative.draft.cache_type_v    = GGML_TYPE_Q8_0;
    params.speculative.draft.kv_window       = 0;

    LOG_INF("densex-auto-draft: profile=%s source=auto reason=exact-tested-metadata series=ista-gsq-rco tier=iq3xxs mtp=1 type=draft-dflash n_max=7 draft=%s kv=q8_0/q8_0 window=0 placement=untouched\n",
            DENSEX_AUTO_DRAFT_PROFILE, draft_path.c_str());
    return true;
}
