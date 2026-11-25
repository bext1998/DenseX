#pragma once

#include <cstdint>

// DenseX (#75): 依主模型 GGUF 標頭自動決定草稿設定（首版只有一組已實測 profile）
// 輸入: common_params（主模型路徑、草稿相關參數與 densex_draft_explicit 遮罩）
// 輸出: 命中精確已測 profile 時更新 params.speculative（type／草稿模型／n_max／草稿 KV／window）並回傳 true；
//       其他情況只寫一行決策 log、params 不動，仍回傳 true（開關不該讓主模型載入失敗）。
// 開關與預設值: --densex-auto-draft（預設關閉）、--densex-auto-draft-dry（預設關閉）
// 依賴模組: gguf、common_params、log

struct common_params;

// densex_draft_explicit 的位元：使用者明確指定了哪些草稿相關參數。
// 由 common/arg.cpp 的選項 handler 設定（CLI、LLAMA_ARG_* 環境變數與設定檔都走同一條 handler）。
enum densex_draft_explicit_flag : uint32_t {
    DENSEX_DRAFT_EXPLICIT_TYPE      = 1u << 0, // --spec-type
    DENSEX_DRAFT_EXPLICIT_MODEL     = 1u << 1, // --spec-draft-model / -md / --model-draft
    DENSEX_DRAFT_EXPLICIT_N_MAX     = 1u << 2, // --spec-draft-n-max
    DENSEX_DRAFT_EXPLICIT_N_MIN     = 1u << 3, // --spec-draft-n-min
    DENSEX_DRAFT_EXPLICIT_TYPE_K    = 1u << 4, // --spec-draft-type-k / -ctkd
    DENSEX_DRAFT_EXPLICIT_TYPE_V    = 1u << 5, // --spec-draft-type-v / -ctvd
    DENSEX_DRAFT_EXPLICIT_KV_WINDOW = 1u << 6, // --spec-draft-kv-window
    DENSEX_DRAFT_EXPLICIT_VOCAB     = 1u << 7, // DENSEX_DRAFT_VOCAB / DENSEX_DRAFT_VOCAB_ORDER（環境變數）
};

bool densex_auto_draft_apply(common_params & params);
