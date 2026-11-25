#pragma once

// DenseX (#36): 依顯存與 ctx 自動決定 CPU FFN 層數 N 與 KV 格式
// 輸入: common_params（含模型路徑、ctx、草稿設定、使用者覆寫旗標）
// 輸出: 成功時更新 params（N、KV 型別、tensor_buft_overrides、串流環境變數）並回傳 true
// 開關與預設值: --densex-auto-place（預設關閉）、--densex-auto-place-dry（預設關閉）
// 依賴模組: gguf、ggml-backend、common_params、log

struct common_params;

bool densex_auto_place_apply(common_params & params);
