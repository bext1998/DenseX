# MTP draft KV 視窗首版實作

工作分支：`densex-draftwin`；基底 `5833ba84d56038c7d8fdf5d79cbc01c9d67ba0a8`。所有程式、測試與本報告僅修改 `G:/InferenceReaserch/DenseX/wt-draftwin`。建置輸出依交辦放在 `DenseX/build/draftwin`；GPU測試只在取得 `DenseX/bench/GPU_BUSY` 後進行，使用8094，不向8080／8081送請求。未讀Mirai目錄、未修改主工作樹、未合併或push。

## 設計

新增 `--spec-draft-kv-window N`，0為舊路徑。Qwen35單一MTP head、單序列的draft context可設W；target的logical context／attention與shared model hparams不變。用plain KV cache的context專用STANDARD SWA mask，位置維持絕對值。

physical cells = pad256(min(logical_ctx, W + R + n_ubatch))，R取實際全域draft上限。attention可見W；覆寫判斷與state保存使用W+R，保護試算後回退重新可見的KV。ctx131072、W8192、R8、ub128時為8448 cells，q8/q8 payload17.53125MiB；不是把logical context縮為8192。

預設0不進新分支；新cache retention參數省略時等於原n_swa，所以其他cache的mask、覆寫與保存政策不變。配置算術用uint64並在縮回uint32前檢查；不支援的model／多序列／model本身SWA在context建立時明確拒絕。

## 驗證計畫與結果

### 紅燈替代證據

基底沒有context-local MTP視窗／retention入口，GPU當時由Claude Q1鎖定，無法先用真模型跑新功能。採用CPU cache oracle的舊策略negative control：同樣physical cells，但retention=W而非W+R，應在回退後與足容量reference分歧。這用來確認測試能捕捉缺rollback reserve，不將單純新增API造成的compile failure當作正確性證據。

### CPU oracle

已通過。W247／R8／U1／C256，實際slot回繞5次，v_trans=false/true皆通過；reference與被測cache使用相同W mask；reference容量／retention足夠而不淘汰。另有獨立scalar mask／attention對照，避免兩份cache一起錯而互相通過。比較attention與固定小projection logits；這不是完整Swift模型的draft logits。最大絕對誤差5.96046448e-8，容許上限2e-5；W0對照跑滿256個token，超過W247且不淘汰。測試不探索CUDA backend，不需要GPU。

negative control已實際測到兩次回退後誤差0.0121020749；PARTIAL_ONLY使用舊retention保存的state，還原後誤差0.0151825445，正確W+R保存則約7.45e-9。這兩項是實際的舊策略失敗對照，不冒稱修改前曾跑過舊binary。

既有test-batch-alloc通過；既有test-llama-context的Qwen35 dummy model，baseline及seq_rm/state_load/copy/roundtrip/position/rotation/fragmentation/rollback/replay/shared等15欄全部OK。CPU runner最終紀錄見build-cpu-final.log；有意使用CUDA=OFF，沒有GPU探索。

### GPU smoke

已通過。`tests/run-draft-window-smoke.ps1` 使用atomic CreateNew取得GPU_BUSY，確認沒有其他推論／測試程序，持鎖依序啟動W0及W8192；ctx131072、MTP8、ub128、K8V4、draft q8/q8、39組FFN spread至CPU、prefetch4。比較temp0的token IDs及A→B→A prompt cache續接，並檢查17.53MiB draft KV日誌。明確設定GGML_CUDA_FA_NATIVE_KV=1，與既有97.5MiB target compute基準一致。腳本僅停止自己啟動的PID，finally歸還自己持有的鎖。

## 缺hidden時的安全選擇

Window模式記錄pending_h對應的絕對位置。新checkpoint保存magic／position／F32 hidden；空、舊格式或無效payload先失效carry與verify rows。RAM cache、清prompt、slot restore及speculative checkpoint restore在缺carry時明確失效。

process若缺首列所需的前一位置hidden，清draft KV，跳過無法配對的首列，後續仍依原target索引k-1取hidden。若只有一列，避免空batch decode，但記下該列target hidden，使下一輪可以恢復drafting。seed前若hidden位置仍不對，清本輪draft結果與q，警告並放棄這輪草稿。不是靜默使用先前slot的hidden；代價是還原後的draft suffix可能暫時不完整、接受率下降。

W0不配置位置追蹤、不保存新payload，set_state直接返回，server也不送新的失效通知；相關cache與Qwen context回歸均通過。這不是跨版本所有輸入逐bit等同的完整證明，GPU段再核對實際greedy token IDs。

## 改動地圖與合併位置

路徑皆相對此worktree；行號為本次修改後位置。

| 檔案／位置 | 內容 |
|---|---|
| common/common.h:348；common/arg.cpp的新旗標與common_params_parse | draft kv_window欄位與CLI；在parser的exception邊界內驗證策略／序列數，避免不合法配置到模型載入時才拋出 |
| common/common.cpp:1240,1281 | 共同範圍驗證、fit draft試算傳相同W/R；target參數維持0 |
| common/speculative.cpp:1457,1631,1748,1926,2748 | hidden位置追蹤、缺hidden重建／跳過、checkpoint payload、runtime傳W/R |
| include/llama.h:372；src/llama-cparams.h:17 | context專用kv_window／kv_rollback，預設0 |
| src/llama-context.cpp:192,3992 | **集中兩處**：copy／範圍驗證與default params；供Claude合併時留意 |
| src/llama-model.cpp:3279 | plain MTP KV配置C、W、STANDARD與retention；不改hparams |
| src/llama-kv-cache.h:118,280；src/llama-kv-cache.cpp:86,1079,2106 | 新retention參數；覆寫／state保存保護W+R，mask仍用W |
| tools/server/server-context.cpp:259,336,352,1445,2789,4588 | 只在window模式處理缺hidden的cache／slot邊界 |
| tests/test-draft-kv-window.cpp；tests/CMakeLists.txt | CPU scalar／足容量cache oracle、兩次回退、PARTIAL_ONLY roundtrip、W0與負對照 |
| tests/test-draft-kv-window-args.cpp | 新旗標／非法值／範圍驗證，確認draft選項未流入target params |
| tests/test-batch-alloc.cpp:913 | 既有MSVC C2131一行修正：陣列長度由初始化內容推導，仍是相同4個位置 |
| tests/run-draft-window-cpu.ps1；tests/run-draft-window-smoke.ps1 | 可重跑CPU驗證與持GPU鎖的8094 smoke／舊slot還原 |

**沒有修改src/llama-graph.cpp、memory-recurrent、delta-net-base或ggml op。** 公開llama_context_params新增兩欄，因此使用本fork的client／bindings需隨庫重建，不能混用舊DLL ABI；本次server採完整static rebuild。

## 建置與失敗修正紀錄

- prescribed CUDA build成功：DenseX/build/build-20261109-193511.log，19:59:23 BUILD_FINISHED；parser提前驗證後incremental build也成功（20:17:22 BUILD_FINISHED）。
- 初次CPU build在既有test-batch-alloc.cpp:913遇MSVC C2131，以上一行修正解除；沒有移除或放寬測試。
- 初次dummy fixture輸出目錄未建立，生成工具雖報錯但返回0；CPU runner已先建目錄並檢查實際GGUF存在。補跑context回歸及修正後完整runner成功；早期失敗紀錄與最終結果分列，不把失敗當通過。
- GPU_BUSY由Claude Q1持有，使用者確認仍在跑且約20:30前結束、自動刪鎖；實際progress.log於20:16:48寫Q1_FINISHED後釋放。等待程式只看鎖，消失後才由smoke script原子建立自己的鎖並核對推論程序；未把樣本間空檔視為可用。


## 實際驗收結果

| 項目 | 結果 |
|---|---|
| prescribed CUDA llama-server build | 通過；輸出DenseX/build/draftwin/bin/llama-server.exe |
| CPU CTest | 3/3：test-draft-kv-window、test-draft-kv-window-args、既有test-batch-alloc |
| 既有Qwen35 context回歸 | 1個生成model、15欄全部OK（含baseline），沒有失敗或skip欄 |
| 非法CLI | 負值、uint32 overflow、尾隨字元、非MTP、正值多序列拒絕；W0多序列仍可parse，draft flag不流入target params |
| W0／W8192 GPU短生成 | 各3個請求，輸出token數64／6／64；對應token IDs逐個完全相同 |
| A→B→A快取 | 每組第一／第三次A輸出IDs相同；W0與W8192的第三次皆只處理4個prompt token |
| target完整context | 兩個context的logical n_ctx皆131072；target KV維持3328.00MiB、target compute97.50MiB |
| draft bounded KV | W8192、retention8200、8448 cells；272.00→17.53MiB，約省254.47MiB |
| draft compute reserve | 同時保留的一份49.27→32.83MiB；額外約16.44MiB，兩次印出仍是重建而非兩份 |
| 無hidden舊slot | draft KV成功載入，精確觸發pos25重建警告；target prompt_n=1，沒有空batch decode錯誤 |
| 完整hidden slot正控制 | 同pos25續接，prompt_n=1，不新增missing-hidden警告；兩者生成16 token IDs完全相同 |
| 清理 | 結束時GPU_BUSY不存在；llama-server／llama-perplexity／test-backend-ops程序數0 |

GPU證據：本worktree `build-draft-window-results/w0.json`、`w8192.json`、`w0.stderr.log`、`w8192.stderr.log`、`restore.json`、`restore.stderr.log`；完成標記 `result.log` 與 `restore-result.log`。這些是本機測試輸出，位於git忽略的build目錄，不把模型／slot binary／大日誌commit進分支。

W8192日誌：`w8192.stderr.log:223–224`印window8192／retention8200／cells8448及17.53MiB；`:173,209`印兩個logical context131072；`:187`target KV3328。還原證據：`restore.stderr.log:328,343–344,354,369`分別為舊draft KV載入、missing-hidden警告與1-token prompt eval、新檔載入與1-token prompt eval。

無hidden slot試驗停用RAM cache／context checkpoints，保存25-token prefix後，把live SDFT末尾的spec payload長度改為0，保留draft KV；先以不同內容、相同長度prefix污染舊carry，再restore並只追加一token。新檔正控制則保留MTPW／pos／hidden payload。不能以兩個有效checkpoint的A→B→A取代這個缺hidden試驗。

初次自動GPU嘗試恰逢舊binary尚未含parser提前驗證，非法參數檢查以異常退出結束；incremental build後已通過。之後第一次完整W0測試漏設native KV環境，使target compute膨脹至582.5MiB；主線只停止自己啟動、父PID匹配的server，finally歸還鎖，保留該組為native0-*診斷紀錄，補齊環境後重新完成兩組比較。未將這些中止／錯配配置當成功或速度數據。

## 重跑方式

```powershell
# CPU-only，無需GPU鎖
pwsh -NoProfile -File tests/run-draft-window-cpu.ps1

# 以下腳本自行檢查並原子取得GPU_BUSY；忙碌時拒絕，不強刪別人的鎖
pwsh -NoProfile -File tests/run-draft-window-smoke.ps1
pwsh -NoProfile -File tests/run-draft-window-smoke.ps1 -RestoreOnly
```

## 限制與未完成項目

本交辦的首版與驗收已完成。未做32K／120K接受率、品質或速度量測；真Swift模型只做短生成，實際跨兩次ring wrap由CPU cache／投影logits oracle驗證，不能冒稱完成長文模型品質驗證。fit已同步W/R/C政策，但本輪GPU smoke固定fit off，尚未實測自動放置能回填幾層FFN。

greedy的已測輸出完全一致，不擴張為所有prompt、非零溫度、所有hardware的逐bit證明。多序列、其他drafter／多MTP head與model原生SWA仍不支援正值window；沒有變更target注意力、draft實際q的保存或驗證器。新hidden payload沒有提供跨model／舊DLL ABI相容保證。缺hidden的重建會暫時犧牲draft歷史與接受率，後續由Claude安排長文量測評估。

只commit至densex-draftwin，不合併densex、不push。

DRAFT_KV_WINDOW_FINISHED
