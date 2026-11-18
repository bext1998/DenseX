---
name: add-new-model
description: 在 DenseX 接入新的 Qwen 3.8 系列模型檔、fine-tune 或推測解碼用的 draft 模型，並驗證它和既有路徑一樣準、一樣穩。使用者要支援新的模型檔、新的權重來源或量化版本、新的 draft 模型，或問「能不能跑某個模型」時使用。
---

# 在 DenseX 接入新模型

DenseX 是給低 VRAM 顯示卡用的家用稠密模型推論引擎，只支援兩種架構：
- `qwen35`：Qwen 3.8 系列的主模型，包括 27B 的各家權重與量化版本（例如 ISTA GSQ-RCO、Unsloth UD 各檔位）。
- `dflash`：`--spec-type draft-dflash` 用的 draft 模型。

其他架構在 issue #10、#86 都已經刪掉：載入時 factory 會直接拒絕（`src/llama-model.cpp` 的 `llama_model_mapping`），部分轉檔路徑也會在轉檔時就擋下（例如 MiniCPM 的 MoE 變體、DeepSeek-V4 文字模型）。其他已刪架構的轉檔類別還留著，轉得出 GGUF，但引擎載入不了。所以「接入新模型」在 DenseX 通常不是「新增一種架構」，而是「讓一個新的模型檔在既有架構上正確、穩定地跑起來」。

開始前，如果 `AGENTS.md`、`docs/densex/rules/CODE_STYLE.md`、`docs/densex/rules/TESTING.md` 還不在上下文裡，先讀完。

## 全程要遵守的事

- **精度和穩定度是門檻**（`AGENTS.md` 基本規則 10）：新模型要和參考實作或參考路徑比對過，才算支援。「能載入、能回答」不等於正確。
- **行為改變放在預設關閉的開關後面**（基本規則 2）：為了新模型改到共用路徑時，既有模型的輸出必須逐字不變。
- **設計由使用者（擁有者）決定**：要支援哪個模型、要不要為它改共用程式，是擁有者的決定。AI 負責查清楚、提出做法和實作，不替擁有者拍板。
- **用一個 GitHub issue 追蹤**：關閉前留言結果（數字、commit hash）；issue 是公開的，不寫 agent 花費、本機絕對路徑或個人資料。
- **模型檔和原始結果是唯讀的**：不改、不刪、不覆寫研究工作區裡的模型檔；新轉出的 GGUF 放在新的檔名。
- **上游凍結**（基本規則 7）：上游 llama.cpp 之後才支援的模型，只能當參考；要移植上游程式碼前先問擁有者。

## 第 0 步：判斷是哪一種情況

先問清楚（使用者沒講的，自己去查 `config.json`、GGUF metadata 或模型卡）：
1. 哪個模型？HF repo id 或檔名、來源（HF 為主，Pirate Face 備用）。
2. 是現成的 GGUF，還是要從 HF 權重自己轉？
3. 有沒有視覺編碼器（mmproj）？
4. 它是主模型，還是 draft 模型（DFlash、MTP 頭）？

然後對照下表，決定走哪條路：

| 情況 | 怎麼判斷 | 走哪條路 |
|---|---|---|
| Qwen 3.8 系列的新權重來源、新量化檔位，架構與 tensor 都和現有的一樣 | GGUF 的 `general.architecture` 是 `qwen35`，tensor 名稱與形狀和已支援的模型一致 | 路線 A |
| 同架構，但 hparam、tensor、chat template 或 tokenizer 有差異 | 載入失敗、缺 tensor、多出 tensor、template 不同，或 fine-tune 改了特殊 token | 路線 B |
| 新的 draft 模型 | 要當 `--draft-model` 用，或是 MTP 頭 | 路線 C |
| 不是 `qwen35`／`dflash`，例如 MoE、其他廠商、Qwen 的其他世代 | 架構名稱不在上面兩個裡 | 路線 D：停下來問擁有者 |

不確定屬於哪一條時，先實際載入一次看錯誤訊息，再判斷；不要先寫程式。

## 路線 A：同架構的新模型檔（通常不改程式）

1. 檢查 GGUF metadata：用 `gguf-py` 的 dump 工具看 `general.architecture`、hparam、tensor 清單、內嵌的 chat template、tokenizer 類型。和一個已支援的 Qwen 3.8 27B 檔逐項比對，差異列出來。
2. 載入並跑基本對話：temperature 0、思考關閉，確認回答正常、會停在 EOS。
3. 跑精度與穩定度的驗證（見下面「驗證清單」）。
4. 不需要改程式時，不要為了「支援」而加程式碼。結果寫進 issue，必要時更新對外的支援模型說明。

## 路線 B：同架構，但有差異

先找出差異到底在哪一層，再決定改哪裡：

| 差異 | 改哪裡 | 原則 |
|---|---|---|
| 轉檔時 tensor 名稱或形狀不同 | `conversion/qwen.py`（`model_arch = gguf.MODEL_ARCH.QWEN35` 的類別）、`gguf-py` 的 tensor mapping | 常數性的 tensor 修改、置換、切塊在轉檔時做，不要放進計算圖 |
| 載入時缺少或多出 hparam／tensor | `src/models/qwen35.cpp` 的載入段、`src/llama-model.cpp` | 選填的用選填方式讀；關鍵的缺少時直接報錯 |
| chat template | 轉檔時寫進 GGUF | 不要在 C++ 裡寫死預設 template |
| 工具調用或輸出解析 | 先用既有的 autoparser 試（`test-chat-auto-parser <jinja>`） | 既有的能處理就不寫專用 parser |
| tokenizer、特殊 token | 轉檔程式的 tokenizer 段 | 重用既有的 pre-tokenizer 要說明理由並測試 |

每一處要改的地方：
- 先給使用者看既有的對應程式碼，說明這個模型哪裡不同，再寫新版本；
- 改到 `qwen35.cpp` 或共用路徑時，既有的 Qwen 3.8 27B 輸出必須逐字不變（regress 的 golden 比對）；
- 新程式碼遵守 `CODE_STYLE.md`：錯誤訊息開頭標模組名稱、不默默退回預設值、新註解用中文。

## 路線 C：draft 模型（DFlash、MTP）

- DFlash 的 draft 模型走 `dflash` 架構：轉檔類別是 `conversion/deepseek.py` 的 DSpark 路徑（`model_arch = DFLASH`），計算圖在 `src/models/dflash.cpp`。
- MTP 頭屬於 `qwen35` 主模型的一部分，相關邏輯在 `src/models/qwen35.cpp`。
- draft 模型的驗證重點和主模型不同：
  - temperature 0 時，開不開推測解碼，輸出必須逐字相同（推測解碼只能改變速度，不能改變結果）；
  - 量接受率與實際速度，和現有的 draft 模型比較；
  - DFlash 跑 `regress.ps1 -Full -Long`，MTP 跑 `regress.ps1 -Draft mtp -Full -Long`。
- 不要在主模型的改動裡附帶未驗證的 draft 骨架；draft 支援另外做成一次改動。

## 路線 D：新架構（超出範圍）

DenseX 只支援稠密的 Qwen 3.8 系列，這是擁有者定的方向；擴大支援範圍屬於重大決策，要走 `AGENTS.md` 基本規則 11 的重新審議流程。遇到其他架構時：
- 不要動手加架構，也不要把已刪除的架構從 git 歷史救回來。
- 停下來，依 `AGENTS.md` 基本規則 8 給擁有者 2～3 個做法，例如「不支援」「改用已支援的同級模型」「開 issue 評估新增架構的成本」，附上利弊與前提，傾向放最後。
- 擁有者決定要支援時，另外開 issue 討論設計，再參考 `docs/development/HOWTO-add-model.md`（上游的新增架構流程，部分步驟在 DenseX 已不適用）。

## 視覺編碼器（mmproj）

DenseX 保留看圖功能（`tools/mtmd/`）。Qwen 3.8 系列附 mmproj 時：
- mmproj 用 `--mmproj` 載入；誤用 `-m` 載入時，引擎會提示改用 `--mmproj`。
- 用固定圖片、固定問題、temperature 0 做端到端測試，記錄圖片 SHA256，和參考路徑比對回答。可以用 repo 裡的 `tools/mtmd/test-1.jpeg`。
- 需要新的前處理器、新的 projector 邏輯或改動 `libmtmd` 共用部分時，先完成文字模型，編碼器另外做成一次改動。

## 常見陷阱

- 同一個 hparam 不要在 Python 轉檔和 C++ 載入兩邊都驗證；選一層負責。
- 關鍵 hparam（少了會輸出錯誤或當掉的，例如 norm-eps、layer 配置）缺少時直接報錯，不默默用預設值。
- 轉檔時把自訂的結束 token 標成 `eot` 不一定夠：長篇或 agent 式生成時，模型可能把結束序列當成一般文字輸出。要實測長生成會不會正常停止。
- 不要在 layer 迴圈裡建立不隨層變化的 view／index tensor，會造成過多的計算圖分割。
- 傳給 flash attention 的自訂 KQ mask，FA 開啟時要先轉成 F16。
- 不用自訂的 sin/cos 繞過 RoPE；`ggml_rope_ext` 表達不了時先開 issue。
- 單純乘上常數 scale（`weight * scale`）通常在推論時套用，不要折進權重：會改變權重的數值範圍，讓量化變差。
- 量化檔位不同，精度也不同：比較時用同一個檔位，或明確寫出檔位差異。

## 驗證清單

依改動範圍執行；沒跑的照實寫，不要宣稱通過。

1. **轉檔正確**（有自己轉檔時）：用 `examples/model-conversion` 的工具比對原始 HF 模型與轉出的 GGUF 的 logits。
2. **輸出正確**：temperature 0、固定 seed，和參考實作或參考路徑比對輸出；改到共用路徑時，用 `../bench/ab.py` 對改動前的 build 比對（`../` 以 repo 根目錄為基準，同 `docs/densex/rules/TESTING.md`），`exact_match` 必須全部是 true。
3. **分佈接近**：量化檔跑 KLD 或 perplexity，和同模型的參考檔比較。
4. **長上下文**：至少跑一次 4K 和 120K，檢查輸出品質與穩定性，不只看速度。
5. **工具調用**：`toolcall_check.py` 14/14（regress 第 6 層）。
6. **回歸**：`regress.ps1` 結尾是 `REGRESS_PASS`；改到推測解碼時，DFlash 與 MTP 都跑 `-Full -Long`。不准為了讓測試通過而改 golden 或放寬門檻。
7. **效能與記憶體**：用 `../bench/ab.py` 量 TTFT、prefill、生成速度、VRAM、Shared VRAM，交錯至少 3 輪，報中位數與範圍。目標機器是 12 GB VRAM，要寫出能跑的最大上下文。
8. **程式風格**：對照 `CODE_STYLE.md` 重新檢查每個改到的檔案，這是和功能測試分開的一輪。

## commit 之前

- 對 diff 跑 `code-review` skill。
- 依 `TESTING.md` 用建置腳本建置，並跑完上面適用的驗證。
- 一個 commit 只做一件事；commit 訊息寫改了什麼、為什麼、開關與預設值，以及根據的量測（基本規則 4）。
- 確認使用者能解釋每一行改動，也完整看過 diff。
- 關閉 issue 前留言結果（數字、commit hash）。
