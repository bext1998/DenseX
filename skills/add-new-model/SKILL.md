---
name: add-new-model
description: 引導在 DenseX（獨立的 llama.cpp fork）新增模型架構的流程。使用者要新增或移植模型架構時使用。
---

# 在 DenseX 新增模型架構

DenseX 是獨立的 llama.cpp fork：
- 不追蹤也不合併上游，也不收 pull request。
- 工作以 GitHub issue（https://github.com/bext1998/DenseX/issues）加上 `densex` 分支的 commit 完成，見 `AGENTS.md`（具權威的規則）與 `CONTRIBUTING.md`。
- 本 skill 沿用上游文字時提到的「PR」，請讀成「這次改動」，也就是它的 issue 加上 commit。

本 skill 帶貢獻者一步步新增模型架構。本專案允許 AI 產生的程式碼，所以下面每一步你都可以寫出完整實作，而不只是指出可參考的模式；但全程都要遵守 `AGENTS.md`：

- **貢獻者對每一行負完全責任**，不論是怎麼寫出來的。他們必須能向審查者解釋並辯護任何一段。邊做邊和他們確認，不要默默產生全部再交出完整的 diff，這樣他們才真的吸收了寫出來的東西。
- **設計由貢獻者決定**：寫程式之前，確認這個架構的設計選擇由貢獻者決定，例如要參考哪個模型、RoPE 變體或 MoE 路由這類非標準部分要怎麼處理。AI 是加速貢獻者已經做好的設計，不是替他們做設計。
- **用一個 GitHub issue 追蹤**：關閉前留言結果（數字、commit hash）。
  - issue 是公開的，不寫 agent 花費、本機絕對路徑或個人資料。
  - 本 repo 沒有 PR 範本，也沒有 AI 揭露段落。
- **commit 訊息自己寫**，依 `AGENTS.md` 基本規則 4：
  - 內容寫改了什麼、為什麼、開關與預設值，以及根據的量測。
  - 交辦給的 commit trailer 要照抄。本 repo 用 `Co-Authored-By:` 表示 AI 共同作者（見 `git log`）。
- **改動看起來很大**，或引入了這裡沒涵蓋的新模式：先停下來，告訴使用者這類改動要先在 issue 裡把設計定案。
- **改動要自成一體。**
  - 如果需要在新模型檔以外做很多非慣例的修改，例如以其他模型不需要的方式改動共用的建圖程式、sampler 或核心 API：停下來，先在 issue 定案設計。
  - 侵入性的改動要拆開。
- **不要把無關的工作綁進這次改動**：多模態、chat template／解析的細節見下面第 4、5 步。
- **絕不用自訂的 sin/cos 實作繞過 RoPE。** 如果現有的 `ggml_rope_ext`（見第 2 步的 RoPE 提示）真的表達不了這個模型的需求，先開 issue 討論，不要自己寫 RoPE 實作。

開始前，如果 `AGENTS.md`、`CONTRIBUTING.md` 和 `docs/development/HOWTO-add-model.md` 還不在上下文裡，先讀完。另外跑 `git log --oneline -- src/models`，讀最近幾個新增模型的 commit 和 diff。這比文件更能反映目前的慣例，因為文件可能落後。

## 第 0 步：範圍與重複檢查

問貢獻者：
1. 哪個模型（HF repo id 或名稱）？只有文字，還是有多模態（視覺／音訊）編碼器？
2. 本機是否已經有 HF 的 `config.json` 和權重？
3. 是否查過有沒有這個模型的既有 issue？建議用 `gh search issues "<model name>" --repo bext1998/DenseX`。已有 issue 涵蓋時，到那裡留言、協作，不要另開重複的。
4. 這個模型最接近哪個既有支援的架構？例如「類 Llama、有滑動視窗」「類 DBRX 的 MoE」「BERT 式編碼器」。

如果貢獻者不知道最接近的參考架構：
- 你可以在 `conversion/*.py` 和 `src/models/*.cpp` 裡找設定形狀相近的架構，例如層數、head 數、MoE expert 數、norm 位置，建議 1～2 個候選。
- 但要讓貢獻者確認，不要自己挑。這個選擇是他們要負責的設計決定。

貢獻者回答完這些問題、並指定參考架構之前，不要進行第 1 步。

## 第 1 步：把模型轉成 GGUF

實際要改的地方照 HOWTO-add-model.md 第 1 節做，例如轉檔類別的註冊、`constants.py`、`tensor_mapping.py`。不要在這裡重新推導，從文件讀。

本 skill 額外要求：
- 每個要改的地方，先給貢獻者看他們在第 0 步指定的參考架構中對應的程式碼，再寫新版本。
- 確認他們理解自己的模型有哪裡不同，例如非標準的 tensor 形狀、額外的 hparam；不要只是默默照抄模式。

## 第 2 步：在 llama.cpp 定義架構

實際要改的地方照 HOWTO-add-model.md 第 2 節做，例如 `llm_arch` enum、`LLM_ARCH_NAMES`、hparam 載入、RoPE 類型的 case；包括它的「Tips and tricks」一節裡關於 `ggml_rope_ext` 的陷阱。

本 skill 額外要求：絕不用自訂的 sin/cos 實作繞過 RoPE，見上面的 RoPE 規則。

## 第 3 步：建立 GGML 計算圖

實際要改的地方照 HOWTO-add-model.md 第 3 節做，例如 `src/models/<name>.cpp` 的 struct、`llama_model_mapping` 的註冊。

本 skill 額外要求：寫 `src/models/<name>.cpp` 之前，至少讀 `src/models/` 底下 10 個其他檔案。要混著挑，不能只看參考架構那一個；用來確認你要寫的 struct 配置、命名和風格符合目前的慣例。模式會隨時間漂移，HOWTO 文件可能落後。

## 第 4 步（選做）：多模態編碼器

只有貢獻者在第 0 步提到有視覺／音訊編碼器時才做。實際要改的地方照 HOWTO-add-model.md 第 4 節和 `docs/multimodal.md` 做，例如 `MmprojModel` 子類別、`clip.cpp`、`mtmd.cpp`，以及 `tools/mtmd/models` 裡的編碼器計算圖。

本 skill 額外要求，請仔細讀：**多模態編碼器能不能和文字模型的基本支援放在同一次改動，取決於改動有多符合慣例。**
- **可以放在一起**：編碼器支援符合慣例，也就是不需要新的基礎設施或邏輯，只是一個新的 cgraph，重用既有的前處理／projector 機制。例如 siglip、pixtral、qwen 只加一個新 projector。
- **要拆開**：只要超出這個範圍，例如新的前處理器、非標準的 projector 邏輯，或改動共用的 `libmtmd` 基礎設施／邏輯：
  - 停下來，告訴貢獻者這不符合慣例；
  - 先完成文字模型，編碼器另外做成後續的獨立改動。
- 不要讓這個決定默默帶過：寫任何 `clip.cpp`／`mtmd.cpp` 程式碼之前，明確向貢獻者說明。

## 第 5 步（選做）：chat template／解析支援

只有模型需要新的內建 chat template（`src/llama-chat.cpp`），或新的輸出解析器時才做；解析器見 `docs/development/parsing.md` 和 `docs/autoparser.md`。

如果需要的超出使用者自備的 Jinja template 已經能涵蓋的範圍，就當成後續的獨立改動，不要放進基本的模型支援改動。要明確向貢獻者說明，不要默默綁進去。

## 常見陷阱（來自過去的審查）

這些在過去新增模型的審查意見中很常出現，值得主動檢查，不要等審查者抓：

- 同一個 hparam／設定假設，不要在 Python 轉檔腳本和 C++ 載入路徑都驗證一次。選一層負責檢查，重複只會增加維護面。
- 某些設定裡確實可能不存在的選填 hparam，例如 shared-expert 數，要用明確的選填／fallback 存取方式讀取，不能假設一定有。
- 真正關鍵的 hparam，也就是少了它模型會輸出錯誤或當掉的，例如 `sliding_window_pattern`、norm-eps：缺少時必須直接報錯，不能默默退回預設值。
- 不要把預設 chat template 寫死在 C++ 執行檔裡，要在轉檔時寫進 GGUF。同一個 `llm_arch` 可能被多個 template 不同的 fine-tune 重用，寫死的 C++ 預設會對它們默默失效。
- 寫專用的工具調用／輸出解析器之前，先確認既有的 autoparser 是否已經能處理這個 template。`test-chat-auto-parser <jinja>` 會顯示它偵測到什麼。
- 轉檔時把自訂的 EOS／結束標籤 token 標成 `eot`，不一定足夠。
  - 在長篇或 agent 式生成中，模型可能把結束序列當成一般文字輸出，而不是那個 token；結果生成不會因 EOG 停止，原始文字會漏過解析器。
  - 要驗證這種情況，不能只驗證 token 路徑。
- 為了方便而重用或別名既有的 pre-tokenizer 時，要明確說明理由並測試。默默重用很容易造成細微的 tokenizer 錯誤。
- 注意在 layer 迴圈裡建立每層的 view／index tensor，造成過多的計算圖分割。不隨層變化的 tensor 要移到迴圈外。遇到 `GGML_SCHED_MAX_SPLIT_INPUTS` 時特別相關。
- 傳給 flash attention 的自訂 KQ mask，dtype 必須符合 FA 的要求：FA 開啟時，傳給 `build_attn_mha` 之前先轉成 F16。
- 把自訂 KV cache 大小補齊到對齊值時（例如 `GGML_PAD(..., 256)`），要在其他大小調整全部完成後才補齊，不能在之前。否則後面的邏輯可能又讓它失去對齊。
- 非標準的 cache／SWA（滑動視窗注意力）語意，要覆寫專用的 hook，例如 `llama_model_n_swa()`，不要改 hparam 來假裝這個行為。hparam 可能在別處被用在無關的用途。
- 基本模型改動不要附帶未完成或未驗證的推測解碼（例如 MTP）骨架。如果還沒確認它能用，就拿掉，另外做成後續改動。
- 轉檔程式應該呼叫基底類別既有的 hparam 邏輯，例如 `super().set_gguf_parameters()`，而不是重新推導。大段重複 `TextModel`／`MmprojModel` 已有功能的程式碼會被標成多餘。
- 常數性的 tensor 修改（例如 `norm(1 + weight)`）、置換和切塊，要在轉檔時做，不要放在計算圖裡。
  - 見 HOWTO-add-model.md 的「Prefer conversion-time tensor modifications」提示：Gemma 3 把 `1 +` 折進權重，Qwen3-Next 在 `modify_tensors` 裡做置換。
  - 在計算圖裡於執行期做這些事，很可能被認為過度複雜而被拒絕。
  - 真的沒辦法在轉檔時做，先開討論說明原因，不要直接在計算圖裡實作。
  - 例外：單純乘上常數 scale（`weight * scale`），通常在推論時套用比較好，不要在轉檔時折進權重。概念上這個 scale 作用在 activation 而不是權重；折進去可能損害數值穩定性，也會改變權重的數值範圍，讓量化變差。

## 驗證清單

參考：`examples/model-conversion/README.md`。

1. 轉成 GGUF，然後檢查並執行原始與轉換後的 tensor。
2. 驗證 logits（原始對轉換後）。
   - 如果這個模型是已支援系列的新版本，先確認「前一版」仍通過 logits 驗證。數值差異可能早就存在，不是這次的改動造成的。
   - 完整 logits 驗證的工具在 `examples/model-conversion`。
3. 量化（相關時包括 QAT 變體），再驗證一次。
4. 跑 perplexity 評估（簡易版與完整版）。
5. 在 `tools/cli`、`tools/completion`、`tools/imatrix`、`tools/quantize` 和 `tools/server` 各做一次基本檢查。
6. 先做 CPU backend；其他 backend 可以是後續的獨立改動（`docs/densex/rules/TESTING.md`）。
7. 對照 `docs/densex/rules/CODE_STYLE.md` 的程式與命名規範，重新檢查每個改到的檔案。
   - 這是和功能測試分開的一輪，同樣重要。
   - 檢查項目包括：不強制換行、程式碼不用 unicode 標點、註解語言符合規定（`ggml/` 用英文 ASCII，其他用中文）、註解精簡不重複、`snake_case` 命名（檔名用 `kebab-case`）、縮排與大括號風格一致等。

## commit 之前

- 先對 diff 跑 `code-review` skill。它能抓到審查者最常指出的慣例與範圍問題，值得在 commit 前先在本機做。
- 跑這次改動需要的檢查（`CONTRIBUTING.md` 的「Testing before you commit」）：
  - 用建置腳本建置：`scripts/densex/build.ps1`，研究工作區裡是 `../build.ps1`；
  - 再跑 `regress.ps1`，結尾必須是 `REGRESS_PASS`。
- 確認貢獻者能向審查者解釋每一行改動，並準備好被問到任何一部分。不論有多少程式碼是 AI 產生的，這都是必要條件。
- 確認他們完整、仔細地看過整份 diff，而不只是掃過。
- 關閉 issue 前留言結果（數字、commit hash）；issue 裡不寫花費、本機絕對路徑和個人資料。
