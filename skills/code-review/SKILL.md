---
name: code-review
description: 在 commit 之前，依專案慣例與審查者常見的意見審查 DenseX 的改動。使用者要審查 diff、commit 或分支時使用。
---

# 審查 DenseX 的改動

DenseX 是獨立的 llama.cpp fork：
- 不送上游 PR；工作以 GitHub issue（https://github.com/bext1998/DenseX/issues）加上 `densex` 的 commit 完成。
- 本 skill 沿用上游關於 PR 的文字時，「PR」請讀成「這次改動」，也就是它的 issue 加上 commit。

本 skill 依專案慣例，以及審查者最常指出的陷阱審查改動，讓作者能在 commit 前修好。有兩種模式：

- **自我審查（預設）**：把貢獻者自己的本機改動當成 commit 前的檢查。改動可以是還沒 commit 的工作，或正在審查的 commit。
  - 不清楚是哪一種時就問。
  - 預設看 `git diff HEAD`；有指定 commit 時，再加上 `git show <commit>`。
- **唯讀審查某個改動或檔案**：使用者指定 commit 或特定檔案時（包括不是他們寫的程式碼），審查那些內容並回報發現。

兩種模式的產出都是**給使用者自己閱讀、處理的私人審查筆記**，絕不是要張貼出去的東西。
- 透過 `gh` 張貼任何內容（審查留言、issue 回覆）或 push，都需要擁有者明確同意，依 `AGENTS.md` 基本規則 4。
- 發現只在對話中呈現。

開始前，如果 `AGENTS.md` 和 `CONTRIBUTING.md` 還不在上下文裡，先讀完。本審查執行的基準是 `docs/densex/rules/CODE_STYLE.md` 和 `CONTRIBUTING.md` 的「AI usage」一節。如果 diff 新增了模型架構，也要讀 `docs/development/HOWTO-add-model.md`，並考慮改用專用的 `add-new-model` skill。

## 第 0 步：確定 diff 範圍，挑選清單

找出實際改了什麼、要套用下面哪些領域清單。跑 `git diff --stat`（或 `git show --stat <commit>`），把改到的路徑分類：

- `conversion/`、`gguf-py/`、`src/models/`、`src/llama-arch.*` -> **新模型／架構**
- `ggml/`（任何 backend、op 或 `ggml.h`）-> **ggml／backend**
- `include/llama.h` 與其他公開標頭 -> **公開 API**
- `tools/server/` -> **Server**
- `tools/ui/`（或舊的 `tools/server/webui/`）-> **WebUI**
- `tests/`、`tools/server/tests/`，或任何新增、修改的測試 -> **測試**
- 其他所有東西，加上以上全部 -> **一般**（一定要跑）

固定要跑三份清單：**範圍與快速退件關卡**、**安全審查**、**一般**。

另外：
- 改到哪些路徑，就跑對應的領域清單。
- 如果 diff 引入了新元件、子系統或基礎設施，就加跑**做法與設計**審查。新元件包括新檔案、類別或模組、新的抽象，或手刻的機制。
- 告訴使用者你要跑哪些清單、為什麼。

## 範圍與快速退件關卡（一定要跑）

這些模式會讓改動不經完整審查就被退回。先檢查這些：這裡的發現比任何程式細節都重要，因為它可能代表這個改動在目前的形式下根本不該做。

- **有沒有先開 issue？** 工作應該從 issue 開始，而不是從程式碼開始（`CONTRIBUTING.md` 的「Where work comes from」）。改動不簡單又沒有連到 issue 時，指出來，建議先開一個。
- **是否和既有或進行中的工作重複？** 建議用 `gh search issues --repo bext1998/DenseX <term>` 查這個功能。與已排入的工作重複的會被關閉。
- **是否自成一體、只有單一目的？** 多個無關的改動或最佳化綁在一起時，應該拆成不同的 issue／commit。指出無關的改動。
- **是否同時改了多個 ggml backend？** 初次支援應該只做 CPU，其他 backend 當後續改動。指出功能第一次改動就綁進的 CUDA／Metal／Vulkan 等改動。
- **是否新增了 `ggml_type`／量化類型？**
  - 這會帶來不成比例的維護負擔，需要完整的理由資料：GGUF 樣本上傳、perplexity 對 FP16／BF16 與相近大小的比較、KL 散度資料、CPU 效能數字。
  - 缺少這些，不論程式品質如何都會被拒絕。
- **是否有侵入性？** 例如新子系統、改變核心 API 的形狀，或改動其他模型不需要的共用計算圖／sampler 程式碼。指出來，建議先和維護者討論再繼續投入。
- **是否在任何執行檔或轉檔腳本加了模型專用的 CLI 參數？**
  - 範圍包括執行檔（`common/arg.cpp`、tools、examples）和轉檔腳本（`convert_*.py`）。
  - 這是不允許的：模型專用的行為必須來自 GGUF metadata，或自動偵測，不能靠針對單一模型的旗標。
- **是否小眾或綁定特定廠商，帶來沒人會長期負責的維護負擔？** 指出維護歸屬的問題。
- **改動在語意上正確嗎？** 還是看起來合理、其實誤解了程式碼的「修正」？要檢查實際行為，不能只看能不能編譯。
- **AI 使用**：如果 AI 有實質貢獻，commit trailer 和 issue 留言裡是否看得出來（`CONTRIBUTING.md` 的「AI usage」）？絕不隱藏 AI 的參與。

## 安全審查（必做）

每次審查都必做，這裡的任何發現都是**阻斷性**的。

原則：GGUF metadata、tensor 形狀、tokenizer／grammar 輸入，以及所有 server／RPC 欄位都可能被攻擊者控制，使用前必須設上限。

- **從 tensor 維度算出的大小／數量**：配置前先驗證。
  - 像 `ne[i]*nb[i]`／nbytes 這類乘積，在刻意構造的維度下可能溢位，配置出太小的空間，接著發生 heap overflow。
  - 溢位檢查必須在它保護的運算「之前」執行。補齊／對齊的巨集在接近 `SIZE_MAX` 時會繞回 0，所以放在補齊之後的檢查會直接通過。
- **GGUF 字串／陣列**：用宣告的長度和元素數決定迴圈或緩衝區大小之前，先設上限。把陣列轉成指標，或讀固定索引（`[i+1]`、`[0..2]`）之前，先驗證元素類型和長度。
- **元素類型混淆**：把 `gguf_get_arr_data()` 或 `tensor->data` 轉成 `float *`／`int32_t *` 之前，要先檢查元素類型。
  - GGUF 陣列：先確認 `gguf_get_kv_type() == GGUF_TYPE_ARRAY`，再看 `gguf_get_arr_type()`；tensor：確認 `type == GGML_TYPE_F32`。
  - `UINT8` 陣列或 `I8` tensor 能通過所有長度檢查，接著卻被每個元素讀 4 bytes。旁邊有長度檢查，不等於有類型檢查。
- **載入器**：對來自檔案的值用 `GGML_ASSERT` 會讓整個程序中止。呼叫端已經會接例外的地方，改用 throw，例如 vocab、模型載入器、clip。
- **用檔案提供的數量索引固定陣列**：索引之前先限制數量，例如用層數／區塊數索引 `LLAMA_MAX_*` 陣列。注意那些只在某個選填鍵存在時才會觸發的檢查。
- **宣告長度與實際陣列長度**：要拿 GGUF 陣列宣告的長度和實際讀到的數量比對，不能只和緩衝區大小比。
- **邊界比較**：指出縮窄轉型（`size_t`->`int32_t`）和有號／無號混用。這些可能繞過長度檢查，複製超出緩衝區。
- **解析或推導出的索引**：檢查 `stoi`／`atoi` 結果的範圍，並接住解析時拋出的例外。沒有邊界檢查，絕不把預設或推導出的 token id（EOS／BOS 等）當索引。
- **重用或預留的緩衝區**：緩衝區縮小或重用後，要重新檢查邊界。
  - 注意先 `reserve()` 再依假設的大小索引；
  - 注意在檢查長度之前就讀取的標頭欄位。
- **Server 的 JSON 整數**：客戶端提供的整數（token／捨棄數、偏移量），在進入索引或指標運算之前，要限制在非負數且有上限。
- **RPC 反序列化的欄位**：每個欄位（type、buffer、data、ne、nb、op_params）都當成惡意輸入，使用前先驗證。下面這些情況都會造成任意讀寫：
  - null 或零值的 buffer 跳過驗證；
  - 攻擊者給的 data 指標；
  - 超出範圍的類型索引；
  - 負的 stride 在符號延伸後，越過只檢查角落的 assert。
- **生命週期／UAF（use-after-free）**：指出以下情況：
  - 儲存指向呼叫端或暫存空間的原始指標；
  - 快取指向之後會被釋放的緩衝區的指標；
  - 來源可能在完成前被釋放的非同步操作；
  - 在 free／realloc 時沒有失效的結構。
  - 另外，有條件建立或「非必要」的 tensor，解參考前要先檢查是否為 null。

## 做法與設計（引入新元件或基礎設施時）

只要 diff 新增了元件、子系統或基礎設施，就跑這一段。

審查太常停在「能不能動」：diff 可能正確，但做法是錯的；而長期來看，混亂的設計比 bug 代價更高。
- 要評估「做法」，不只是行為。提出更乾淨的做法是高價值的發現，不是挑小毛病。
- 如果看到更好的設計，具體描述它，不要只說目前的設計不好。

- **更上游的簡單做法**：最大的改善常常是換一種資料模型或設計，直接去掉整個子系統，而不是微調已經寫好的程式碼。複雜度必須由問題本身來證明有必要，而不是因為第一個能動的做法剛好比較複雜。
- **重用優先於重造**：新增之前，先搜尋有沒有現成的 helper、函式庫、物件或機制。重新實作 codebase 已有的東西，會把已解決的 bug 帶回來，也增加維護面。
- **清楚的擁有權與生命週期**：優先用 RAII 和明顯的擁有權，不要手動的存活旗標、手動追蹤的指標和「它還活著嗎？」的檢查。手動追蹤生命週期是細微 bug 的常見來源。
- **大小剛好的機制**：指出多餘、殺雞用牛刀或比需要更重的原語與抽象；只用設計真正需要的最少機制。
- **正確的結構與契合度**：
  - 新型別要有存在的理由，一個型別扮演兩種角色就拆開；
  - 遵循既有的模式、慣用法和命名，避免專案刻意不用的寫法。
- **根因與症狀**：修正疊在修正上，代表設計需要修正，而不是再加一層防護。

## 新模型／架構

完整流程見 `add-new-model` skill 和 `docs/development/HOWTO-add-model.md`；這裡是審查時最常被抓到的部分：

- 真正的依賴是設定或能力值時，不要依 `model.arch` 分支：依 hparam／能力判斷，不要依架構 enum。
- 如果模型是既有架構的近親變體，差異有正當理由嗎？優先重用或繼承既有的架構／模型類別，不要複製一份。近乎重複的類別或 `src/models/<name>.cpp` 會被要求和它的兄弟合併。
- 新的 tensor 名稱要經過 `tensor_mapping.py`，不要臨時做名稱比對。
- QKV 要用 `ggml_view` 切 *activation*，不要切 *權重* tensor；依賴 ggml 的 broadcasting，不要手動複製 tensor。
- 新的計算圖輸入要在建圖函式的開頭宣告，不要在第一次用到的地方才宣告。
- 模型少了就無法正確執行的 hparam 必須是必填：缺少時直接報錯，不能用預設值默默帶過。只有在不同設定間真的可能沒有的值，才用 fallback 存取方式。
- 新的或選填的權重 tensor（scale 等）必須經過 `build_lora_mm` 和既有的 helper，符合慣例；不要留下從其他架構複製來的原始 matmul。
- 不要用自訂的 sin/cos 實作改 RoPE。如果 `ggml_rope_ext` 真的表達不了，那是要開 issue 討論的事，不是直接改。
- 要測量化 KV 的路徑（`-ctk`／`-ctv q8_0`），不能只測預設的 f16。新的推測解碼或注意力功能常常在那裡默默壞掉。
- 複製程式碼時，保留既有說明模型特殊行為的註解，並註明來源（「copied from X, with Y added」）。
- 移除改寫參考實作後留下的死程式碼和分支。

## ggml／backend

- `supports_op`，以及任何 dispatch 或門檻條件，範圍必須剛好是這次要改的情況。只針對幾種量化類型的條件，不能默默停用或啟用其他所有情況。
- 不要寫死 warp／lane 大小：用 `ggml_cuda_get_physical_warp_size()`（CUDA 是 32，HIP／ROCm 是 64）和可移植的 helper。
- 審查前移除殘留的除錯、profiling 和 log 程式碼。
- 新增或修改了 op？更新 `docs/ops.md`，以及受影響 backend 的 `docs/ops/*.csv`。
- 新 op 或 op 的修改：
  - 要有對應的 `test-backend-ops` 案例；
  - DenseX 仍然附帶的 backend（CPU 和 CUDA）之間要一致。
- 新 kernel 應該附上具體的效能資料，也就是在實際 tensor 形狀下的吞吐量，而不只是正確性。
- 不要讓 backend 修改 cgraph 當捷徑。這是還沒解決的架構問題，不能偷渡進來。
- `ggml/` 的改動風險較高：跑受影響的 `test-backend-ops` 案例；CUDA 的話再跑 `sanitize.ps1`（`CONTRIBUTING.md` 的「Testing before you commit」）。
- CUDA：避免過度模板化 kernel，只在有明顯效能提升的地方使用。

## 公開 API（`include/llama.h`）

公開 API 的改動門檻比內部改動高，要先在 issue 裡把理由講清楚。審查重點：

- **理由**：為什麼既有機制不夠用？例如 `cb_eval`、既有的 batch／sampler 設定。
  - 如果夠用，這個改動多半不該新增公開介面。
  - 這是這類改動被拒絕最常見的單一原因。
- **實驗性或權宜的介面**放在旁邊的標頭（`src/llama-ext.h`），不要放進 `llama.h`。
- **保持精簡、通用**：
  - 一個通用的呼叫優於好幾個狹窄的便利包裝；
  - 新呼叫要能向前相容（例如混合模態的 batch），不要假設今天的形狀不會變。
- **C API 是一等、穩定、定義 ABI 的介面**：不要提議用平行的 C++ API 取代它。`llama-cpp.h` 維持一層薄的便利包裝。
- **型別與命名**：
  - 大小用固定大小的整數型別（`int32_t`；大小與偏移量用 `size_t`）；
  - 命名用 `snake_case`，函式是 `<class>_<method>`，也就是 `<class>_<action>_<noun>`；
  - enum 值全大寫，並以 enum 名稱為前綴；不透明型別加 `_t` 後綴。
  - 避免對既有的匯出函式做不必要的簽名或 ABI 改動。
- **每個新 API 都要在同一次改動裡附上實際使用它的範例或工具**：審查者要求把它接進 `server`、`embedding`、`perplexity` 等，常因此找到真正的 bug。

## Server（`tools/server/`）

- 這個功能在 server 定義的範圍內嗎？看 `tools/server/README-dev.md`，超出範圍的功能會被拒絕。
- 安全：
  - 不要信任客戶端提供的標頭，例如 `X-Forwarded-For`，也不要加入容易誤用的功能。
  - IP 白名單這類功能應該放在反向代理，除非有可信代理的設計。
- 新行為要正確接進既有的請求／回應和 checkpoint 路徑；注意跨請求的資源洩漏。

## WebUI（`tools/ui/`）

這些只適用於 `tools/ui/`（以前是 `tools/server/webui/`）底下的改動，也就是 SvelteKit + Svelte 5 前端。它們是前端慣例，不是整個 repo 的慣例，不要套用到 C++ 或其他工具的程式碼。

這個領域最主要的審查主題是「放在哪裡」：大多數被接受的細節意見，都是關於把某個值、函式或邏輯放到它應該在的那一層。

### 常數與 enum：判斷規則

**觸發條件**：
- 一個字面值被重用、具有領域意義，或跨越邊界，就要處理。跨越邊界的例子：線上／儲存格式、selector、key、共用的 UI 文字。
- 只在單一元件裡用一次的值留在原地，邊界見本節最後；指出這種情況只是雜訊。

新增之前，先搜尋 `$lib/constants`／`$lib/enums`：重用勝過新增一個幾乎重複的。各種東西該變成什麼：

- **字串 enum**（`$lib/enums`，`enum X { KEY = 'value' }`）：程式會據以分支、比較，或當類型判別值儲存的封閉類別值集合。例如角色、狀態、種類、類型、動作、排序欄位。
  - 任何寫成 `x.type === 'some-string'` 的東西，或 `.d.ts` 裡的封閉字串聯集，都應該是 enum。
  - key 用 SCREAMING_SNAKE，加領域前綴（`ModelDownloadStopRequest.CANCEL`）。
  - value 預設小寫，是儲存或線上傳輸的表示法，所以可以刻意和 key 不同（`TOOL_USE = 'tools'`、`MTP = 'mtp'`），或完全對應外部 API。不同時，要在 enum 的 JSDoc 說明（見 `HfModelSort`）。
  - 不用數值 enum。
- **以 enum 為 key 的常數物件**（constants）：查表或對應。例如每種檔案類型的圖示、每種能力的標籤、HTTP 代碼對應的訊息（`FILE_TYPE_ICONS`、`HTTP_CODE_TO_STRING`、`Record<Enum, string>`）。enum 負責列舉，物件負責對應。
- **宣告式設定陣列／物件**（constants）：一組相關的項目，被不只一個 renderer 使用，例如下拉選單加 sheet。
  - 選單項目的 `action`／`id`／`enabledWhen` 欄位是 enum 值，不是字串（`ATTACHMENT_FILE_ITEMS`）。
  - 兩個介面呈現同一個選單時，兩者都必須從這一份設定產生。
- **分組的 `as const` 物件**（constants）：屬於同一個功能的相關字面值。
  - 例如錯誤訊息分類（`ERROR_MESSAGES`，含 `HTTP`／`NETWORK` 群組）、多個介面共用的 UI 文字（`ATTACHMENT_MENU_TEXT`）、某功能的 prompt、fallback 和 pattern（`TITLE_GENERATION`）。
  - 優先用一個分組物件，不要把同一個功能的常數散落各處。
- **純量常數**：
  - 字串：CSS class 字串、selector、`data-*` 屬性名、標頭名、儲存 key、API 路由／路徑、副檔名。
  - 數字：任何門檻、逾時、大小、上限或數量，加上單位或情境後綴（`_MS`、`_PX`、`_BYTES`、`_CHARS`、`_MAX_LENGTH`）。
  - 正規表示式：任何不簡單的字面 regex，加 `_REGEX` 後綴，集中在該功能的 constants 檔。
  - 字元／分隔符：依它在領域中的意義命名，不依字元本身。例如 `PATH_SEPARATOR`、`ORG_SEPARATOR`、`SSE_RECORD_SEPARATOR`、`ISO_DATE_TIME_SEPARATOR`、`NEWLINE`。不要有通用的 `SLASH` 常數。
- **key 名稱常數**：被當成另一個物件或已儲存設定的 key 使用的字串，要有一個對應屬性名的物件常數（`SETTINGS_KEYS.CUSTOM_CSS = 'customCss'`）。這樣改名時會讓建置失敗，而不是默默對不上。

放置與組合：

- **放置**：
  - 跨功能共用的放在 `$lib/constants/<topic>.constants.ts`／`$lib/enums/<topic>.enums.ts`。
  - 功能專屬的值可以放在功能資料夾，但只限真正專屬於它的。
- **組合，不要重複**：常數可以從其他常數或 enum 推導。例如 `SANDBOX_TOOL_NAME = BuiltInTool.BROWSER_RUN_JAVASCRIPT`；儲存 key 由 `STORAGE_APP_NAME` 前綴組成。
- **測試和 Storybook story 也用同樣的常數／enum**：做結構性斷言時，不重新宣告它們要比對的字面值。
  - 例外：儲存或線上格式的契約斷言，保留獨立的字面值。那裡字面值本身就是契約。
  - 如果重用正式程式的常數，測試會變成同義反覆：常數一改，期望值跟著改，回歸照樣通過。

**什麼留在原地**：
- 單一元件裡只用一次的使用者文字、console／log 訊息，以及 import 路徑。
- 原地文字有了第二個使用者時，就升級成分組常數物件。

**相關的資料形狀規則**：空字串不是有效的「未設定」初始狀態，要用 `undefined`。

### 放置與分層

- **元件負責呈現，不擁有商業邏輯。**
  - 簡單的、呈現用的本地推導留在元件裡，用 `$derived` 從 props 和 store 值推導。
  - 要移出去的是：共用的商業邏輯、多個 store 的協調，以及命名／解析邏輯。這些應該由 store 方法、`$lib/utils` 函式或 `$lib/hooks` hook 負責。
  - 元件應該呼叫 `store.doSomething()`，而不是自己重新實作。
- **各層要分明**：
  - stores（`*.svelte.ts`，有狀態）、services（無狀態的客戶端）、utils（純函式）；
  - hooks（`*.svelte.ts`，可重用的反應性）、`$lib/types`（型別）、`$lib/constants`（常數）。
  - 如果新檔案模糊了這些分層，指出來，並說明它應該屬於哪一層。
- **有狀態的「service」是命名錯誤**：它其實是 store，或應該放在使用它的範圍裡。反過來，只有一個元件用的邏輯，做成一次性的 `.svelte.ts` 模組（實際上就是 store）是殺雞用牛刀；留在那個元件或 context 裡。
- **引入新抽象（store／service／hook／context）之前**，先問它會不會有第二個使用者；不會的話，邏輯留在既有範圍。先 KISS，之後再迭代。
- **Barrel 檔（`index.ts`）要符合既有的 barrel 風格**：
  - store 用檔案層級的 JSDoc，加上每個匯出群組的註解；constants／enums 用單純的 re-export 列表。
  - 只在匯出內容不夠自明時才加 JSDoc。
  - 常數和型別不從 store 的 barrel re-export。

### DRY 與死程式碼

- **先重用再寫**：新增之前，先搜尋有沒有現成的 helper、元件或 store 方法。截斷、複製／下載、檔案預覽、清理和格式化的 helper 都已經有了。
- **兄弟變體要共用來源**：例如同一個選擇器的 Dropdown 版與 Sheet 版，必須從同一份設定產生，不能重複 markup 和邏輯。
- **指出死程式碼**：包括註解掉的 markup，以及改名或移除的功能留下的殘跡。刻意保留但只是暫時的殘留，要加 `@deprecated` JSDoc，之後才找得到、移得掉。

### 元件與 Svelte 5

- 把臃腫的元件拆成專注的子元件，載入狀態另外做專用的 `*Skeleton.svelte`；元件內重複的 markup 用 `{#snippet}`。
- 用既有的 `$lib/components/ui` 原語，不要手刻 dialog、tooltip、input 或下拉選單。
- 只用 rune 語法：
  - `$props()` 解構、`$derived`、`$state`；只有簡單運算式做不到時才用 `$derived.by`。
  - `$state`／`$derived` 能表達的事，不要用 `$effect`。一整排 `$effect` 是設計的壞味道。
- window／document 事件用 `<svelte:window>`／`<svelte:document>` 處理，不要手動 add／remove listener。
- Svelte 5 的 `class` 屬性支援陣列：簡單的條件 class 不需要 `cn()`。
- 反應式集合用 `SvelteMap`／`SvelteSet`。

### 註解

- 和「一般」清單的標準相同，這裡執行得更嚴格：
  - 不寫重述程式碼的註解，明顯的函式不寫冗長的 JSDoc，不留註解掉的程式碼。
  - 範例和邊界情況的列舉放在單元測試，不寫成文字。
  - 不明顯的「為什麼」或理由註解要保留，因為測試沒辦法在程式碼的位置傳達那個背景。

### 型別、命名與風格

- 用 `===` 不用 `==`，用 `includes()` 不用 index 檢查；`Array.isArray` 檢查之後，不需要多餘的 null 合併。
- 同一個概念在整個 codebase 只用一種拼法，例如在 `cwd` 和 `currentWorkingDirectory` 之間選一個，到處都用它。名稱要明確、自我說明，不要耍聰明。

### 效能（串流 UI）

- 每個串流片段都會執行的東西必須便宜：
  - 先節流再計算，不是算完再節流；
  - 區塊層級的 regex 一次全域套用，逐行掃描時從後往前、能提早結束就結束；
  - 不要每次更新都重新解析整則訊息。
- 高度／展開的過場動畫要平順，並與 UI 其他部分一致。
  - auto-height 用 `interpolate-size: allow-keywords` 做動畫；
  - 不支援的瀏覽器用 `grid-template-rows: 0fr/1fr` 當 fallback（見 `CollapsibleRegion.svelte`）。

### 安全（前端）

- 清理任何來自模型或遠端資料的呈現內容：rehype／`{@html}` 管線、SVG 內容和 URL。用 `$lib/utils` 的 URL 清理函式，不要臨時自己檢查，簡單的情況也不要為此引入第三方函式庫。
- 代理標頭和 agent／工具的檔案系統存取是安全邊界。
  - 範圍限制必須端對端驗證，不能只在 UI 裡做；例如瀏覽根目錄的 sandbox。
  - C++ 那一側要找 server 的審查者一起看。

### `tools/ui` 專屬的規範

- **不 commit 建置產物**：UI 在本機建置，由 `scripts/ui-assets.cmake` 打包。
  - `npm run build` 在本機仍必須成功。
  - DenseX 唯一的 CI workflow（`.github/workflows/densex-ci.yml`）只跑 CPU，不建置 UI；所以 UI 建置壞掉只能在本機抓到，CI 抓不到。
- `npm run lint && npm run check`（以及 `npm run format`）必須通過；Storybook 測試失敗屬於阻斷性問題。
- 有使用者可見的互動或視覺效果的改動，要在 issue 留言附上簡短示範（影片或截圖），呈現改動實際運作的樣子。只動測試、重構、文件和設定的改動不需要示範。
- **範圍紀律**：屬於其他工作的改動，不要順便放進這次。指出來，移到該去的地方。也要指出對已棄用或舊檔案的無關修改，問是不是刻意的。
- 新的、不簡單的 utils 和解析邏輯要有 Vitest 單元測試；可重用的 UI 元件，可行時要有 Storybook story／測試。

## 多模態（`tools/mtmd/`）

- tensor 名稱必須以 `v.`、`a.`、`mm.` 或 `a.mm.` 為前綴。舊的命名不符合這個慣例，這是預期中的；但新程式碼要遵守。
- RoPE 不要用明確的 sin/cos，改用 `ggml_rope_ext`，見 `HOWTO-add-model.md`。如果它表達不了需要的行為，那是設計討論，不是快速修補。
- 新的 GGML op 不能在同一次改動引入，要另外做成獨立的改動。
- 大多數情況下，`build_vit` 足以建立視覺模型的 transformer 計算圖。
  - 除非有非常好的理由，不要自己寫迴圈手動建 transformer 計算圖。
  - 真的要這樣做，在 issue 留言和 commit 訊息裡說明原因。
- 如果需要專用的前處理器，它很有可能可以從既有的前處理器衍生。新增前處理器類別之前要仔細確認。
- 如果模型需要 `mtmd.h` 裡新的公開 API，先在 issue 定案。
- 音訊生成模型見 `tools/mtmd/README-dev.md`。

## 測試

- **遵循既有的測試模式**：不要新增新的測試系統。在 `tests/*` 底下新增檔案之前，先仔細想想測試能不能放進既有檔案。大多數情況下，新的測試案例屬於既有的測試檔：新增前先找有沒有涵蓋同一元件的檔案。
- **只加有意義的測試**：太瑣碎的測試只會讓測試套件和 CI 膨脹。
- **不要有時間敏感的測試**：例如時間門檻、sleep、與實際時間競速。它們在 CI 上不穩定。
- **注意會明顯增加 CI 時間的測試**：例如昂貴的計算、大型輸入、長時間 sleep／等待，或從網路下載大量資料（大模型、資料集）。指出來，並問較小的模型／輸入或既有的 fixture 是否就夠了。
- **Server 測試放哪裡**：
  - 內部元件的小測試（helper、`server_tokens`、queue、格式轉換器等）放在 `tests/test-server-component.cpp`。
  - `tools/server/tests` 的端對端 pytest 套件很花 CI 時間，只有在測試需要用真實模型驗證端對端行為時才加。

## 一般（一定要跑）

對每一行改動執行 `docs/densex/rules/CODE_STYLE.md` 的規範。這是和「程式能不能動」分開的一輪，對審查速度同樣重要：

- 程式碼本身只用 ASCII：不用 em dash、unicode 箭頭，也不要把 `x`、`...` 寫成 unicode 字元；改用 ASCII 的 `-`、`->`、`x`、`...`。
- 註解語言：
  - 新註解用正體中文；
  - `ggml/`（含 `.cu`／`.cuh`）例外，維持英文 ASCII，因為 nvcc 那條路徑沒有 `/utf-8`。在 `ggml/` 看到非 ASCII 的新註解要指出來。
- 註解要精簡，說明不明顯的「為什麼」，而不是「做了什麼」。要指出：
  - 冗長的註解；
  - 重述程式碼的註解；
  - 提到目前任務或 PR 的註解；
  - 依固定欄寬硬換行的註解。
- 文字和註解不要強制依固定字數換行，也不要把一句話拆成好幾行。
- 命名：
  - 名稱用 `snake_case`；
  - C/C++ 檔名用 `kebab-case`（小寫加連字號），標頭用 `.h`；Python 檔名用小寫加底線；
  - 命名採「最長共同前綴」（`number_small`，不是 `small_number`）。
- 格式：4 個空白縮排，大括號放同一行，`void * ptr`、`int & a`，行尾不留空白；配合周圍的風格。
- commit 之前，跑本 repo 現有的檢查（`CONTRIBUTING.md` 的「Testing before you commit」）：
  - 用建置腳本建置：`scripts/densex/build.ps1`，研究工作區裡是 `../build.ps1`；
  - 回歸測試 `regress.ps1`（`REGRESS_PASS`）；
  - 在本機跑 editorconfig 檢查（`.ecrc`）。
  - DenseX 唯一的 workflow `.github/workflows/densex-ci.yml` 跑的是 CPU 的 configure／build／test 路徑。
- 重用既有的基礎設施，不要引入新元件；除非有明確理由，不新增第三方相依、額外標頭或檔案。
- 保持簡單：做到 90% 的簡單改動，常常比做到 100% 的複雜改動更好。指出不必要的模板或花俏的 STL；這裡用基本的 `for` 迴圈就很好。
- 每一行新增的程式碼，都應該是貢獻者不靠 AI 也能向審查者解釋和辯護的。指出任何看起來是貼進來、但不理解的東西。
- commit trailer 遵循交辦和 `AGENTS.md`。
  - 本 repo 的 AI 共同作者寫成 `Co-Authored-By: <agent> <noreply@densex.local>`（見 `git log`）。
  - 上游的 `Assisted-by:` 規則不適用於這裡：不要把 `Co-Authored-By:` 的 AI trailer 當成發現。
- 任何提到 Minja 的地方都要當成阻斷性問題，原因見 `docs/densex/rules/CODE_STYLE.md`。

## 回報

依嚴重程度分組，讓使用者知道真正擋住合併的是什麼：

1. **阻斷性**：快速退件／範圍問題和正確性 bug。不論其他部分如何，這些都可能讓改動被否決。
2. **會拖慢審查**：違反慣例、命名、註解規範，缺少測試、文件或效能資料，缺少 API 的理由或範例。
3. **小毛病**：次要的風格問題、可選的清理。

每項發現要指出檔案和行號，具體說明要改什麼、為什麼。

不要主動改寫整份 diff：讓貢獻者自己修，這樣他們才擁有並理解這些修改。也不要自己張貼任何內容：審查筆記只在對話中交付，絕不透過 `gh` 張貼。
