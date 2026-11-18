# 專案結構

對內文件（給 agent 與維護者看），說明每個資料夾的用途、DenseX 會不會改它、以及相關的開關或規則。基本規則在根目錄的 `AGENTS.md`。

每一列都是 2026-11-18 實際對照檔案寫的（清點時的 HEAD 是 `451280ff9`，#86 Phase 7），不是照上游 README 抄的。「會不會改」看 `git log` 與 #86 各階段報告。

## 頂層資料夾

| 路徑 | 用途 | DenseX 會改嗎 | 相關開關／規則 |
|---|---|---|---|
| `.github/` | 唯一的 CI：`.github/workflows/densex-ci.yml`（Windows、CPU-only、allowlist 測試＋qwen35 dummy 的 context 測試） | 會（建置選項變動時要跟著改） | 觸發分支是 `main`；測試清單見 `docs/densex/rules/TESTING.md` |
| `.pi/` | pi agent 設定（`.pi/gg/SYSTEM.md`：commit 慣例、規則入口指向 `AGENTS.md`） | 會（對內文件，正體中文） | AGENTS 基本規則 9 |
| `app/` | 統一入口 `llama` exe（`llama.cpp`、`download.cpp`），把各工具的 `*-impl` 串成一個程式 | 不改 | `LLAMA_BUILD_APP`（CMake 預設 `${LLAMA_STANDALONE}`；`scripts/densex/build.ps1` 與 CI 都傳 OFF） |
| `benches/` | 上游在別台機器做的歷史量測（`dgx-spark/`、`mac-m2-ultra/`、`nemotron/`），不在任何 `add_subdirectory` 裡 | 不改、不當本機基準 | `.editorconfig` 的 `[benches/**]` 排除格式；量測一律用 `../bench/` 的腳本 |
| `cmake/` | 建置輔助：`cmake/build-info.cmake`、`cmake/llama-config.cmake.in`、`cmake/llama.pc.in`、`cmake/license.cmake`、toolchain（`cmake/x64-windows-llvm.cmake` 等） | 必要時才改 | `CMakePresets.json` 的 `CMAKE_TOOLCHAIN_FILE` 都指向這裡 |
| `common/` | `llama-common` 共用庫：參數解析（`arg.cpp`）、chat／jinja、speculative、sampler | 會 | 改到工具調用要過 14/14（AGENTS 基本規則 6） |
| `conversion/` | HF→GGUF 的 Python 轉檔器與註冊表 | 只維護 Qwen 3.8／DFlash 用得到的部分 | 見下方「conversion/」 |
| `docs/` | 文件：`docs/densex/` 是 DenseX 自己的，其餘是上游文件 | `docs/densex/` 會；其餘只修失效引用 | 見下方「docs/」 |
| `examples/` | 示範程式＋被根目錄轉檔器依賴的 Python 工具（`examples/convert_legacy_llama.py`、`examples/model-conversion/`、`examples/pydantic_models_to_grammar*.py`） | 不改 | `LLAMA_BUILD_EXAMPLES` **預設 OFF**（#86 P3.1 改的，DenseX 獨有） |
| `ggml/` | 張量庫與 backend，只剩 `ggml-cpu`、`ggml-cuda` 兩個 backend 目錄 | #86 不改這裡 | 註解一律英文 ASCII（`docs/densex/rules/CODE_STYLE.md`）；改到 `ggml-cuda/` 要跑 `../bench/sanitize.ps1` |
| `gguf-py/` | Python 的 GGUF 讀寫庫（`gguf-py/gguf/`），有自己的 `LICENSE` | 不改（#86 明令不刪其常數） | `pyproject.toml` 的本地依賴 `gguf @ ./gguf-py` |
| `grammars/` | 範例 grammar 與 `README.md` | 不改 | `--grammar-file` 的示範，不是 grammar 實作 |
| `include/` | 公開 API：`llama.h`、`llama-cpp.h` | **不改**（API／ABI 凍結） | AGENTS 與 #86 交辦的禁改清單 |
| `licenses/` | 額外授權檔 `LICENSE-jsonhpp`（對應 `vendor/nlohmann/json.hpp`） | **不刪**（#86 明令） | `app/CMakeLists.txt:19` 用 glob 打包 |
| `media/` | 圖片：`media/densex/` 是 README 用的 logo，其餘是上游 banner | 只用 `media/densex/` | README 的 `srcset` 指向 `media/densex/*.svg` |
| `models/` | tokenizer 測試資產（`ggml-vocab-*.gguf` 與同名的 `.inp`／`.out`）、`templates/`（chat template，測試會掃描）、`models/densex/qwen38-draft-vocab-freq.tsv` | 只增 `models/densex/` 的資料 | 測試資產不可刪；`DENSEX_DRAFT_VOCAB_ORDER` 用到 `models/densex/` 那份表 |
| `pocs/` | `vdot` 小工具 | 不改 | 跟 `examples/` 共用 `LLAMA_BUILD_EXAMPLES`（預設 OFF） |
| `requirements/` | Python 依賴分檔，`requirements.txt` 用 `-r` 串起來 | 只在新增／移除工具時改 | #86 P3.3 實測：所有 `-r` 都解析得到（0 失效） |
| `scripts/` | 建置與維護腳本 | 會（正式建置入口在 `scripts/densex/`） | 見下方「scripts/densex/」 |
| `skills/` | agent 技能：`add-new-model/`（Qwen 3.8 專用的新增模型流程）、`code-review/` | 會（對內文件，正體中文） | AGENTS 基本規則 9；不收上游 PR |
| `src/` | 引擎核心：model、context、graph、KV、loader、sampler | 會（#86 Phase 7 收尾時不改） | 見下方「src/models/」 |
| `tests/` | C++ 測試與 CI allowlist | 會（不刪測試、不降門檻） | 見下方「tests/」 |
| `tools/` | 可執行工具（server、cli、ui、mtmd…） | 會 | 見下方「tools/」 |
| `vendor/` | 第三方 header：`cpp-httplib`、`hash`、`miniaudio`、`nlohmann`、`sheredom`、`stb` | 不改（#86 不刪任何 vendor） | 每個元件都有授權檔或檔頭，見 #86 Phase 7 報告的授權檢查 |
| `AGENTS.md` | 本 repo 唯一的規則入口與規則目錄 | 會 | 對內文件，正體中文 |
| `README.md`、`CONTRIBUTING.md`、`SECURITY.md` | 對外文件 | 會 | 語言維持原狀（英文為主），見 AGENTS 基本規則 9 |
| `LICENSE`、`AUTHORS` | MIT 授權與作者清單 | **不刪不改** | `AUTHORS` 由 `scripts/gen-authors.sh` 產生 |
| `CMakeLists.txt`、`CMakePresets.json` | 建置主入口與 preset | 會（小心改） | project 名稱 `llama.cpp` 與公開 target 名稱不動 |
| `pyproject.toml`、`requirements.txt` | Python 套件設定與依賴 | 會 | `mypy.ini`、`pyrightconfig.json`、`ty.toml` 是 Python 型別檢查設定，路徑都實測存在 |
| `convert_hf_to_gguf.py`、`convert_lora_to_gguf.py`、`convert_llama_ggml_to_gguf.py`、`convert_hf_to_gguf_update.py` | 根目錄的 Python 轉檔入口（呼叫 `conversion/` 與 `gguf-py`） | 會（跟著 Qwen 3.8／DFlash 轉檔需求） | 對應 `pyproject.toml` 的 3 個 console script；依賴見 `requirements/` |
| `.editorconfig`、`.clang-format`、`.clang-tidy`、`.flake8`、`.pre-commit-config.yaml`、`.ecrc`、`.gitignore` | 編輯器、lint 與忽略設定 | 必要時才改 | 有幾個 glob 指向不存在的路徑（`prompts/`、`examples/llama.swiftui/`），見 #86 Phase 7 報告「待決策」 |

**不在 repo 內、但在 repo 上一層**（`docs/densex/rules/TESTING.md` 的 `../` 都是以 repo 根目錄為基準）：`../bench/`（`../bench/regress.ps1`、`../bench/gpu_lock.ps1`、`../bench/run_locked.ps1`、`../bench/ab.py`、鎖檔 `../bench/GPU_BUSY`）、`../build.ps1`（研究工作區的建置包裝）、`../build/`（建置輸出）。

## 重點子資料夾

### `src/models/`

| 檔案 | 用途 |
|---|---|
| `qwen35.cpp` | Qwen 3.8 主模型（factory 的 `LLM_ARCH_QWEN35`） |
| `dflash.cpp` | DFlash2 草稿模型（`LLM_ARCH_DFLASH`，`--spec-type draft-dflash`） |
| `deepseek4.cpp` | **只保留 DFlash 繼承的 graph 基底類別**；本身不是可載入架構（#86 Phase 4） |
| `delta-net-base.cpp` | GDN（Gated DeltaNet）層的計算圖基底類別；`qwen35.cpp` 的主模型 graph 繼承它（`models.h:232`） |
| `models.h` | 以上類別的宣告 |

factory 在 `src/llama-model.cpp:46` 的 `llama_model_mapping()`：只建 `qwen35` 與 `dflash`，其他一律拋 `DenseX only supports qwen35 and dflash architectures, ...`。`src/llama-arch.h` 的 `enum llm_arch` 也只剩 `QWEN35`、`DFLASH`、`UNKNOWN` 三項（#86 Phase 4b＋批 4）。

### `ggml/src/ggml-cuda/`

165 個檔案的 CUDA backend，`ggml/src/` 底下只有它與 `ggml-cpu/`。

- DenseX 的開關集中在這裡與 `src/llama-densex-env.cpp`、`ggml/src/ggml-cuda/densex-env.cuh`：`GGML_CUDA_FA_NATIVE_KV`、`DENSEX_FA_DECODE_MMA`、`DENSEX_MMQ_SLOTS`、`DENSEX_MMVQ_MAX`（README 的開關表）。
- **註解一律英文 ASCII**：`.cu` 由 nvcc 交給 MSVC 編譯，這條路徑沒設 `/utf-8`，中文可能被當 cp950 誤解碼（`docs/densex/rules/CODE_STYLE.md`）。
- 改這裡必須加跑 `../bench/sanitize.ps1`（`docs/densex/rules/TESTING.md`）；改 CUDA 要另外驗 sm89／sm120 的編譯。

### `tools/`

預設 `all` 會建（與 `tools/CMakeLists.txt` 逐一核對）：`server`、`cli`、`ui`、`mtmd`、`perplexity`、`quantize`、`llama-bench`、`imatrix`、`tokenize`、`gguf-split`。

`EXCLUDE_FROM_ALL`（#86 P3.2 加的，仍可用 `cmake --build . --target <name>` 單獨建）：`batched-bench`、`completion`、`fit-params`、`export-lora`、`results`。

相關開關：`LLAMA_BUILD_SERVER`（`scripts/densex/build.ps1` 與 CI 都是 ON）、`LLAMA_BUILD_UI`（預設 OFF，`-UI` 才跑 npm）、`LLAMA_USE_PREBUILT_UI`（CMake 預設 ON，但 `scripts/densex/build.ps1` 與 CI 都傳 OFF）。

### `conversion/`

100 個 Python 檔（`find conversion -type f -not -path "*__pycache__*"`，2026-11-18）。分兩類：

- **DenseX 維護**：`conversion/base.py`、`conversion/qwen.py`（Qwen 3.8）、`conversion/deepseek.py` 的 DFlash／DSpark 路徑、`conversion/minicpm.py` 的 dense 文字分支。
- **上游留下、DenseX 不維護但保留**：其餘 80 多個架構的轉檔類別（`conversion/llama.py`、`conversion/falcon.py`、`conversion/gemma.py`、`conversion/qwen4exp.py`…）與 `conversion/__init__.py` 的 `TEXT_MODEL_MAP` 註冊表。這些類別會寫出**引擎拒載**的 GGUF；#86 Phase 4b5 只在兩條已知路徑攔截（`conversion/minicpm.py` 的 MoE 文字變體、`conversion/deepseek.py` 的 DeepSeek-V4），其餘路徑沒攔，是已知的待決策。

### `tests/`

`ls tests` 76 個項目、實際檔案 105 個（C++ 測試、資產與子目錄）。

- CI 的 allowlist 是 `tests/densex-ci-tests.txt`（37 項）；#86 Phase 7 實測它與研究工作區的 `../bench/cpu_tests_allowlist.txt` **完全相同**（`diff` 輸出為空）。
- `test-llama-archs`：產生 qwen35 dummy 模型；`test-llama-context`：用 dummy 跑 context 測試（CI 的後三個步驟，default 與 `DENSEX_GDN_DEFERRED=0` 各一次）。
- `tests/test-metal-graph-optimize.cpp` 檔案保留，但沒有 CMake 條件會建它（#86 Phase 4 刪掉 Metal gate，`../bench/run_cpu_tests.ps1` 本來就把它排除）。
- 規則：不刪測試、不降門檻、不改 golden（`AGENTS.md` 基本規則 3）。

### `scripts/densex/`

只有一個 `scripts/densex/build.ps1`，是**正式建置入口**：MSVC＋CUDA＋Ninja、靜態連結、`-Arch` 預設 86（輸出 `build/sm86/`），`-Src <checkout> -Name <build>` 建別的工作樹，`-Tests` 打開測試 target，`-LineInfo` 禁止搭配 `sm86`。研究工作區改用外層的 `../build.ps1`（輸出到 `../build/`），原因寫在 `docs/densex/rules/TESTING.md`。

`scripts/` 其餘檔案多是上游留下的工具（`scripts/get_chat_template.py`、`scripts/create_ops_docs.py`、`scripts/ui-assets.cmake`、`scripts/sync_vendor.py` 等）；上游的 release／sync 腳本已於 #86 Phase 1–2 刪除，沒有留下引用。

### `docs/`

- `docs/densex/`：DenseX 自己的文件 —— 對外的 `INSTALL.md`、`USAGE.md`、`QUALITY.md`、`AI_SETUP.md`；對內的 `rules/CODE_STYLE.md`、`rules/TESTING.md`、本檔 `STRUCTURE.md`。
- 其餘 `docs/*.md`：上游文件（`docs/build.md`、`docs/speculative.md`、`docs/multimodal/`…），保留供參考。只有在指向已刪除東西時才修，改寫整段屬另一類工作。

## 相關規則

| 情境 | 讀哪份 |
|---|---|
| 寫程式、審查 | `docs/densex/rules/CODE_STYLE.md` |
| 建置、測試、commit 前 | `docs/densex/rules/TESTING.md` |
| 目錄／文件放哪、找資料 | `AGENTS.md` 規則目錄，以及研究工作區的 `../../docs/rules/FILES.md` |
