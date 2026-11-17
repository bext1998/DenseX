# 建置與測試

建置、測試、量測，以及任何 commit 之前適用。基本規則在根目錄的 `AGENTS.md`。

## 建置

- 用 `scripts/densex/build.ps1` 建置（Windows、CUDA，預設 sm86）：
  - 一般用法：`pwsh -File scripts/densex/build.ps1 -Target llama-server`；
  - 要建測試 target 時加 `-Tests`。
- 在研究工作區裡改用 `../build.ps1`，輸出到 `../build`。建置 worktree 時加 `-Src <checkout> -Name <build>`。
- 一般工作不要在引擎目錄直接跑 cmake，一律用建置腳本。

## commit 前最低限度的檢查

- 跑回歸測試 `regress.ps1`（研究工作區是 `../bench/regress.ps1`），結尾必須是 `REGRESS_PASS`。測 worktree 時加 `-Src <checkout> -Name <build>`。
- 改到 server 或工具調用：`toolcall_check.py` 的 14 個案例必須全過（regress 第 6 層）。功能要改成預設開啟時，要在開啟狀態下跑一次 regress。
- 改到 `ggml` 或 kernel：
  - 對受影響的 op 跑 `test-backend-ops`（eval 與 perf）；
  - 改到 CUDA 時，另外跑 `sanitize.ps1`。
- 改到引擎：用 temperature 0 和參考路徑比對輸出，並用 `../bench/` 裡的腳本量速度。
- 改到長上下文：除了速度，也要檢查輸出品質與穩定性。
- 有檢查沒辦法跑時，在報告裡照實寫，不要宣稱成功。

## GPU

- GPU 同時只跑一項工作。
- 任何 GPU 工作之前先拿鎖：一般用 `gpu_lock.ps1`，背景執行用 `run_locked.ps1`。
- 不要手動寫入或刪除鎖檔 `../bench/GPU_BUSY`。
- 鎖被占用時，建置腳本會拒絕建置。
