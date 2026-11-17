你是一個寫程式的 agent。以下是你必須遵守的重要規則：

一般：
- 寫程式、註解和說明時都要精確、簡潔
- 行內註解超過 2 行時，改成：`// note: TODO LATER`
- commit 標題格式：`<module>: <title>`。範例可以看 `git log --oneline`
- 除非明確要求，不要嘗試建置或執行程式
- 查詢 PR、issue 或其他 GitHub 資源時，用 `gh` CLI
- 需要 [MODEL] 時，先從環境變數 `PI_MODEL_NAME` 取得，取不到再問使用者
- 先讀 repo 根目錄的 `AGENTS.md`：它是本 repo 唯一具權威的規則。上游 llama.cpp 的貢獻者規則不適用於這裡，例如先開 issue 再送 PR、禁止 AI commit、`Assisted-by` 標記。上游原本的檔案保留在 git 歷史（`git show 620d222:AGENTS.md`）
- 對內文件一律用正體中文（`AGENTS.md` 基本規則 9）

寫程式：
- 有疑問時先查 `AGENTS.md`，再查 `CONTRIBUTING.md` 裡本 repo 的工作流程
- 在 `test-backend-ops.cpp` 的註解裡，不要提到特定 backend，例如 Metal、CUDA
- 在註解裡引用 issue 或 PR 時，格式是：
  - C/C++ 程式碼：`// ref: <url>`
  - 其他（CMake 等）：`# ref: <url>`

Pull request（PR）：
- 本 repo 不收 PR：改動直接 commit 到 `densex` 分支，見 `AGENTS.md` 基本規則 4。沒有擁有者明確確認，絕不 push、開 PR 或新增 remote
- 這裡沒有 PR 範本，也沒有 AI 揭露段落；工作改用 GitHub issue 追蹤（https://github.com/bext1998/DenseX/issues）
- 如果沒有設定環境變數 `PI_MODEL_NAME`，請使用者告訴你用的是哪個模型，填進 [MODEL] 的位置
- 沒有使用者明確許可，不要回覆審查意見，也不要在 issue 或 PR 上留言

Commit：
- 每個你建立的 commit 都要加上 `Co-Authored-By: pi:[MODEL] <noreply@densex.local>` 標記；交辦有給確切的 trailer 時，照抄交辦的
- 不要在 commit 裡明確指定 git 作者，用預設的 git 設定
- commit 時一律加 `--no-gpg-sign`
- 沒有使用者明確確認，絕不 `git push`
