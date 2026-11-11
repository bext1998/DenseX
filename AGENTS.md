# DenseX engine (fork of llama.cpp)

This repository is DenseX, a home inference engine for dense models on low-VRAM GPUs. It started from llama.cpp `620d2225` and is maintained as an independent project on branch `densex`. The upstream contributor rules (issues before PRs, no AI commits, `Assisted-by`, PR etiquette) do not apply here. The original upstream file is in git history (`git show 620d222:AGENTS.md`).

## Ownership (owner rule, 2026-11-11)

這個推論引擎的程式碼以 llama.cpp 為基礎修改，並獨立於 llama.cpp 之外。你（AI agent）與人類共同擁有並管理這些程式碼：在能力所及的範圍內修改、重構、維護、清理不必要的程式碼，並持續提升推論引擎的性能與效率。

In practice:
- Refactoring and removing code that DenseX does not use (other backends, model architectures, tools) is in scope; git history keeps it recoverable. Deleting untracked files, models or data outside git still needs the owner's approval.
- Behavior changes (speed paths, numerics, memory layout) sit behind a switch that is off by default until measured and accepted, with the old path kept as the reference.
- Pure refactors and cleanups must keep behavior identical: same temperature-0 output and the affected `test-backend-ops` cases passing, verified before commit.
- Prefer one focused change per commit, so a regression can be bisected.

## Read first

`README.md` (what DenseX is, switches, build). When this checkout sits inside the owner's private research workspace, project rules live there as well:
- `../../AGENTS.md`: project rules, collaboration and the task queue (`docs/tasks/QUEUE.md`).
- `../SPEC.md` (section 12: version control), `../NOTES.md` (experiment log), `../MEMORY_AUDIT.md`.

## Version control

- Commit directly to `densex` when a change is done and verified. No push, no PR, no new remote without the owner's approval.
- Commit message: what changed, why, the switch and its default, and the measurement it is based on (result paths, numbers).
- Every behavior change sits behind a switch (env var or flag) that is off by default, with the original path kept as the reference. Pure refactors and cleanups need no switch (see Ownership).

## Upstream

DenseX is frozen at llama.cpp `620d2225` and no longer merges or tracks upstream (owner decision 2026-11-11, `../SPEC.md` section 12). Later upstream changes are outside references only: evaluate one at a time, write our own version, and ask the owner before porting.

## Code style

- Blend in with the surrounding code. Reuse existing infrastructure, or replace it when it blocks a DenseX need; do not add parallel subsystems that duplicate it.
- Keep code comments short (1-2 lines), only where needed, in simple English. Mark DenseX changes with a `DenseX` prefix in the comment.
- Use ASCII only in code and comments: `-`, `->`, `x`, `...` instead of em dash, arrows, multiplication sign or ellipsis.
- Do not hard-wrap comments or split a line mid-sentence to fit a column width.
- llama.cpp has its own Jinja engine in `common/jinja` (not Minja).

## Build and test

- Build with `scripts/densex/build.ps1` (Windows, CUDA, sm86 by default): `pwsh -File scripts/densex/build.ps1 -Target llama-server`, add `-Tests` for test targets. In the research workspace use `../build.ps1`, which builds into `../build`.
- One GPU job at a time on the target machine: check `../bench/GPU_BUSY` and running engines before any GPU work.
- Before committing a refactor or cleanup, run the regression net `../bench/regress.ps1` (research workspace).
- Kernel changes: `test-backend-ops` (eval and perf) for the affected ops. Engine changes: compare temperature-0 outputs with the reference path, and measure speed with the scripts in `../bench/`.
