# DenseX engine (fork of llama.cpp)

This repository is DenseX, a home inference engine for dense models on low-VRAM GPUs. It started from llama.cpp `620d2225` and is maintained as an independent project on branch `densex`. The upstream contributor rules (issues before PRs, no AI commits, `Assisted-by`, PR etiquette) do not apply here. The original upstream file is in git history (`git show 620d222:AGENTS.md`).

## Read first

Project rules live outside this repository, in the research workspace that contains it:
- `../../AGENTS.md`: project rules, collaboration and the task queue (`docs/tasks/QUEUE.md`).
- `../SPEC.md` (section 12: version control), `../NOTES.md` (experiment log), `../MEMORY_AUDIT.md`.

## Version control

- Commit directly to `densex` when a change is done and verified. No push, no PR, no new remote without the owner's approval.
- Commit message: what changed, why, the switch and its default, and the measurement it is based on (result paths, numbers).
- Every behavior change sits behind a switch (env var or flag) that is off by default, with the original path kept as the reference.

## Code style (kept from upstream, so upstream updates keep merging cleanly)

- Blend in with the surrounding code; reuse existing infrastructure instead of adding subsystems.
- Keep code comments short (1-2 lines), only where needed, in simple English. Mark DenseX changes with a `DenseX` prefix in the comment.
- Use ASCII only in code and comments: `-`, `->`, `x`, `...` instead of em dash, arrows, multiplication sign or ellipsis.
- Do not hard-wrap comments or split a line mid-sentence to fit a column width.
- llama.cpp has its own Jinja engine in `common/jinja` (not Minja).

## Build and test

- Build with `../build.ps1` (Windows, CUDA, sm86 by default): `pwsh -File ../build.ps1 -Target llama-server`, add `-Tests` for test targets.
- One GPU job at a time on the target machine: check `../bench/GPU_BUSY` and running engines before any GPU work.
- Kernel changes: `test-backend-ops` (eval and perf) for the affected ops. Engine changes: compare temperature-0 outputs with the reference path, and measure speed with the scripts in `../bench/`.
