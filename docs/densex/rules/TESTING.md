# Build and test

Applies when building, testing, measuring, or before any commit. Base rules are in the root `AGENTS.md`.

## Build

- Build with `scripts/densex/build.ps1` (Windows, CUDA, sm86 by default): `pwsh -File scripts/densex/build.ps1 -Target llama-server`, add `-Tests` for test targets. In the research workspace use `../build.ps1`, which builds into `../build`; `-Src <checkout> -Name <build>` builds a worktree.
- Do not run raw cmake in the engine tree for normal work; use the build script.

## Minimum checks before a commit

- Run the regression net `regress.ps1` (research workspace: `../bench/regress.ps1`); it must end in `REGRESS_PASS`. Use `-Src <checkout> -Name <build>` to test a worktree.
- Server / tool-calling changes: `toolcall_check.py` must pass all 14 cases (regress layer 6). A feature switched on by default needs one regress run with it on.
- `ggml` / kernel changes: `test-backend-ops` (eval and perf) for the affected ops, plus `sanitize.ps1` for CUDA changes.
- Engine changes: compare temperature-0 outputs with the reference path, and measure speed with the scripts in `../bench/`.
- Long-context changes: check output quality and stability, not only speed.
- If a check cannot be run, say so in the report instead of claiming success.

## GPU

- One GPU job at a time: take the lock (`gpu_lock.ps1`, or `run_locked.ps1` for background runs) before any GPU work, and never write or delete the lock file `../bench/GPU_BUSY` by hand. The build script refuses to build while the lock is held.
