# DenseX engine (fork of llama.cpp)

This is the single rule file for this repository. It holds the base rules every agent follows. Detailed rules live in `docs/densex/rules/`; read them when the task needs them (see the index below).

DenseX is a home inference engine for dense models on low-VRAM GPUs. It started from llama.cpp `620d2225` and is maintained as an independent project on branch `densex`. The upstream contributor rules (issues before PRs, no AI commits, `Assisted-by`, PR etiquette) do not apply here. The original upstream file is in git history (`git show 620d222:AGENTS.md`).

## Ownership (owner rule, 2026-11-11)

這個推論引擎的程式碼以 llama.cpp 為基礎修改，並獨立於 llama.cpp 之外。你（AI agent）與人類共同擁有並管理這些程式碼：在能力所及的範圍內修改、重構、維護、清理不必要的程式碼，並持續提升推論引擎的性能與效率。

## Base rules

1. **Refactor and clean up freely, inside git.** Removing code DenseX does not use (other backends, model architectures, tools) is in scope; git history keeps it recoverable. Deleting untracked files, models or data outside git needs the owner's approval.
2. **Behavior changes sit behind a switch that is off by default** (speed paths, numerics, memory layout), with the old path kept as the reference, until measured and accepted.
3. **Pure refactors keep behavior identical**: same temperature-0 output and the affected tests passing, verified before commit. Never weaken a test, golden value or threshold to make a run pass.
4. **Version control**: commit directly to `densex` when a change is done and verified, one focused change per commit. No push, no PR, no new remote without the owner's approval. Commit messages say what changed, why, the switch and its default, and the measurement it is based on.
5. **Issues are public** (https://github.com/bext1998/DenseX/issues): take work from an issue, label it `in-progress`, comment the result (numbers, commit) and close it. No agent costs, local absolute paths or personal data.
6. **One GPU job at a time** on the target machine; tool calling must keep passing 14/14 (see `docs/densex/rules/TESTING.md`).
7. **Upstream is frozen** at llama.cpp `620d2225` (owner decision 2026-11-11). Later upstream changes are outside references only: evaluate one at a time, write our own version, and ask the owner before porting.
8. **Delegate execution, not judgment.** Owner involvement scales with risk. Low-risk, easily reverted details: decide and state the assumption. Choices that change direction, scope, cost, quality bars or are hard to undo: do not decide alone; lay out 2-3 options with their trade-offs and the assumptions behind each, and put any leaning of your own last. The owner's own proposals may be challenged too, once, with reasons.

## Rule index (read when needed)

| File | When to read it |
|---|---|
| `docs/densex/rules/CODE_STYLE.md` | Writing or reviewing code: style, comments, naming |
| `docs/densex/rules/TESTING.md` | Building, testing, measuring, or before any commit |
| `CONTRIBUTING.md` | How work flows through this repo, licensing and credits |
| `README.md` | What DenseX is, its switches, how to build |

When this checkout sits inside the owner's private research workspace, the workspace rules also apply: `../../AGENTS.md` (entry point), `../SPEC.md` (section 12: version control), `../NOTES.md` (experiment log), `../MEMORY_AUDIT.md`.
