# Contributing to DenseX

DenseX is an independent fork of llama.cpp, frozen at upstream commit `620d2225` and maintained on the `densex` branch. It does not track or merge upstream, and it is **not** a contribution path to llama.cpp: do not open PRs, issues or patches against `ggml-org/llama.cpp` for work done here.

`AGENTS.md` is the authoritative rule set for this repository (ownership, version control, code style, build and test). This file only describes how work is handled in this repo and never overrides `AGENTS.md`; if the two ever disagree, `AGENTS.md` wins. The upstream contributor guide is kept in git history (`git show 620d222:CONTRIBUTING.md`).

## Where work comes from

- Work is tracked as GitHub issues: https://github.com/bext1998/DenseX/issues. Pick an issue, label it `in-progress`, do the work, then comment the result (numbers, commit hashes) and close it.
- One issue is one piece of work; one focused change is one commit, so a regression can be bisected.
- Issues are public: no agent costs, no local absolute paths, no personal data.

## Making a change

- Commit directly to `densex`. No push, no PR and no new remote without the owner's approval (`AGENTS.md`, "Version control").
- Behavior changes (speed paths, numerics, memory layout) go behind a switch that is off by default, with the old path kept as the reference. Pure refactors and cleanups must keep output identical.
- Commit messages say what changed, why, the switch and its default, and the measurement the change is based on (result paths, numbers).
- Keep third-party attribution intact: `LICENSE`, `AUTHORS`, `licenses/` and per-vendor headers are not to be trimmed as "unused".

## Testing before you commit

Minimum bar for any code change. `AGENTS.md` ("Build and test") has the exact paths - inside a standalone checkout it is `scripts/densex/build.ps1`; in the research workspace it is `../build.ps1` plus `../bench/*.ps1`:

- Build through the build script above. Do not run raw cmake in the engine tree.
- Run the regression net `regress.ps1`; it must end in `REGRESS_PASS`. Use `-Src <checkout> -Name <build>` to test a worktree.
- `ggml` / kernel changes: also run `test-backend-ops` (eval and perf) for the affected ops, plus `sanitize.ps1` for CUDA changes.
- Server / tool-calling changes: `toolcall_check.py` must pass all 14 cases.
- Long-context changes: check output quality and stability, not only speed.
- One GPU job at a time: take the lock (`gpu_lock.ps1`) before any GPU work, and never delete the lock file.
- Never weaken a test, golden value or threshold to make a run pass. If a check cannot be run, say so in the report instead of claiming success.

## Code style

Follow `AGENTS.md` ("Code style"): blend in with the surrounding code, short ASCII-only comments that explain *why*, a `DenseX` prefix on DenseX-specific comments, no hard-wrapped comment columns, and reuse existing infrastructure instead of adding parallel subsystems. Naming keeps the usual llama.cpp conventions: `snake_case` functions, longest-common-prefix naming, `kebab-case` file names for C/C++, enum values upper case and prefixed with the enum name.

## AI usage

AI-generated code is allowed; whoever submits it is responsible for every line and must be able to explain it. See `AGENTS.md` for the rules that apply to agents working in this repository (including which files to read first and how commits are written).

## Security

See [SECURITY.md](SECURITY.md).

## License and credits

MIT, see [LICENSE](LICENSE). DenseX modifications Copyright (c) 2026 mazemaze. DenseX is based on llama.cpp / ggml (Copyright (c) 2023-2026 The ggml authors, MIT); the original llama.cpp README is kept in [docs/README.llama.cpp.md](docs/README.llama.cpp.md) and the upstream authors are listed in [AUTHORS](AUTHORS).
