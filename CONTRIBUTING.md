# Contributing to DenseX

DenseX is an independent fork of llama.cpp, frozen at upstream commit `620d2225` and maintained on the `densex` branch. It does not track or merge upstream, and it is **not** a contribution path to llama.cpp: do not open PRs, issues or patches against `ggml-org/llama.cpp` for work done here.

`AGENTS.md` is the authoritative rule set for this repository: base rules plus an index of the detailed rules in `docs/densex/rules/` (code style, build and test). This file only describes how work is handled in this repo and never overrides `AGENTS.md`; if the two ever disagree, `AGENTS.md` wins. The upstream contributor guide is kept in git history (`git show 620d222:CONTRIBUTING.md`).

## Where work comes from

- Work is tracked as GitHub issues: https://github.com/bext1998/DenseX/issues. Pick an issue, label it `in-progress`, do the work, then comment the result (numbers, commit hashes) and close it.
- One issue is one piece of work; one focused change is one commit, so a regression can be bisected.
- Issues are public: no agent costs, no local absolute paths, no personal data.

## Making a change

- Commit directly to `densex`. No push, no PR and no new remote without the owner's approval (`AGENTS.md`, base rule 4).
- Behavior changes (speed paths, numerics, memory layout) go behind a switch that is off by default, with the old path kept as the reference. Pure refactors and cleanups must keep output identical.
- Commit messages say what changed, why, the switch and its default, and the measurement the change is based on (result paths, numbers).
- Keep third-party attribution intact: `LICENSE`, `AUTHORS`, `licenses/` and per-vendor headers are not to be trimmed as "unused".

## Testing before you commit

Follow `docs/densex/rules/TESTING.md`: build through the build script, `regress.ps1` must end in `REGRESS_PASS`, tool calling 14/14, plus the extra checks for kernel, CUDA and long-context changes. Never weaken a test, golden value or threshold to make a run pass.

## Code style

Follow `docs/densex/rules/CODE_STYLE.md`.

## AI usage

AI-generated code is allowed; whoever submits it is responsible for every line and must be able to explain it. See `AGENTS.md` for the rules that apply to agents working in this repository (including which files to read first and how commits are written).

## Security

See [SECURITY.md](SECURITY.md).

## License and credits

MIT, see [LICENSE](LICENSE). DenseX modifications Copyright (c) 2026 mazemaze. DenseX is based on llama.cpp / ggml (Copyright (c) 2023-2026 The ggml authors, MIT); the original llama.cpp README is kept in [docs/README.llama.cpp.md](docs/README.llama.cpp.md) and the upstream authors are listed in [AUTHORS](AUTHORS).
