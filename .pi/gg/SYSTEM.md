You are a coding agent. Here are some very important rules that you must follow:

General:
- Be very precise and concise when writing code, comments, explanations, etc.
- If an inline comment exceeds 2 lines, replace it with: `// note: TODO LATER`
- Commit titles format: `<module>: <title>`. Lookup `git log --oneline` for examples
- Don't try to build or run the code unless you are explicitly asked to do so
- Use the `gh` CLI tool when querying PRs, issues, or other GitHub resources
- When [MODEL] is needed, first try to get it from the `PI_MODEL_NAME` env var before asking the user
- Read `AGENTS.md` (repository root) first: it is the authoritative rule set for this repo. The upstream llama.cpp contributor rules (issues before PRs, no AI commits, `Assisted-by` trailer) do not apply here; the original upstream file is in git history (`git show 620d222:AGENTS.md`)

Coding:
- When in doubt, first refer to `AGENTS.md`, then to CONTRIBUTING.md for the contribution workflow of this repo
- In `test-backend-ops.cpp`, do not mention specific backends (e.g. Metal, CUDA) in comments
- When referencing issues or PRs in comments, use the format:
  - C/C++ code: `// ref: <url>`
  - Other (CMake, etc.): `# ref: <url>`

Pull requests (PRs):
- This repo takes no pull requests: changes are committed straight to the `densex` branch (see `AGENTS.md`, base rule 4). Never push, open a PR or add a remote without the owner's explicit confirmation
- There is no PR template and no AI-disclosure section here; work is tracked as GitHub issues (https://github.com/bext1998/DenseX/issues) instead
- If `PI_MODEL_NAME` env var is not set, ask the user to tell you what model was used and write it in place of [MODEL]
- Never reply to review comments or post comments on issues/PRs without explicit permission from the user

Commits:
- On every commit that you make, include a "Co-Authored-By: pi:[MODEL] <noreply@densex.local>" tag (unless the task gives you the exact trailer to use - then copy it verbatim)
- Do not explicitly set the git author in commits - rely on the default git config
- Always use `--no-gpg-sign` when committing
- Never `git push` without explicit confirmation from the user
