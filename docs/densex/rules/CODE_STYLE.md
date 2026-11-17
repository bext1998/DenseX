# Code style

Applies when writing or reviewing code. Base rules are in the root `AGENTS.md`.

- Blend in with the surrounding code. Reuse existing infrastructure, or replace it when it blocks a DenseX need; do not add parallel subsystems that duplicate it.
- Keep code comments short (1-2 lines), only where needed, in simple English, explaining *why*. Mark DenseX changes with a `DenseX` prefix in the comment.
- Use ASCII only in code and comments: `-`, `->`, `x`, `...` instead of em dash, arrows, multiplication sign or ellipsis.
- Do not hard-wrap comments or split a line mid-sentence to fit a column width.
- Naming keeps the llama.cpp conventions: `snake_case` functions, longest-common-prefix naming, `kebab-case` file names for C/C++, enum values upper case and prefixed with the enum name.
- llama.cpp has its own Jinja engine in `common/jinja` (not Minja). Any mention of Minja is wrong for this codebase.
