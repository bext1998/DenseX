---
name: add-new-model
description: Guided workflow for adding a new model architecture to DenseX (an independent llama.cpp fork). Use when the user wants to add/port a new model architecture.
---

# Add a new model architecture to DenseX

DenseX is an independent fork of llama.cpp: it does not track or merge upstream and it takes no pull requests. Work here lands as GitHub issues (https://github.com/bext1998/DenseX/issues) and commits on the `densex` branch - see `AGENTS.md` (authoritative rules) and `CONTRIBUTING.md`. Where this skill says "PR" in inherited upstream text, read it as "the change" (its issue plus its commit).

This skill walks a contributor through adding a new model architecture. AI-generated code is permitted in this project, so you may write full implementations for the steps below rather than only pointing at patterns - but follow `AGENTS.md` throughout:

- The contributor is 100% responsible for every line, however it was produced. They must be able to explain and defend any part of it to a reviewer. Check in with them as you go (don't silently generate everything and hand over a finished diff) so they actually absorb what was written.
- Before writing code, make sure the contributor owns the design choices for this architecture (which reference model to follow, how non-standard bits like RoPE variants or MoE routing should be handled) - AI accelerates a design the contributor has already made, it doesn't make the design for them.
- Track the work in one GitHub issue and comment the result (numbers, commit hash) before closing it; issues are public, so no agent costs, local absolute paths or personal data. There is no PR template and no AI-disclosure section in this repo.
- Write commit messages yourself, per `AGENTS.md` (base rule 4): what changed, why, the switch and its default, and the measurement it is based on. Copy the commit trailer the task gives you verbatim - in this repo AI co-authorship is expressed with `Co-Authored-By:` (see `git log`).
- If the requested change looks large or introduces a new pattern not covered here, pause and tell the user this kind of change needs the design settled in an issue first.
- Keep the change self-contained. If it would require a lot of unconventional changes outside the new model file(s) (e.g. touching shared graph-building code, the sampler, or core APIs in ways other models don't), STOP and settle the design in an issue first - invasive changes get split up.
- Do not bundle unrelated work into this change - see Step 4 and Step 5 below for the specifics on multimodal and chat-template/parsing work.
- Never hack around RoPE with a custom sin/cos implementation. If the existing `ggml_rope_ext` (see Step 2's RoPE tips) genuinely cannot express what this model needs, open an issue to discuss it first - do not write a custom RoPE implementation.

Before starting, read `AGENTS.md`, `CONTRIBUTING.md` and `docs/development/HOWTO-add-model.md` if they are not already in context. Also run `git log --oneline -- src/models` and read the recent commits that added a model (their diffs) - this shows current convention more reliably than the docs, which can lag behind.

## Step 0 - Scope and dedup check

Ask the contributor:
1. Which model (HF repo id or name)? Is it text-only or does it have a multimodal (vision/audio) encoder?
2. Do they already have the HF `config.json`/weights available locally?
3. Have they checked for an existing issue on this model? Suggest `gh search issues "<model name>" --repo bext1998/DenseX`. If an existing issue covers it, comment there and collaborate rather than opening a duplicate.
4. What existing supported architecture is this model closest to (e.g. "Llama-like with sliding window", "MoE like DBRX", "BERT-style encoder")?

If the contributor doesn't know the closest reference architecture, you may grep `conversion/*.py` and `src/models/*.cpp` for architectures with a similar config shape (layer count, head count, MoE expert count, norm placement) and suggest 1-2 candidates - but let the contributor confirm the choice rather than picking one yourself; this choice is a design decision they need to own.

Do not proceed to Step 1 until the contributor has answered these and named a reference architecture.

## Step 1 - Convert the model to GGUF

Follow HOWTO-add-model.md section 1 for the actual touch points (conversion class registration, `constants.py`, `tensor_mapping.py`, etc.) - don't re-derive them here, read them from the doc.

Skill-specific addition: for each touch point, show the contributor the equivalent code in the reference architecture they named in Step 0 before writing the new version, and check that they understand what's different about their model (e.g. non-standard tensor shapes, extra hparams) rather than just copying the pattern silently.

## Step 2 - Define the architecture in llama.cpp

Follow HOWTO-add-model.md section 2 for the actual touch points (`llm_arch` enum, `LLM_ARCH_NAMES`, hparam loading, RoPE type case, etc.), including its "Tips and tricks" section for `ggml_rope_ext` gotchas.

Skill-specific addition: never hack around RoPE with a custom sin/cos implementation - see the RoPE rule above.

## Step 3 - Build the GGML graph

Follow HOWTO-add-model.md section 3 for the actual touch points (`src/models/<name>.cpp` struct, `llama_model_mapping` registration, etc.).

Skill-specific addition: before writing `src/models/<name>.cpp`, read at least 10 other files under `src/models/` (pick a mix, not just the one reference architecture) to confirm the struct layout, naming, and style you're about to write actually matches current convention - the pattern drifts over time and the HOWTO doc can lag behind it.

## Step 4 - Optional: multimodal encoder

Only do this if the contributor flagged a vision/audio encoder in Step 0. Follow HOWTO-add-model.md section 4 and `docs/multimodal.md` for the actual touch points (`MmprojModel` subclass, `clip.cpp`, `mtmd.cpp`, encoder graph in `tools/mtmd/models`, etc.).

Skill-specific addition, and read this carefully: **whether the multimodal encoder can be bundled into the same change as the base text-model support depends on how conventional the change is.** It's OK to bundle it if the encoder support is conventional - i.e. no new infra or logic is needed, it's just a new cgraph reusing existing preprocessing/projector machinery (e.g. siglip/pixtral/qwen with just a new projector). If it requires anything beyond that - a new preprocessor, non-standard projector logic, or changes to shared `libmtmd` infra/logic - STOP, tell the contributor this is non-conventional, and have them land the text model first with the encoder as a dedicated follow-up change. Do not let this decision pass silently - call it out explicitly to the contributor before writing any `clip.cpp`/`mtmd.cpp` code.

## Step 5 - Optional: chat template / parsing support

Only do this if the model needs a new built-in chat template (`src/llama-chat.cpp`) or a new output parser (see `docs/development/parsing.md` and `docs/autoparser.md`). If either is needed beyond what a user-supplied Jinja template already covers, treat it as its own dedicated follow-up change, not part of the base model-support change - call this out explicitly to the contributor rather than silently bundling it in.

## Common pitfalls (from past reviews)

These recur often enough in review comments on past add-model changes that they're worth checking proactively, not just waiting for a reviewer to catch them:

- Don't validate the same hparam/config assumption in both the Python conversion script and the C++ load path - pick one layer to own the check, duplicating it just adds maintenance surface.
- Optional hparams that are genuinely absent from some configs (e.g. a shared-expert count) should be read with an explicit optional/fallback accessor, not assumed present.
- Hparams that are actually load-bearing (the model produces wrong output or crashes without them, e.g. `sliding_window_pattern`, norm-eps) must hard-error if missing, not silently fall back to a default.
- Don't bake a default chat template into the C++ binary - inject it into the GGUF at conversion time instead, since one `llm_arch` can be reused by multiple fine-tunes with different templates, and a baked-in C++ default fails silently for those.
- Before writing a dedicated tool-call/output parser, check whether the existing autoparser already handles the template (`test-chat-auto-parser <jinja>` shows what it detects).
- Marking a custom EOS/closing-tag token as `eot` at conversion time isn't always sufficient - in long/agentic generations a model can emit the closing sequence as literal text instead of the token, so generation never stops on EOG and raw text leaks past the parser. Verify this case, not just the token path.
- If reusing or aliasing an existing pre-tokenizer for convenience, justify and test that choice explicitly - silent reuse is an easy source of subtle tokenizer bugs.
- Watch for excessive graph splits caused by building per-layer view/index tensors inside the layer loop - hoist tensors that don't vary per layer out of the loop (relevant if you hit `GGML_SCHED_MAX_SPLIT_INPUTS`).
- A custom KQ mask fed into flash attention must match FA's expected dtype - cast it to F16 before passing it to `build_attn_mha` when FA is enabled.
- When padding a custom KV-cache size to an alignment (e.g. `GGML_PAD(..., 256)`), apply the padding after all other size adjustments, not before - otherwise later logic can un-align it again.
- For non-standard cache/SWA (sliding-window-attention) semantics, override the dedicated hook (e.g. `llama_model_n_swa()`) rather than mutating hparams to fake the behavior - hparams may be read elsewhere for unrelated purposes.
- Don't ship unfinished or unverified speculative-decoding (e.g. MTP) scaffolding in the base model change - if it hasn't actually been confirmed to work, pull it out and land it as its own follow-up.
- Conversion code should call into the base class's existing hparam logic (e.g. `super().set_gguf_parameters()`) rather than re-deriving it - large blocks of code that duplicate what `TextModel`/`MmprojModel` already provide will get flagged as redundant.
- Do constant tensor modifications (e.g. `norm(1 + weight)`) and permutations/chunking at conversion time, not in the graph - see HOWTO-add-model.md's "Prefer conversion-time tensor modifications" tip (Gemma 3 folds its `1 +` into the weights, Qwen3-Next permutes in `modify_tensors`). Doing these at runtime in the graph is very likely to be rejected as over-complicated; if you genuinely can't do it at conversion time, open a discussion first explaining why rather than implementing it in the graph.
  - Exception: a plain `weight * scale` with a constant scale is usually better applied at inference time instead of being folded into the weight at conversion. The scale conceptually applies to the activation, not the weight, so folding it in can hurt numerical stability, and it shifts the weight's value range in a way that can make quantization worse.

## Validation checklist

Reference: `examples/model-conversion/README.md`.

1. Convert to GGUF, then inspect/run both the original and converted tensors.
2. Run logits verification (original vs converted). If this model is a new version of an already-supported family, verify the *previous* version still passes logits verification first - numerical differences may be pre-existing, not caused by the new work. The tools to perform full logits validation are available in `examples/model-conversion`.
3. Quantize (including QAT variants if relevant) and re-verify.
4. Run perplexity evaluation (simple and full).
5. Sanity-check across `tools/cli`, `tools/completion`, `tools/imatrix`, `tools/quantize`, and `tools/server`.
6. CPU backend first; other backends can be separate follow-up changes (`docs/densex/rules/TESTING.md`).
7. Re-review every changed file against the coding/naming guidelines in `docs/densex/rules/CODE_STYLE.md` - this is a separate pass from functional testing and is just as important: no forced line-wrapping, no unicode punctuation, minimal/non-redundant comments, `snake_case` naming (`kebab-case` for file names), matching indentation/brace style, etc.

## Before committing

- Run the `code-review` skill on the diff first - it catches the convention and scope issues reviewers flag most often, and it's worth doing locally before the commit.
- Run the required checks for this change (`CONTRIBUTING.md`, "Testing before you commit"): build through the build script (`scripts/densex/build.ps1`, or `../build.ps1` in the research workspace), then `regress.ps1` ending in `REGRESS_PASS`.
- Confirm the contributor can explain every changed line to a reviewer and is prepared to be asked about any of it - this is required regardless of how much of the code was AI-generated.
- Confirm they did a comprehensive manual review of the full diff, not just a skim.
- Comment the result on the issue (numbers, commit hash) before closing it; keep costs, local absolute paths and personal data out of the issue.
