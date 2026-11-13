<h1 align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="media/densex/densex-logo-dark.svg">
    <img alt="DenseX" src="media/densex/densex-logo-light.svg" width="480">
  </picture>
</h1>

State of The Art for Dense Model Inference at Home.

DenseX is an experimental inference engine for running large **dense** models on low-VRAM consumer GPUs at long context. Its primary model scope is the full Qwen 3.8 27B **GSQ-RCO** range, including ISTA-DASLab's original tiers and GSQ-RCO variants such as Swift 1.5. **Unsloth Dynamic v3.1**, across its full tier range, is the second series added in [#24](https://github.com/bext1998/DenseX/issues/24).

The results below use **Swift 1.5 IQ3_XXS-mtp on an RTX 3060 12 GB** as the measured sample. Other tiers and VRAM-based presets are **not yet measured**; validation is tracked in issues [#24](https://github.com/bext1998/DenseX/issues/24) and [#28](https://github.com/bext1998/DenseX/issues/28). See [model selection](docs/densex/INSTALL.md#2-obtain-the-model) before changing files.

DenseX started from [llama.cpp](https://github.com/ggml-org/llama.cpp) commit `620d2225` and is now an independent project: it does not track or merge upstream.

> **Status: research / experimental.** Tested only on Windows 11 + RTX 3060 12 GB + Ryzen 5 3600X (32 GB RAM). Most DenseX features are opt-in switches, off by default. The code base is being cleaned up (unused backends and model architectures will be removed).

## Guides

- [Install and verify](docs/densex/INSTALL.md): toolchain, model download and SHA256, first launch.
- [Usage and troubleshooting](docs/densex/USAGE.md): short/128K settings, switches, memory and speed problems.
- [Quality results](docs/densex/QUALITY.md): the 18-question 120K evaluation and its limits.
- [AI agent setup](docs/densex/AI_SETUP.md): executable steps, success checks, and setup boundaries.

## Results (RTX 3060 12 GB, measured)

Qwen 3.8 27B IQ3_XXS-mtp, context 131072, MTP draft length 12, K/V cache q4_0, 16 FFN layers streamed from host RAM, draft KV window 32768, frequency-ranked draft vocabulary (49152), temperature 0, two runs each:

| Prompt depth | Generation tok/s | Prompt tok/s | Time to first token |
|---|---|---|---|
| 4K | 32.2 / 32.3 | 409 | 10 s |
| 32K | 29.0 / 29.4 | 360 | 89 s |
| 64K | 29.0 / 29.2 | 311 | 206 s |
| 96K | 26.2 / 26.2 | 275 | 349 s |
| 120K | 28.1 / 27.9 | 253 | 474 s |

Peak VRAM 11857 MiB. At 120K, generation is 13% slower than at 4K.

## What DenseX adds

| Feature | Switch (default) | Notes |
|---|---|---|
| Host-weight prefetch into device slots | `GGML_SCHED_PREFETCH=<slots>` (off) | Weights kept in pinned host RAM (`-ot ...=CPU --load-mode none`) are copied to the GPU each forward, overlapped with compute |
| FlashAttention reads quantized K/V in place | `GGML_CUDA_FA_NATIVE_KV=1` (off) | No F16 copy of the KV cache in the compute buffer (about 500 MiB saved per context at 128K) |
| Single-token decode attention on the MMA kernel | `DENSEX_FA_DECODE_MMA=0` disables (on when native K/V is on) | +8% decode at 120K, same output |
| Gated DeltaNet deferred commit | `DENSEX_GDN_DEFERRED` (on for dense Qwen35) | Cheap rollback of rejected MTP draft tokens |
| Bounded KV window for MTP drafting | `--spec-draft-kv-window N` (off) | The draft head attends to the last N tokens only |
| Draft over a reduced vocabulary | `DENSEX_DRAFT_VOCAB_ORDER=<tsv>`, `DENSEX_DRAFT_VOCAB=<K>` (off) | Draft logits over the K most frequent tokens, scattered back to original ids. Ranking for Qwen 3.8: [models/densex/qwen38-draft-vocab-freq.tsv](models/densex/qwen38-draft-vocab-freq.tsv) (token id, source: special / generated / corpus, count) |
| MMQ stream-k over SM x resident-block slots | `DENSEX_MMQ_SLOTS=0` disables (on) | |
| MMVQ / MMQ switch point per type for small verify batches | `DENSEX_MMVQ_MAX=<n>` overrides (built-in table) | Thresholds measured on sm86 |

Every change is described, with its measurement, in the commit message (`git log`).

## Build (Windows, CUDA)

Requirements: Visual Studio 2022 (C++ tools), CUDA Toolkit 13.0, CMake, Ninja.

```powershell
pwsh -File scripts/densex/build.ps1 -Target llama-server        # output build/sm86/bin/
pwsh -File scripts/densex/build.ps1 -Target test-backend-ops -Tests
```

`-Arch 89` or `-Arch 120` builds for other GPUs (compile-tested only). Each build appends a line to `build/<name>/BUILD_INFO.txt` with the commit, a hash of uncommitted changes and the executable hash.

## Example: 120K on a 12 GB GPU

```powershell
$env:GGML_SCHED_PREFETCH = '4'; $env:GGML_OP_OFFLOAD_MIN_BATCH = '1'; $env:GGML_CUDA_FA_NATIVE_KV = '1'
$env:DENSEX_DRAFT_VOCAB_ORDER = 'models/densex/qwen38-draft-vocab-freq.tsv'; $env:DENSEX_DRAFT_VOCAB = '49152'
llama-server -m Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf -c 131072 -np 1 -ngl 99 -t 6 -tb 6 -b 512 -ub 128 -fa on `
  -ctk q4_0 -ctv q4_0 --spec-type draft-mtp --spec-draft-n-max 12 --spec-draft-type-k q8_0 --spec-draft-type-v q8_0 `
  --spec-draft-kv-window 32768 --temp 0 --jinja --fit off --chat-template-kwargs '{"enable_thinking":false}' `
  -ot "blk\.(3|7|11|15|19|23|27|31|35|39|43|47|51|55|59|63)\.ffn_(gate|up|down)\.weight=CPU" --load-mode none
```

This is the configuration of the results table (thinking off, 512 generated tokens per request, one request per server start).

## License and credits

MIT, see [LICENSE](LICENSE). DenseX modifications Copyright (c) 2026 mazemaze. DenseX is based on llama.cpp / ggml (Copyright (c) 2023-2026 The ggml authors, MIT). The original llama.cpp README is kept in [docs/README.llama.cpp.md](docs/README.llama.cpp.md).

Design references (ideas studied, re-implemented in DenseX's own way): Mirai S server (MIT, Copyright (c) 2026 Cary Palmer), Strata (MIT).

**About the contributor count:** This repository was created from a clone of llama.cpp, and its history was later rewritten so that `main` starts from a single commit importing llama.cpp `620d2225`. GitHub's sidebar may still show the contributor count cached from the original llama.cpp history (about 300). Those people are the authors of llama.cpp / ggml, not of DenseX's own changes. Their work is credited above and in [AUTHORS](AUTHORS). The commits in this repository are listed by `git shortlog -sn`.

## 繁體中文摘要

DenseX 是讓 12 GB 等級顯示卡跑大型稠密模型長上下文的實驗性推論引擎，由 llama.cpp `620d2225` 分出後獨立發展、不追蹤上游。目前在 RTX 3060 12 GB 上，Qwen 3.8 27B 於 120K 上下文可達約 28 tok/s（4K 約 32 tok/s）。功能多為預設關閉的開關，詳見上表與 commit 紀錄。

側欄的貢獻者人數（約 300）是 GitHub 從 llama.cpp 原始歷史留下的快取，那些是 llama.cpp／ggml 的作者，並非 DenseX 的開發者；DenseX 本身的提交可用 `git shortlog -sn` 查看。

主要支援 Qwen 3.8 27B 的 GSQ-RCO 全系列，Unsloth Dynamic v3.1 是後來加入的第二系列；Swift 1.5 是目前實測樣本，其他檔位與顯存配置仍待量測。首次使用請依[安裝文件](docs/densex/INSTALL.md)建置、下載並驗證模型，再測短請求；長文設定與記憶體問題見[使用說明](docs/densex/USAGE.md)，實際品質分數與限制見[品質報告](docs/densex/QUALITY.md)。要請 AI agent 協助安裝，請提供 [AI_SETUP](docs/densex/AI_SETUP.md)。
