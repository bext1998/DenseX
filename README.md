<h1 align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="media/densex/densex-logo-dark.svg">
    <img alt="DenseX" src="media/densex/densex-logo-light.svg" width="480">
  </picture>
</h1>

State of The Art for Dense Model Inference at Home.

DenseX is an experimental inference engine for running large **dense** models on low-VRAM consumer GPUs at long context. The first target is Qwen 3.8 27B (GSQ-RCO IQ3_XXS GGUF with MTP head) on a single **RTX 3060 12 GB**.

DenseX started from [llama.cpp](https://github.com/ggml-org/llama.cpp) commit `620d2225` and is now an independent project: it does not track or merge upstream.

> **Status: research / experimental.** Tested only on Windows 11 + RTX 3060 12 GB + Ryzen 5 3600X (32 GB RAM). Most DenseX features are opt-in switches, off by default. The code base is being cleaned up (unused backends and model architectures will be removed).

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

## 繁體中文摘要

DenseX 是讓 12 GB 等級顯示卡跑大型稠密模型長上下文的實驗性推論引擎，由 llama.cpp `620d2225` 分出後獨立發展、不追蹤上游。目前在 RTX 3060 12 GB 上，Qwen 3.8 27B 於 120K 上下文可達約 28 tok/s（4K 約 32 tok/s）。功能多為預設關閉的開關，詳見上表與 commit 紀錄。
