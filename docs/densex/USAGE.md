# DenseX settings and troubleshooting

Build and verify a short request with [INSTALL](INSTALL.md) first. Commands below
run from the checkout root, use the model from that guide, and serve one sequence
on loopback. Change settings between server processes: several environment
variables are read once and cached.

## Short context

For short requests on the measured RTX 3060 12 GB / 32 GB RAM machine, start with
`-c 16384`, all layers on GPU (`-ngl 99`, no `-ot` CPU override), and
`--spec-draft-kv-window 0`. The complete command is in
[INSTALL](INSTALL.md#3-first-launch-short-context).

Historical L1 measurements used a 4000-token prompt, context 16384, IQ3_XXS-mtp,
MTP-12, K4V4, and frequency vocabulary 49152, with two runs:

| FFN layers kept on host (N) | Generation tok/s | Peak VRAM MiB |
|---|---|---|
| 0 | 40.1 / 39.3 | 11077 |
| 4 | 38.9 / 38.0 | 10863 |
| 8 | 39.0 / 31.9 | 10497 |
| 16 | 32.1 / 32.4 | 9769 |

N=8's slower second run has no confirmed cause. These figures are measurements,
not a promise of 40 tok/s for arbitrary prompts or a guarantee for 8 GB cards.
The input and output together must fit the configured context; 16384 is not
16384 input tokens plus an unlimited answer.

## 128K context

For the measured 12 GB setup, use context 131072 and reserve output space:
the published speed run used about 120000 input tokens and 512 generated tokens.
The following reproduces the README's placement and performance switches, with
an explicit executable path, local address, and model alias:

```powershell
$server = '.\build\sm86\bin\llama-server.exe'
$model = '.\models\Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf'
$env:GGML_SCHED_PREFETCH = '4'
$env:GGML_OP_OFFLOAD_MIN_BATCH = '1'
$env:GGML_CUDA_FA_NATIVE_KV = '1'
$env:DENSEX_GDN_DEFERRED = '1'
$env:DENSEX_FA_DECODE_MMA = '1'
$env:DENSEX_MMQ_SLOTS = '1'
$env:DENSEX_MMVQ_MAX = '-1'
$env:DENSEX_DRAFT_VOCAB_ORDER = 'models/densex/qwen38-draft-vocab-freq.tsv'
$env:DENSEX_DRAFT_VOCAB = '49152'
$env:DENSEX_KV_VMM = '0'
$serverArgs = @(
    '-m', $model, '--alias', 'densex-local', '--host', '127.0.0.1', '--port', '8080',
    '-c', '131072', '-np', '1', '-ngl', '99', '-t', '6', '-tb', '6',
    '-b', '512', '-ub', '128', '-fa', 'on', '-ctk', 'q4_0', '-ctv', 'q4_0',
    '--spec-type', 'draft-mtp', '--spec-draft-n-max', '12',
    '--spec-draft-type-k', 'q8_0', '--spec-draft-type-v', 'q8_0',
    '--spec-draft-kv-window', '32768', '--temp', '0', '--jinja', '--fit', 'off',
    '--chat-template-kwargs', '{"enable_thinking":false}',
    '-ot', 'blk\.(3|7|11|15|19|23|27|31|35|39|43|47|51|55|59|63)\.ffn_(gate|up|down)\.weight=CPU',
    '--load-mode', 'none'
)
& $server @serverArgs
if ($LASTEXITCODE -ne 0) { throw 'Server exited with an error' }
```

Set `$server` to your named-build output if needed. Send the smoke request in
INSTALL to confirm startup; a short request alone does not validate 128K quality.
This exact API smoke sequence is **untested** in build-only checks. Historical
speed and quality results are listed separately below.

`-ot` keeps the listed FFN weights in pinned host RAM. With `--load-mode none`,
`GGML_OP_OFFLOAD_MIN_BATCH=1`, and prefetch 4, those weights are streamed back to
the GPU for computation. This is not the slower path that computes the moved
FFN weights on the CPU. Keep all three settings together.

The draft window limits only MTP's guesses; the target still uses full attention.
Do not truncate the user's document to imitate that setting.

The [README results](../../README.md#results-rtx-3060-12-gb-measured) are C34:
120K generation 28.1 / 27.9 tok/s, prompt processing 253 tok/s, TTFT 474 s,
peak VRAM 11857 MiB. The first response to a new long document spends minutes
on prefill; fast generation does not remove that delay. Quality Q2 used a
different placement and MTP length; see [QUALITY](QUALITY.md).

## Other memory capacities

The primary model series is the full Qwen 3.8 27B GSQ-RCO range; Unsloth
Dynamic v3.1 is the subsequently added second series. Swift 1.5 IQ3_XXS-mtp
is the measured sample used by the commands above. Provider catalogs and
download verification are in [INSTALL](INSTALL.md#2-obtain-the-model).
Other tiers are **not yet measured**, with compatibility and hardware validation
in progress under #24 / #28.

| Situation | Recommended starting point | Evidence limit |
|---|---|---|
| 12 GB GPU / 32 GB RAM | Short setup first; use the 128K setup for long documents | Measured on RTX 3060; other 12 GB GPUs are runtime **untested** here |
| More than 12 GB VRAM | Keep the short setup as a smoke check, then tune placement on the actual machine | Capacity and speed at 128K are **untested**; no automatic layer-count recommendation |
| Less than 12 GB VRAM or less system RAM | Do not assume this model/configuration fits; assess available memory before loading | **Untested**; no validated 128K preset in this guide |

More CPU overrides can free VRAM while increasing host memory and transfer cost.
Fewer can improve generation if they fit. Do not silently replace the model or
lower quantization precision to obtain a successful install.

The private tier/VRAM survey supplied these **estimates**, not validated presets:

| GPU capacity | Primary GSQ-RCO candidate | Second-series Unsloth candidate | Evidence limit |
|---|---|---|---|
| 12 GB | IQ3_XXS-mtp; IQ2_S-mtp for a smaller file | UD-Q2_K_XL | Outside the Swift sample: **not yet measured** |
| 16 GB | IQ3_S-mtp | UD-Q3_K_XL | Estimated 120K all-GPU candidates; **not yet measured** |
| 24 GB | IQ3_S-mtp (highest GSQ-RCO tier in this catalog) | UD-Q5_K_XL | Estimated 120K all-GPU candidates; **not yet measured** |

These estimates assumed main K4V4, one sequence, MTP-8, and roughly 300 MiB
VRAM headroom. They do not transfer unchanged to the MTP-12 examples or a busy
desktop. The survey's 12 GB placement counts predate C34; do not replace C34's
16-layer override with an older estimate. Model-file size is not runtime VRAM.

### MTP for other model files

- **Embedded head:** GSQ-RCO/Swift `-mtp` files, and Unsloth files whose metadata
  includes an MTP head, can use `--spec-type draft-mtp` without a separate draft
  file. The header survey includes Unsloth UD-Q2_K_XL among those with a head.
  Runtime behavior and performance on other tiers are **not yet measured**.
- **No embedded head:** do not blindly use the sample's MTP settings. To start
  without drafting, replace `--spec-type draft-mtp` with `--spec-type none`,
  remove the other `--spec-draft-*` options (including the window), and leave
  `DENSEX_DRAFT_VOCAB=0` with `DENSEX_DRAFT_VOCAB_ORDER` unset. This is a
  code-supported route, **not yet measured** for those tiers; preserve the
  user's chosen model rather than silently replacing it.
- **Separate Unsloth MTP file:** the repository publishes
  `MTP/mtp-Qwen3.8-27B-Q4_0.gguf`. The parser and draft loader support
  `--spec-type draft-mtp --model-draft <verified-MTP-file>` for a separate draft
  model. This route is **untested / not yet measured** here; verify both files
  and budget the extra model memory. Do not count it twice when the main file
  already embeds a head. Check ranked-vocabulary compatibility for the chosen
  main/draft pair before enabling the sample's reduced head.

This distinction follows the header survey, the
[argument parser](../../common/arg.cpp), and the
[draft context loader](../../common/speculative.cpp). Runtime validation remains
part of #24 / #28, including behavior when MTP is requested from a headless file.

## Switches and defaults

These are the README's user-facing DenseX switches plus the required scheduler
threshold and the opt-in VMM switch present in this checkout. Values listed as
defaults are code defaults, not the recommended preset. Process environment
values can override them.

| Setting | Default | Meaning / measured use |
|---|---|---|
| `GGML_SCHED_PREFETCH` | `0` (off) | Number of GPU staging slots for host weights; long preset uses `4`. Slots cost VRAM; lower if slot allocation fails. |
| `GGML_OP_OFFLOAD_MIN_BATCH` | `32` | Minimum batch for GPU offload of otherwise host operations; `1` is required for the measured streamed decode path. |
| `GGML_CUDA_FA_NATIVE_KV` | Off | `1` reads quantized K/V directly in the supported CUDA FlashAttention path, avoiding the full F16 staging copy. README reports about 500 MiB saved at 128K. |
| `DENSEX_FA_DECODE_MMA` | On when native K/V is enabled and the path is eligible | `0` restores the single-token vector attention path. README reports +8% at 120K for the MMA path. |
| `DENSEX_GDN_DEFERRED` | Automatic: on for dense `qwen35`, single sequence | Deferred commit of recurrent state during speculative verification; `0` restores snapshot rollback. Presets use `-np 1`. |
| `--spec-draft-kv-window` | `0` (full draft context) | Limit MTP attention to the last N tokens with absolute positions; long preset uses `32768`. Does not restrict target attention. |
| `DENSEX_DRAFT_VOCAB_ORDER` | Unset | TSV order for the shared output head; use the bundled Qwen frequency ranking. Reordering is in memory, not a model-file edit. |
| `DENSEX_DRAFT_VOCAB` | `0` (off) | Draft over the first K head rows; use `49152` with the ranking. Without an order file, these are the first original token IDs, not the K most frequent tokens. Target logits retain the full vocabulary. |
| `DENSEX_MMQ_SLOTS` | On | MMQ stream-k scheduling over SM/resident-block slots; `0` restores the previous scheduling. |
| `DENSEX_MMVQ_MAX` | Unset / `-1` uses the built-in type/GPU table | A nonnegative N forces MMVQ through batch N. Leave at its default for the measured presets; `8` is not the tuned default. |
| `DENSEX_KV_VMM` | `0` (off) | Experimental CUDA virtual-memory KV allocation. Not part of these measured presets; leave off. Runtime use with these guides is **untested**. |

The reduced vocabulary requires a compatible qwen35 model, an untied output head
without per-row scales, and the output head on the GPU. Invalid ranking files
or incompatible placement fail at model load. Do not put `output.weight` on the
CPU to make these examples fit.

Source definitions:
[scheduler](../../ggml/src/ggml-backend.cpp),
[CUDA environment/offload](../../ggml/src/ggml-cuda/ggml-cuda.cu),
[native KV attention](../../ggml/src/ggml-cuda/fattn.cu),
[MMVQ selection](../../ggml/src/ggml-cuda/mmvq.cu),
[model environment](../../src/llama-densex-env.cpp),
[GDN state](../../src/llama-memory-recurrent.cpp),
[ranking loader](../../src/llama-model.cpp),
[draft logits](../../src/models/qwen35.cpp), and
[draft argument parser](../../common/arg.cpp).

### Launch parameters in the examples

| Parameter | Purpose / selected value |
|---|---|
| `-m`, `--alias` | Model file and API identifier (`densex-local`) |
| `--host`, `--port` | Loopback `127.0.0.1:8080` |
| `-c`, `-np` | Total context capacity (16384 or 131072), one sequence |
| `-ngl 99` | Request all transformer layers on GPU, subject to explicit tensor overrides |
| `-t 6`, `-tb 6` | CPU generation / batch thread counts used in the measured setup |
| `-b 512`, `-ub 128` | Logical and physical prompt batch sizes |
| `-fa on` | Enable FlashAttention |
| `-ctk q4_0`, `-ctv q4_0` | Main attention K/V quantization (K4V4) |
| `--spec-type draft-mtp`, `--spec-draft-n-max 12` | Built-in MTP drafting, at most 12 draft tokens |
| `--model-draft` | Optional separate draft model path; unused by the measured Swift sample |
| `--spec-draft-type-k q8_0`, `--spec-draft-type-v q8_0` | Draft cache types, separate from main K/V |
| `--temp 0` | Temperature-zero generation |
| `--jinja`, `--chat-template-kwargs` | Model chat template; examples set `enable_thinking:false` |
| `--fit off` | Disable automatic fitting so explicit placement stays reproducible |
| `-ot`, `--load-mode none` | Explicit host FFN placement and pinned, non-mmap weight loading |

These choices are in the README or L1 setup; argument semantics are in the
[parser](../../common/arg.cpp) and [server reference](../../tools/server/README.md).
To enable thinking, explicitly set `enable_thinking:true` and
`reasoning_effort:"medium"` in request `chat_template_kwargs`; also send
`reasoning_effort:"medium"` at the request top level. Do not rely on the model
template's implicit effort. See QUALITY for the historical budget and scoring.

## Build options

The [build script](../../scripts/densex/build.ps1) accepts:

| Option | Default / purpose |
|---|---|
| `-Target` | `llama-server`; the CMake target to build |
| `-Arch` | `86`; CUDA architecture; `89` and `120` are compile-tested only |
| `-Jobs` | `6`; parallel build jobs |
| `-Name` | `sm<Arch>`; output subdirectory name |
| `-Src` | The script's checkout; select another checkout explicitly |
| `-BuildRoot` | `<checkout>/build`; standalone script only, wrapper supplies its own root |
| `-CudaPath` | Standard CUDA Toolkit 13.0 directory; point to the installed toolkit if different |
| `-Tests` | Off; configure tests when explicitly building a test target |
| `-UI` | Off; run `npm ci` and build/embed the current DenseX web interface. Requires Node/npm and registry access. Without it, existing dist assets may still be embedded; no UI download is performed. |
| `-Concurrent` | Off; permits building during inference, which can perturb measurements. Do not use to bypass another worker's lock. |

## Troubleshooting

### Allocation failure or model will not fit

Read the first allocation error. Check free dedicated VRAM and RAM, the exact
model, context, other GPU processes, and whether the intended CPU tensor override
was applied. If prefetch allocation fails, its log asks for fewer
`GGML_SCHED_PREFETCH` slots. Stop and restart after a change; an existing failed
process is not a clean capacity check. Keep a known working short preset.

### Sudden slowdown or Shared GPU memory growth

Compare prompt speed, generation speed, dedicated VRAM, and per-process Shared
GPU memory. In the historical SLOW1 investigation, both an earlier and a newer
build fell to 3.4–3.5 tok/s under the same environment, while dedicated VRAM was
near its limit. Other desktop applications used additional VRAM. Recovery after
closing them was **unconfirmed**, so do not claim a guaranteed fix.

Shared GPU memory alone does not indicate VRAM overflow: pinned host weights are
intentional in the streaming setup. Look for growth relative to the same setup's
baseline together with a slowdown. Do not kill unrelated applications or change
driver settings without the user's authorization. Retest after the user frees
resources. A newer commit or an answer that still matches does not prove speed
is healthy.

### Host path is unexpectedly slow

Check all of `--load-mode none`, `GGML_OP_OFFLOAD_MIN_BATCH=1`, and
`GGML_SCHED_PREFETCH=4` for the long preset. mmap loading can prevent asynchronous
pinned-weight prefetch and produce a warning. Do not interpret `=CPU` in `-ot`
as a request for CPU computation in this configuration.

### Missing answer, busy port, or long wait

Wait for `/health` to return 200/`ok`; 503 means still loading. Confirm which
process owns the requested port rather than replacing it. For a new 120K
document, prefill took about 474 seconds in C34; a request can be busy long before
the first token. Distinguish loading, prefill, reasoning, and final content.

## Measurement provenance

The README preserves C34's public speed table. The private research log recorded
L1 (short-context table) on 2026-11-11 and SLOW1 (slowdown investigation) on
2026-11-12. The model/tier estimates and MTP header survey came from the private `QUANT_TIERS_VRAM.md` research record (2026-11-12, no GPU run). Those logs are
not in this public repository; this page records their conditions and
measurements without adding new runtime runs. For quality scores and their
limits, use [QUALITY](QUALITY.md).
