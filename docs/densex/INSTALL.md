# Install DenseX on Windows

DenseX is experimental. The measured platform is Windows 11, an RTX 3060
12 GB, a Ryzen 5 3600X, and 32 GB of system RAM. Start with the short-context
configuration below before trying the [128K configuration](USAGE.md#128k-context).
Linux, AMD/Intel GPU backends, other RAM capacities, and runtime performance on
other NVIDIA cards are **untested** here. sm89 and sm120 have compile checks,
not a runtime performance guarantee.

## Requirements

- Windows x64 and PowerShell 7 (`pwsh`) for the commands on this page.
- Visual Studio 2022 with the C++ build tools. The build script locates its x64
  developer shell automatically; a full Visual Studio IDE is not required.
- CUDA Toolkit 13.0 at its standard installation location, or pass `-CudaPath`
  to the build script. Use an NVIDIA driver compatible with your toolkit.
- Git, CMake, and Ninja. CMake and Ninja must be available on `PATH` after the
  Visual Studio developer shell is loaded.
- An NVIDIA GPU with sufficient free VRAM and system RAM for the selected
  model, context, and host-weight placement. The examples target 12 GB VRAM
  and 32 GB RAM; they are not minimum-memory guarantees.
- Disk space for the source, CUDA build output, and model. The selected model
  alone is about 10.44 GB according to its publisher; leave additional space
  for the build rather than treating model size as the total requirement.

Keep the checkout, model, and build paths ASCII-only. Do not run a build while
another inference measurement is using the computer.

## 1. Get the source and build

Run in PowerShell, from the directory where you want the checkout:

```powershell
git clone https://github.com/bext1998/DenseX.git
if ($LASTEXITCODE -ne 0) { throw 'Clone failed' }
Set-Location -LiteralPath DenseX
pwsh -File scripts/densex/build.ps1 -Target llama-server
if ($LASTEXITCODE -ne 0) { throw 'Build failed; inspect build/build-*.log' }
Test-Path -LiteralPath build/sm86/bin/llama-server.exe
```

Success means exit code 0, `BUILD_FINISHED` at the end of the build log, and
`Test-Path` returning `True`. The executable is `build/sm86/bin/llama-server.exe`.
`build/sm86/BUILD_INFO.txt` records the source commit and executable provenance.
The server build disables the bundled web UI; verify it through HTTP below.

The default CUDA architecture is sm86. `-Arch 89` and `-Arch 120` select other
architectures (**runtime untested**); do not assume sm86 binaries suit every GPU.
Other build options are in [USAGE](USAGE.md#build-options).

For an explicitly fresh build, choose an unused name; do not delete an existing
build tree or overwrite another worker's output:

```powershell
pwsh -File scripts/densex/build.ps1 -Name docs-clean -Target llama-server
if ($LASTEXITCODE -ne 0) { throw 'Clean build failed' }
Test-Path -LiteralPath build/docs-clean/bin/llama-server.exe
```

### Existing research workspace

If a workspace has a `build.ps1` wrapper alongside its engine checkouts, the
in-repository script refuses its default build location. From the engine checkout,
use the wrapper so output stays in the workspace build directory:

```powershell
$checkout = (Get-Location).Path
$workspace = Split-Path -Parent $checkout
pwsh -File (Join-Path $workspace 'build.ps1') -Src $checkout -Name wt-codex -Target llama-server
if ($LASTEXITCODE -ne 0) { throw 'Workspace build failed' }
Test-Path -LiteralPath (Join-Path $workspace 'build/wt-codex/bin/llama-server.exe')
```

This assumes a direct child checkout and a previously unused `wt-codex` build
name. The wrapper honors the workspace GPU lock. Wait for its owner to finish;
do not use `-Concurrent` to bypass an active measurement.

## 2. Obtain the model

DenseX primarily supports the **full Qwen 3.8 27B GSQ-RCO range**, including
ISTA-DASLab's original tiers and GSQ-RCO variants such as Swift 1.5. **Unsloth
Dynamic v3.1** is the second series added in #24, across its full tier range.
Swift 1.5 is the current measured sample, not the limit of model support.
Choose the user's requested provider and tier from these original repositories:

| Family | Original source / full tier catalog | Verification level |
|---|---|---|
| ISTA-DASLab GSQ-RCO | [All tiers](https://huggingface.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF/tree/main): IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, with and without bundled MTP | Other tiers/providers: **not yet measured** |
| Unsloth Dynamic v3.1 | [All tiers](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/tree/main), including low-bit UD tiers through high-bit tiers and BF16 shards | **Not yet measured**; hardware/tier validation in progress |
| Swift 1.5 GSQ-RCO | [All published tiers](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-27B-GSQ-RCO-GGUF/tree/main) | README numbers measured only on IQ3_XXS-mtp |

Support scope does not mean every tier fits every GPU, or that every feature has
been runtime-verified for every file. Tier validation and VRAM-based selection
are ongoing in [#24](https://github.com/bext1998/DenseX/issues/24) and
[#28](https://github.com/bext1998/DenseX/issues/28). See
[memory selection estimates](USAGE.md#other-memory-capacities). Do not assign the
Swift sample's speed or quality to the other downloads.

The executable examples below use the measured sample, **Swift 1.5 Qwen3.8 27B
GSQ-RCO IQ3_XXS with the MTP head**, specifically
`Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf` from the
[publisher's file list](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-27B-GSQ-RCO-GGUF/tree/main).
The examples use the model's built-in drafting head; this sample's `-mtp`
suffix identifies that variant. MTP layout differs by file:

- GSQ-RCO and Swift `-mtp` variants bundle the head; their non-MTP variants do not.
- Unsloth also publishes an independent
  [`MTP/mtp-Qwen3.8-27B-Q4_0.gguf`](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/tree/main/MTP).
  Its existence does **not** mean every Unsloth main GGUF lacks a head: the
  header survey found one embedded in UD-Q2_K_XL and higher tiers, while
  UD-IQ1_S, UD-IQ1_M, UD-IQ2_XXS, UD-IQ2_XS, and UD-IQ2_S lack it. Embedded-head
  and separate-file runtime configurations remain **not yet measured** for
  these other tiers. See [MTP selection](USAGE.md#mtp-for-other-model-files).

ISTA-DASLab and Unsloth repositories identify their license as Apache 2.0.
Swift 1.5 weights use Swift Open License v1.0; the original Qwen portions use
Apache 2.0. See the publisher's
[LICENSE](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-27B-GSQ-RCO-GGUF/blob/main/LICENSE)
and [LICENSE-APACHE-2.0](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-27B-GSQ-RCO-GGUF/blob/main/LICENSE-APACHE-2.0).
These model licenses are separate from DenseX's MIT license.

Place the downloaded file in a `models` directory under the checkout, or supply
its existing path through `$model` below. Do not modify, re-quantize, or replace an
existing model to follow this guide. A different model or quantization can be
within the supported family while still being **not yet measured**. Check its
MTP layout and memory needs before adapting the sample's launch preset.

### Verify the download before loading

For **every download source**, verify every selected model file, shard, and
separate MTP file against its original repository's published SHA-256.
For Swift, compare against the publisher's
[`SHA256SUMS`](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-27B-GSQ-RCO-GGUF/blob/main/SHA256SUMS).
Run this block after placing the model in `models`, or adjust `$modelPath` to
the existing file. It reads the model without modifying it:

```powershell
$modelName = 'Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf'
$modelPath = Join-Path 'models' $modelName
$checksumUri = 'https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-27B-GSQ-RCO-GGUF/resolve/main/SHA256SUMS'
$sums = (Invoke-WebRequest -Uri $checksumUri -TimeoutSec 60).Content
$escapedName = [regex]::Escape($modelName)
$entries = @($sums -split '\r?\n' | Where-Object {
    $_ -match "^[0-9a-fA-F]{64}\s+\*?${escapedName}$"
})
if ($entries.Count -ne 1) { throw 'Expected exactly one checksum for this model' }
$expected = ($entries[0] -split '\s+')[0]
$actual = (Get-FileHash -LiteralPath $modelPath -Algorithm SHA256).Hash
if ($actual -ine $expected) { throw 'SHA256 mismatch: do not load this model' }
'SHA256 verified against the publisher SHA256SUMS'
```

If Hugging Face is unavailable, use a previously saved publisher `SHA256SUMS`
from the same revision, replacing only the `$sums` assignment with
`$sums = Get-Content -LiteralPath '.\SHA256SUMS' -Raw`. Do not trust a checksum
provided only by an unrelated mirror. If the original manifest is unavailable
or the hash differs, stop before loading; preserve the file and report the
problem rather than changing the expected hash. This Swift manifest is not
the checksum source for ISTA-DASLab or Unsloth files.

For **ISTA-DASLab or Unsloth**, open the selected file's page in its original
Hugging Face repository, record the revision, and copy the displayed SHA-256
(not a Git commit ID or abbreviated hash). Compare it locally:

```powershell
$modelPath = Read-Host 'Path to the downloaded GGUF file or shard'
$expected = (Read-Host 'Full SHA-256 from its original Hugging Face file page').Trim()
if ($expected -notmatch '^[0-9a-fA-F]{64}$') { throw 'A full SHA-256 is required' }
$actual = (Get-FileHash -LiteralPath $modelPath -Algorithm SHA256).Hash
if ($actual -ine $expected) { throw 'SHA256 mismatch: do not load this file' }
'SHA256 verified against the original Hugging Face file page'
```

Repeat for each shard and any separate MTP file. If the original page is down,
use its previously recorded SHA-256 and revision. Without a trusted original
checksum, stop before loading. Do not substitute another provider's checksum.

### Backup source when Hugging Face is unavailable

[Pirate Face](https://pirateface.co) is a backup source when Hugging Face is
unavailable; it distributes models through community torrents. As checked on
**2026-11-13**:

| Repository | Pirate Face availability | Displayed license |
|---|---|---|
| [ISTA-DASLab GSQ-RCO](https://pirateface.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF) | Magnet listed | apache-2.0 |
| [Unsloth](https://pirateface.co/unsloth/Qwen3.8-27B-GGUF) | Magnet listed | apache-2.0 |
| Swift 1.5 | No torrent; Swift Open License v1.0 absent from the site's [allowed-license list](https://pirateface.co/policy) | Swift Open License v1.0 at the original source |

Check that the torrent contains the chosen file and revision; a repository-level
magnet does not guarantee every tier is included. Verify all files against the
**original repository's** SHA-256 using the procedures above, even when obtained
through a torrent. Do not silently change provider or tier when a file is missing.

The draft frequency table is already in the checkout at
[`models/densex/qwen38-draft-vocab-freq.tsv`](../../models/densex/qwen38-draft-vocab-freq.tsv).
Use it with this model family; it is not a universal ranking for other tokenizers.

## 3. First launch: short context

Run from the checkout root in a dedicated PowerShell terminal. The following
uses a 16384-token context and keeps all transformer layers on the GPU. It is
based on the measured short-context configuration; see [USAGE](USAGE.md#short-context).
It does not allocate the 128K context merely to answer a short prompt.

```powershell
$server = '.\build\sm86\bin\llama-server.exe'
# For a named/workspace build, set $server to the executable printed above.
$model = '.\models\Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf'
if (!(Test-Path -LiteralPath $server)) { throw 'Server executable missing' }
if (!(Test-Path -LiteralPath $model)) { throw 'Model file missing' }

$env:GGML_SCHED_PREFETCH = '0'
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
    '-c', '16384', '-np', '1', '-ngl', '99', '-t', '6', '-tb', '6',
    '-b', '512', '-ub', '128', '-fa', 'on', '-ctk', 'q4_0', '-ctv', 'q4_0',
    '--spec-type', 'draft-mtp', '--spec-draft-n-max', '12',
    '--spec-draft-type-k', 'q8_0', '--spec-draft-type-v', 'q8_0',
    '--spec-draft-kv-window', '0', '--temp', '0', '--jinja', '--fit', 'off',
    '--chat-template-kwargs', '{"enable_thinking":false}', '--load-mode', 'none'
)
& $server @serverArgs
if ($LASTEXITCODE -ne 0) { throw 'Server exited with an error; inspect its log' }
```

Keep that terminal running. If startup fails, preserve the log and check
[memory and speed troubleshooting](USAGE.md#troubleshooting). Do not start a
second server on another port to work around insufficient memory.

## 4. Verify from a second PowerShell terminal

These are normal API operations implemented by the server. The documentation
change was build-checked only; this exact smoke request is **untested** in that
check and is a step for the installer to execute.

```powershell
$health = Invoke-RestMethod -Uri 'http://127.0.0.1:8080/health'
if ($health.status -ne 'ok') { throw 'Server is not ready' }

$body = @{
    model = 'densex-local'
    messages = @(@{ role = 'user'; content = 'Reply with one short sentence.' })
    temperature = 0
    max_tokens = 64
    stream = $false
    chat_template_kwargs = @{ enable_thinking = $false }
} | ConvertTo-Json -Depth 6
$reply = Invoke-RestMethod -Method Post -Uri 'http://127.0.0.1:8080/v1/chat/completions' `
    -ContentType 'application/json' -Body $body
$text = $reply.choices[0].message.content
if ([string]::IsNullOrWhiteSpace($text)) { throw 'No final answer returned' }
$text
```

`/health` returns HTTP 503 while loading; wait for HTTP 200 and `status: ok`
before sending the request. Successful setup requires a nonempty final answer,
not just a live process or a completed build. This short smoke check is not a
long-context quality test.

The API is bound to loopback. Do not expose it to the network as an installation
side effect. Press Ctrl+C in the server terminal to stop it before changing
environment variables or launch arguments, then start a new process.

## Evidence

- [README](../../README.md): measured platform, model, feature switches, build,
  and 128K example.
- [Build script](../../scripts/densex/build.ps1): toolchain discovery, defaults,
  named output directories, UI build flags, and `BUILD_INFO.txt`.
- [Server API](../../tools/server/README.md): `/health` and chat completions.
- [USAGE](USAGE.md): source links for switches and short-context measurements.
