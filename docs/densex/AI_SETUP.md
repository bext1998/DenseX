# Set up DenseX for a user

Use this checklist to build and install the source checkout for a user who has
authorized local setup. Read [INSTALL](INSTALL.md) for executable command blocks,
[USAGE](USAGE.md) for supported presets and defaults, and [QUALITY](QUALITY.md)
for the limits of existing results. DenseX has no installer workflow in this
guide: build the `llama-server` target from this repository.

The primary model series is **Qwen 3.8 27B GSQ-RCO, all tiers**, including
ISTA-DASLab's original files and variants such as Swift 1.5. **Unsloth Dynamic
v3.1, all tiers**, is the second series added later in #24. Preserve the user's
selection; Swift 1.5 IQ3_XXS-mtp is the current measured example, not a requirement
to replace other supported tiers. Their runtime and VRAM presets are **not yet
measured**, pending #24 / #28.

## 1. Inspect the machine and existing work

In PowerShell:

```powershell
$PSVersionTable.PSVersion
Get-CimInstance Win32_OperatingSystem | Select-Object Caption, OSArchitecture
Get-CimInstance Win32_ComputerSystem | Select-Object TotalPhysicalMemory
Get-Command git, cmake, ninja -ErrorAction Stop
nvidia-smi
if ($LASTEXITCODE -ne 0) { throw 'Cannot inspect the NVIDIA GPU/driver' }
Get-Process -Name llama-server,llama-kvmem-server -ErrorAction SilentlyContinue
Get-NetTCPConnection -LocalPort 8080 -State Listen -ErrorAction SilentlyContinue
```

Success: identify Windows x64, the GPU/driver, available memory, build tools,
and whether a server already owns the intended port. Check free disk space for
the model plus build output. The measured system is RTX 3060 12 GB / 32 GB
RAM; other machines are **untested**, not automatically supported by matching
VRAM alone. Visual Studio's developer shell can supply CMake/Ninja if absent
from the initial shell; confirm their availability there before building.

If another worker is measuring inference, respect the workspace GPU lock and
wait. Do not take over its server, port, or terminal. Diagnose missing tools
before asking the user to install them; substantial downloads, driver changes,
and system tool installation must stay within the user's authorization.

## 2. Select and build the checkout

Use the user's existing DenseX checkout when present; inspect its instructions
and uncommitted changes first:

```powershell
git status --short --branch
if ($LASTEXITCODE -ne 0) { throw 'Not a Git checkout' }
git rev-parse HEAD
if ($LASTEXITCODE -ne 0) { throw 'Cannot identify source revision' }
```

For a new checkout, execute [INSTALL step 1](INSTALL.md#1-get-the-source-and-build).
Use the existing-workspace wrapper when that layout applies. To verify a clean
build, select an unused `-Name`; do not delete another build to make it clean.
Do not use `-Concurrent` to get past another worker's active measurement.

Success: build exits 0, log ends `BUILD_FINISHED`, the expected executable
exists, and `BUILD_INFO.txt` identifies the source. Keep the executable path
for the next step; a named build does not live in `build/sm86` automatically.

## 3. Obtain and verify the selected model

Prefer an existing copy of the user's selected provider/tier. Otherwise follow
[INSTALL step 2](INSTALL.md#2-obtain-the-model) after confirming the download
and storage fit the user's request. The primary source is the linked Hugging
Face publisher for that provider. Determine whether the file embeds an MTP
head before choosing launch arguments. The Swift example bundles one; Unsloth
also publishes `MTP/mtp-Qwen3.8-27B-Q4_0.gguf` separately, but some main GGUFs
already embed a head. An existing filename alone does not verify contents.

**Execute the SHA256 verification block in INSTALL for every source**, including
an existing file or a backup download. Success requires exactly one matching
entry in Swift's `SHA256SUMS` and the local SHA256 equal to that entry. For
ISTA-DASLab or Unsloth, use the full SHA-256 from the selected file's original
Hugging Face page instead; record the revision and verify every shard and
separate MTP file. If the original source is unavailable, use a previously saved
original manifest or recorded hash for that revision; otherwise stop before
loading. Never substitute a mirror's unsupported checksum or edit the expected
hash after a mismatch.

Pirate Face is a backup when Hugging Face is unavailable. On 2026-11-13,
ISTA-DASLab's and Unsloth's repositories both had magnets and displayed
`apache-2.0`; Swift Open License v1.0 was absent from the allowed-license list
and Swift had no torrent. INSTALL links all sources and the policy. Check the
exact file/revision and verify against the original repository, regardless of
download source. Do not invent a magnet or silently change provider/tier.
The original ISTA-DASLab/Unsloth repositories identify Apache 2.0; Swift weights
use Swift Open License v1.0, with original Qwen portions under Apache 2.0.
State these facts without interpreting the license terms for the user.

Success also requires the bundled
`models/densex/qwen38-draft-vocab-freq.tsv` to exist when selecting the ranked
vocabulary preset. Do not generate a replacement ranking as an installation
side effect.

## 4. Launch and check a short request

Run the complete [first-launch block](INSTALL.md#3-first-launch-short-context),
using the verified model from step 3 and executable from step 2. For a different tier,
first adapt memory placement and MTP options using
[USAGE](USAGE.md#mtp-for-other-model-files); those presets are **not yet measured**.
Use a dedicated terminal and leave the server running while checking HTTP
from another terminal. The short preset uses context 16384, all layers on GPU,
one sequence, and loopback. The documented request disables thinking.

Execute [INSTALL step 4](INSTALL.md#4-verify-from-a-second-powershell-terminal).
Success requires `/health` HTTP 200 with `status: ok` **and a nonempty final
answer** from `/v1/chat/completions`. Loading returns 503. Do not call it working
merely because the process exists or the CUDA build succeeded.

This documentation change checked the build, not GPU inference. These API
smoke commands are **untested in that check** and must be executed during an
actual setup before reporting runtime success. Do not run them during another
worker's GPU job.

## 5. Apply the requested context and verify again

Only choose the [128K preset](USAGE.md#128k-context) when the user needs long
documents and the memory budget permits it. Stop the server you started with
Ctrl+C; restart with the complete long preset. Do not change environment
variables and assume the already-running server has adopted them.

Keep `--load-mode none`, `GGML_OP_OFFLOAD_MIN_BATCH=1`, and prefetch 4 together
for streamed FFN weights. Retain full target attention. If thinking is wanted,
explicitly send `reasoning_effort:"medium"` and matching template settings;
do not let the template silently use its default effort.

Success: the short API smoke check still works and the actual requested
document completes within context without memory failure. Record TTFT,
generation rate, RAM/VRAM and whether the final answer addresses the document.
A short smoke check does not establish 120K quality; Q2 is historical evidence
with a different setup. On untested hardware, report measured behavior as a new
local observation rather than borrowing the README's speed.

## Do not do these during setup

- Do not edit, re-quantize, truncate, or delete model files, including hard links.
- Do not install another inference framework or silently change model precision,
  context length, prompt contents, or attention coverage to make setup pass.
- Do not modify source code, Git configuration, or another person's uncommitted
  work merely to install. Do not commit, push, or merge without authorization.
- Do not delete caches/builds or stop unrelated GPU applications without consent.
- Do not expose the API beyond loopback or transmit user documents externally as
  part of a local smoke check.
- Do not claim hash, build, runtime, long-context quality, or speed checks passed
  unless each was actually performed. Mark remaining checks **untested**.

## Report back

Give the source commit, executable/model locations, checksum result and manifest
source, preset and changes, local endpoint, build/runtime checks actually run,
and any memory or quality limitation. Do not include credentials in logs or
publish the user's paths, hardware identifiers, or documents to an issue.
If blocked, give the failing step, exact relevant error, and the smallest action
needed from the user. Preserve the working files and logs.
