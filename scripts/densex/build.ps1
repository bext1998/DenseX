# Build DenseX (Windows, MSVC + CUDA, Ninja, static, CUDA FA for all KV quant types).
# usage: pwsh -File scripts/densex/build.ps1 [-Target llama-server] [-Arch 86] [-Jobs 6] [-Name sm86] [-Concurrent] [-Tests]
#        [-Src <checkout>] [-BuildRoot <dir>] [-UI]
# Writes <BuildRoot, default <repo>/build>/<Name, default sm<Arch>>/bin/<target>.exe and appends a BUILD_INFO.txt line
# (commit, uncommitted-diff hash, exe hash); log <BuildRoot>/build-<time>.log, last line BUILD_FINISHED or BUILD_FAILED.
# -Tests also configures the tests (e.g. -Target test-backend-ops). -Src builds another checkout (e.g. a git worktree).
# -Concurrent builds while an inference process runs (its speed may be perturbed).
# DenseX: -UI runs npm ci + build from current sources; off skips npm but may embed existing dist assets.
param([string]$Target = 'llama-server', [string]$Arch = '86', [int]$Jobs = 6, [string]$Name = '', [switch]$Concurrent, [switch]$Tests,
      [string]$CudaPath = 'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0', [string]$Src = '', [string]$BuildRoot = '', [switch]$UI)
$ErrorActionPreference = 'Stop'
# Windows PowerShell 5.1 turns a native tool's stderr into a terminating error while $ErrorActionPreference is 'Stop'
# and the output is redirected with *>>. A fresh cmake configure writes warnings to stderr ("Manually-specified
# variables were not used by the project") and the 5.1 run died with "ERROR: CMAKE_BUILD_TYPE=Release" before the build
# started, although cmake exited 0 (issue #26). Native tools therefore run with 'Continue' and only their exit code
# decides success; their output still lands in the log.
function Invoke-Native([string]$label, [scriptblock]$run) {
    $prev = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try { $out = @(& $run) } finally { $ErrorActionPreference = $prev }
    $rc = $LASTEXITCODE
    if ($rc -ne 0) { throw "$label failed (exit $rc)" }
    return $out
}
$src = if ($Src) { $Src } else { (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path }
# A checkout inside the research workspace (a build.ps1 wrapper next to it) builds into the workspace's build root only;
# building here by default left a stray <repo>/build tree that later runs picked up by mistake.
if (!$BuildRoot -and (Test-Path (Join-Path $src '../build.ps1'))) { throw "use $(Resolve-Path (Join-Path $src '../build.ps1')) (or pass -BuildRoot)" }
$root = if ($BuildRoot) { $BuildRoot } else { Join-Path $src 'build' }
$build = Join-Path $root ($(if ($Name) { $Name } else { 'sm' + ($Arch -replace ';', '-') }))
New-Item -ItemType Directory -Force $root | Out-Null
$log = Join-Path $root ("build-" + (Get-Date -Format 'yyyyMMdd-HHmmss') + '.log')
function Log($m) { "$(Get-Date -Format 'HH:mm:ss') $m" | Tee-Object -Append -FilePath $log }
try {
    if (!$Concurrent -and (Get-Process -Name llama-server,llama-kvmem-server,ninfer-serve,ffn_probe -ErrorAction SilentlyContinue)) { throw 'an inference process is running; do not build during measurements' }
    if (!$env:VSCMD_VER) {
        $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/Installer/vswhere.exe'
        $vs = & $vswhere -version '[17.0,18.0)' -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
        & (Join-Path $vs 'Common7/Tools/Launch-VsDevShell.ps1') -Arch amd64 -HostArch amd64 -SkipAutomaticLocation | Out-Null
    }
    $env:PATH = "$CudaPath/bin;$CudaPath/bin/x64;$env:PATH"
    Log "build dir: $build; concurrent: $Concurrent"
    Log "UI source build: $UI; prebuilt download: off; when UI is off, existing source/build dist assets may still be embedded"
    Log "commit: $(git -C $src rev-parse HEAD); dirty files: $((git -C $src status --short) -join ', ')"
    $configure = @('-S', $src, '-B', $build, '-G', 'Ninja', '-DCMAKE_BUILD_TYPE=Release',
        '-DCMAKE_C_COMPILER=cl.exe', '-DCMAKE_CXX_COMPILER=cl.exe',
        "-DCMAKE_CUDA_COMPILER=$CudaPath/bin/nvcc.exe", "-DCUDAToolkit_ROOT=$CudaPath",
        "-DCMAKE_CUDA_ARCHITECTURES=$Arch", '-DGGML_CUDA=ON', '-DGGML_CUDA_FA_QUANTS=all',
        '-DBUILD_SHARED_LIBS=OFF', $(if ($UI) { '-DLLAMA_BUILD_UI=ON' } else { '-DLLAMA_BUILD_UI=OFF' }), '-DLLAMA_USE_PREBUILT_UI=OFF',
        '-DLLAMA_LLGUIDANCE=OFF', '-DFETCHCONTENT_FULLY_DISCONNECTED=ON',
        '-DLLAMA_BUILD_SERVER=ON', '-DLLAMA_BUILD_COMMON=ON', '-DLLAMA_BUILD_TOOLS=ON',
        $(if ($Tests) { '-DLLAMA_BUILD_TESTS=ON' } else { '-DLLAMA_BUILD_TESTS=OFF' }), '-DLLAMA_BUILD_EXAMPLES=OFF', '-DLLAMA_BUILD_APP=OFF')
    Log "configure"
    Invoke-Native 'configure' { cmake @configure *>> $log } | Out-Null
    # Objects whose header deps ninja did not record (cl's localized /showIncludes prefix does not always match, e.g.
    # when built from a UTF-8 console) are not rebuilt after a header change, which mixed old and new llama_model
    # layouts on 2026-11-11 ("bad array new length" at load). Delete them so they are always rebuilt.
    $deps = Invoke-Native 'ninja -t deps' { ninja -C $build -t deps 2>> $log }
    $stale = @($deps | Select-String '^(\S+\.obj): #deps 0,' | ForEach-Object { $_.Matches[0].Groups[1].Value } |
               Where-Object { $_ -notmatch 'ggml-cuda' })
    foreach ($o in $stale) { Remove-Item -LiteralPath (Join-Path $build $o) -ErrorAction SilentlyContinue }
    Log "objects without recorded header deps, rebuilt: $($stale.Count)"
    Log "build target=$Target jobs=$Jobs"
    Invoke-Native 'build' { cmake --build $build --parallel $Jobs --target $Target *>> $log } | Out-Null
    # BUILD_INFO.txt: what source each exe came from (read by bench/common.ps1 Write-Provenance)
    $diff = (Invoke-Native 'git diff' { git -C $src diff HEAD 2>> $log }) -join "`n"
    # SHA256::HashData is .NET 5+; PowerShell 5.1 runs on .NET Framework, so hash through the instance API. Both
    # produce the same digest, which keeps BUILD_INFO.txt's diff= comparable with earlier records (issue #26).
    $diffHash = if ($diff) {
        $sha = [Security.Cryptography.SHA256]::Create()
        try { $digest = $sha.ComputeHash([Text.Encoding]::UTF8.GetBytes($diff)) } finally { $sha.Dispose() }
        [BitConverter]::ToString($digest).Replace('-', '').Substring(0, 16)
    } else { 'clean' }
    $info = Join-Path $build 'BUILD_INFO.txt'
    $exe = Join-Path $build "bin/$Target.exe"
    $line = "$(Get-Date -Format s) target=$Target commit=$(git -C $src rev-parse --short HEAD) diff=$diffHash exe_sha256=$((Get-FileHash $exe).Hash.Substring(0, 16)) src=$src"
    Add-Content $info $line
    Log "build info: $line"
    Log 'BUILD_FINISHED'
} catch {
    Log "ERROR: $_"
    Log 'BUILD_FAILED'
    exit 1
}
