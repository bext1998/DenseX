param([int]$Jobs = 2)
$ErrorActionPreference = 'Stop'
$src = Split-Path $PSScriptRoot -Parent
$build = Join-Path $src 'build-draft-window-cpu'
if (!$env:VSCMD_VER) {
    $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/Installer/vswhere.exe'
    $vs = & $vswhere -version '[17.0,18.0)' -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if ($LASTEXITCODE -ne 0) { throw "vswhere failed: $LASTEXITCODE" }
    & (Join-Path $vs 'Common7/Tools/Launch-VsDevShell.ps1') -Arch amd64 -HostArch amd64 -SkipAutomaticLocation | Out-Null
}
cmake -S $src -B $build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=cl.exe -DCMAKE_CXX_COMPILER=cl.exe `
    -DGGML_CUDA=OFF -DBUILD_SHARED_LIBS=OFF -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_COMMON=ON `
    -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_APP=OFF `
    -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_LLGUIDANCE=OFF -DFETCHCONTENT_FULLY_DISCONNECTED=ON
if ($LASTEXITCODE -ne 0) { throw "CPU configure failed: $LASTEXITCODE" }
cmake --build $build --parallel $Jobs --target test-draft-kv-window test-draft-kv-window-args test-batch-alloc test-llama-archs test-llama-context
if ($LASTEXITCODE -ne 0) { throw "CPU build failed: $LASTEXITCODE" }
ctest --test-dir $build --output-on-failure -R '^(test-draft-kv-window|test-draft-kv-window-args|test-batch-alloc)$'
if ($LASTEXITCODE -ne 0) { throw "CPU tests failed: $LASTEXITCODE" }
$models = Join-Path $build 'qwen-models'
New-Item -ItemType Directory -Force -Path $models | Out-Null
& (Join-Path $build 'bin/test-llama-archs.exe') -a qwen35 -o $models
if ($LASTEXITCODE -ne 0) { throw "CPU Qwen model generation failed: $LASTEXITCODE" }
if (!(Get-ChildItem -LiteralPath $models -Filter '*.gguf')) { throw 'No Qwen model was generated' }
& (Join-Path $build 'bin/test-llama-context.exe') --models $models --test '.*' -ngl 0 --fit off
if ($LASTEXITCODE -ne 0) { throw "CPU context regression failed: $LASTEXITCODE" }
'DRAFT_WINDOW_CPU_PASSED'
