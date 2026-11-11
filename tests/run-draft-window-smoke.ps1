param(
    [Parameter(Mandatory)][string]$Model,
    [int]$Port = 8094,
    [int]$TimeoutSeconds = 300,
    [switch]$RestoreOnly,
    [int]$HiddenSize = 5120
)
$ErrorActionPreference = 'Stop'
if ($Port -in 8080, 8081) { throw 'Ports 8080 and 8081 are reserved' }
$worktree = Split-Path $PSScriptRoot -Parent
$densex = Split-Path $worktree -Parent
$exe = Join-Path $densex 'build/draftwin/bin/llama-server.exe'
$lockPath = Join-Path $densex 'bench/GPU_BUSY'
$out = Join-Path $worktree 'build-draft-window-results'
$names = 'llama-server', 'llama-perplexity', 'test-backend-ops'
if (Test-Path -LiteralPath $lockPath) { throw "GPU is busy: $lockPath" }
if (Get-Process -Name $names -ErrorAction SilentlyContinue) { throw 'Another GPU test process is running' }
if (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue) { throw "Port $Port is already in use" }
if (!(Test-Path -LiteralPath $exe)) { throw "Missing server: $exe" }
$owner = "Codex draft window test`nPID=$PID"
$lock = [IO.File]::Open($lockPath, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write, [IO.FileShare]::None)
try { $bytes = [Text.Encoding]::UTF8.GetBytes($owner); $lock.Write($bytes, 0, $bytes.Length) } finally { $lock.Dispose() }
$child = $null
$oldPrefetch = $env:GGML_SCHED_PREFETCH
$oldOffload = $env:GGML_OP_OFFLOAD_MIN_BATCH
$oldNative = $env:GGML_CUDA_FA_NATIVE_KV
$results = @{}
try {
    if (Get-Process -Name $names -ErrorAction SilentlyContinue) { throw 'Another GPU test started before lock acquisition' }
    New-Item -ItemType Directory -Force -Path $out | Out-Null
    $env:GGML_SCHED_PREFETCH = '4'
    $env:GGML_OP_OFFLOAD_MIN_BATCH = '1'
    $env:GGML_CUDA_FA_NATIVE_KV = '1'
    if (!$RestoreOnly) {
        $invalid = @(
            @{name='negative'; args=@('--spec-draft-kv-window', '-1'); expected='must be between'},
            @{name='overflow'; args=@('--spec-draft-kv-window', '4294967296'); expected='must be between'},
            @{name='not-mtp'; args=@('--spec-draft-kv-window', '8192'); expected='requires only --spec-type'},
            @{name='multi-seq'; args=@('--spec-type', 'draft-mtp', '-np', '2', '--spec-draft-kv-window', '8192'); expected='requires a single sequence'}
        )
        foreach ($case in $invalid) {
            $cliLog = Join-Path $out "invalid-$($case.name).log"
            $cliError = "$cliLog.err"
            $cliArgs = @('-m', $Model, '--host', '127.0.0.1', '--port', "$Port") + $case.args
            $child = Start-Process -FilePath $exe -ArgumentList $cliArgs -PassThru -WindowStyle Hidden `
                -RedirectStandardOutput $cliLog -RedirectStandardError $cliError
            if (!$child.WaitForExit(10000)) { throw "Invalid CLI case $($case.name) did not exit" }
            $rc = $child.ExitCode
            $child = $null
            if ($rc -eq 0 -or !(Select-String -LiteralPath @($cliLog, $cliError) -SimpleMatch $case.expected)) {
                throw "Invalid CLI case $($case.name) did not reject as expected; exit=$rc"
            }
        }
    }
    $layers = 0..38 | ForEach-Object { [int][Math]::Floor(($_ + 1) * 64 / 39) - 1 }
    $override = 'blk\.(' + ($layers -join '|') + ')\.ffn_(gate|up|down)\.weight=CPU'
    $prompt = "<|im_start|>user`nWhat is 17 + 25? Explain briefly.<|im_end|>`n<|im_start|>assistant`n<think>`n</think>`n`n"
    $windows = if ($RestoreOnly) { @(8192) } else { @(0, 8192) }
    foreach ($window in $windows) {
        $tag = if ($RestoreOnly) { 'restore' } else { "w$window" }
        $stdout = Join-Path $out "$tag.stdout.log"
        $stderr = Join-Path $out "$tag.stderr.log"
        $serverArgs = @('-m', $Model, '--host', '127.0.0.1', '--port', "$Port", '-ngl', '99',
            '-c', '131072', '-b', '512', '-ub', '128', '-np', '1', '--fit', 'off',
            '--load-mode', 'none', '-ctk', 'q8_0', '-ctv', 'q4_0', '-fa', 'on',
            '--spec-type', 'draft-mtp', '--spec-draft-n-max', '8',
            '--cache-type-k-draft', 'q8_0', '--cache-type-v-draft', 'q8_0',
            '--spec-draft-kv-window', "$window", '-ot', $override, '-lv', '4')
        if ($RestoreOnly) {
            $slotDir = Join-Path $out 'slots'
            New-Item -ItemType Directory -Force -Path $slotDir | Out-Null
            $serverArgs += @('--ctx-checkpoints', '0', '--cache-ram', '0', '--no-cache-idle-slots', '--slots', '--slot-save-path', $slotDir)
        }
        $child = Start-Process -FilePath $exe -ArgumentList $serverArgs -PassThru -WindowStyle Hidden `
            -RedirectStandardOutput $stdout -RedirectStandardError $stderr
        $deadline = [DateTime]::UtcNow.AddSeconds($TimeoutSeconds)
        $ready = $false
        while ([DateTime]::UtcNow -lt $deadline) {
            if ($child.HasExited) { throw "Server exited with $($child.ExitCode); see $stderr" }
            try {
                $health = Invoke-RestMethod "http://127.0.0.1:$Port/health" -TimeoutSec 2
                if ($health.status -eq 'ok') { $ready = $true; break }
            } catch { }
            Start-Sleep -Milliseconds 500
        }
        if (!$ready) { throw "Server readiness timed out; see $stderr" }
        $requests = if ($RestoreOnly) { @() } else { @($prompt, "<|im_start|>user`nName three colors.<|im_end|>`n<|im_start|>assistant`n<think>`n</think>`n`n", $prompt) }
        $responses = @()
        foreach ($text in $requests) {
            $body = @{prompt=$text; n_predict=64; temperature=0; seed=42; cache_prompt=$true;
                return_tokens=$true; reasoning_effort='medium'} | ConvertTo-Json -Compress
            $r = Invoke-RestMethod "http://127.0.0.1:$Port/completion" -Method Post -ContentType 'application/json' `
                -Body $body -TimeoutSec $TimeoutSeconds
            if (@($r.tokens).Count -eq 0) { throw 'Completion returned no token IDs' }
            $responses += $r
        }
        if (!$RestoreOnly) {
            $results["$window"] = $responses
            $responses | ConvertTo-Json -Depth 30 | Set-Content -LiteralPath (Join-Path $out "w$window.json") -Encoding utf8
            if (($responses[0].tokens -join ',') -ne ($responses[2].tokens -join ',')) {
                throw "W=$window changed greedy tokens after prompt-cache restore"
            }
        } else {
            function Complete-Tokens($tokens, $count, $cache) {
                $body = @{id_slot=0; prompt=@($tokens); n_predict=$count; temperature=0; seed=42;
                    cache_prompt=$cache; return_tokens=$true; reasoning_effort='medium'} | ConvertTo-Json -Compress -Depth 8
                Invoke-RestMethod "http://127.0.0.1:$Port/completion" -Method Post -ContentType 'application/json' -Body $body -TimeoutSec $TimeoutSeconds
            }
            function Slot-Action($action, $filename) {
                Invoke-RestMethod "http://127.0.0.1:$Port/slots/0?action=$action" -Method Post -ContentType 'application/json' `
                    -Body (@{filename=$filename} | ConvertTo-Json -Compress) -TimeoutSec $TimeoutSeconds
            }
            $tokenized = Invoke-RestMethod "http://127.0.0.1:$Port/tokenize" -Method Post -ContentType 'application/json' `
                -Body (@{content=$prompt; add_special=$true} | ConvertTo-Json -Compress)
            $prefix = @($tokenized.tokens)
            if ($prefix.Count -lt 2) { throw 'Restore test needs a nonempty prefix' }
            $initial = Complete-Tokens $prefix 1 $true
            $saved = Slot-Action 'save' 'good.bin'
            if ($saved.n_saved -ne $prefix.Count) { throw "Unexpected saved token count $($saved.n_saved)" }
            $data = [IO.File]::ReadAllBytes((Join-Path $slotDir 'good.bin'))
            $hiddenBytes = 8L + 4L * $HiddenSize
            $lengthOffset = $data.LongLength - $hiddenBytes - 8L
            if ($lengthOffset -lt 0 -or $lengthOffset -gt [int]::MaxValue - 8) { throw 'Invalid hidden trailer offset' }
            if ([BitConverter]::ToUInt64($data, [int]$lengthOffset) -ne $hiddenBytes -or
                [BitConverter]::ToUInt32($data, [int]$lengthOffset + 8) -ne 0x5750544d -or
                [BitConverter]::ToInt32($data, [int]$lengthOffset + 12) -ne $prefix.Count - 1) {
                throw 'Saved file does not have the expected MTP hidden trailer'
            }
            $oldFile = [IO.File]::Open((Join-Path $slotDir 'old-nohidden.bin'), [IO.FileMode]::Create)
            try {
                $oldFile.Write($data, 0, [int]$lengthOffset)
                $zeroLength = [BitConverter]::GetBytes([uint64]0)
                $oldFile.Write($zeroLength, 0, $zeroLength.Length)
            } finally { $oldFile.Dispose() }
            $ordinary = Invoke-RestMethod "http://127.0.0.1:$Port/tokenize" -Method Post -ContentType 'application/json' `
                -Body '{"content":" blue red","add_special":false}'
            $polluted = @($prefix)
            $polluted[[int]($prefix.Count / 2)] = $ordinary.tokens[0]
            $ignored = Complete-Tokens $polluted 1 $false
            $ignored = Slot-Action 'restore' 'old-nohidden.bin'
            $continued = @($prefix) + @($ordinary.tokens[-1])
            $oldResponse = Complete-Tokens $continued 16 $true
            $oldResponse | ConvertTo-Json -Depth 30 | Set-Content -LiteralPath (Join-Path $out 'restore-old.json') -Encoding utf8
            if ($oldResponse.timings.prompt_n -ne 1) { throw 'Old-file restore did not process exactly one new target token' }
            $warning = "windowed MTP rebuilding seq 0 at pos $($prefix.Count) without the first hidden pair"
            $before = @(Select-String -LiteralPath $stderr -SimpleMatch $warning).Count
            if ($before -lt 1) { throw 'Missing-hidden fallback was not exercised' }
            $ignored = Slot-Action 'restore' 'good.bin'
            $goodResponse = Complete-Tokens $continued 16 $true
            $goodResponse | ConvertTo-Json -Depth 30 | Set-Content -LiteralPath (Join-Path $out 'restore-good.json') -Encoding utf8
            if ($goodResponse.timings.prompt_n -ne 1) { throw 'Good-file restore did not process one new target token' }
            if (@(Select-String -LiteralPath $stderr -SimpleMatch $warning).Count -ne $before) {
                throw 'Valid hidden roundtrip incorrectly entered the missing-hidden fallback'
            }
            $oldTokens = @($oldResponse.tokens)
            $goodTokens = @($goodResponse.tokens)
            if ($oldTokens.Count -eq 0 -or $goodTokens.Count -eq 0) { throw 'Restore completion returned no token IDs' }
            for ($i = 0; $i -lt [Math]::Max($oldTokens.Count, $goodTokens.Count); $i++) {
                if ($i -ge $oldTokens.Count -or $i -ge $goodTokens.Count -or $oldTokens[$i] -ne $goodTokens[$i]) {
                    throw "First restore divergence: position=$i no-hidden=$($oldTokens[$i]) good-hidden=$($goodTokens[$i])"
                }
            }
            @{prefix_length=$prefix.Count; warning=$warning; old=$oldResponse; good=$goodResponse} |
                ConvertTo-Json -Depth 30 | Set-Content -LiteralPath (Join-Path $out 'restore.json') -Encoding utf8
        }
        Stop-Process -Id $child.Id
        $child.WaitForExit()
        $child = $null
    }
    for ($case = 0; !$RestoreOnly -and $case -lt 3; $case++) {
        $a = @($results['0'][$case].tokens)
        $b = @($results['8192'][$case].tokens)
        for ($i = 0; $i -lt [Math]::Max($a.Count, $b.Count); $i++) {
            if ($i -ge $a.Count -or $i -ge $b.Count -or $a[$i] -ne $b[$i]) {
                throw "First token divergence: case=$case position=$i W0=$($a[$i]) W8192=$($b[$i])"
            }
        }
    }
    $windowLog = if ($RestoreOnly) { 'restore.stderr.log' } else { 'w8192.stderr.log' }
    $kv = Select-String -LiteralPath (Join-Path $out $windowLog) -Pattern 'KV buffer size'
    if (!($kv.Line -match '17\.53 MiB')) { throw 'Expected 17.53 MiB draft KV was not logged' }
    if (!($kv.Line -match '3328\.00 MiB')) { throw 'Target full KV allocation changed unexpectedly' }
    $marker = if ($RestoreOnly) { 'DRAFT_WINDOW_RESTORE_PASSED' } else { 'DRAFT_WINDOW_SMOKE_PASSED' }
    $resultLog = if ($RestoreOnly) { 'restore-result.log' } else { 'result.log' }
    $marker | Set-Content -LiteralPath (Join-Path $out $resultLog)
    Write-Output $marker
} finally {
    if ($child -and !$child.HasExited) { Stop-Process -Id $child.Id; $child.WaitForExit() }
    $env:GGML_SCHED_PREFETCH = $oldPrefetch
    $env:GGML_OP_OFFLOAD_MIN_BATCH = $oldOffload
    $env:GGML_CUDA_FA_NATIVE_KV = $oldNative
    if ((Test-Path -LiteralPath $lockPath) -and (Get-Content -LiteralPath $lockPath -Raw) -eq $owner) {
        Remove-Item -LiteralPath $lockPath
    }
}
