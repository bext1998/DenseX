# DenseX：Ninja＋MSVC 標頭相依的防呆（2026-11-19，#80 合併時 regress 抓到）。
#
# cl /showIncludes 的前綴（「注意: 包含檔案:」）用主控台 code page 編碼：950 是 Big5，65001 是 UTF-8。
# CMake 設定時把當下的位元組寫進 CMakeFiles/rules.ninja 的 msvc_deps_prefix，Ninja 只認這組位元組。
# 之後如果在另一個 code page 下建置（例如從不同的 shell 啟動），Ninja 認不出 include 訊息，物件檔的標頭相依
# 記成 0，改了標頭也不會重編，結構體版面不一致就會出現莫名其妙的錯誤。
#
# 用法：載入 VS 環境（cl 在 PATH）之後、cmake 設定之前呼叫 Assert-ClDepsPrefix <建置目錄>。
# 前綴對不上，或已經有大量物件檔沒有相依紀錄時，刪掉 CMakeCache.txt、最上層 CMakeFiles 與 .ninja_deps，
# 迫使接下來的 cmake 重新偵測前綴、Ninja 全部重編。只刪這三項設定／紀錄，不動其他建置產物以外的東西。

function Get-DenseXBytesAfter([byte[]]$Data, [byte[]]$Pattern) {
    for ($i = 0; $i -le $Data.Length - $Pattern.Length; $i++) {
        $ok = $true
        for ($j = 0; $j -lt $Pattern.Length; $j++) { if ($Data[$i + $j] -ne $Pattern[$j]) { $ok = $false; break } }
        if ($ok) { return $i + $Pattern.Length }
    }
    return -1
}

# cl 現在實際輸出的前綴位元組（到包含檔路徑的磁碟機代號之前為止）
function Get-DenseXClPrefix {
    $tmp = Join-Path ([IO.Path]::GetTempPath()) "densex-cl-prefix-$PID"
    New-Item -ItemType Directory -Force $tmp | Out-Null
    try {
        [IO.File]::WriteAllText((Join-Path $tmp 'h.h'), '')
        [IO.File]::WriteAllText((Join-Path $tmp 'a.c'), "#include `"h.h`"`n")
        $outFile = Join-Path $tmp 'out.txt'
        cmd /c "cl /nologo /showIncludes /c `"$tmp\a.c`" /Fo`"$tmp\a.obj`" > `"$outFile`" 2>&1" | Out-Null
        $b = [IO.File]::ReadAllBytes($outFile)
        # 找「X:\」：前綴裡的冒號後面接空白，不會誤判
        for ($i = 1; $i -lt $b.Length - 1; $i++) {
            if ($b[$i] -eq 0x3A -and $b[$i + 1] -eq 0x5C -and (($b[$i - 1] -bor 0x20) -ge 0x61) -and (($b[$i - 1] -bor 0x20) -le 0x7A)) {
                $start = $i - 1
                while ($start -gt 0 -and $b[$start - 1] -ne 0x0A) { $start-- }
                return [byte[]]$b[$start..($i - 2)]
            }
        }
        throw "cl-deps-guard: cannot find the /showIncludes line in cl output ($outFile)"
    } finally {
        Remove-Item -Recurse -Force $tmp -ErrorAction SilentlyContinue
    }
}

function Assert-ClDepsPrefix([Parameter(Mandatory)][string]$Build) {
    $rules = Join-Path $Build 'CMakeFiles/rules.ninja'
    if (!(Test-Path $rules)) { return }   # 還沒設定過，cmake 會用現在的 code page 偵測

    # Ninja 比對的是 rules.ninja 的 msvc_deps_prefix（CMakeCXXCompiler.cmake 裡那份一律是 UTF-8，不能拿來比）
    $reason = $null
    $data = [IO.File]::ReadAllBytes($rules)
    $at = Get-DenseXBytesAfter $data ([Text.Encoding]::ASCII.GetBytes('msvc_deps_prefix = '))
    if ($at -ge 0) {
        $end = $at
        while ($end -lt $data.Length -and $data[$end] -ne 0x0A -and $data[$end] -ne 0x0D) { $end++ }
        $saved = [Convert]::ToHexString([byte[]]$data[$at..($end - 1)]) -replace '(20)+$', ''
        $now = [Convert]::ToHexString((Get-DenseXClPrefix)) -replace '(20)+$', ''
        if ($saved -ne $now) {
            $reason = "cl /showIncludes prefix differs from rules.ninja (configured $saved, now $now; console code page differs)"
        }
    }

    # 前綴對得上也可能是早先錯位時編出來的：正常建置目錄只有極少數物件檔（hbm.cpp 等）沒有相依
    if (!$reason -and (Test-Path (Join-Path $Build '.ninja_deps'))) {
        Push-Location $Build
        try { $zero = @(ninja -t deps 2>$null | Select-String '\.obj: #deps 0,').Count } finally { Pop-Location }
        if ($zero -gt 10) { $reason = "$zero object files have no recorded header dependencies" }
    }

    if ($reason) {
        Write-Warning "cl-deps-guard: $Build : $reason; resetting the CMake cache and Ninja deps log (full rebuild)"
        foreach ($p in 'CMakeCache.txt', 'CMakeFiles', '.ninja_deps') {
            $f = Join-Path $Build $p
            if (Test-Path $f) { Remove-Item -Recurse -Force $f }
        }
    }
}
