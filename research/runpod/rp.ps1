# Helper for driving the RunPod pod over its proxy SSH.
#
# The RunPod proxy (ssh.runpod.io) gives an interactive shell only: it ignores a
# remote command argument, has no sftp subsystem (so scp fails), and readline
# mangles long input lines. So:
#   * files are shipped as short base64 chunks and decoded remotely
#   * commands are shipped the same way, then executed with output redirected to
#     a log, which is echoed back between markers
#
# Usage:
#   . .\research\runpod\rp.ps1
#   Send-RpFile -Local research\scripts\09_x.py -Remote /workspace/VeriPromiseESG/research/scripts/09_x.py
#   Invoke-Rp -Script "cd /workspace/VeriPromiseESG && python research/scripts/09_x.py"

$RpHost = "qj8r5sz9zgzf8o-644115e0@ssh.runpod.io"
$RpKey = "$env:USERPROFILE\.ssh\id_ed25519"
$RpRepo = "/workspace/VeriPromiseESG"

function Invoke-RpRaw {
    param([string]$Stdin, [int]$TimeoutSec = 3600)
    # PowerShell prepends a UTF-8 BOM when piping a string to a native command.
    # bash then sees a stray U+FEFF token on the first line, which trips readline
    # completion and swallows the rest of that line. Force a BOM-less encoding and
    # lead with a bare newline as insurance.
    # Piping a PowerShell string to a native command emits a BOM regardless of
    # [Console]::OutputEncoding on 5.1, so write exact bytes to a file and let
    # cmd.exe do the input redirection instead.
    $pf = Join-Path $env:TEMP "rp_stdin_$([guid]::NewGuid().ToString('N')).txt"
    $of = Join-Path $env:TEMP "rp_stdout_$([guid]::NewGuid().ToString('N')).txt"
    try {
        $bytes = [System.Text.Encoding]::UTF8.GetBytes("`n" + ($Stdin -replace "`r`n", "`n"))
        [System.IO.File]::WriteAllBytes($pf, $bytes)
        $cmd = "ssh -tt -o StrictHostKeyChecking=no -o ServerAliveInterval=30 " +
        "-i `"$RpKey`" $RpHost < `"$pf`" > `"$of`" 2>&1"
        cmd.exe /c $cmd | Out-Null
        if (Test-Path $of) { [System.IO.File]::ReadAllText($of) } else { "" }
    }
    finally {
        Remove-Item $pf, $of -Force -ErrorAction SilentlyContinue
    }
}

function Get-RpClean {
    param([string]$Text)
    # Windows PowerShell 5.1 has no `e escape -- it yields a literal "e", so the
    # original patterns here matched nothing and every ANSI sequence survived
    # into the parsed output. Build the escape character explicitly.
    $ESC = [char]27
    $t = $Text -replace "$ESC\[[0-9;?]*[a-zA-Z]", ""
    $t = $t -replace "$ESC\]0;[^$([char]7)]*$([char]7)", ""
    $t = $t -replace "`r", ""
    return $t
}

function Get-RpBetween {
    param([string]$Text, [string]$Tag = "RPMARK")
    $t = Get-RpClean $Text
    $start = "<<<$Tag-START>>>"
    $end = "<<<$Tag-END>>>"
    # The markers also appear in the shell's echo of our own input, so anchor on
    # the last START that still has an END after it.
    $j = $t.LastIndexOf($end)
    if ($j -lt 0) { return $t }
    $i = $t.LastIndexOf($start, $j)
    if ($i -lt 0) { return $t }
    $body = $t.Substring($i + $start.Length, $j - $i - $start.Length)
    # drop the shell prompt line that carries the trailing `echo <<<END>>>`
    $body = ($body -split "`n" | Where-Object { $_ -notmatch "root@.*[#$]\s*$" -and $_ -notmatch "^\s*\[K?$" }) -join "`n"
    return $body.Trim("`n")
}

function Send-RpFile {
    <#  Ship a local file to the pod as short base64 chunks.

        Some pods echo back a duplicated or dropped character when input
        arrives quickly -- a doubled slash in a path, a repeated letter in a
        keyword. One corrupted line is enough to make `base64 -d` emit nothing,
        so the md5 is checked and the whole transfer retried.
    #>
    param(
        [Parameter(Mandatory = $true)][string]$Local,
        [Parameter(Mandatory = $true)][string]$Remote,
        [int]$ChunkSize = 120,
        [int]$Retries = 4
    )
    if (-not (Test-Path $Local)) { throw "no such local file: $Local" }
    $bytes = [System.IO.File]::ReadAllBytes((Resolve-Path $Local))
    $b64 = [Convert]::ToBase64String($bytes)
    $dir = [System.IO.Path]::GetDirectoryName($Remote) -replace "\\", "/"

    # One heredoc rather than a printf per chunk. Some pods duplicate characters
    # in the echoed input -- "runpod" came back as "runpodd", "x.b64" as
    # "x..b64" -- and with the destination path repeated on every line most
    # chunks then appended to a mistyped filename, leaving the real one empty.
    # Inside a heredoc the path appears once and every other line is pure
    # base64, where a corrupted character is caught by the md5 check.
    $stage = "/tmp/rpx_" + [guid]::NewGuid().ToString("N").Substring(0, 8) + ".b64"
    $sb = New-Object System.Text.StringBuilder
    [void]$sb.AppendLine("mkdir -p $dir")
    [void]$sb.AppendLine("cat > $stage <<'RPB64EOF'")
    for ($i = 0; $i -lt $b64.Length; $i += $ChunkSize) {
        $n = [Math]::Min($ChunkSize, $b64.Length - $i)
        [void]$sb.AppendLine($b64.Substring($i, $n))
    }
    [void]$sb.AppendLine("RPB64EOF")
    # Mention the destination path once, in a cd, then work on the bare
    # basename. Repeating a long path invites the character duplication these
    # pods show, and a mistyped path silently writes the decode somewhere else.
    $base = [System.IO.Path]::GetFileName($Remote)
    [void]$sb.AppendLine("cd $dir || exit 9")
    [void]$sb.AppendLine("base64 -d $stage > $base && rm -f $stage")
    [void]$sb.AppendLine("echo '<<<RPMARK-START>>>'; wc -c $base; md5sum $base; echo '<<<RPMARK-END>>>'")
    [void]$sb.AppendLine("exit")

    $localMd5 = (Get-FileHash -Algorithm MD5 $Local).Hash.ToLower()
    for ($try = 1; $try -le $Retries; $try++) {
        $res = Get-RpBetween -Text (Invoke-RpRaw -Stdin $sb.ToString())
        if ($res -match $localMd5) {
            Write-Output "OK $Remote ($($bytes.Length) bytes, md5 $localMd5)"
            return
        }
        Write-Output "  transfer attempt $try/$Retries did not verify, retrying"
        Start-Sleep -Seconds 2
    }
    Write-Output "FAILED $Remote after $Retries attempts (local md5 $localMd5)"
}

function Invoke-Rp {
    <#  Run a shell script on the pod, return its output cleanly. #>
    param(
        [Parameter(Mandatory = $true)][string]$Script,
        [string]$LogPath = "/tmp/rp_run.log",
        [switch]$Background
    )
    $tmp = [System.IO.Path]::GetTempFileName()
    $body = "set -o pipefail`nsource /workspace/venv/bin/activate 2>/dev/null`ncd $RpRepo`n$Script`n"
    [System.IO.File]::WriteAllText($tmp, $body, (New-Object System.Text.UTF8Encoding $false))
    $remote = "/tmp/rp_cmd_$(Get-Random).sh"
    Send-RpFile -Local $tmp -Remote $remote | Out-Null
    Remove-Item $tmp -Force

    if ($Background) {
        $run = "nohup bash $remote > $LogPath 2>&1 & echo STARTED_PID=`$!"
    }
    else {
        $run = "bash $remote > $LogPath 2>&1; echo EXIT=`$?"
    }
    $sb = "export PAGER=cat`n$run`necho '<<<RPMARK-START>>>'`ntail -c 12000 $LogPath`necho '<<<RPMARK-END>>>'`nexit`n"
    $out = Invoke-RpRaw -Stdin $sb
    return Get-RpBetween -Text $out
}

function Get-RpLog {
    param([string]$LogPath = "/tmp/rp_run.log", [int]$Bytes = 12000)
    $sb = "export PAGER=cat`necho '<<<RPMARK-START>>>'`ntail -c $Bytes $LogPath`necho '<<<RPMARK-END>>>'`nexit`n"
    return Get-RpBetween -Text (Invoke-RpRaw -Stdin $sb)
}

function Get-RpGpu {
    $sb = "echo '<<<RPMARK-START>>>'`nnvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader`nps aux | grep -c '[p]ython'`necho '<<<RPMARK-END>>>'`nexit`n"
    return Get-RpBetween -Text (Invoke-RpRaw -Stdin $sb)
}

function Receive-RpFile {
    <#  Pull a file back from the pod.

        The proxy gives a pty, and a pty truncates a long base64 stream: an
        unverified pull of a 65 KB array silently produced a file 165 bytes
        longer than the original. So the file is fetched in verified chunks --
        each one md5-checked on both ends and retried on its own -- and only
        assembled once every chunk matches. Fine for artifacts and reports,
        not for weights. #>
    param(
        [Parameter(Mandatory = $true)][string]$Remote,
        [Parameter(Mandatory = $true)][string]$Local,
        [int]$ChunkBytes = 16384,
        [int]$Retries = 4
    )
    $probe = "export PAGER=cat`necho '<<<RPMARK-START>>>'`nwc -c < $Remote`nmd5sum $Remote | cut -d' ' -f1`necho '<<<RPMARK-END>>>'`nexit`n"
    # Assign before splitting: inline, PowerShell binds -split as a parameter
    # of Get-RpBetween instead of as the operator.
    $probeOut = Get-RpBetween -Text (Invoke-RpRaw -Stdin $probe)
    $info = ($probeOut -split "`n" | ForEach-Object { $_.Trim() })
    $size = [int]($info | Where-Object { $_ -match '^[0-9]+$' } | Select-Object -First 1)
    $wantAll = ($info | Where-Object { $_ -match '^[0-9a-f]{32}$' } | Select-Object -First 1)
    if (-not $size -or -not $wantAll) { throw "cannot stat $Remote" }

    $dir = Split-Path -Parent $Local
    if ($dir -and -not (Test-Path $dir)) { New-Item -ItemType Directory -Force $dir | Out-Null }
    $buf = New-Object System.Collections.Generic.List[byte]
    for ($off = 0; $off -lt $size; $off += $ChunkBytes) {
        $n = [Math]::Min($ChunkBytes, $size - $off)
        $cut = "tail -c +$($off + 1) $Remote | head -c $n"
        $ok = $false
        for ($try = 1; $try -le $Retries; $try++) {
            $sb = "export PAGER=cat`necho '<<<RPMARK-START>>>'`n$cut | md5sum | cut -d' ' -f1`n$cut | base64 -w 76`necho`necho '<<<RPMARK-END>>>'`nexit`n"
            $chunkOut = Get-RpBetween -Text (Invoke-RpRaw -Stdin $sb)
            $lines = ($chunkOut -split "`n" | ForEach-Object { $_.Trim() } |
                Where-Object { $_ -match '^[A-Za-z0-9+/=]+$' })
            if ($lines.Count -lt 2) { continue }
            $want = ($lines | Where-Object { $_ -match '^[0-9a-f]{32}$' } | Select-Object -First 1)
            if (-not $want) { continue }
            $b64 = (($lines | Where-Object { $_ -ne $want }) -join '')
            if ($b64.Length % 4 -ne 0) { continue }
            try { $bytes = [System.Convert]::FromBase64String($b64) } catch { continue }
            if ($bytes.Length -ne $n) { continue }
            $md5 = [System.Security.Cryptography.MD5]::Create()
            $got = ([BitConverter]::ToString($md5.ComputeHash($bytes)) -replace '-', '').ToLower()
            if ($got -ne $want) { continue }
            $buf.AddRange($bytes)
            $ok = $true
            break
        }
        if (-not $ok) { throw "chunk at offset $off of $Remote would not verify" }
    }
    [System.IO.File]::WriteAllBytes($Local, $buf.ToArray())
    $final = (Get-FileHash -Algorithm MD5 $Local).Hash.ToLower()
    if ($final -ne $wantAll) { Remove-Item $Local -Force; throw "assembled file md5 $final != $wantAll" }
    "OK $Local ($size bytes, md5 $final)"
}
