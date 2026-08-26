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

$RpHost = "77ye2k3ml7upns-64410e2b@ssh.runpod.io"
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
    $t = $Text -replace "`e\[[0-9;?]*[a-zA-Z]", ""
    $t = $t -replace "`e\]0;[^`a]*`a", ""
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
    <#  Ship a local file to the pod as short base64 chunks. #>
    param(
        [Parameter(Mandatory = $true)][string]$Local,
        [Parameter(Mandatory = $true)][string]$Remote,
        [int]$ChunkSize = 240
    )
    if (-not (Test-Path $Local)) { throw "no such local file: $Local" }
    $bytes = [System.IO.File]::ReadAllBytes((Resolve-Path $Local))
    $b64 = [Convert]::ToBase64String($bytes)
    $dir = [System.IO.Path]::GetDirectoryName($Remote) -replace "\\", "/"

    $sb = New-Object System.Text.StringBuilder
    [void]$sb.AppendLine("mkdir -p $dir && : > $Remote.b64")
    for ($i = 0; $i -lt $b64.Length; $i += $ChunkSize) {
        $n = [Math]::Min($ChunkSize, $b64.Length - $i)
        [void]$sb.AppendLine("printf %s '" + $b64.Substring($i, $n) + "' >> $Remote.b64")
    }
    [void]$sb.AppendLine("base64 -d $Remote.b64 > $Remote && rm -f $Remote.b64")
    [void]$sb.AppendLine("echo '<<<RPMARK-START>>>'; wc -c $Remote; md5sum $Remote; echo '<<<RPMARK-END>>>'")
    [void]$sb.AppendLine("exit")

    $out = Invoke-RpRaw -Stdin $sb.ToString()
    $res = Get-RpBetween -Text $out
    $localMd5 = (Get-FileHash -Algorithm MD5 $Local).Hash.ToLower()
    if ($res -notmatch $localMd5) {
        Write-Output "WARN md5 mismatch. local=$localMd5"
        Write-Output $res
    }
    else {
        Write-Output "OK $Remote ($($bytes.Length) bytes, md5 $localMd5)"
    }
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
