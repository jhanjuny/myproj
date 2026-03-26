param(
    [int]$Port = 8010,
    [string]$PythonExe = "D:\conda_envs\torch\python.exe",
    [string]$RootDir = "",
    [string]$StateDir = "",
    [ValidateSet("auto", "tailscale", "cloudflared")]
    [string]$Backend = "auto"
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RootDir)) {
    $RootDir = $PSScriptRoot
}

if ([string]::IsNullOrWhiteSpace($StateDir)) {
    $StateDir = Join-Path $env:TEMP "tight_binding_public"
}

New-Item -ItemType Directory -Force -Path $StateDir | Out-Null

$CloudflaredExe = Join-Path $StateDir "cloudflared.exe"
$ServerStdout = Join-Path $StateDir "server_stdout.log"
$ServerStderr = Join-Path $StateDir "server_stderr.log"
$TunnelStdout = Join-Path $StateDir "cloudflared_stdout.log"
$TunnelStderr = Join-Path $StateDir "cloudflared_stderr.log"
$TailscaleStdout = Join-Path $StateDir "tailscale_stdout.log"
$TailscaleStderr = Join-Path $StateDir "tailscale_stderr.log"
$ServerPidFile = Join-Path $StateDir "server.pid"
$TunnelPidFile = Join-Path $StateDir "tunnel.pid"
$UrlFile = Join-Path $StateDir "public_url.txt"
$BackendFile = Join-Path $StateDir "backend.txt"

function Stop-ExistingProcess {
    param([string]$PidFile)

    if (-not (Test-Path -LiteralPath $PidFile)) {
        return
    }

    $PidValue = (Get-Content -LiteralPath $PidFile -ErrorAction SilentlyContinue | Select-Object -First 1).Trim()
    if ([string]::IsNullOrWhiteSpace($PidValue)) {
        Remove-Item -LiteralPath $PidFile -Force -ErrorAction SilentlyContinue
        return
    }

    try {
        Stop-Process -Id ([int]$PidValue) -Force -ErrorAction Stop
    } catch {
    }

    Remove-Item -LiteralPath $PidFile -Force -ErrorAction SilentlyContinue
}

function Resolve-TailscaleExe {
    $Candidates = @()

    try {
        $Command = Get-Command tailscale.exe -ErrorAction Stop
        if ($Command -and $Command.Source) {
            $Candidates += $Command.Source
        }
    } catch {
    }

    if ($env:ProgramFiles) {
        $Candidates += (Join-Path $env:ProgramFiles "Tailscale IPN\tailscale.exe")
    }

    if ($env:ProgramFiles -and (Test-Path Env:"ProgramFiles(x86)")) {
        $Candidates += (Join-Path ${env:ProgramFiles(x86)} "Tailscale IPN\tailscale.exe")
    }

    foreach ($Candidate in ($Candidates | Select-Object -Unique)) {
        if (-not [string]::IsNullOrWhiteSpace($Candidate) -and (Test-Path -LiteralPath $Candidate)) {
            return $Candidate
        }
    }

    return $null
}

function Test-TcpPort {
    param(
        [string]$HostName = "127.0.0.1",
        [int]$Port,
        [int]$TimeoutMs = 1000
    )

    $Client = New-Object System.Net.Sockets.TcpClient
    try {
        $AsyncResult = $Client.BeginConnect($HostName, $Port, $null, $null)
        if (-not $AsyncResult.AsyncWaitHandle.WaitOne($TimeoutMs, $false)) {
            return $false
        }
        $Client.EndConnect($AsyncResult)
        return $true
    } catch {
        return $false
    } finally {
        $Client.Close()
    }
}

function Wait-TcpPort {
    param(
        [string]$HostName = "127.0.0.1",
        [int]$Port,
        [int]$Attempts = 15,
        [int]$TimeoutMs = 1000,
        [int]$DelayMs = 400
    )

    for ($Index = 0; $Index -lt $Attempts; $Index++) {
        if (Test-TcpPort -HostName $HostName -Port $Port -TimeoutMs $TimeoutMs) {
            return $true
        }
        Start-Sleep -Milliseconds $DelayMs
    }

    return $false
}

function Get-LogTail {
    param([string]$Path, [int]$Lines = 20)

    if (-not (Test-Path -LiteralPath $Path)) {
        return @()
    }

    return Get-Content -LiteralPath $Path -Tail $Lines -ErrorAction SilentlyContinue
}

function Find-RegexInFiles {
    param(
        [string[]]$Paths,
        [string]$Pattern
    )

    foreach ($Path in $Paths) {
        if (-not (Test-Path -LiteralPath $Path)) {
            continue
        }
        $Content = Get-Content -LiteralPath $Path -Raw -ErrorAction SilentlyContinue
        if ($Content -match $Pattern) {
            return $matches[0]
        }
    }

    return $null
}

function Invoke-NativeCapture {
    param(
        [string]$FilePath,
        [string[]]$Arguments,
        [string]$StdoutPath,
        [string]$StderrPath
    )

    foreach ($Path in @($StdoutPath, $StderrPath)) {
        if (Test-Path -LiteralPath $Path) {
            Remove-Item -LiteralPath $Path -Force
        }
    }

    $Proc = Start-Process `
        -FilePath $FilePath `
        -ArgumentList $Arguments `
        -PassThru `
        -Wait `
        -WindowStyle Hidden `
        -RedirectStandardOutput $StdoutPath `
        -RedirectStandardError $StderrPath

    $Stdout = if (Test-Path -LiteralPath $StdoutPath) {
        Get-Content -LiteralPath $StdoutPath -Raw -ErrorAction SilentlyContinue
    } else {
        ""
    }

    $Stderr = if (Test-Path -LiteralPath $StderrPath) {
        Get-Content -LiteralPath $StderrPath -Raw -ErrorAction SilentlyContinue
    } else {
        ""
    }

    return [pscustomobject]@{
        ExitCode = $Proc.ExitCode
        Stdout   = $Stdout
        Stderr   = $Stderr
    }
}

function Write-LinkSummary {
    param(
        [string]$PublicUrl,
        [string]$BackendName,
        [string]$ServerPid = "",
        [string]$TunnelPid = ""
    )

    Write-Host "tight binding root is being served from: $RootDir"
    if (-not [string]::IsNullOrWhiteSpace($ServerPid)) {
        Write-Host "local root:   http://localhost:$Port/"
    }
    Write-Host "public root:  $PublicUrl/"
    Write-Host "backend:      $BackendName"
    Write-Host "graphene:     $PublicUrl/single_layer_graphene/outputs/report.html"
    Write-Host "dimerization: $PublicUrl/graphene_bond_dimerization/outputs/report.html"
    Write-Host "1d chain:     $PublicUrl/1d_chain_dimerization/outputs/report.html"
    Write-Host "ab cluster:   $PublicUrl/graphene_ab_pair_cluster/outputs/report.html"
    if (-not [string]::IsNullOrWhiteSpace($ServerPid)) {
        Write-Host "server pid:   $ServerPid"
    }
    if (-not [string]::IsNullOrWhiteSpace($TunnelPid)) {
        Write-Host "tunnel pid:   $TunnelPid"
    }
    Write-Host "state dir:    $StateDir"
    Write-Host "note:         rerun any tight binding project and refresh the page; the same root server will serve the updated files."
}

function Start-LocalServer {
    Stop-ExistingProcess -PidFile $ServerPidFile

    if (-not (Test-Path -LiteralPath $PythonExe)) {
        throw "Python executable not found: $PythonExe"
    }

    if (Test-Path -LiteralPath $ServerStdout) { Remove-Item -LiteralPath $ServerStdout -Force }
    if (Test-Path -LiteralPath $ServerStderr) { Remove-Item -LiteralPath $ServerStderr -Force }

    $QuotedRootDir = '"' + $RootDir + '"'
    $ServerProc = Start-Process `
        -FilePath $PythonExe `
        -ArgumentList @("-m", "http.server", "$Port", "--bind", "0.0.0.0", "--directory", $QuotedRootDir) `
        -PassThru `
        -WindowStyle Hidden `
        -RedirectStandardOutput $ServerStdout `
        -RedirectStandardError $ServerStderr

    $ServerProc.Id | Set-Content -LiteralPath $ServerPidFile -Encoding ascii

    $PortReady = Wait-TcpPort -HostName "127.0.0.1" -Port $Port -Attempts 15 -TimeoutMs 1000 -DelayMs 400
    if (-not $PortReady) {
        Stop-Process -Id $ServerProc.Id -Force -ErrorAction SilentlyContinue
        throw "The local HTTP server did not start on port $Port."
    }

    return $ServerProc
}

function Start-TailscaleFunnel {
    param([string]$TailscaleExe)

    if ([string]::IsNullOrWhiteSpace($TailscaleExe)) {
        throw "Tailscale CLI was not found."
    }

    Write-Host "Starting Tailscale Funnel on the tight binding root..."

    Stop-ExistingProcess -PidFile $ServerPidFile
    Stop-ExistingProcess -PidFile $TunnelPidFile

    try {
        [void](Invoke-NativeCapture -FilePath $TailscaleExe -Arguments @("funnel", "reset") -StdoutPath $TailscaleStdout -StderrPath $TailscaleStderr)
    } catch {
    }

    $FunnelResult = Invoke-NativeCapture -FilePath $TailscaleExe -Arguments @("funnel", "--bg", "--yes", $RootDir) -StdoutPath $TailscaleStdout -StderrPath $TailscaleStderr
    if ($FunnelResult.ExitCode -ne 0) {
        $JoinedOutput = (($FunnelResult.Stdout + [Environment]::NewLine + $FunnelResult.Stderr).Trim())
        throw "tailscale funnel failed: $JoinedOutput"
    }

    Start-Sleep -Seconds 2

    $StatusResult = Invoke-NativeCapture -FilePath $TailscaleExe -Arguments @("status", "--json") -StdoutPath $TailscaleStdout -StderrPath $TailscaleStderr
    if ($StatusResult.ExitCode -ne 0) {
        $JoinedOutput = (($StatusResult.Stdout + [Environment]::NewLine + $StatusResult.Stderr).Trim())
        throw "tailscale status failed: $JoinedOutput"
    }

    $StatusJson = $StatusResult.Stdout | ConvertFrom-Json
    $DnsName = $StatusJson.Self.DNSName
    if ([string]::IsNullOrWhiteSpace($DnsName)) {
        throw "Could not determine the Tailscale DNS name."
    }

    $PublicUrl = "https://" + $DnsName.TrimEnd(".")
    $PublicUrl | Set-Content -LiteralPath $UrlFile -Encoding ascii
    "tailscale" | Set-Content -LiteralPath $BackendFile -Encoding ascii

    Write-LinkSummary -PublicUrl $PublicUrl -BackendName "tailscale funnel (stable ts.net URL)"

    $CombinedOutput = (($FunnelResult.Stdout + [Environment]::NewLine + $FunnelResult.Stderr).Trim())
    if ($CombinedOutput) {
        Write-Host ""
        Write-Host "tailscale output:"
        ($CombinedOutput -split "`r?`n") | ForEach-Object { if ($_ -ne "") { Write-Host "  $_" } }
    }
}

function Start-CloudflaredTunnel {
    Write-Host "Starting local HTTP server for tight binding..."
    $ServerProc = Start-LocalServer

    Write-Host "Starting cloudflared public tunnel..."

    Stop-ExistingProcess -PidFile $TunnelPidFile

    if (-not (Test-Path -LiteralPath $CloudflaredExe)) {
        Invoke-WebRequest `
            -Uri "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe" `
            -OutFile $CloudflaredExe
    }

    foreach ($Path in @($TunnelStdout, $TunnelStderr)) {
        if (Test-Path -LiteralPath $Path) {
            Remove-Item -LiteralPath $Path -Force
        }
    }

    $TunnelProc = Start-Process `
        -FilePath $CloudflaredExe `
        -ArgumentList @("tunnel", "--url", "http://127.0.0.1:$Port", "--no-autoupdate") `
        -PassThru `
        -WindowStyle Hidden `
        -RedirectStandardOutput $TunnelStdout `
        -RedirectStandardError $TunnelStderr

    $TunnelProc.Id | Set-Content -LiteralPath $TunnelPidFile -Encoding ascii

    $PublicUrl = $null
    for ($Index = 0; $Index -lt 45; $Index++) {
        Start-Sleep -Seconds 1
        if ($TunnelProc.HasExited) {
            break
        }
        $PublicUrl = Find-RegexInFiles -Paths @($TunnelStdout, $TunnelStderr) -Pattern "https://[-a-z0-9]+\.trycloudflare\.com"
        if ($PublicUrl) {
            break
        }
    }

    if (-not $PublicUrl) {
        $StdoutTail = Get-LogTail -Path $TunnelStdout
        $StderrTail = Get-LogTail -Path $TunnelStderr
        Stop-Process -Id $TunnelProc.Id -Force -ErrorAction SilentlyContinue
        Stop-ExistingProcess -PidFile $ServerPidFile

        $MessageLines = @(
            "Failed to obtain a public cloudflared URL.",
            "stdout: $TunnelStdout",
            "stderr: $TunnelStderr"
        )
        if ($StdoutTail.Count -gt 0) {
            $MessageLines += ""
            $MessageLines += "cloudflared stdout tail:"
            $MessageLines += $StdoutTail
        }
        if ($StderrTail.Count -gt 0) {
            $MessageLines += ""
            $MessageLines += "cloudflared stderr tail:"
            $MessageLines += $StderrTail
        }

        throw ($MessageLines -join [Environment]::NewLine)
    }

    $PublicUrl | Set-Content -LiteralPath $UrlFile -Encoding ascii
    "cloudflared" | Set-Content -LiteralPath $BackendFile -Encoding ascii

    Write-LinkSummary -PublicUrl $PublicUrl -BackendName "cloudflared (temporary public URL)" -ServerPid "$($ServerProc.Id)" -TunnelPid "$($TunnelProc.Id)"
    Write-Host "warning:      this public URL changes each time the tunnel is restarted."
}

if (Test-Path -LiteralPath $UrlFile) { Remove-Item -LiteralPath $UrlFile -Force -ErrorAction SilentlyContinue }
if (Test-Path -LiteralPath $BackendFile) { Remove-Item -LiteralPath $BackendFile -Force -ErrorAction SilentlyContinue }

$TailscaleExe = Resolve-TailscaleExe
$SelectedBackend = $Backend
if ($Backend -eq "auto") {
    if ($TailscaleExe) {
        $SelectedBackend = "tailscale"
    } else {
        $SelectedBackend = "cloudflared"
    }
}

if ($SelectedBackend -eq "tailscale") {
    try {
        Start-TailscaleFunnel -TailscaleExe $TailscaleExe
        exit 0
    } catch {
        if ($Backend -eq "tailscale") {
            throw
        }

        Write-Warning "Tailscale Funnel startup failed. Falling back to temporary cloudflared URL. Details: $($_.Exception.Message)"
        Start-CloudflaredTunnel
        exit 0
    }
}

Start-CloudflaredTunnel
