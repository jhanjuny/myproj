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
$TunnelLog = Join-Path $StateDir "cloudflared.log"
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
        $Proc = Get-Process -Id ([int]$PidValue) -ErrorAction Stop
        Stop-Process -Id $Proc.Id -Force -ErrorAction SilentlyContinue
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
        [string]$Host = "127.0.0.1",
        [int]$Port,
        [int]$TimeoutMs = 1000
    )

    $Client = New-Object System.Net.Sockets.TcpClient
    try {
        $AsyncResult = $Client.BeginConnect($Host, $Port, $null, $null)
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
        [string]$Host = "127.0.0.1",
        [int]$Port,
        [int]$Attempts = 15,
        [int]$TimeoutMs = 1000,
        [int]$DelayMs = 400
    )

    for ($Index = 0; $Index -lt $Attempts; $Index++) {
        if (Test-TcpPort -Host $Host -Port $Port -TimeoutMs $TimeoutMs) {
            return $true
        }
        Start-Sleep -Milliseconds $DelayMs
    }

    return $false
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

    $PortReady = Wait-TcpPort -Host "127.0.0.1" -Port $Port -Attempts 15 -TimeoutMs 1000 -DelayMs 400
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

    try {
        & $TailscaleExe funnel reset | Out-Null
    } catch {
    }

    $FunnelOutput = & $TailscaleExe funnel --bg --yes "http://127.0.0.1:$Port" 2>&1
    Start-Sleep -Seconds 2

    $StatusJson = & $TailscaleExe status --json | ConvertFrom-Json
    $DnsName = $StatusJson.Self.DNSName
    if ([string]::IsNullOrWhiteSpace($DnsName)) {
        throw "Could not determine the Tailscale DNS name."
    }

    $PublicUrl = "https://" + $DnsName.TrimEnd(".")
    $PublicUrl | Set-Content -LiteralPath $UrlFile -Encoding ascii
    "tailscale" | Set-Content -LiteralPath $BackendFile -Encoding ascii
    if (Test-Path -LiteralPath $TunnelPidFile) {
        Remove-Item -LiteralPath $TunnelPidFile -Force -ErrorAction SilentlyContinue
    }

    Write-Output "tight binding root is being served from: $RootDir"
    Write-Output "local root:   http://localhost:$Port/"
    Write-Output "public root:  $PublicUrl/"
    Write-Output "backend:      tailscale funnel"
    Write-Output "graphene:     $PublicUrl/single_layer_graphene/outputs/report.html"
    Write-Output "dimerization: $PublicUrl/graphene_bond_dimerization/outputs/report.html"
    Write-Output "1d chain:     $PublicUrl/1d_chain_dimerization/outputs/report.html"
    Write-Output "ab cluster:   $PublicUrl/graphene_ab_pair_cluster/outputs/report.html"
    Write-Output "server pid:   $((Get-Content -LiteralPath $ServerPidFile | Select-Object -First 1).Trim())"
    Write-Output "state dir:    $StateDir"
    Write-Output "note:         rerun any tight binding project and refresh the page; the same root server will serve the updated files."

    if ($FunnelOutput) {
        Write-Output ""
        Write-Output "tailscale output:"
        $FunnelOutput | ForEach-Object { Write-Output "  $_" }
    }
}

function Start-CloudflaredTunnel {
    Stop-ExistingProcess -PidFile $TunnelPidFile

    if (-not (Test-Path -LiteralPath $CloudflaredExe)) {
        Invoke-WebRequest `
            -Uri "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe" `
            -OutFile $CloudflaredExe
    }

    if (Test-Path -LiteralPath $TunnelLog) { Remove-Item -LiteralPath $TunnelLog -Force }

    $QuotedTunnelLog = '"' + $TunnelLog + '"'
    $TunnelProc = Start-Process `
        -FilePath $CloudflaredExe `
        -ArgumentList @("tunnel", "--url", "http://127.0.0.1:$Port", "--no-autoupdate", "--logfile", $QuotedTunnelLog) `
        -PassThru `
        -WindowStyle Hidden

    $TunnelProc.Id | Set-Content -LiteralPath $TunnelPidFile -Encoding ascii

    $PublicUrl = $null
    for ($Index = 0; $Index -lt 60; $Index++) {
        Start-Sleep -Seconds 1
        if ($TunnelProc.HasExited) {
            break
        }
        if (Test-Path -LiteralPath $TunnelLog) {
            $Content = Get-Content -LiteralPath $TunnelLog -Raw -ErrorAction SilentlyContinue
            if ($Content -match "https://[-a-z0-9]+\.trycloudflare\.com") {
                $PublicUrl = $matches[0]
                break
            }
        }
    }

    if (-not $PublicUrl) {
        Stop-Process -Id $TunnelProc.Id -Force -ErrorAction SilentlyContinue
        throw "Failed to obtain a public cloudflared URL. Check $TunnelLog"
    }

    $PublicUrl | Set-Content -LiteralPath $UrlFile -Encoding ascii
    "cloudflared" | Set-Content -LiteralPath $BackendFile -Encoding ascii

    Write-Output "tight binding root is being served from: $RootDir"
    Write-Output "local root:   http://localhost:$Port/"
    Write-Output "public root:  $PublicUrl/"
    Write-Output "backend:      cloudflared (temporary public URL)"
    Write-Output "graphene:     $PublicUrl/single_layer_graphene/outputs/report.html"
    Write-Output "dimerization: $PublicUrl/graphene_bond_dimerization/outputs/report.html"
    Write-Output "1d chain:     $PublicUrl/1d_chain_dimerization/outputs/report.html"
    Write-Output "ab cluster:   $PublicUrl/graphene_ab_pair_cluster/outputs/report.html"
    Write-Output "server pid:   $((Get-Content -LiteralPath $ServerPidFile | Select-Object -First 1).Trim())"
    Write-Output "tunnel pid:   $($TunnelProc.Id)"
    Write-Output "state dir:    $StateDir"
    Write-Output "note:         rerun any tight binding project and refresh the page; the same root server will serve the updated files."
    Write-Output "warning:      this public URL changes each time the tunnel is restarted."
}

if (Test-Path -LiteralPath $UrlFile) { Remove-Item -LiteralPath $UrlFile -Force -ErrorAction SilentlyContinue }
if (Test-Path -LiteralPath $BackendFile) { Remove-Item -LiteralPath $BackendFile -Force -ErrorAction SilentlyContinue }

$ServerProc = Start-LocalServer
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
            Stop-ExistingProcess -PidFile $ServerPidFile
            throw
        }

        Write-Warning "Tailscale Funnel startup failed. Falling back to temporary cloudflared URL. Details: $($_.Exception.Message)"
        Start-CloudflaredTunnel
        exit 0
    }
}

Start-CloudflaredTunnel
