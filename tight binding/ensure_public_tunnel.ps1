param(
    [int]$Port = 8010,
    [ValidateSet("auto", "tailscale", "cloudflared")]
    [string]$Backend = "auto",
    [string]$StateDir = "",
    [string]$StartScriptPath = ""
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($StateDir)) {
    $StateDir = Join-Path $env:TEMP "tight_binding_public"
}

if ([string]::IsNullOrWhiteSpace($StartScriptPath)) {
    $StartScriptPath = Join-Path $PSScriptRoot "start_public_tunnel.ps1"
}

$ServerPidFile = Join-Path $StateDir "server.pid"
$TunnelPidFile = Join-Path $StateDir "tunnel.pid"
$BackendFile = Join-Path $StateDir "backend.txt"
$UrlFile = Join-Path $StateDir "public_url.txt"

function Read-FirstLine {
    param([string]$Path)

    if (-not (Test-Path -LiteralPath $Path)) {
        return ""
    }

    $Line = Get-Content -LiteralPath $Path -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($null -eq $Line) {
        return ""
    }

    return $Line.Trim()
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

function Get-RunningProcess {
    param([string]$PidValue)

    if ([string]::IsNullOrWhiteSpace($PidValue)) {
        return $null
    }

    try {
        return Get-Process -Id ([int]$PidValue) -ErrorAction Stop
    } catch {
        return $null
    }
}

$StoredBackend = Read-FirstLine -Path $BackendFile
$SelectedBackend = if ($Backend -eq "auto" -and $StoredBackend) { $StoredBackend } else { $Backend }
$ServerPid = Read-FirstLine -Path $ServerPidFile
$TunnelPid = Read-FirstLine -Path $TunnelPidFile
$PublicUrl = Read-FirstLine -Path $UrlFile
$ServerProc = Get-RunningProcess -PidValue $ServerPid
$TunnelProc = Get-RunningProcess -PidValue $TunnelPid
$LocalPortUp = Test-TcpPort -HostName "127.0.0.1" -Port $Port -TimeoutMs 1000

$Healthy = if ($SelectedBackend -eq "tailscale") {
    $ServerProc -and $LocalPortUp -and (-not [string]::IsNullOrWhiteSpace($PublicUrl))
} else {
    $ServerProc -and $TunnelProc -and $LocalPortUp
}

if ($Healthy) {
    Write-Host "tight binding public tunnel already healthy."
    Write-Host "backend:      $SelectedBackend"
    if ($PublicUrl) {
        Write-Host "public root:  $PublicUrl/"
    }
    exit 0
}

Write-Host "tight binding public tunnel is down or partial. Restarting..."

if (-not (Test-Path -LiteralPath $StartScriptPath)) {
    throw "Start script not found: $StartScriptPath"
}

$QuotedScript = "'" + $StartScriptPath.Replace("'", "''") + "'"
$Command = "& $QuotedScript -Port $Port -Backend $SelectedBackend"
powershell.exe -ExecutionPolicy Bypass -Command $Command
