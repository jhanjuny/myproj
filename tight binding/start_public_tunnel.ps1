param(
    [int]$Port = 8010,
    [string]$PythonExe = "D:\conda_envs\torch\python.exe",
    [string]$RootDir = "",
    [string]$StateDir = ""
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

Stop-ExistingProcess -PidFile $ServerPidFile
Stop-ExistingProcess -PidFile $TunnelPidFile

if (-not (Test-Path -LiteralPath $PythonExe)) {
    throw "Python executable not found: $PythonExe"
}

if (-not (Test-Path -LiteralPath $CloudflaredExe)) {
    Invoke-WebRequest `
        -Uri "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe" `
        -OutFile $CloudflaredExe
}

if (Test-Path -LiteralPath $ServerStdout) { Remove-Item -LiteralPath $ServerStdout -Force }
if (Test-Path -LiteralPath $ServerStderr) { Remove-Item -LiteralPath $ServerStderr -Force }
if (Test-Path -LiteralPath $TunnelLog) { Remove-Item -LiteralPath $TunnelLog -Force }
if (Test-Path -LiteralPath $UrlFile) { Remove-Item -LiteralPath $UrlFile -Force }

$QuotedRootDir = '"' + $RootDir + '"'
$QuotedTunnelLog = '"' + $TunnelLog + '"'

$ServerProc = Start-Process `
    -FilePath $PythonExe `
    -ArgumentList @("-m", "http.server", "$Port", "--bind", "0.0.0.0", "--directory", $QuotedRootDir) `
    -PassThru `
    -WindowStyle Hidden `
    -RedirectStandardOutput $ServerStdout `
    -RedirectStandardError $ServerStderr

$ServerProc.Id | Set-Content -LiteralPath $ServerPidFile -Encoding ascii

Start-Sleep -Seconds 2

$PortReady = Test-NetConnection 127.0.0.1 -Port $Port -WarningAction SilentlyContinue
if (-not $PortReady.TcpTestSucceeded) {
    Stop-Process -Id $ServerProc.Id -Force -ErrorAction SilentlyContinue
    throw "The local HTTP server did not start on port $Port."
}

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
    Stop-Process -Id $ServerProc.Id -Force -ErrorAction SilentlyContinue
    throw "Failed to obtain a public cloudflared URL. Check $TunnelLog"
}

$PublicUrl | Set-Content -LiteralPath $UrlFile -Encoding ascii

Write-Output "tight binding root is being served from: $RootDir"
Write-Output "local root:   http://localhost:$Port/"
Write-Output "public root:  $PublicUrl/"
Write-Output "graphene:     $PublicUrl/single_layer_graphene/outputs/report.html"
Write-Output "dimerization: $PublicUrl/graphene_bond_dimerization/outputs/report.html"
Write-Output "1d chain:     $PublicUrl/1d_chain_dimerization/outputs/report.html"
Write-Output "server pid:   $($ServerProc.Id)"
Write-Output "tunnel pid:   $($TunnelProc.Id)"
Write-Output "state dir:    $StateDir"
