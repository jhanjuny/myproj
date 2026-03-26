param(
    [string]$StateDir = ""
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($StateDir)) {
    $StateDir = Join-Path $env:TEMP "tight_binding_public"
}

$ServerPidFile = Join-Path $StateDir "server.pid"
$TunnelPidFile = Join-Path $StateDir "tunnel.pid"

function Stop-ByPidFile {
    param([string]$PidFile, [string]$Label)
    if (-not (Test-Path -LiteralPath $PidFile)) {
        Write-Output "${Label}: no pid file"
        return
    }
    $PidValue = (Get-Content -LiteralPath $PidFile -ErrorAction SilentlyContinue | Select-Object -First 1).Trim()
    if ([string]::IsNullOrWhiteSpace($PidValue)) {
        Remove-Item -LiteralPath $PidFile -Force -ErrorAction SilentlyContinue
        Write-Output "${Label}: empty pid file removed"
        return
    }
    try {
        Stop-Process -Id ([int]$PidValue) -Force -ErrorAction Stop
        Write-Output "${Label}: stopped pid $PidValue"
    } catch {
        Write-Output "${Label}: pid $PidValue was not running"
    }
    Remove-Item -LiteralPath $PidFile -Force -ErrorAction SilentlyContinue
}

Stop-ByPidFile -PidFile $TunnelPidFile -Label "tunnel"
Stop-ByPidFile -PidFile $ServerPidFile -Label "server"
