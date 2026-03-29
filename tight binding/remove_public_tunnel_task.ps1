param(
    [string]$TaskName = "TightBindingPublicTunnel",
    [string]$WatchdogTaskName = "TightBindingPublicTunnelWatchdog",
    [string]$StateDir = ""
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($StateDir)) {
    $StateDir = Join-Path $env:TEMP "tight_binding_public"
}

try {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction Stop
    Write-Host "Scheduled task removed: $TaskName"
} catch {
    Write-Host "Scheduled task not found or could not be removed: $TaskName"
}

try {
    & schtasks.exe /Delete /TN $WatchdogTaskName /F | Out-Null
    Write-Host "Scheduled task removed: $WatchdogTaskName"
} catch {
    Write-Host "Scheduled task not found or could not be removed: $WatchdogTaskName"
}

foreach ($LauncherName in @("start_public_tunnel_launcher.cmd", "ensure_public_tunnel_launcher.cmd")) {
    $LauncherPath = Join-Path $StateDir $LauncherName
    if (Test-Path -LiteralPath $LauncherPath) {
        Remove-Item -LiteralPath $LauncherPath -Force -ErrorAction SilentlyContinue
        Write-Host "Removed launcher: $LauncherPath"
    }
}
