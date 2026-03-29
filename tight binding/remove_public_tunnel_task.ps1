param(
    [string]$TaskName = "TightBindingPublicTunnel",
    [string]$WatchdogTaskName = "TightBindingPublicTunnelWatchdog",
    [string]$StateDir = ""
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($StateDir)) {
    $StateDir = Join-Path $env:TEMP "tight_binding_public"
}

foreach ($CurrentTaskName in @($TaskName, $WatchdogTaskName)) {
    $QueryOutput = & schtasks.exe /Query /TN $CurrentTaskName 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Scheduled task not found or could not be removed: $CurrentTaskName"
        continue
    }

    $DeleteOutput = & schtasks.exe /Delete /TN $CurrentTaskName /F 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Scheduled task removed: $CurrentTaskName"
    } else {
        Write-Host "Scheduled task not found or could not be removed: $CurrentTaskName"
        if ($DeleteOutput) {
            Write-Host ($DeleteOutput -join [Environment]::NewLine)
        }
    }
}

foreach ($LauncherName in @("start_public_tunnel_launcher.cmd", "ensure_public_tunnel_launcher.cmd")) {
    $LauncherPath = Join-Path $StateDir $LauncherName
    if (Test-Path -LiteralPath $LauncherPath) {
        Remove-Item -LiteralPath $LauncherPath -Force -ErrorAction SilentlyContinue
        Write-Host "Removed launcher: $LauncherPath"
    }
}
