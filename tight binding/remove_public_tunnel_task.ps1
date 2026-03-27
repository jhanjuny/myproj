param(
    [string]$TaskName = "TightBindingPublicTunnel",
    [string]$WatchdogTaskName = "TightBindingPublicTunnelWatchdog"
)

$ErrorActionPreference = "Stop"

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
