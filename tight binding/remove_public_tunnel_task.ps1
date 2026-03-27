param(
    [string]$TaskName = "TightBindingPublicTunnel"
)

$ErrorActionPreference = "Stop"

try {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction Stop
    Write-Host "Scheduled task removed: $TaskName"
} catch {
    Write-Host "Scheduled task not found or could not be removed: $TaskName"
    throw
}
