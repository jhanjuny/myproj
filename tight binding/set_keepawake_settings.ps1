param(
    [switch]$AlsoDisableDisplaySleep
)

$ErrorActionPreference = "Stop"

powercfg /change standby-timeout-ac 0
powercfg /change hibernate-timeout-ac 0

if ($AlsoDisableDisplaySleep) {
    powercfg /change monitor-timeout-ac 0
}

Write-Host "AC power sleep timeout: disabled"
Write-Host "AC power hibernate timeout: disabled"
if ($AlsoDisableDisplaySleep) {
    Write-Host "AC power display timeout: disabled"
} else {
    Write-Host "AC power display timeout: unchanged"
}
Write-Host ""
Write-Host "Use show_keepawake_settings.ps1 to confirm the current values."
