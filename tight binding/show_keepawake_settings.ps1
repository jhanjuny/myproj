param()

$ErrorActionPreference = "Stop"

Write-Host "Current AC sleep / hibernate settings"
Write-Host "-------------------------------------"
powercfg /query SCHEME_CURRENT SUB_SLEEP STANDBYIDLE
Write-Host ""
powercfg /query SCHEME_CURRENT SUB_SLEEP HIBERNATEIDLE
