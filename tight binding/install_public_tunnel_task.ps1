param(
    [string]$TaskName = "TightBindingPublicTunnel",
    [string]$WatchdogTaskName = "TightBindingPublicTunnelWatchdog",
    [int]$Port = 8010,
    [int]$WatchdogMinutes = 5,
    [ValidateSet("auto", "tailscale", "cloudflared")]
    [string]$Backend = "auto",
    [string]$StartScriptPath = "",
    [string]$EnsureScriptPath = "",
    [switch]$RunWithHighest
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($StartScriptPath)) {
    $StartScriptPath = Join-Path $PSScriptRoot "start_public_tunnel.ps1"
}

if ([string]::IsNullOrWhiteSpace($EnsureScriptPath)) {
    $EnsureScriptPath = Join-Path $PSScriptRoot "ensure_public_tunnel.ps1"
}

if (-not (Test-Path -LiteralPath $StartScriptPath)) {
    throw "Start script not found: $StartScriptPath"
}
if (-not (Test-Path -LiteralPath $EnsureScriptPath)) {
    throw "Ensure script not found: $EnsureScriptPath"
}

$UserId = if ($env:USERDOMAIN) { "$($env:USERDOMAIN)\$($env:USERNAME)" } else { $env:USERNAME }
$QuotedScript = "'" + $StartScriptPath.Replace("'", "''") + "'"
$Arguments = "-ExecutionPolicy Bypass -Command ""& $QuotedScript -Port $Port -Backend $Backend"""
$QuotedEnsureScript = "'" + $EnsureScriptPath.Replace("'", "''") + "'"
$WatchdogArguments = "-ExecutionPolicy Bypass -Command ""& $QuotedEnsureScript -Port $Port -Backend $Backend"""

$Action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument $Arguments
$Trigger = New-ScheduledTaskTrigger -AtLogOn -User $UserId
$Settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
$RunLevel = if ($RunWithHighest) { "Highest" } else { "Limited" }
$Principal = New-ScheduledTaskPrincipal -UserId $UserId -LogonType Interactive -RunLevel $RunLevel

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $Action `
    -Trigger $Trigger `
    -Settings $Settings `
    -Principal $Principal `
    -Description "Auto-start the tight binding public tunnel on user logon." `
    -Force | Out-Null

Write-Host "Scheduled task registered."
Write-Host "task name:   $TaskName"
Write-Host "user:        $UserId"
Write-Host "trigger:     AtLogOn"
Write-Host "run level:   $RunLevel"
Write-Host "script:      $StartScriptPath"
Write-Host "arguments:   -Port $Port -Backend $Backend"
Write-Host ""

$WatchdogCreateArgs = @(
    "/Create",
    "/TN", $WatchdogTaskName,
    "/SC", "MINUTE",
    "/MO", "$WatchdogMinutes",
    "/TR", "powershell.exe $WatchdogArguments",
    "/F"
)
if ($RunWithHighest) {
    $WatchdogCreateArgs += @("/RL", "HIGHEST")
}
& schtasks.exe @WatchdogCreateArgs | Out-Null

Write-Host "Watchdog task registered."
Write-Host "watchdog:    $WatchdogTaskName"
Write-Host "interval:    every $WatchdogMinutes minute(s)"
Write-Host "ensure:      $EnsureScriptPath"
Write-Host ""
Write-Host "To test it now:"
Write-Host "  Start-ScheduledTask -TaskName '$TaskName'"
