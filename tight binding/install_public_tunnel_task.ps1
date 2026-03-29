param(
    [string]$TaskName = "TightBindingPublicTunnel",
    [string]$WatchdogTaskName = "TightBindingPublicTunnelWatchdog",
    [int]$Port = 8010,
    [int]$WatchdogMinutes = 5,
    [ValidateSet("auto", "tailscale", "cloudflared")]
    [string]$Backend = "auto",
    [string]$StartScriptPath = "",
    [string]$EnsureScriptPath = "",
    [string]$StateDir = "",
    [switch]$RunWithHighest
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($StartScriptPath)) {
    $StartScriptPath = Join-Path $PSScriptRoot "start_public_tunnel.ps1"
}

if ([string]::IsNullOrWhiteSpace($EnsureScriptPath)) {
    $EnsureScriptPath = Join-Path $PSScriptRoot "ensure_public_tunnel.ps1"
}

if ([string]::IsNullOrWhiteSpace($StateDir)) {
    $StateDir = Join-Path $env:TEMP "tight_binding_public"
}

if (-not (Test-Path -LiteralPath $StartScriptPath)) {
    throw "Start script not found: $StartScriptPath"
}
if (-not (Test-Path -LiteralPath $EnsureScriptPath)) {
    throw "Ensure script not found: $EnsureScriptPath"
}

New-Item -ItemType Directory -Force -Path $StateDir | Out-Null

function Write-Launcher {
    param(
        [string]$LauncherPath,
        [string]$ScriptPath
    )

    $Content = @(
        "@echo off",
        "powershell.exe -ExecutionPolicy Bypass -File `"$ScriptPath`" -Port $Port -Backend $Backend"
    )
    Set-Content -LiteralPath $LauncherPath -Value $Content -Encoding ascii
}

$StartLauncherPath = Join-Path $StateDir "start_public_tunnel_launcher.cmd"
$EnsureLauncherPath = Join-Path $StateDir "ensure_public_tunnel_launcher.cmd"
Write-Launcher -LauncherPath $StartLauncherPath -ScriptPath $StartScriptPath
Write-Launcher -LauncherPath $EnsureLauncherPath -ScriptPath $EnsureScriptPath

$RunLevel = if ($RunWithHighest) { "Highest" } else { "Limited" }
$RunLevelValue = if ($RunWithHighest) { "HIGHEST" } else { "LIMITED" }

$StartCreateArgs = @(
    "/Create",
    "/TN", $TaskName,
    "/SC", "ONLOGON",
    "/TR", $StartLauncherPath,
    "/RL", $RunLevelValue,
    "/F"
)
$StartOutput = & schtasks.exe @StartCreateArgs 2>&1
if ($LASTEXITCODE -ne 0) {
    throw "Failed to register logon task '$TaskName':`n$($StartOutput -join [Environment]::NewLine)"
}

Write-Host "Scheduled task registered."
Write-Host "task name:   $TaskName"
Write-Host "trigger:     AtLogOn"
Write-Host "run level:   $RunLevel"
Write-Host "script:      $StartScriptPath"
Write-Host "launcher:    $StartLauncherPath"
Write-Host ""

$WatchdogCreateArgs = @(
    "/Create",
    "/TN", $WatchdogTaskName,
    "/SC", "MINUTE",
    "/MO", "$WatchdogMinutes",
    "/TR", $EnsureLauncherPath,
    "/F"
)
if ($RunWithHighest) {
    $WatchdogCreateArgs += @("/RL", "HIGHEST")
} else {
    $WatchdogCreateArgs += @("/RL", "LIMITED")
}
$WatchdogOutput = & schtasks.exe @WatchdogCreateArgs 2>&1
if ($LASTEXITCODE -ne 0) {
    throw "Failed to register watchdog task '$WatchdogTaskName':`n$($WatchdogOutput -join [Environment]::NewLine)"
}

Write-Host "Watchdog task registered."
Write-Host "watchdog:    $WatchdogTaskName"
Write-Host "interval:    every $WatchdogMinutes minute(s)"
Write-Host "ensure:      $EnsureScriptPath"
Write-Host "launcher:    $EnsureLauncherPath"
Write-Host ""
Write-Host "To test it now:"
Write-Host "  Start-ScheduledTask -TaskName '$TaskName'"
