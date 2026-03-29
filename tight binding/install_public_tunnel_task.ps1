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

$CurrentIdentity = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
$UserId = if ([string]::IsNullOrWhiteSpace($CurrentIdentity)) { $env:USERNAME } else { $CurrentIdentity }

$StartLauncherPath = Join-Path $StateDir "start_public_tunnel_launcher.cmd"
$EnsureLauncherPath = Join-Path $StateDir "ensure_public_tunnel_launcher.cmd"
Write-Launcher -LauncherPath $StartLauncherPath -ScriptPath $StartScriptPath
Write-Launcher -LauncherPath $EnsureLauncherPath -ScriptPath $EnsureScriptPath

$Action = New-ScheduledTaskAction -Execute $StartLauncherPath
$Trigger = New-ScheduledTaskTrigger -AtLogOn -User $UserId
$Settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
$RunLevel = if ($RunWithHighest) { "Highest" } else { "Limited" }
$Principal = New-ScheduledTaskPrincipal -UserId $UserId -LogonType InteractiveToken -RunLevel $RunLevel

try {
    Register-ScheduledTask `
        -TaskName $TaskName `
        -Action $Action `
        -Trigger $Trigger `
        -Settings $Settings `
        -Principal $Principal `
        -Description "Auto-start the tight binding public tunnel on user logon." `
        -Force `
        -ErrorAction Stop | Out-Null
} catch {
    throw "Failed to register logon task '$TaskName': $($_.Exception.Message)"
}

Write-Host "Scheduled task registered."
Write-Host "task name:   $TaskName"
Write-Host "user:        $UserId"
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
}
$null = & schtasks.exe @WatchdogCreateArgs
if ($LASTEXITCODE -ne 0) {
    throw "Failed to register watchdog task '$WatchdogTaskName' via schtasks.exe."
}

Write-Host "Watchdog task registered."
Write-Host "watchdog:    $WatchdogTaskName"
Write-Host "interval:    every $WatchdogMinutes minute(s)"
Write-Host "ensure:      $EnsureScriptPath"
Write-Host "launcher:    $EnsureLauncherPath"
Write-Host ""
Write-Host "To test it now:"
Write-Host "  Start-ScheduledTask -TaskName '$TaskName'"
