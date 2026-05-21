#Requires -Version 5.1
<#
.SYNOPSIS
    Codex PTY bug-rescue: sends file + error context to Codex interactive TUI.

.EXAMPLE
    .\codex-rescue.ps1 -File apps\rheed_monitor\main.py -Error "AttributeError: 'NoneType'"
    .\codex-rescue.ps1 -Prompt "Why does my loop never exit?"
#>
param(
    [string]$File   = "",
    [string]$Error  = "",
    [string]$Prompt = ""
)

$Driver = Join-Path $PSScriptRoot "codex-pty.mjs"

if (-not (Test-Path $Driver)) {
    Write-Error "Driver not found at $Driver — run install.ps1 first."
    exit 1
}

Write-Host "[codex-rescue] Starting Codex PTY session..." -ForegroundColor Cyan

if ($Prompt) {
    $Prompt | node $Driver --stdin
} elseif ($File) {
    node $Driver --mode rescue --file $File --error $Error
} else {
    Write-Error "Usage: .\codex-rescue.ps1 -File path.py [-Error 'msg']"
    Write-Error "       .\codex-rescue.ps1 -Prompt 'free text question'"
    exit 1
}
