#Requires -Version 5.1
<#
.SYNOPSIS
    Codex PTY code-review: sends a file to Codex interactive TUI for review.

.EXAMPLE
    .\codex-review.ps1 -File apps\rheed_monitor\main.py
    .\codex-review.ps1 -File src\datasets\npz_classification.py
#>
param(
    [Parameter(Mandatory=$true)]
    [string]$File
)

$Driver = Join-Path $PSScriptRoot "codex-pty.mjs"

if (-not (Test-Path $Driver)) {
    Write-Error "Driver not found at $Driver — run install.ps1 first."
    exit 1
}

if (-not (Test-Path $File)) {
    Write-Error "File not found: $File"
    exit 1
}

Write-Host "[codex-review] Reviewing: $File" -ForegroundColor Cyan

node $Driver --mode review --file $File
