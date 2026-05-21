#Requires -Version 5.1
# One-time setup: installs node-pty and verifies the environment.

$ScriptDir = $PSScriptRoot
Set-Location $ScriptDir

Write-Host "=== codex-driver setup ===" -ForegroundColor Yellow

# Check node
$nodeVer = node --version 2>&1
if ($LASTEXITCODE -ne 0) { Write-Error "node not found in PATH"; exit 1 }
Write-Host "Node: $nodeVer" -ForegroundColor Green

# Check codex
$codexPath = (Get-Command codex -ErrorAction SilentlyContinue).Source
if ($codexPath) {
    Write-Host "Codex: $codexPath" -ForegroundColor Green
} else {
    Write-Warning "codex not found in PATH — install with: npm install -g @openai/codex"
}

# Install node-pty
Write-Host "Installing node-pty..." -ForegroundColor Yellow
npm install
if ($LASTEXITCODE -ne 0) { Write-Error "npm install failed"; exit 1 }

Write-Host ""
Write-Host "Setup complete. Usage examples:" -ForegroundColor Green
Write-Host '  .\codex-rescue.ps1 -File apps\rheed_monitor\main.py -Error "ImportError"'
Write-Host '  .\codex-review.ps1 -File apps\rheed_monitor\main.py'
Write-Host '  node codex-pty.mjs "explain this error: ..."'
