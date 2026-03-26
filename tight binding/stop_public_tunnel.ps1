param(
    [string]$StateDir = ""
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($StateDir)) {
    $StateDir = Join-Path $env:TEMP "tight_binding_public"
}

$ServerPidFile = Join-Path $StateDir "server.pid"
$TunnelPidFile = Join-Path $StateDir "tunnel.pid"
$BackendFile = Join-Path $StateDir "backend.txt"
$UrlFile = Join-Path $StateDir "public_url.txt"

function Resolve-TailscaleExe {
    $Candidates = @()

    try {
        $Command = Get-Command tailscale.exe -ErrorAction Stop
        if ($Command -and $Command.Source) {
            $Candidates += $Command.Source
        }
    } catch {
    }

    if ($env:ProgramFiles) {
        $Candidates += (Join-Path $env:ProgramFiles "Tailscale IPN\tailscale.exe")
    }

    if ($env:ProgramFiles -and (Test-Path Env:"ProgramFiles(x86)")) {
        $Candidates += (Join-Path ${env:ProgramFiles(x86)} "Tailscale IPN\tailscale.exe")
    }

    foreach ($Candidate in ($Candidates | Select-Object -Unique)) {
        if (-not [string]::IsNullOrWhiteSpace($Candidate) -and (Test-Path -LiteralPath $Candidate)) {
            return $Candidate
        }
    }

    return $null
}

function Stop-ByPidFile {
    param([string]$PidFile, [string]$Label)

    if (-not (Test-Path -LiteralPath $PidFile)) {
        Write-Output "${Label}: no pid file"
        return
    }

    $PidValue = (Get-Content -LiteralPath $PidFile -ErrorAction SilentlyContinue | Select-Object -First 1).Trim()
    if ([string]::IsNullOrWhiteSpace($PidValue)) {
        Remove-Item -LiteralPath $PidFile -Force -ErrorAction SilentlyContinue
        Write-Output "${Label}: empty pid file removed"
        return
    }

    try {
        Stop-Process -Id ([int]$PidValue) -Force -ErrorAction Stop
        Write-Output "${Label}: stopped pid $PidValue"
    } catch {
        Write-Output "${Label}: pid $PidValue was not running"
    }

    Remove-Item -LiteralPath $PidFile -Force -ErrorAction SilentlyContinue
}

$Backend = ""
if (Test-Path -LiteralPath $BackendFile) {
    $Backend = (Get-Content -LiteralPath $BackendFile -ErrorAction SilentlyContinue | Select-Object -First 1).Trim()
}

if ($Backend -eq "tailscale") {
    $TailscaleExe = Resolve-TailscaleExe
    if ($TailscaleExe) {
        try {
            & $TailscaleExe funnel reset | Out-Null
            Write-Output "tailscale: funnel reset"
        } catch {
            Write-Output "tailscale: failed to reset funnel ($($_.Exception.Message))"
        }
    } else {
        Write-Output "tailscale: CLI not found"
    }
    Stop-ByPidFile -PidFile $TunnelPidFile -Label "tunnel"
} else {
    Stop-ByPidFile -PidFile $TunnelPidFile -Label "tunnel"
}

Stop-ByPidFile -PidFile $ServerPidFile -Label "server"

if (Test-Path -LiteralPath $BackendFile) {
    Remove-Item -LiteralPath $BackendFile -Force -ErrorAction SilentlyContinue
}

if (Test-Path -LiteralPath $UrlFile) {
    Remove-Item -LiteralPath $UrlFile -Force -ErrorAction SilentlyContinue
}
