param(
    [int]$Port = 8010,
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

function Read-FirstLine {
    param([string]$Path)

    if (-not (Test-Path -LiteralPath $Path)) {
        return ""
    }

    $Line = Get-Content -LiteralPath $Path -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($null -eq $Line) {
        return ""
    }

    return $Line.Trim()
}

function Test-TcpPort {
    param(
        [string]$HostName = "127.0.0.1",
        [int]$Port,
        [int]$TimeoutMs = 1000
    )

    $Client = New-Object System.Net.Sockets.TcpClient
    try {
        $AsyncResult = $Client.BeginConnect($HostName, $Port, $null, $null)
        if (-not $AsyncResult.AsyncWaitHandle.WaitOne($TimeoutMs, $false)) {
            return $false
        }
        $Client.EndConnect($AsyncResult)
        return $true
    } catch {
        return $false
    } finally {
        $Client.Close()
    }
}

function Get-RunningProcess {
    param([string]$PidValue)

    if ([string]::IsNullOrWhiteSpace($PidValue)) {
        return $null
    }

    try {
        return Get-Process -Id ([int]$PidValue) -ErrorAction Stop
    } catch {
        return $null
    }
}

function Test-PublicUrl {
    param([string]$Url)

    if ([string]::IsNullOrWhiteSpace($Url)) {
        return [pscustomobject]@{
            Reachable = $false
            Message   = "no public_url.txt entry"
        }
    }

    try {
        $Uri = [System.Uri]$Url
    } catch {
        return [pscustomobject]@{
            Reachable = $false
            Message   = "invalid URL syntax"
        }
    }

    try {
        [void][System.Net.Dns]::GetHostAddresses($Uri.Host)
    } catch {
        return [pscustomobject]@{
            Reachable = $false
            Message   = "DNS lookup failed"
        }
    }

    try {
        $Request = [System.Net.HttpWebRequest]::Create($Url)
        $Request.Method = "HEAD"
        $Request.Timeout = 4000
        $Request.ReadWriteTimeout = 4000
        $Response = $Request.GetResponse()
        $StatusCode = [int]$Response.StatusCode
        $Response.Close()
        return [pscustomobject]@{
            Reachable = $true
            Message   = "HTTP $StatusCode"
        }
    } catch [System.Net.WebException] {
        $StatusText = $_.Exception.Status.ToString()
        return [pscustomobject]@{
            Reachable = $false
            Message   = "web error: $StatusText"
        }
    } catch {
        return [pscustomobject]@{
            Reachable = $false
            Message   = $_.Exception.Message
        }
    }
}

$Backend = Read-FirstLine -Path $BackendFile
$PublicUrl = Read-FirstLine -Path $UrlFile
$ServerPid = Read-FirstLine -Path $ServerPidFile
$TunnelPid = Read-FirstLine -Path $TunnelPidFile

$ServerProc = Get-RunningProcess -PidValue $ServerPid
$TunnelProc = Get-RunningProcess -PidValue $TunnelPid
$LocalPortUp = Test-TcpPort -HostName "127.0.0.1" -Port $Port -TimeoutMs 1000
$PublicCheck = Test-PublicUrl -Url $PublicUrl
$TailscaleSoftPass = $false

if ($Backend -eq "tailscale" -and (-not $PublicCheck.Reachable) -and $PublicUrl -and $ServerProc -and $LocalPortUp) {
    if ($PublicCheck.Message -match "Timeout|The operation has timed out|web error: Timeout") {
        $TailscaleSoftPass = $true
    }
}

Write-Host "tight binding public status"
Write-Host "---------------------------"
Write-Host "state dir:      $StateDir"
Write-Host "backend:        $(if ($Backend) { $Backend } else { '<missing>' })"
Write-Host "public url:     $(if ($PublicUrl) { $PublicUrl } else { '<missing>' })"
Write-Host "server pid:     $(if ($ServerPid) { $ServerPid } else { '<missing>' })"
Write-Host "server running: $(if ($ServerProc) { 'yes' } else { 'no' })"
if ($Backend -eq "tailscale") {
    Write-Host "tunnel pid:     <managed by tailscale>"
    Write-Host "tunnel running: n/a"
} else {
    Write-Host "tunnel pid:     $(if ($TunnelPid) { $TunnelPid } else { '<missing>' })"
    Write-Host "tunnel running: $(if ($TunnelProc) { 'yes' } else { 'no' })"
}
Write-Host "local port up:  $(if ($LocalPortUp) { 'yes' } else { 'no' })"
if ($TailscaleSoftPass) {
    Write-Host "public check:   $($PublicCheck.Message) (self-check soft-pass for tailscale)"
} else {
    Write-Host "public check:   $($PublicCheck.Message)"
}

$Overall = if ($Backend -eq "tailscale") {
    if ($PublicUrl -and $ServerProc -and $LocalPortUp -and ($PublicCheck.Reachable -or $TailscaleSoftPass)) {
        "UP"
    } else {
        "DOWN_OR_MISCONFIGURED"
    }
} else {
    if ($ServerProc -and $TunnelProc -and $LocalPortUp -and $PublicCheck.Reachable) {
        "UP"
    } elseif ($PublicUrl -and -not $PublicCheck.Reachable -and -not $ServerProc -and -not $TunnelProc) {
        "STALE_URL_ONLY"
    } else {
        "DOWN_OR_PARTIAL"
    }
}

Write-Host "overall:        $Overall"

if ($Overall -eq "STALE_URL_ONLY") {
    Write-Host ""
    Write-Host "note: public_url.txt still exists, but the actual server/tunnel processes are gone."
    Write-Host "      start the server again or clear the stale files with stop_public_tunnel.ps1."
}

if ($Backend -eq "tailscale" -and $Overall -eq "UP") {
    Write-Host ""
    Write-Host "note: tailscale funnel is managed by the Tailscale service, so a separate tunnel.pid is not expected."
    if ($TailscaleSoftPass) {
        Write-Host "      the public self-check timed out from this same host, but the local server and saved ts.net URL are in place."
    }
}
