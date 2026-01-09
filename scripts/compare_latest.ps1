param(
  [string]$Python = "D:\conda_envs\torch\python.exe",
  [string]$OutputsDir = "outputs",
  [string]$Sort = "last_val_acc",
  [switch]$Desc
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if (!(Test-Path $Python)) { throw "Python not found: $Python" }
if (!(Test-Path $OutputsDir)) { throw "Outputs dir not found: $OutputsDir" }

$latest = Get-ChildItem -Directory $OutputsDir -Filter "run_*" |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 1

if ($null -eq $latest) { throw "No run_* found under $OutputsDir" }

$day = $latest.Name.Substring(4,8)  # run_YYYYMMDD_HHMMSS -> YYYYMMDD
$glob = "$OutputsDir/run_${day}_*"

$args = @("src/tools/compare_runs_table.py", "--glob", $glob, "--sort", $Sort)
if ($Desc) { $args += "--desc" }

& $Python @args
