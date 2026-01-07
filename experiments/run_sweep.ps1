param(
  [string]$Dataset = "demo",
  [int]$Epochs = 3,
  [string[]]$Exps = @("configs/exp/lr_1e3.json", "configs/exp/lr_1e4.json"),
  [string]$Python = "D:\conda_envs\torch\python.exe",
  [string]$TrainScript = "src/train.py",
  [string]$RunsMd = "experiments/runs.md"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Read-Json([string]$Path) {
  if (!(Test-Path $Path)) { return $null }
  # BOM 대응
  $txt = Get-Content -Raw -Encoding UTF8 $Path
  try { return ($txt | ConvertFrom-Json) } catch { return ($txt | ConvertFrom-Json) }
}

function Get-TagFromExp([string]$ExpPath) {
  $cfg = Read-Json $ExpPath
  if ($null -eq $cfg) { return "exp" }
  if ($cfg.PSObject.Properties.Name -contains "lr") {
    $lr = [double]$cfg.lr
    if ($lr -gt 0) {
      $log10 = [math]::Log10($lr)
      $r = [math]::Round($log10, 0)
      if ([math]::Abs($log10 - $r) -lt 1e-10) {
        return ("lr=1e{0}" -f ([int]$r))
      }
      return ("lr={0}" -f $lr)
    }
  }
  return "exp"
}

function Ensure-RunsMd([string]$Path) {
  if (!(Test-Path $Path)) {
    New-Item -ItemType Directory -Force (Split-Path $Path) | Out-Null
    @"
# Experiment runs
"@ | Set-Content -Encoding UTF8 $Path
  }
}

# outputs_dir 읽기
$paths = Read-Json "configs/paths.json"
if ($null -eq $paths -or $null -eq $paths.outputs_dir) {
  throw "configs/paths.json에서 outputs_dir를 읽지 못했습니다."
}
$outputsDir = $paths.outputs_dir

Ensure-RunsMd $RunsMd

$stampDate = (Get-Date -Format "yyyy-MM-dd")
$stampTime = (Get-Date -Format "HH:mm:ss")
Add-Content -Encoding UTF8 $RunsMd ""
Add-Content -Encoding UTF8 $RunsMd "## $stampDate sweep ($Dataset) @ $stampTime"
Add-Content -Encoding UTF8 $RunsMd ""
Add-Content -Encoding UTF8 $RunsMd "| tag | exp | run_dir | last_val_acc | last_train_loss |"
Add-Content -Encoding UTF8 $RunsMd "|---|---|---|---:|---:|"

foreach ($exp in $Exps) {
  if (!(Test-Path $exp)) { throw "exp 파일이 없습니다: $exp" }

  Write-Host "=== RUN exp=$exp ==="
  $tag = Get-TagFromExp $exp

  # 실행 (출력 캡쳐)
  $out = & $Python $TrainScript --dataset $Dataset --exp $exp --epochs $Epochs 2>&1
  $out | ForEach-Object { $_ }

  # run_dir 파싱 (없으면 최신 run_*로 fallback)
  $runDir = $null
  foreach ($line in $out) {
    if ($line -match "^\[env\]\s+run_dir:\s+(.*)$") {
      $runDir = $Matches[1].Trim()
    }
  }
  if ([string]::IsNullOrWhiteSpace($runDir)) {
    $latest = Get-ChildItem -Directory -Path $outputsDir -Filter "run_*" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($null -eq $latest) { throw "outputs_dir 아래에 run_*가 없습니다: $outputsDir" }
    $runDir = $latest.FullName
  }

  # tag.txt 남기기
  "$tag ($exp)" | Set-Content -Encoding UTF8 (Join-Path $runDir "tag.txt")

  # summary.json 읽기
  $summaryPath = Join-Path $runDir "summary.json"
  $s = Read-Json $summaryPath
  $acc = $null
  $loss = $null
  if ($null -ne $s) {
    $acc = $s.last_val_acc
    $loss = $s.last_train_loss
  }

  $leaf = Split-Path $runDir -Leaf
  $relRun = "outputs/$leaf"

  Add-Content -Encoding UTF8 $RunsMd ("| {0} | {1} | {2} | {3} | {4} |" -f $tag, $exp, $relRun, $acc, $loss)
}

Write-Host "Updated: $RunsMd"
