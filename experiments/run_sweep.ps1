param(
  [string]$Dataset = "demo",
  [int]$Epochs = 3,
  [string[]]$Exps = @(),
  [string]$Python = "D:\conda_envs\torch\python.exe",
  [string]$TrainScript = "src/train.py",
  [string]$RunsMd = "experiments/runs_local.md",
  [object[]]$Seeds = @(),
  [object[]]$BatchSizes = @(),
  [object[]]$HiddenDims = @(),
  [object[]]$StepsPerEpochs = @()
)


if ($Exps.Count -eq 0) {
  $Exps = Get-ChildItem -Path "configs/exp" -Filter "*.json" |
    Sort-Object Name |
    ForEach-Object { Join-Path "configs/exp" $_.Name }
}




Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Read-Json([string]$Path) {
  if (!(Test-Path $Path)) { return $null }
  # BOM 대응
  
  $txt = Get-Content -Raw -Encoding UTF8 $Path

  # UTF-8 BOM 제거 (JSONDecodeError 예방)
  if ($txt.Length -gt 0 -and [int]$txt[0] -eq 0xFEFF) {
    $txt = $txt.Substring(1)
  }

  return ($txt | ConvertFrom-Json)

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
if ($Seeds.Count -eq 0) { $Seeds = @($null) }
if ($BatchSizes.Count -eq 0) { $BatchSizes = @($null) }
if ($HiddenDims.Count -eq 0) { $HiddenDims = @($null) }
if ($StepsPerEpochs.Count -eq 0) { $StepsPerEpochs = @($null) }


$stampDate = (Get-Date -Format "yyyy-MM-dd")
$stampTime = (Get-Date -Format "HH:mm:ss")
Add-Content -Encoding UTF8 $RunsMd ""
Add-Content -Encoding UTF8 $RunsMd "## $stampDate sweep ($Dataset) @ $stampTime"
Add-Content -Encoding UTF8 $RunsMd ""
Add-Content -Encoding UTF8 $RunsMd "| tag | exp | run_dir | last_val_acc | last_train_loss |"
Add-Content -Encoding UTF8 $RunsMd "|---|---|---|---:|---:|"

foreach ($exp in $Exps) {
  if (!(Test-Path $exp)) { throw "exp 파일이 없습니다: $exp" }

  foreach ($seed in $Seeds) {
    foreach ($bs in $BatchSizes) {
      foreach ($hd in $HiddenDims) {
        foreach ($spe in $StepsPerEpochs) {

          Write-Host "=== RUN exp=$exp seed=$seed bs=$bs hd=$hd spe=$spe ==="

          $tag = Get-TagFromExp $exp
          $tagFull = $tag
          if ($seed -ne $null) { $tagFull += " seed=$seed" }
          if ($bs -ne $null)   { $tagFull += " bs=$bs" }
          if ($hd -ne $null)   { $tagFull += " hd=$hd" }
          if ($spe -ne $null)  { $tagFull += " spe=$spe" }

          $extra = @()
          if ($seed -ne $null) { $extra += @("--seed", "$seed") }
          if ($bs -ne $null)   { $extra += @("--batch_size", "$bs") }
          if ($hd -ne $null)   { $extra += @("--hidden_dim", "$hd") }
          if ($spe -ne $null)  { $extra += @("--steps_per_epoch", "$spe") }

          # 실행 (출력 캡쳐)
          $out = & $Python $TrainScript --dataset $Dataset --exp $exp --epochs $Epochs @extra 2>&1
          $out | ForEach-Object { $_ }

          # run_dir 파싱 (없으면 최신 run_*로 fallback)
          $runDir = $null
          foreach ($line in $out) {
            if ($line -match "^\[env\]\s+run_dir:\s+(.*)$") {
              $runDir = $Matches[1].Trim()
            }
          }
          if ([string]::IsNullOrWhiteSpace($runDir)) {
            $latest = Get-ChildItem -Directory -Path $outputsDir -Filter "run_*" |
              Sort-Object LastWriteTime -Descending | Select-Object -First 1
            if ($null -eq $latest) { throw "outputs_dir 아래에 run_*가 없습니다: $outputsDir" }
            $runDir = $latest.FullName
          }

          # tag.txt 남기기 (tagFull로)
          "$tagFull ($exp)" | Set-Content -Encoding UTF8 (Join-Path $runDir "tag.txt")

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

          $accStr  = if ($null -ne $acc)  { "{0:F4}" -f [double]$acc } else { "" }
          $lossStr = if ($null -ne $loss) { "{0:F4}" -f [double]$loss } else { "" }

          Add-Content -Encoding UTF8 $RunsMd ("| {0} | {1} | {2} | {3} | {4} |" -f $tagFull, $exp, $relRun, $accStr, $lossStr)

        }
      }
    }
  }
}

Write-Host "Updated: $RunsMd"
