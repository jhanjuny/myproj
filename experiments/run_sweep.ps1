param(
  [string]$Dataset = "demo",
  [int]$Epochs = 3
)

$py = "D:\conda_envs\torch\python.exe"
$exps = @(
  "configs/exp/lr_1e3.json",
  "configs/exp/lr_1e4.json"
)

foreach ($e in $exps) {
  Write-Host "=== RUN exp=$e ==="
  & $py src/train.py --dataset $Dataset --exp $e --epochs $Epochs
}
