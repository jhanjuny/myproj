# Experiment runs

## 2026-01-07 LR sweep (demo)

| tag | exp | run_dir | last_val_acc | last_train_loss |
|---|---|---|---:|---:|
| lr=1e-3 | configs/exp/lr_1e3.json | outputs/run_20260107_151542 | 0.1240 | 1.0072 |
| lr=1e-4 | configs/exp/lr_1e4.json | outputs/run_20260107_151545 | 0.1190 | 2.0231 |

### Commands
- Sweep:
  - `powershell -ExecutionPolicy Bypass -File experiments\run_sweep.ps1 -Dataset demo -Epochs 3`

### Per-run outputs (run_dir)
- stdout.log / metrics.jsonl / summary.json / checkpoint_last.pt / args_effective.json / run_info.json
