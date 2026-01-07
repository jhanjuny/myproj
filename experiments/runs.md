# Experiment runs

## 2026-01-07 LR sweep (demo)

| tag | exp | run_dir | last_val_acc | last_train_loss |
|---|---|---|---:|---:|
| lr=1e-3 | configs/exp/lr_1e3.json | outputs/run_20260107_145519 | 0.124 | 1.0072 |
| lr=1e-4 | configs/exp/lr_1e4.json | outputs/run_20260107_145522 | 0.119 | 2.0231 |

### Commands
- lr=1e-3:
  - `D:\conda_envs\torch\python.exe src\train.py --dataset demo --exp configs\exp\lr_1e3.json --epochs 3`
- lr=1e-4:
  - `D:\conda_envs\torch\python.exe src\train.py --dataset demo --exp configs\exp\lr_1e4.json --epochs 3`
