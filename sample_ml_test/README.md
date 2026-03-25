# sample_ml_test

Minimal Python classification project for validating the Codex workflow.

## What It Does

- generates a synthetic 2D binary classification dataset
- trains a small logistic regression model
- prints train/validation metrics
- runs a few sample predictions at the end

## Run On The Remote PC

From the repository root:

```powershell
cd C:\Users\hanjunpy\ml\projects\myproj
D:\conda_envs\torch\python.exe sample_ml_test\train.py
```

## Faster Smoke Test

```powershell
cd C:\Users\hanjunpy\ml\projects\myproj
D:\conda_envs\torch\python.exe sample_ml_test\train.py --epochs 12 --train-size 512 --val-size 256
```

## Example Git Pull Workflow

```powershell
cd C:\Users\hanjunpy\ml\projects\myproj
git pull
D:\conda_envs\torch\python.exe sample_ml_test\train.py
```

## Notes

- no external dataset download is required
- no output files are created unless `--save-dir` is provided
