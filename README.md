# myproj

This repository contains a small experiment framework for training/evaluating models and tracking runs.

## Environment

Install dependencies:
```powershell
python -m pip install -r requirements.txt
```

## Sweep

Run sweep (lab PC / remote):

```powershell
powershell -ExecutionPolicy Bypass -File experiments/run_sweep.ps1 -Dataset demo -Epochs 5 -Exps "configs\exp\lr_1e3.json,configs\exp\lr_1e4.json"
```