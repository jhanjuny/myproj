# 1d_chain_dimerization

Comparison example for:

- a uniform 1D tight-binding chain
- a dimerized 1D chain with hopping modulation `delta`

## Outputs

Running the script produces:

- `outputs/real_space.svg`
- `outputs/model_summary.txt`
- `outputs/dos_overlay.csv`
- `outputs/dos_overlay.svg`

## Run

From the repository root:

```powershell
python "tight binding\1d_chain_dimerization\run_comparison.py" --delta 0.25
```

On the remote PC:

```powershell
D:\conda_envs\torch\python.exe "tight binding\1d_chain_dimerization\run_comparison.py" --delta 0.25
```

## Model

Uniform chain:

```text
H(k) = -2 t cos(k)
```

Dimerized chain:

```text
t1 = t (1 - delta)
t2 = t (1 + delta)
H(k) = [[0, t1 + t2 e^(-ik)],
        [t1 + t2 e^(+ik), 0]]
```
