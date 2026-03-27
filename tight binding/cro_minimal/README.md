# CrO Minimal Tight-Binding Project

This project builds a **structure-derived minimal tight-binding model** from the supplied `CrO_source.cif`.

Assumptions used in this project:

- one effective orbital per crystallographic site
- `4 Cr + 4 O` sites after symmetry expansion
- Cr and O use different onsite energies
- only nearest `Cr-O` hoppings are kept
- hopping amplitudes depend on bond length through an exponential decay law
- no spin, no SOC, no fitted multi-orbital chemistry yet

This is a **playground model from the CIF geometry**, not a quantitatively fitted ab-initio Hamiltonian.

## Remote Run

```powershell
cd C:\ml\projects\myproj
git pull origin main
D:\conda_envs\torch\python.exe "tight binding\cro_minimal\run_cro_minimal.py"
```

## Key Outputs

- `outputs/real_space.svg`
- `outputs/real_space_interactive.html`
- `outputs/band_structure.svg`
- `outputs/dos.svg`
- `outputs/reciprocal_space_map.svg`
- `outputs/reciprocal_space_interactive.html`
- `outputs/calculation_formulas.html`
- `outputs/report.html`

## Main Parameters

- `--onsite-cr`
- `--onsite-o`
- `--t0`
- `--beta`
- `--nn-cutoff`
- `--dos-grid`
- `--slice-grid`

Example:

```powershell
D:\conda_envs\torch\python.exe "tight binding\cro_minimal\run_cro_minimal.py" --onsite-cr 1.3 --onsite-o -1.1 --t0 1.0 --beta 8.0 --nn-cutoff 2.05
```
