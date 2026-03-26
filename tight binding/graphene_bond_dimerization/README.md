# graphene_bond_dimerization

Tight-binding study for graphene with one strengthened nearest-neighbor bond per unit cell.

## Model

This project assumes a minimal bond-dimerization pattern:

- one bond changes from `t` to `t + delta_t`
- the other two nearest-neighbor bonds remain `t`

The Hamiltonian is

```text
H(k) = [[0, -g(k)],
        [-g*(k), 0]]

g(k) = (t + delta_t) exp(i k.d1) + t exp(i k.d2) + t exp(i k.d3)
```

For positive hoppings, this model has:

```text
Eg = 0                  if delta_t <= t
Eg = 2 * (delta_t - t) if delta_t > t
```

So small bond dimerization does not immediately open a band gap. Instead, the Dirac points move in reciprocal space and the cones become anisotropic.

## Outputs

Running the script produces:

- `outputs/real_space.svg`
- `outputs/real_space_interactive.html`
- `outputs/model_summary.txt`
- `outputs/band_structure.csv`
- `outputs/band_structure.svg`
- `outputs/dos.csv`
- `outputs/dos.svg`
- `outputs/reciprocal_space_map.csv`
- `outputs/reciprocal_space_map.svg`
- `outputs/reciprocal_space_interactive.html`
- `outputs/pristine_reciprocal_space_interactive.html`
- `outputs/reciprocal_space_comparison.html`
- `outputs/report.html`
- `outputs/interactive_status.txt`

## Run

From the repository root:

```powershell
python "tight binding\graphene_bond_dimerization\run_dimerized_graphene.py" --delta-t 0.30
```

On the remote PC:

```powershell
D:\conda_envs\torch\python.exe "tight binding\graphene_bond_dimerization\run_dimerized_graphene.py" --delta-t 0.30
```

For the interactive 3D HTML export, install:

```powershell
D:\conda_envs\torch\python.exe -m pip install "pyvista[jupyter]"
```

## Serve The Report

This project includes a helper script that prints laptop/mobile-friendly URLs for the generated report:

```powershell
D:\conda_envs\torch\python.exe "tight binding\graphene_bond_dimerization\serve_report.py" --port 8012
```

Then open:

- `http://localhost:8012/report.html`
- or one of the printed network URLs from the script output
