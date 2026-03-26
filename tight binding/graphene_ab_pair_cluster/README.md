# graphene_ab_pair_cluster

Single-layer graphene with a pure real-space AB-pair clustering distortion.

## Interpretation

This project follows the user's "pure 2" interpretation:

- start from single-layer graphene
- enlarge to a 6-site supercell
- explicitly move each original AB pair closer together in real space
- relabel the 6 sites as `A, A', B, B', C, C'`
- use equal onsite energies
- use nearest-neighbor hopping only
- generate hopping amplitudes from the distorted bond lengths through

```text
t(d) = t0 * exp[-beta * (d/a0 - 1)]
```

So this is not a hopping-only Kekule pattern. The coordinates themselves are distorted and the Hamiltonian is built from that distorted geometry.

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
- `outputs/report.html`
- `outputs/interactive_status.txt`

## Run

From the repository root:

```powershell
python "tight binding\graphene_ab_pair_cluster\run_graphene_ab_pair_cluster.py" --pair-shift 0.12
```

On the remote PC:

```powershell
D:\conda_envs\torch\python.exe "tight binding\graphene_ab_pair_cluster\run_graphene_ab_pair_cluster.py" --pair-shift 0.12
```

For the interactive 3D HTML export, install:

```powershell
D:\conda_envs\torch\python.exe -m pip install "pyvista[jupyter]"
```
