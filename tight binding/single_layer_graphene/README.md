# single_layer_graphene

Nearest-neighbor tight-binding model for single-layer graphene.

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
- `outputs/report.html`

## Run

From the repository root:

```powershell
python "tight binding\single_layer_graphene\run_graphene.py"
```

On the remote PC:

```powershell
D:\conda_envs\torch\python.exe "tight binding\single_layer_graphene\run_graphene.py"
```

For the interactive 3D HTML export, install:

```powershell
D:\conda_envs\torch\python.exe -m pip install "pyvista[jupyter]"
```

## Model

Nearest-neighbor graphene Hamiltonian:

```text
H(k) = [[0, -t f(k)],
        [-t f*(k), 0]]

f(k) = exp(i k·d1) + exp(i k·d2) + exp(i k·d3)
```

with the three nearest-neighbor bond vectors:

```text
d1 = (0, 1)
d2 = (sqrt(3)/2, -1/2)
d3 = (-sqrt(3)/2, -1/2)
```

The script generates:

- real-space honeycomb patch
- interactive HTML view of the structure
- `G-K-M-G` band structure
- DOS
- 2D reciprocal-space band map inside the first Brillouin zone
