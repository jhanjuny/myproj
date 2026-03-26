# tight binding

Minimal tight-binding workspace for solid-state calculations and simulations.

## Included

- reusable tight-binding model code
- built-in example models: `chain`, `square`, `ssh`
- band structure calculation
- density of states (DOS) calculation
- CSV and SVG export without external plotting dependencies

## Quick Start

From the repository root:

```powershell
cd C:\Users\hanjunpy\ml\projects\myproj
python "tight binding\run_tb.py" --model square --mode both
```

On the remote PC:

```powershell
cd C:\ml\projects\myproj
D:\conda_envs\torch\python.exe "tight binding\run_tb.py" --model square --mode both
```

## Browse Results

To browse the whole `tight binding` workspace through one server:

```powershell
cd C:\ml\projects\myproj
D:\conda_envs\torch\python.exe "tight binding\serve_tight_binding.py" --port 8010
```

Then open:

- `http://localhost:8010/`
- `http://localhost:8010/single_layer_graphene/outputs/report.html`
- `http://localhost:8010/graphene_bond_dimerization/outputs/report.html`
- `http://localhost:8010/1d_chain_dimerization/outputs/report.html`

## Public Link

If you want a public HTTPS link that keeps working even after your local SSH/port-forwarding session closes,
run the tunnel directly on the remote PC:

```powershell
cd C:\ml\projects\myproj
powershell -ExecutionPolicy Bypass -File "tight binding\start_public_tunnel.ps1" -Port 8010
```

Behavior:

- the whole `tight binding` root is served, so rerunning a project updates the same server immediately
- the script now prefers a stable Tailscale Funnel `https://<machine>.<tailnet>.ts.net/` URL when available
- if Tailscale Funnel is unavailable, it falls back to a temporary Cloudflare tunnel URL

To stop the public tunnel later:

```powershell
cd C:\ml\projects\myproj
powershell -ExecutionPolicy Bypass -File "tight binding\stop_public_tunnel.ps1"
```

To check whether the saved public URL is still alive or only a stale record:

```powershell
cd C:\ml\projects\myproj
powershell -ExecutionPolicy Bypass -File "tight binding\status_public_tunnel.ps1" -Port 8010
```

## Example Commands

Square lattice band structure + DOS:

```powershell
python "tight binding\run_tb.py" --model square --mode both
```

SSH chain with custom hopping:

```powershell
python "tight binding\run_tb.py" --model ssh --mode both --t1 0.7 --t2 1.3 --delta 0.2
```

1D chain, band only:

```powershell
python "tight binding\run_tb.py" --model chain --mode band --samples-per-segment 120
```

## Outputs

Generated files are written under:

```text
tight binding/outputs/<model>/
```

Each run can create:

- `band_structure.csv`
- `band_structure.svg`
- `dos.csv`
- `dos.svg`
- `calculation_formulas.html`
- `report.html`

Project-specific reports now include a dedicated formula panel so the Hamiltonian, dispersion, gap expression,
Brillouin-zone integral, numerical sampling approximation, and DOS broadening equation can be reviewed as math,
not just inferred from the code.

## Notes

- no external datasets are needed
- `numpy` is required
- `matplotlib` is not required
