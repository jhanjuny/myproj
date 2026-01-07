import argparse
import json
from pathlib import Path

def read_last_metrics(run_dir: Path):
    p = run_dir / "metrics.jsonl"
    if not p.exists():
        return None
    last = None
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            last = json.loads(line)
    return last

def read_summary(run_dir: Path):
    p = run_dir / "summary.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="run dirs (e.g. outputs/run_... outputs/run_...)")
    args = ap.parse_args()

    rows = []
    for r in args.runs:
        run_dir = Path(r)
        s = read_summary(run_dir) or {}
        m = read_last_metrics(run_dir) or {}
        rows.append({
            "run_dir": str(run_dir),
            "epochs": s.get("epochs"),
            "last_epoch": s.get("last_epoch"),
            "last_step": s.get("last_step") or m.get("step"),
            "last_train_loss": s.get("last_train_loss") or m.get("train_loss"),
            "last_val_acc": s.get("last_val_acc") or m.get("val_acc"),
        })

    # pretty print
    print(json.dumps(rows, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
