from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore").strip()


def collect(run_dir: Path) -> Dict[str, Any]:
    s = read_json(run_dir / "summary.json") or {}
    tag = read_text(run_dir / "tag.txt")
    return {
        "tag": tag,
        "run_dir": run_dir.as_posix(),
        "last_val_acc": s.get("last_val_acc"),
        "last_train_loss": s.get("last_train_loss"),
        "epoch_next": s.get("epoch_next"),
    }


def fmt(v: Any, nd: int = 4) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="outputs/run_*", help="glob for run dirs (relative to repo root)")
    ap.add_argument("--sort", default="last_val_acc", choices=["last_val_acc", "last_train_loss", "run_dir"])
    ap.add_argument("--desc", action="store_true")
    args = ap.parse_args()

    root = Path(".")
    run_dirs = sorted([p for p in root.glob(args.glob) if p.is_dir()])
    rows: List[Dict[str, Any]] = [collect(p) for p in run_dirs]

    def key_fn(r: Dict[str, Any]):
        v = r.get(args.sort)
        if args.sort == "run_dir":
            return r["run_dir"]
        return -1e9 if v is None else v

    rows.sort(key=key_fn, reverse=args.desc)

    headers = ["tag", "run_dir", "last_val_acc", "last_train_loss", "epoch_next"]
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join(["---"] * len(headers)) + "|")
    for r in rows:
        print(
            "| "
            + " | ".join(
                [
                    r.get("tag", ""),
                    r.get("run_dir", ""),
                    fmt(r.get("last_val_acc")),
                    fmt(r.get("last_train_loss")),
                    fmt(r.get("epoch_next"), nd=0),
                ]
            )
            + " |"
        )


if __name__ == "__main__":
    main()
