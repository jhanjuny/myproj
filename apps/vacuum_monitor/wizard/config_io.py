from __future__ import annotations

from pathlib import Path
import yaml


def load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data or {}


def save_yaml(path: Path, cfg: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False)
    path.write_text(text, encoding="utf-8")
