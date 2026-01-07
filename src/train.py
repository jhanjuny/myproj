
import argparse
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from datasets.npz_classification import NpzClassificationDataset


@dataclass
class RunInfo:
    started_at: str
    git_head: str | None
    torch_version: str
    cuda_available: bool
    cuda_device: str | None
    args: dict
    paths: dict


def now_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def get_git_head() -> str | None:
    # git이 없는 환경에서도 돌아가게 예외 처리
    try:
        import subprocess

        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:
        return None


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_logger(run_dir: Path) -> None:
    # 아주 단순한 “stdout을 파일로도 남기는” 방식
    # (추후 logging 모듈로 확장 가능)
    log_path = run_dir / "stdout.log"
    class Tee:
        def __init__(self, *files):
            self.files = files
        def write(self, data):
            for f in self.files:
                f.write(data)
                f.flush()
        def flush(self):
            for f in self.files:
                f.flush()

    import sys
    log_f = open(log_path, "a", encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, log_f)
    sys.stderr = Tee(sys.__stderr__, log_f)
    print(f"[log] stdout/stderr -> {log_path}")


def save_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def checkpoint_path(run_dir: Path) -> Path:
    return run_dir / "checkpoint_last.pt"


def save_checkpoint(run_dir: Path, epoch: int, step: int, model: nn.Module, optimizer: optim.Optimizer) -> None:
    ckpt = {
        "epoch": epoch,
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "rng_torch": torch.get_rng_state(),
        "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    p = checkpoint_path(run_dir)
    torch.save(ckpt, p)
    print(f"[ckpt] saved -> {p}")


def load_checkpoint(run_dir: Path, model: nn.Module, optimizer: optim.Optimizer) -> tuple[int, int]:
    p = checkpoint_path(run_dir)
    if not p.exists():
        raise FileNotFoundError(f"checkpoint not found: {p}")

    ckpt = torch.load(p, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])

    # RNG 복원(가능한 범위)
    if "rng_torch" in ckpt and ckpt["rng_torch"] is not None:
        torch.set_rng_state(ckpt["rng_torch"])
    if torch.cuda.is_available() and "rng_cuda" in ckpt and ckpt["rng_cuda"] is not None:
        torch.cuda.set_rng_state_all(ckpt["rng_cuda"])

    epoch = int(ckpt.get("epoch", 0))
    step = int(ckpt.get("step", 0))
    print(f"[ckpt] loaded <- {p} (epoch={epoch}, step={step})")
    return epoch, step


def build_dummy_model(input_dim=1024, hidden=2048, out_dim=10) -> nn.Module:
    return nn.Sequential(
        nn.Linear(input_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, out_dim),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", default="configs/paths.json", help="path json file containing data_dir/outputs_dir")
    ap.add_argument("--run_dir", default=None, help="existing run dir to resume or custom output dir")
    ap.add_argument("--resume", action="store_true", help="resume from checkpoint_last.pt in run_dir")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--steps_per_epoch", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--exp", default=None, help="experiment config json (overrides defaults)")
    ap.add_argument("--dataset", default=None, help="dataset name under data_dir (expects processed/train.npz)")
    args = ap.parse_args()
    # experiment config (json) 적용: exp 파일 값으로 args를 덮어씀
    if args.exp:
        exp_path = Path(args.exp)
        exp_cfg = json.loads(exp_path.read_text(encoding="utf-8"))
        for k, v in exp_cfg.items():
            if hasattr(args, k):
                setattr(args, k, v)



    # paths.json 로드
    paths_path = Path(args.paths)
    paths = json.loads(paths_path.read_text(encoding="utf-8"))
    data_dir = Path(paths["data_dir"])
    outputs_dir = Path(paths["outputs_dir"])

    # run_dir 결정
    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        run_dir = outputs_dir / f"run_{now_tag()}"
    run_dir.mkdir(parents=True, exist_ok=True)

    setup_logger(run_dir)

    # 환경 정보 출력/저장
    cuda_available = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_available else None
    device = torch.device("cuda" if cuda_available else "cpu")

    set_seed(args.seed)

    info = RunInfo(
        started_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        git_head=get_git_head(),
        torch_version=torch.__version__,
        cuda_available=cuda_available,
        cuda_device=device_name,
        args=vars(args),
        paths=paths,
    )
    save_json(run_dir / "run_info.json", asdict(info))
    save_json(run_dir / "args_effective.json", vars(args))
    

    print("[env] data_dir:", str(data_dir))
    print("[env] outputs_dir:", str(outputs_dir))
    print("[env] run_dir:", str(run_dir))
    print("[env] torch:", torch.__version__)
    print("[env] cuda available:", cuda_available)
    print("[env] device:", device_name)

    # dataset 로드 (있으면 사용, 없으면 랜덤 데이터 fallback)
    train_loader = None
    val_loader = None
    input_dim = 1024
    num_classes = 10

    if args.dataset:
        train_npz = data_dir / args.dataset / "processed" / "train.npz"
        if train_npz.exists():
            ds = NpzClassificationDataset(train_npz)
            input_dim = ds.input_dim
            num_classes = ds.num_classes
            train_loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
            print(f"[data] loaded: {train_npz} (N={len(ds)}, D={input_dim}, C={num_classes})")
            # ===== 6-2B: val 로드 추가 (여기에 넣는 게 정답) =====
            val_npz = data_dir / args.dataset / "processed" / "val.npz"
            if val_npz.exists():
                val_ds = NpzClassificationDataset(val_npz)
                val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
                print(f"[data] loaded: {val_npz} (N={len(val_ds)}, D={val_ds.input_dim}, C={val_ds.num_classes})")
            else:
                print(f"[data] val not found: {val_npz}. skip evaluation.")
             # ===================================================

        else:
            print(f"[data] not found: {train_npz}. fallback to random data.")



    # 더미 학습 루프 (체크포인트/재시작 동작 검증용)
    model = build_dummy_model(input_dim=input_dim, out_dim=num_classes).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    start_epoch = 0
    global_step = 0

    if args.resume:
        start_epoch, global_step = load_checkpoint(run_dir, model, optimizer)

    epoch = start_epoch
    try:
        for epoch in range(start_epoch, args.epochs):
            model.train()
            running_loss = 0.0
            it = iter(train_loader) if train_loader is not None else None

            for _ in range(args.steps_per_epoch):
                if train_loader is None:
                    x = torch.randn(args.batch_size, input_dim, device=device)
                    y = torch.randint(0, num_classes, (args.batch_size,), device=device)
                else:
                    try:
                        x_cpu, y_cpu = next(it)
                    except StopIteration:
                        it = iter(train_loader)
                        x_cpu, y_cpu = next(it)
                    x = x_cpu.to(device, non_blocking=True)
                    y = y_cpu.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                logits = model(x)
                loss = loss_fn(logits, y)
                loss.backward()
                optimizer.step()

                global_step += 1
                running_loss += loss.item()

            avg = running_loss / max(1, args.steps_per_epoch)
            print(f"[train] epoch={epoch} step={global_step} loss={avg:.6f}")
            
            if val_loader is not None:
                model.eval()
                correct = 0
                total = 0
                with torch.no_grad():
                    for vx_cpu, vy_cpu in val_loader:
                        vx = vx_cpu.to(device, non_blocking=True)
                        vy = vy_cpu.to(device, non_blocking=True)
                        logits = model(vx)
                        pred = logits.argmax(dim=1)
                        correct += (pred == vy).sum().item()
                        total += vy.numel()
                acc = correct / max(1, total)
                print(f"[eval] epoch={epoch} val_acc={acc:.4f}")
                model.train()

            # 매 epoch마다 항상 최신 체크포인트 저장
            save_checkpoint(run_dir, epoch=epoch, step=global_step, model=model, optimizer=optimizer)

        print("[done] training finished.")

    except KeyboardInterrupt:
        print("[interrupt] caught Ctrl+C. Saving checkpoint then exiting...")
        save_checkpoint(run_dir, epoch=epoch, step=global_step, model=model, optimizer=optimizer)
        print("[interrupt] checkpoint saved. safe exit.")


if __name__ == "__main__":
    main()
