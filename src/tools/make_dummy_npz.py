from pathlib import Path
import argparse
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output .npz path")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--d", type=int, default=1024)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    X = rng.normal(size=(args.n, args.d)).astype(np.float32)
    y = rng.integers(low=0, high=args.k, size=(args.n,), dtype=np.int64)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, X=X, y=y)
    print(f"wrote: {out_path} (X={X.shape}, y={y.shape}, classes={args.k})")


if __name__ == "__main__":
    main()
