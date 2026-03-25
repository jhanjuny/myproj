import argparse
import json
import math
import random
from pathlib import Path
from typing import List, Sequence, Tuple


Sample = Tuple[float, float, int]


def make_blob_dataset(num_samples: int, noise: float, seed: int) -> List[Sample]:
    rng = random.Random(seed)
    half = num_samples // 2
    samples: List[Sample] = []

    for _ in range(half):
        x1 = rng.gauss(-2.0, noise)
        x2 = rng.gauss(-2.0, noise)
        samples.append((x1, x2, 0))

    for _ in range(num_samples - half):
        x1 = rng.gauss(2.0, noise)
        x2 = rng.gauss(2.0, noise)
        samples.append((x1, x2, 1))

    rng.shuffle(samples)
    return samples


def sigmoid(value: float) -> float:
    if value >= 0:
        exp_value = math.exp(-value)
        return 1.0 / (1.0 + exp_value)
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def predict_prob(weights: Sequence[float], bias: float, x1: float, x2: float) -> float:
    logit = (weights[0] * x1) + (weights[1] * x2) + bias
    return sigmoid(logit)


def binary_loss(prob: float, label: int) -> float:
    clipped = min(max(prob, 1e-8), 1.0 - 1e-8)
    return -((label * math.log(clipped)) + ((1 - label) * math.log(1.0 - clipped)))


def evaluate(weights: Sequence[float], bias: float, samples: Sequence[Sample]) -> Tuple[float, float]:
    total_loss = 0.0
    correct = 0

    for x1, x2, label in samples:
        prob = predict_prob(weights, bias, x1, x2)
        pred = 1 if prob >= 0.5 else 0
        total_loss += binary_loss(prob, label)
        correct += int(pred == label)

    count = max(1, len(samples))
    return total_loss / count, correct / count


def save_artifacts(save_dir: Path, weights: Sequence[float], bias: float, summary: dict) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "weights": [round(weights[0], 6), round(weights[1], 6)],
        "bias": round(bias, 6),
        "summary": summary,
    }
    (save_dir / "model.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--train-size", type=int, default=512)
    parser.add_argument("--val-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--noise", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--save-dir", type=str, default="")
    args = parser.parse_args()

    train_samples = make_blob_dataset(args.train_size, args.noise, args.seed)
    val_samples = make_blob_dataset(args.val_size, args.noise, args.seed + 1)

    rng = random.Random(args.seed)
    weights = [rng.uniform(-0.5, 0.5), rng.uniform(-0.5, 0.5)]
    bias = rng.uniform(-0.1, 0.1)

    for epoch in range(1, args.epochs + 1):
        rng.shuffle(train_samples)
        total_loss = 0.0
        correct = 0

        for x1, x2, label in train_samples:
            prob = predict_prob(weights, bias, x1, x2)
            pred = 1 if prob >= 0.5 else 0
            error = prob - label

            weights[0] -= args.lr * error * x1
            weights[1] -= args.lr * error * x2
            bias -= args.lr * error

            total_loss += binary_loss(prob, label)
            correct += int(pred == label)

        train_loss = total_loss / max(1, len(train_samples))
        train_acc = correct / max(1, len(train_samples))
        val_loss, val_acc = evaluate(weights, bias, val_samples)

        if epoch == 1 or epoch == args.epochs or epoch % max(1, args.epochs // 5) == 0:
            print(
                f"epoch={epoch:03d} "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}"
            )

    print("\nSample predictions:")
    for index, (x1, x2, label) in enumerate(val_samples[:5]):
        prob = predict_prob(weights, bias, x1, x2)
        pred = 1 if prob >= 0.5 else 0
        print(
            f"sample={index} x=({x1:.3f}, {x2:.3f}) "
            f"true={label} pred={pred} p(class1)={prob:.3f}"
        )

    summary = {
        "epochs": args.epochs,
        "train_size": args.train_size,
        "val_size": args.val_size,
        "lr": args.lr,
        "noise": args.noise,
        "seed": args.seed,
        "final_val_loss": round(val_loss, 6),
        "final_val_acc": round(val_acc, 6),
    }

    if args.save_dir:
        save_artifacts(Path(args.save_dir), weights, bias, summary)
        print(f"\nSaved artifacts to: {Path(args.save_dir).resolve()}")


if __name__ == "__main__":
    main()
