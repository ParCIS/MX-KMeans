#!/usr/bin/env python3
"""Generate a small Gaussian-cluster CSV with no third-party dependencies."""

import argparse
import csv
from pathlib import Path
import random


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-samples", type=int, default=4096)
    parser.add_argument("--dimensions", type=int, default=128)
    parser.add_argument("--clusters", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("data/example.csv"))
    args = parser.parse_args()
    if not (args.dimensions > 0 and 4 <= args.clusters <= args.n_samples and args.clusters % 4 == 0):
        parser.error("Require D > 0 and 4 <= K <= N with K divisible by 4")
    rng = random.Random(args.seed)
    centers = [[rng.uniform(-5, 5) for _ in range(args.dimensions)] for _ in range(args.clusters)]
    labels = [i % args.clusters for i in range(args.n_samples)]
    rng.shuffle(labels)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([f"d{i}" for i in range(args.dimensions)] + ["label"])
        for label in labels:
            writer.writerow([f"{rng.gauss(x, 0.25):.6f}" for x in centers[label]] + [label])
    print(args.output)


if __name__ == "__main__":
    main()
