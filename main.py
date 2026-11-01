"""Fit the SQMM tail model to a legacy INR target.

Examples (from the repo root):
    python sub_tail/main.py --dataset cello --runs 5
    python sub_tail/main.py --dataset camera --growth 20 --epochs 600
    python sub_tail/main.py --dataset coffee --divergence kl
    python sub_tail/main.py --dataset astronaut --config my_config.json

"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

try:
    from .datasets import DATASETS
    from .train import run
except ImportError:
    from datasets import DATASETS
    from train import run


def parse_args():
    parser = argparse.ArgumentParser(description="Fit the SQMM tail model to cello / camera / coffee / astronaut.")
    parser.add_argument("--dataset", choices=DATASETS, default="cello")
    parser.add_argument("--runs", type=int, default=5, help="Independent runs, each from a fresh random initialisation.")
    parser.add_argument("--epochs", type=int, default=None, help="Default: 300 (600 for astronaut).")
    parser.add_argument("--growth", type=int, default=None, help="Updates between band introductions (default: G*).")
    parser.add_argument("--divergence", choices=("squared", "kl"), default=None, help="L_align divergence (default squared).")
    parser.add_argument("--align-weight", type=float, default=None, help="lambda of L_align (default 3).")
    parser.add_argument("--config", type=str, default=None, help="JSON with model_config / train_config overrides.")
    parser.add_argument("--output-dir", type=str, default=None, help="Default: sub_tail/out/<dataset>.")
    parser.add_argument("--save-weights", action="store_true")
    parser.add_argument("--threads", type=int, default=int(os.environ.get("SQINR_THREADS", "2")),
                        help="Torch CPU threads (the paper's runs used 2).")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    torch.set_num_threads(args.threads)
    payload = json.loads(Path(args.config).read_text()) if args.config else {}
    train_config = dict(payload.get("train_config", {}), dataset=args.dataset)
    for key, value in (("epochs", args.epochs), ("growth", args.growth),
                       ("align_divergence", args.divergence), ("align_weight", args.align_weight)):
        if value is not None:
            train_config[key] = value
    output_dir = args.output_dir or Path(__file__).resolve().parent / "out" / args.dataset
    results = run(runs=args.runs, output_dir=output_dir,
                  model_config=payload.get("model_config"), train_config=train_config,
                  save_weights=args.save_weights, verbose=not args.quiet)
    for name, stats in results["summary"].items():
        print(f"{name}: {stats['runs']} run(s) | mean mse_best={stats['mean_mse_best']:.6e} "
              f"(min {stats['min_mse_best']:.6e}) | mean psnr_best={stats['mean_psnr_best']:.2f}")


if __name__ == "__main__":
    main()
