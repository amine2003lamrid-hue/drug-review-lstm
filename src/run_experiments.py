"""Run the experiments described in configs/*.yaml.

    python -m src.run_experiments --config configs/architecture.yaml
    python -m src.run_experiments --config configs/architecture.yaml configs/hparams.yaml --quick

Each experiment = base.yaml defaults + per-run overrides. Runs whose results JSON already exists are
skipped (so a disconnected Colab session can resume); use --force to rerun.
"""
import argparse
import copy
import os
import pickle

import torch
import yaml

from .data import build_data
from .train import train_run

QUICK = {"n_train": 2000, "n_val": 500, "n_test": 500, "epochs": 1}
DATA_KEYS = ("data_dir", "seed", "n_train", "n_val", "n_test", "min_freq", "max_vocab", "max_len")


def load_runs(paths, base_path="configs/base.yaml"):
    with open(base_path) as f:
        base = yaml.safe_load(f)
    runs = []
    for p in paths:
        with open(p) as f:
            spec = yaml.safe_load(f)
        for r in spec["runs"]:
            cfg = copy.deepcopy(base)
            cfg.update(spec.get("defaults", {}))
            cfg.update(r)
            cfg["experiment"] = spec["experiment"]
            runs.append(cfg)
    return runs


def get_data(cfg, cache_dir="data"):
    key = "_".join(str(cfg.get(k)) for k in DATA_KEYS[1:])
    path = os.path.join(cache_dir, f"cache_{key}.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    data = build_data(cfg)
    os.makedirs(cache_dir, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f)
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", nargs="+", required=True)
    ap.add_argument("--quick", action="store_true", help="tiny subset, 1 epoch: pipeline sanity check")
    ap.add_argument("--only", nargs="*", help="run only these run names")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    torch.set_num_threads(max(1, os.cpu_count() or 1))
    out_dir = "results/quick" if args.quick else "results"
    runs = load_runs(args.config)
    seen = set()
    for cfg in runs:
        if cfg["name"] in seen or (args.only and cfg["name"] not in args.only):
            continue
        seen.add(cfg["name"])
        if args.quick:
            cfg.update(QUICK)
        if not args.force and os.path.exists(os.path.join(out_dir, f"{cfg['name']}.json")):
            print(f"[{cfg['name']}] already done, skipping")
            continue
        vocab, ds = get_data(cfg)
        train_run(cfg, vocab, ds, out_dir=out_dir,
                  ckpt_dir="checkpoints/quick" if args.quick else "checkpoints",
                  log_dir="runs/quick" if args.quick else "runs")


if __name__ == "__main__":
    main()
