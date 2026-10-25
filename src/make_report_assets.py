"""Build tables (results/*.md) and figures (figures/*.png) from results/*.json.

    python -m src.make_report_assets                 # full results
    python -m src.make_report_assets --results results/quick --figures figures/quick
"""
import argparse
import glob
import json
import os

import numpy as np

from . import plotstyle as ps

plt = ps.plt
ARCH = [("vanilla", "Vanilla LSTM"), ("cifg", "CIFG-LSTM"), ("lstmp_p128", "LSTMP (P=128)")]
HPARAM = [
    ("vanilla", "Baseline (Adam, lr 1e-3, bs 128, Xavier)", "baseline"),
    ("vanilla_lr5e-3", "Learning rate 5e-3", "learning rate"),
    ("vanilla_sgd", "SGD (momentum 0.9, lr 0.05)", "optimizer"),
    ("vanilla_bs64", "Batch size 64", "batch size"),
    ("vanilla_bs256", "Batch size 256", "batch size"),
    ("vanilla_orthogonal", "Orthogonal recurrent init", "initialization"),
]
BONUS = [("lstmp_p128", "LSTMP (P=128)"), ("lstmp_p64", "LSTMP (P=64)")]


def load(results_dir):
    out = {}
    for p in glob.glob(os.path.join(results_dir, "*.json")):
        with open(p) as f:
            r = json.load(f)
        if "test" in r:
            out[r["name"]] = r
    return out


def row(r, label):
    t = r["test"]
    return (f"| {label} | {r['trainable_params']:,} | {r['recurrent_params']:,} | {t['accuracy']:.4f} | "
            f"{t['macro_f1']:.4f} | {t['precision']:.4f} | {t['recall']:.4f} | {r['best_epoch']} | "
            f"{r['mean_epoch_time_s']:.1f} |")


HEADER = ("| Model | Trainable params | Recurrent params | Test acc. | Test macro-F1 | Precision | Recall | "
          "Best epoch | Time/epoch (s) |\n|---|---|---|---|---|---|---|---|---|")


def tables(res, results_dir):
    device = next(iter(res.values()))["device"] if res else "?"
    note = f"\n\nAll runs: same splits, seed 42, early stopping on val macro-F1. Device: {device}.\n"
    arch = [row(res[k], lab) for k, lab in ARCH if k in res]
    with open(os.path.join(results_dir, "architecture_table.md"), "w") as f:
        f.write(HEADER + "\n" + "\n".join(arch) + note)
    hp = [row(res[k], lab) for k, lab, _ in HPARAM if k in res]
    with open(os.path.join(results_dir, "hparam_table.md"), "w") as f:
        f.write(HEADER.replace("| Model |", "| Vanilla LSTM variant |") + "\n" + "\n".join(hp) + note)
    bonus = [row(res[k], lab) for k, lab in BONUS if k in res]
    if len(bonus) == 2:
        with open(os.path.join(results_dir, "projection_table.md"), "w") as f:
            f.write(HEADER + "\n" + "\n".join(bonus) + note)


def curves(res, fig_dir):
    present = [(k, lab) for k, lab in ARCH if k in res]
    if not present:
        return
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    for k, lab in present:
        h = res[k]["history"]
        ep = [e["epoch"] for e in h]
        color = ps.MODEL_COLORS[res[k]["config"]["cell"]]
        axes[0].plot(ep, [e["val_loss"] for e in h], color=color, marker="o", markersize=4, label=lab)
        axes[1].plot(ep, [e["val_macro_f1"] for e in h], color=color, marker="o", markersize=4, label=lab)
    axes[0].set_title("Validation loss")
    axes[1].set_title("Validation macro-F1")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.xaxis.set_major_locator(ps.matplotlib.ticker.MaxNLocator(integer=True))
    axes[1].legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "val_curves.png"))
    plt.close(fig)


def confusion(res, fig_dir):
    best = max(res.values(), key=lambda r: r["best_val_macro_f1"])  # chosen on validation, not test
    cm = np.array(best["test"]["confusion_matrix"])
    norm = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.grid(False)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}\n({norm[i, j]:.1%})", ha="center", va="center",
                    color="white" if norm[i, j] > 0.5 else ps.TEXT, fontsize=9)
    ax.set_xticks([0, 1], ["negative", "positive"])
    ax.set_yticks([0, 1], ["negative", "positive"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion matrix: {best['name']} (test)")
    fig.savefig(os.path.join(fig_dir, "confusion_matrix.png"))
    plt.close(fig)
    return best["name"]


def params_vs_f1(res, fig_dir):
    keys = [k for k, _ in ARCH + BONUS if k in res]
    keys = list(dict.fromkeys(keys))
    labels = dict(ARCH + BONUS)
    if not keys:
        return
    fig, axes = plt.subplots(1, 2, figsize=(9, 3), gridspec_kw={"width_ratios": [1, 1]})
    y = np.arange(len(keys))[::-1]
    colors = [ps.MODEL_COLORS[res[k]["config"]["cell"]] for k in keys]
    params = [res[k]["recurrent_params"] / 1000 for k in keys]
    f1s = [res[k]["test"]["macro_f1"] for k in keys]
    axes[0].barh(y, params, color=colors, height=0.6)
    axes[0].set_title("Recurrent-layer parameters (k)")
    for yi, v in zip(y, params):
        axes[0].text(v, yi, f" {v:.0f}k", va="center", fontsize=8, color=ps.MUTED)
    axes[1].barh(y, f1s, color=colors, height=0.6)
    lo = max(0.0, min(f1s) - 0.05)
    axes[1].set_xlim(lo, min(1.0, max(f1s) + 0.03))
    axes[1].set_title("Test macro-F1 (axis starts at %.2f)" % lo)
    for yi, v in zip(y, f1s):
        axes[1].text(v, yi, f" {v:.3f}", va="center", fontsize=8, color=ps.MUTED)
    for ax in axes:
        ax.set_yticks(y, [labels[k] for k in keys])
        ax.grid(axis="y", visible=False)
    axes[1].set_yticklabels([])
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "params_vs_f1.png"))
    plt.close(fig)


def hparam_chart(res, fig_dir):
    present = [(k, lab) for k, lab, _ in HPARAM if k in res]
    if not present:
        return
    fig, ax = plt.subplots(figsize=(7, 3))
    y = np.arange(len(present))[::-1]
    f1s = [res[k]["test"]["macro_f1"] for k, _ in present]
    colors = [ps.SERIES[0] if k == "vanilla" else ps.MUTED for k, _ in present]
    ax.barh(y, f1s, color=colors, height=0.6)
    if "vanilla" in res:
        ax.axvline(res["vanilla"]["test"]["macro_f1"], color=ps.SERIES[0], linewidth=1, linestyle="--")
    lo = max(0.0, min(f1s) - 0.05)
    ax.set_xlim(lo, min(1.0, max(f1s) + 0.04))
    for yi, v in zip(y, f1s):
        ax.text(v, yi, f" {v:.3f}", va="center", fontsize=8, color=ps.MUTED)
    ax.set_yticks(y, [lab for _, lab in present])
    ax.grid(axis="y", visible=False)
    ax.set_title("Hyperparameter study (vanilla LSTM): test macro-F1, axis starts at %.2f" % lo)
    fig.savefig(os.path.join(fig_dir, "hparam_comparison.png"))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--figures", default="figures")
    args = ap.parse_args()
    os.makedirs(args.figures, exist_ok=True)
    res = load(args.results)
    if not res:
        print(f"No results in {args.results}; run src.run_experiments first.")
        return
    tables(res, args.results)
    curves(res, args.figures)
    best = confusion(res, args.figures)
    params_vs_f1(res, args.figures)
    hparam_chart(res, args.figures)
    with open(os.path.join(args.results, "summary.json"), "w") as f:
        json.dump({"best_model_by_val_f1": best, "runs": sorted(res)}, f, indent=2)
    print(f"Assets written for {len(res)} runs; best model (val macro-F1): {best}")


if __name__ == "__main__":
    main()
