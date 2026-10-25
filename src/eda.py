"""Exploratory data analysis: figures/ PNGs + results/eda_summary.md + results/eda_stats.json."""
import json
import os

import numpy as np
import pandas as pd

from . import plotstyle as ps
from .data import _read, clean, download, tokenize

K = ps.matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1000:.0f}k" if v else "0")

plt = ps.plt
FIG, RES = "figures", "results"


def main(data_dir="data", max_len=200):
    os.makedirs(FIG, exist_ok=True)
    os.makedirs(RES, exist_ok=True)
    download(data_dir)
    train, test = _read(data_dir, "Train"), _read(data_dir, "Test")
    full = pd.concat([train.assign(split="train"), test.assign(split="test")])
    full = full.dropna(subset=["review", "rating"])

    # 1. rating distribution
    fig, ax = plt.subplots(figsize=(6, 3.2))
    counts = full.rating.astype(int).value_counts().sort_index()
    colors = [ps.NEG if r <= 4 else (ps.POS if r >= 7 else ps.MUTED) for r in counts.index]
    ax.bar(counts.index, counts.values, color=colors, width=0.8)
    ax.set_xticks(range(1, 11))
    ax.yaxis.set_major_formatter(K)
    ax.set_xlabel("Rating")
    ax.set_ylabel("Reviews")
    ax.set_title("Rating distribution (train + test)")
    ax.text(0.02, 0.95, "orange: negative (1-4)   gray: dropped (5-6)   blue: positive (7-10)",
            transform=ax.transAxes, fontsize=8, color=ps.MUTED, va="top")
    fig.savefig(f"{FIG}/rating_distribution.png")
    plt.close(fig)

    # 2. class balance after binarization
    kept = full[(full.rating >= 7) | (full.rating <= 4)].copy()
    kept["label"] = (kept.rating >= 7).astype(int)
    bal = kept.groupby(["split", "label"]).size().unstack()
    fig, ax = plt.subplots(figsize=(6, 2.6))
    y = np.arange(len(bal))
    ax.barh(y, bal[0], color=ps.NEG, label="negative (0)", height=0.6)
    ax.barh(y, bal[1], left=bal[0] + bal.values.sum() * 0.002, color=ps.POS, label="positive (1)", height=0.6)
    for i, s in enumerate(bal.index):
        tot = bal.loc[s].sum()
        ax.text(bal.loc[s, 0] / 2, i, f"{bal.loc[s, 0] / tot:.1%}", ha="center", va="center", color="white", fontsize=7.5)
        ax.text(bal.loc[s, 0] + bal.loc[s, 1] / 2, i, f"{bal.loc[s, 1] / tot:.1%}", ha="center", va="center",
                color="white", fontsize=7.5)
    ax.xaxis.set_major_formatter(K)
    ax.set_yticks(y, bal.index)
    ax.set_xlabel("Reviews")
    ax.set_title("Class balance after binarization")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=2)
    fig.savefig(f"{FIG}/class_balance.png")
    plt.close(fig)

    # 3. review length (tokens)
    lengths = kept[kept.split == "train"].review.map(lambda t: len(tokenize(t)))
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.hist(lengths.clip(upper=300), bins=np.arange(0, 305, 5), color=ps.POS, edgecolor="white", linewidth=0.5)
    ax.yaxis.set_major_formatter(K)
    ax.axvline(max_len, color=ps.TEXT, linewidth=1.2, linestyle="--")
    cov = (lengths <= max_len).mean()
    ax.text(max_len + 8, ax.get_ylim()[1] * 0.9, f"max_len = {max_len}\n{cov:.1%} of reviews fit", fontsize=9)
    ax.set_xlabel("Review length in tokens (clipped at 300)")
    ax.set_ylabel("Reviews")
    ax.set_title("Review length distribution (train)")
    fig.savefig(f"{FIG}/review_length.png")
    plt.close(fig)

    # 4/5. top conditions and drugs
    for col, fname, title in (("condition", "top_conditions.png", "Top 15 conditions"),
                              ("drugName", "top_drugs.png", "Top 15 drugs")):
        top = kept[col].value_counts().head(15)[::-1]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.barh(top.index, top.values, color=ps.POS, height=0.7)
        ax.xaxis.set_major_formatter(K)
        ax.set_xlabel("Reviews")
        ax.set_title(title)
        ax.grid(axis="y", visible=False)
        fig.savefig(f"{FIG}/{fname}")
        plt.close(fig)

    stats = {
        "n_train_raw": int(len(train)), "n_test_raw": int(len(test)),
        "n_train_kept": int((kept.split == "train").sum()), "n_test_kept": int((kept.split == "test").sum()),
        "dropped_neutral": int(len(full) - len(kept)),
        "pos_ratio_train": float(kept[kept.split == "train"].label.mean()),
        "pos_ratio_test": float(kept[kept.split == "test"].label.mean()),
        "len_mean": float(lengths.mean()), "len_median": float(lengths.median()),
        "len_p90": float(lengths.quantile(0.9)), "len_p95": float(lengths.quantile(0.95)),
        "coverage_max_len": float(cov), "max_len": max_len,
        "n_conditions": int(kept.condition.nunique()), "n_drugs": int(kept.drugName.nunique()),
    }
    with open(f"{RES}/eda_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    rng = np.random.default_rng(42)
    lines = ["# EDA summary", "", "| statistic | value |", "|---|---|"]
    lines += [f"| {k} | {v:.3f} |" if isinstance(v, float) else f"| {k} | {v} |" for k, v in stats.items()]
    short = kept[kept.split == "train"].assign(n=lengths)
    short = short[(short.n >= 8) & (short.n <= 35)]
    for lab, name in ((1, "Positive"), (0, "Negative")):
        lines += ["", f"## {name} examples", ""]
        sample = short[short.label == lab].sample(3, random_state=int(rng.integers(1e6)))
        for _, r in sample.iterrows():
            lines.append(f"- ({r.drugName}, {r.condition}, rating {int(r.rating)}) {clean(r.review)}")
    with open(f"{RES}/eda_summary.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
