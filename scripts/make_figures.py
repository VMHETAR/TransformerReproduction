"""Static/data figures (no training needed) + interim training curves.

Generates:
  results/plots/lr_schedule.png      - Vaswani LR schedule, demo window shaded
  results/plots/data_lengths.png      - train subword-length histograms
  results/plots/model_params.png      - parameter count breakdown (Transformer vs GRU)
  results/plots/train_loss_vs_step.png - interim train loss vs step (from current logs)
Official comparison plots (val vs wall-clock, BLEU vs ablations) come from
scripts/plot_results.py and are refreshed after runs finish.
"""
import os
import json
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, ".")


def lr_schedule():
    from src.utils import get_lr
    steps = np.arange(1, 20001)
    lrs = [get_lr(int(s)) for s in steps]
    plt.figure(figsize=(7, 4.5))
    plt.plot(steps, lrs, lw=1.5, label="base schedule (d_model=512, warmup=4000)")
    plt.axvspan(0, 250, color="orange", alpha=0.2, label="demo budget (250 steps)")
    plt.axvline(4000, color="red", ls="--", lw=1, label="warmup end (4000)")
    plt.xscale("log")
    plt.xlabel("Step (log scale)")
    plt.ylabel("Learning rate")
    plt.title("Vaswani LR schedule lr = d^-0.5 * min(step^-0.5, step*warmup^-1.5)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("results/plots/lr_schedule.png", dpi=150)
    plt.close()
    print("lr at step 100/250/4000:", get_lr(100), get_lr(250), get_lr(4000))


def data_lengths():
    import torch
    train_ids = torch.load("data/train_ids.pt", weights_only=False)
    src = np.array([len(s) for s, _ in train_ids])
    tgt = np.array([len(t) for _, t in train_ids])
    fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for a, arr, name in [(ax[0], src, "English (source)"), (ax[1], tgt, "German (target)")]:
        a.hist(arr, bins=50, range=(0, 100), color="steelblue", edgecolor="white", linewidth=0.3)
        a.axvline(100, color="red", ls="--", lw=1.2, label="filter cutoff (100)")
        a.set_xlabel("Subword tokens / sentence")
        a.set_title(f"{name}  (median {np.median(arr):.0f}, kept {len(arr)})")
        a.legend(fontsize=8)
    ax[0].set_ylabel("Pairs")
    fig.suptitle("Training-subset length distribution (BPE-32k, bos/eos included; 5,983 pairs >100 removed)")
    fig.tight_layout()
    fig.savefig("results/plots/data_lengths.png", dpi=150)
    plt.close()


def model_params():
    labels = ["Transformer\nbase", "GRU+Bahdanau\nbaseline"]
    emb = [16.384, 16.384]
    rest = [44.102, 48.286]
    x = np.arange(len(labels))
    plt.figure(figsize=(6, 4.5))
    plt.bar(x, emb, label="Shared embedding (16.38M, tied output)", color="steelblue")
    plt.bar(x, rest, bottom=emb, label="Encoder/decoder/attention", color="darkorange")
    for i, (e, r) in enumerate(zip(emb, rest)):
        plt.text(i, e + r + 0.5, f"{e + r:.2f}M", ha="center", fontsize=10)
    plt.xticks(x, labels)
    plt.ylabel("Parameters (millions)")
    plt.title("Matched model sizes (vocab 32k)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("results/plots/model_params.png", dpi=150)
    plt.close()


def train_loss_so_far():
    plt.figure(figsize=(7, 4.5))
    found = False
    for p in sorted(glob.glob("results/logs/*.json")):
        name = os.path.basename(p)[:-5]
        if name.startswith("smoke_"):
            continue
        with open(p) as f:
            d = json.load(f)
        lg = d["log"]
        xs = [e["step"] for e in lg]
        ys = [e["train_loss"] for e in lg]
        if xs:
            plt.plot(xs, ys, lw=1.2, label=name)
            found = True
    plt.axhline(10.37, color="gray", ls=":", lw=1, label="uniform baseline ln(32k)=10.37")
    plt.xlabel("Step")
    plt.ylabel("Train loss (label smoothing 0.1)")
    plt.title("Training loss vs step (INTERIM - refreshed when runs finish)")
    if found:
        plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("results/plots/train_loss_vs_step.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    os.makedirs("results/plots", exist_ok=True)
    lr_schedule()
    data_lengths()
    model_params()
    train_loss_so_far()
    print("static figures written")
