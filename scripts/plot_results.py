"""Plots: val loss vs wall-clock (Transformer vs GRU), BLEU vs ablations."""
import os
import json
import glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_log(path):
    with open(path) as f:
        d = json.load(f)
    return d["log"]


def main(results="results", out="results/plots"):
    os.makedirs(out, exist_ok=True)
    logs = {os.path.basename(p)[:-5]: load_log(p) for p in glob.glob(f"{results}/logs/*.json")}
    # 1. val loss vs wall-clock for main comparison runs
    plt.figure(figsize=(7, 5))
    for name in sorted(logs):
        if not (name.startswith("trans_base") or name.startswith("gru_base")):
            continue
        lg = logs[name]
        xs = [e["elapsed_s"] / 60 for e in lg if "val_loss" in e]
        ys = [e["val_loss"] for e in lg if "val_loss" in e]
        if xs:
            plt.plot(xs, ys, marker="o", label=name)
    plt.xlabel("Wall-clock time (min)")
    plt.ylabel("Validation loss")
    plt.title("Validation loss vs wall-clock time")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{out}/val_loss_vs_time.png", dpi=150)
    plt.close()

    # 2. BLEU vs ablations
    import json as J
    bleus = {}
    for p in glob.glob(f"{results}/bleu/*.json"):
        with open(p) as f:
            d = J.load(f)
        bleus[d["run"]] = d["bleu"]
    # posenc group
    pos_vals = {}
    for run, b in bleus.items():
        if run.startswith("abl_posenc_"):
            key = run[len("abl_posenc_"):].rsplit("_seed", 1)[0]
            pos_vals.setdefault(key, []).append(b)
    # heads group: 1/4 from abl_heads_*, 8 reuses abl_posenc_sinusoidal_*
    head_vals = {}
    for run, b in bleus.items():
        if run.startswith("abl_heads_"):
            key = run[len("abl_heads_"):].rsplit("_seed", 1)[0]
            head_vals.setdefault(f"heads={key}", []).append(b)
        if run.startswith("abl_posenc_sinusoidal_seed"):
            head_vals.setdefault("heads=8", []).append(b)
    for group, vals in [("posenc", pos_vals), ("heads", head_vals)]:
        if not vals:
            continue
        keys = sorted(vals)
        means = [sum(vals[k]) / len(vals[k]) for k in keys]
        stds = [(sum((x - m) ** 2 for x in vals[k]) / len(vals[k])) ** 0.5 if len(vals[k]) > 1 else 0.0
                for k, m in zip(keys, means)]
        plt.figure(figsize=(6, 4))
        plt.errorbar(keys, means, yerr=stds, fmt="o-", capsize=4)
        for k, m, s in zip(keys, means, stds):
            plt.text(k, m, f" {m:.2f}+/-{s:.2f}", va="bottom", ha="center", fontsize=8)
        plt.ylabel("Test BLEU (subset; see README)")
        plt.title(f"BLEU vs {group} (mean +/- std, 2 seeds)")
        plt.tight_layout()
        plt.savefig(f"{out}/bleu_vs_{group}.png", dpi=150)
        plt.close()
    print("plots written to", out)


if __name__ == "__main__":
    main()
