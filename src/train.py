"""Training loop: Adam schedule, label smoothing, accumulation, AMP, logging, ckpt averaging."""
import os
import time
import math
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data import BucketBatchSampler, collate_fn
from src.utils import get_lr, LabelSmoothingLoss, average_checkpoints, count_params


def train_one_config(model, train_set, val_set, cfg, device, log_path, ckpt_dir, run_name):
    os.makedirs(ckpt_dir, exist_ok=True)
    d_model = cfg["d_model"]
    warmup = cfg.get("warmup", 4000)
    max_steps = cfg["max_steps"]
    tokens_per_batch = cfg["tokens_per_batch"]
    accum = cfg.get("accum_steps", 1)
    log_every = cfg.get("log_every", 10)
    save_every = cfg.get("save_every", 200)
    val_every = cfg.get("val_every", 200)
    pad_idx = cfg["pad_idx"]
    use_amp = cfg.get("amp", False)

    opt = torch.optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
    crit = LabelSmoothingLoss(smoothing=cfg.get("label_smoothing", 0.1), pad_idx=pad_idx)
    # bf16 autocast on CPU if requested (mixed precision per spec; no GradScaler needed for bf16)
    use_bf16 = bool(use_amp)

    sampler = BucketBatchSampler(train_set, tokens_per_batch, shuffle=True, seed=cfg.get("seed", 0))

    def loader():
        while True:
            for batch_idx in sampler:
                batch = [train_set[i] for i in batch_idx]
                yield collate_fn(batch, pad_idx)

    gen = loader()
    model.train()
    step = 0
    total_tokens = 0
    t0 = time.time()
    log = []
    peak_mem = 0.0
    # rough peak CPU mem via torch allocator not available; use rss
    import psutil
    proc = psutil.Process()
    rss0 = proc.memory_info().rss / 1e9

    time_to_target = None
    target_loss = cfg.get("target_val_loss", None)
    ckpt_paths = []

    while step < max_steps:
        step += 1
        lr = get_lr(step, d_model, warmup)
        for pg in opt.param_groups:
            pg["lr"] = lr
        opt.zero_grad()
        accum_loss = 0.0
        step_tokens = 0
        step_t0 = time.time()
        for a in range(accum):
            src, din, lab = next(gen)
            src, din, lab = src.to(device), din.to(device), lab.to(device)
            step_tokens += int((lab != pad_idx).sum())
            if use_bf16:
                with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                    logits = model(src, din)
                    loss = crit(logits, lab) / accum
            else:
                logits = model(src, din)
                loss = crit(logits, lab) / accum
            loss.backward()
            accum_loss += loss.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        dt = time.time() - step_t0
        total_tokens += step_tokens
        toks_per_sec = step_tokens / max(dt, 1e-6)
        rss = proc.memory_info().rss / 1e9
        peak_mem = max(peak_mem, rss)
        elapsed = time.time() - t0
        entry = {"step": step, "lr": lr, "train_loss": accum_loss,
                 "tokens": step_tokens, "tokens_per_sec": toks_per_sec,
                 "elapsed_s": elapsed, "rss_gb": rss}
        if step % log_every == 0 or step == 1:
            print(f"[{run_name}] step {step}/{max_steps} loss={accum_loss:.3f} "
                  f"lr={lr:.2e} tok/s={toks_per_sec:.0f} elapsed={elapsed/60:.1f}min", flush=True)
        if step % val_every == 0 or step == max_steps:
            vl = evaluate_loss(model, val_set, crit, pad_idx, device, max_batches=cfg.get("val_batches", 20))
            entry["val_loss"] = vl
            print(f"[{run_name}] step {step} VAL loss={vl:.3f}", flush=True)
            if target_loss is not None and time_to_target is None and vl <= target_loss:
                time_to_target = elapsed
        log.append(entry)
        if step % save_every == 0 or step == max_steps:
            p = os.path.join(ckpt_dir, f"{run_name}_step{step}.pt")
            torch.save({"model": {k: v.cpu() for k, v in model.state_dict().items()},
                        "step": step, "cfg": cfg}, p)
            ckpt_paths.append(p)
            with open(log_path, "w") as f:
                json.dump({"log": log, "time_to_target_loss": time_to_target,
                           "peak_rss_gb": peak_mem, "rss0_gb": rss0,
                           "total_tokens": total_tokens,
                           "total_time_s": elapsed}, f, indent=1)
    # average last 5 checkpoints
    avg_src = ckpt_paths[-5:]
    avg_path = os.path.join(ckpt_dir, f"{run_name}_avg5.pt")
    average_checkpoints(avg_src, avg_path)
    with open(log_path, "w") as f:
        json.dump({"log": log, "time_to_target_loss": time_to_target,
                   "peak_rss_gb": peak_mem, "rss0_gb": rss0,
                   "total_tokens": total_tokens, "total_time_s": time.time() - t0,
                   "checkpoints": ckpt_paths, "avg_checkpoint": avg_path}, f, indent=1)
    return avg_path, log


@torch.no_grad()
def evaluate_loss(model, val_set, crit, pad_idx, device, max_batches=20):
    model.eval()
    sampler = BucketBatchSampler(val_set, tokens_per_batch=4000, shuffle=False, seed=0)
    tot, n = 0.0, 0
    for bi, batch_idx in enumerate(sampler):
        if bi >= max_batches:
            break
        batch = [val_set[i] for i in batch_idx]
        src, din, lab = collate_fn(batch, pad_idx)
        src, din, lab = src.to(device), din.to(device), lab.to(device)
        logits = model(src, din)
        tot += crit(logits, lab).item()
        n += 1
    model.train()
    return tot / max(n, 1)
