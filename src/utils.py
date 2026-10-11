import math
import torch
import torch.nn.functional as F


def get_lr(step, d_model=512, warmup=4000):
    step = max(step, 1)
    return (d_model ** -0.5) * min(step ** -0.5, step * warmup ** -1.5)


class LabelSmoothingLoss(torch.nn.Module):
    def __init__(self, smoothing=0.1, pad_idx=0):
        super().__init__()
        self.smoothing = smoothing
        self.pad_idx = pad_idx

    def forward(self, logits, target):
        # logits: (B, T, V), target: (B, T)
        V = logits.size(-1)
        logp = F.log_softmax(logits, dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(logp)
            true_dist.fill_(self.smoothing / (V - 1))
            true_dist.scatter_(-1, target.unsqueeze(-1), 1.0 - self.smoothing)
            mask = (target == self.pad_idx).unsqueeze(-1)
            true_dist.masked_fill_(mask, 0.0)
        loss = -(true_dist * logp).sum(-1)
        mask = (target != self.pad_idx)
        return (loss * mask).sum() / mask.sum().clamp_min(1)


def average_checkpoints(paths, out_path=None):
    import torch
    avg = None
    n = 0
    for p in paths:
        sd = torch.load(p, map_location="cpu")
        sd = sd.get("model", sd)
        if avg is None:
            avg = {k: v.clone().float() for k, v in sd.items()}
        else:
            for k in avg:
                avg[k] += sd[k].float()
        n += 1
    for k in avg:
        avg[k] /= n
    if out_path is not None:
        torch.save({"model": avg}, out_path)
    return avg


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
