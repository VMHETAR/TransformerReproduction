import sys
sys.path.insert(0, ".")
import torch
from src.model_transformer import verify_sdpa_match

if __name__ == "__main__":
    torch.manual_seed(0)
    ok, max_diff = verify_sdpa_match(n_trials=50, tol=1e-5)
    print(f"SDPA match: ok={ok} max_abs_diff={max_diff:.3e} tol=1e-5")
    assert ok, f"F.scaled_dot_product_attention differs from naive by {max_diff}"
