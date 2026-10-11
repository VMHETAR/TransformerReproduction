#!/bin/bash
# Ablation launcher: 5 unique configs x 2 seeds = 10 runs, max 100 steps each.
# Usage: bash scripts/run_ablations.sh [test_n]
set -e
TEST_N=${1:-50}
echo "test_n=$TEST_N"
# posenc ablations (heads=8)
TORCH_THREADS=6 /tmp/opencode/venv/bin/python scripts/run_one.py --config configs/demo_transformer.yaml --run-name abl_posenc_sinusoidal_seed0 --pos-encoding sinusoidal --n-heads 8 --seed 0 --max-steps 100 --test-n $TEST_N
TORCH_THREADS=6 /tmp/opencode/venv/bin/python scripts/run_one.py --config configs/demo_transformer.yaml --run-name abl_posenc_sinusoidal_seed1 --pos-encoding sinusoidal --n-heads 8 --seed 1 --max-steps 100 --test-n $TEST_N
TORCH_THREADS=6 /tmp/opencode/venv/bin/python scripts/run_one.py --config configs/demo_transformer.yaml --run-name abl_posenc_none_seed0 --pos-encoding none --n-heads 8 --seed 0 --max-steps 100 --test-n $TEST_N
TORCH_THREADS=6 /tmp/opencode/venv/bin/python scripts/run_one.py --config configs/demo_transformer.yaml --run-name abl_posenc_none_seed1 --pos-encoding none --n-heads 8 --seed 1 --max-steps 100 --test-n $TEST_N
TORCH_THREADS=6 /tmp/opencode/venv/bin/python scripts/run_one.py --config configs/demo_transformer.yaml --run-name abl_posenc_learned_seed0 --pos-encoding learned --n-heads 8 --seed 0 --max-steps 100 --test-n $TEST_N
TORCH_THREADS=6 /tmp/opencode/venv/bin/python scripts/run_one.py --config configs/demo_transformer.yaml --run-name abl_posenc_learned_seed1 --pos-encoding learned --n-heads 8 --seed 1 --max-steps 100 --test-n $TEST_N
# heads ablations (sinusoidal; heads=8 reuses abl_posenc_sinusoidal_* runs to save compute)
TORCH_THREADS=6 /tmp/opencode/venv/bin/python scripts/run_one.py --config configs/demo_transformer.yaml --run-name abl_heads_1_seed0 --pos-encoding sinusoidal --n-heads 1 --seed 0 --max-steps 100 --test-n $TEST_N
TORCH_THREADS=6 /tmp/opencode/venv/bin/python scripts/run_one.py --config configs/demo_transformer.yaml --run-name abl_heads_1_seed1 --pos-encoding sinusoidal --n-heads 1 --seed 1 --max-steps 100 --test-n $TEST_N
TORCH_THREADS=6 /tmp/opencode/venv/bin/python scripts/run_one.py --config configs/demo_transformer.yaml --run-name abl_heads_4_seed0 --pos-encoding sinusoidal --n-heads 4 --seed 0 --max-steps 100 --test-n $TEST_N
TORCH_THREADS=6 /tmp/opencode/venv/bin/python scripts/run_one.py --config configs/demo_transformer.yaml --run-name abl_heads_4_seed1 --pos-encoding sinusoidal --n-heads 4 --seed 1 --max-steps 100 --test-n $TEST_N
echo "ablations done (heads=8 reuses abl_posenc_sinusoidal_seed{0,1})"
