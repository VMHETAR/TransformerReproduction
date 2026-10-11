"""Prepare data: sample 500k subset, train SentencePiece, encode splits."""
import os
import json
import argparse
import sys
sys.path.insert(0, ".")
import numpy as np
import sentencepiece as spm
from datasets import load_dataset
from src.data import sample_subset_indices, write_text_for_sp, encode_pairs

VALID_TEXT = "validation"  # wmt/wmt14 de-en: validation=newstest2013 (3000), test=newstest2014 (3003)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--n-sub", type=int, default=500000)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--vocab-size", type=int, default=32000)
    ap.add_argument("--max-len", type=int, default=100)
    ap.add_argument("--sp-train-pairs", type=int, default=500000,
                    help="pairs used for SP training (subset of the 500k)")
    args = ap.parse_args()
    os.makedirs(args.data_dir, exist_ok=True)

    print("Loading WMT14 de-en ...", flush=True)
    ds = load_dataset("wmt/wmt14", "de-en")
    n_total = len(ds["train"])
    print(f"train size: {n_total}, val: {len(ds['validation'])}, test: {len(ds['test'])}")

    idx = sample_subset_indices(n_total, args.n_sub, args.seed)
    np.save(os.path.join(args.data_dir, "subset_indices_seed%d.npy" % args.seed), np.array(idx))
    with open(os.path.join(args.data_dir, "subset_meta.json"), "w") as f:
        json.dump({"n_total": n_total, "n_sub": len(idx), "seed": args.seed,
                   "dataset": "wmt/wmt14:de-en",
                   "validation": "newstest2013 (3000)", "test": "newstest2014 (3003)"}, f, indent=1)
    print(f"Saved {len(idx)} subset indices (seed {args.seed})")

    # Pull subset texts (streaming in chunks to save RAM)
    print("Extracting subset texts ...", flush=True)
    en_sub, de_sub = [], []
    CH = 20000
    train = ds["train"]
    for s in range(0, len(idx), CH):
        chunk = idx[s:s + CH]
        rows = train[chunk]["translation"]
        for r in rows:
            en_sub.append(r["en"])
            de_sub.append(r["de"])
        print(f"  {min(s+CH, len(idx))}/{len(idx)}", flush=True)

    # SP training corpus
    corpus_prefix = os.path.join(args.data_dir, "sp_train")
    with open(corpus_prefix + ".en", "w", encoding="utf-8") as f:
        for t in en_sub[:args.sp_train_pairs]:
            f.write(t.replace("\n", " ") + "\n")
    with open(corpus_prefix + ".de", "w", encoding="utf-8") as f:
        for t in de_sub[:args.sp_train_pairs]:
            f.write(t.replace("\n", " ") + "\n")
    model_prefix = os.path.join(args.data_dir, f"spm32k_seed{args.seed}")
    if not os.path.exists(model_prefix + ".model"):
        print("Training SentencePiece (32k shared BPE) ...", flush=True)
        spm.SentencePieceTrainer.Train(
            f"--input={corpus_prefix}.en,{corpus_prefix}.de --model_prefix={model_prefix} "
            f"--vocab_size={args.vocab_size} --model_type=bpe --character_coverage=1.0 "
            f"--num_threads=16 --split_digits=true --byte_fallback=true "
            f"--unk_id=0 --bos_id=1 --eos_id=2 --pad_id=3 --hard_vocab_limit=false")
    sp = spm.SentencePieceProcessor(model_file=model_prefix + ".model")
    print(f"SP vocab: {sp.vocab_size()}, bos={sp.bos_id()} eos={sp.eos_id()} pad={sp.pad_id()}")

    # Encode train subset
    print("Encoding training subset ...", flush=True)
    train_ids, n_filt = encode_pairs(sp, en_sub, de_sub, args.max_len)
    print(f"kept {len(train_ids)}, filtered(>{args.max_len})={n_filt}")
    # Save as npz of objects (lengths vary) -> use torch save instead
    import torch
    torch.save(train_ids, os.path.join(args.data_dir, "train_ids.pt"))

    # Encode val/test (full sets)
    for split in ["validation", "test"]:
        en = [r["en"] for r in ds[split]["translation"]]
        de = [r["de"] for r in ds[split]["translation"]]
        ids, nf = encode_pairs(sp, en, de, max_len=512)
        torch.save({"ids": ids, "en": en, "de": de},
                   os.path.join(args.data_dir, f"{split}_ids.pt"))
        print(f"{split}: {len(ids)} pairs (filtered>{512}: {nf})")

    # Also save raw val/test text for sacrebleu
    with open(os.path.join(args.data_dir, "val_en.txt"), "w") as f:
        for r in ds["validation"]["translation"]:
            f.write(r["en"] + "\n")
    with open(os.path.join(args.data_dir, "val_de.txt"), "w") as f:
        for r in ds["validation"]["translation"]:
            f.write(r["de"] + "\n")
    with open(os.path.join(args.data_dir, "test_en.txt"), "w") as f:
        for r in ds["test"]["translation"]:
            f.write(r["en"] + "\n")
    with open(os.path.join(args.data_dir, "test_de.txt"), "w") as f:
        for r in ds["test"]["translation"]:
            f.write(r["de"] + "\n")
    print("Done.")


if __name__ == "__main__":
    main()
