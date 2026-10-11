"""Data pipeline: WMT14 En-De subset sampling, SentencePiece training, batching."""
import os
import json
import random
import numpy as np
import torch
import sentencepiece as spm
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader, Sampler

BOS, EOS, PAD, UNK = 1, 2, 0, 3  # set after sp training; sp default: unk=0,bos=1,eos=2,pad=3


def sample_subset_indices(n_total, n_sub=500_000, seed=1234):
    rng = np.random.RandomState(seed)
    return sorted(rng.choice(n_total, size=min(n_sub, n_total), replace=False).tolist())


def load_wmt14():
    ds = load_dataset("wmt/wmt14", "de-en")
    return ds  # keys: train (4.5M), validation (3000=newstest2013), test (3003=newstest2014)


def write_text_for_sp(pairs_en_de, prefix):
    with open(prefix + ".en", "w", encoding="utf-8") as f:
        for en, _ in pairs_en_de:
            f.write(en.replace("\n", " ") + "\n")
    with open(prefix + ".de", "w", encoding="utf-8") as f:
        for _, de in pairs_en_de:
            f.write(de.replace("\n", " ") + "\n")


def train_sentencepiece(input_files, model_prefix, vocab_size=32000, seed=1234):
    # Train shared BPE on train subset only (joint en+de corpus via two input files)
    cmd = (f"--input={','.join(input_files)} --model_prefix={model_prefix} "
           f"--vocab_size={vocab_size} --model_type=bpe --character_coverage=1.0 "
           f"--num_threads=16 --split_digits=true --byte_fallback=true "
           f"--unk_id=0 --bos_id=1 --eos_id=2 --pad_id=3 "
           f"--hard_vocab_limit=false")
    spm.SentencePieceTrainer.Train(cmd)
    return model_prefix + ".model"


class TranslationDataset(Dataset):
    def __init__(self, pairs_ids, max_len=100):
        # pairs_ids: list of (src_ids, tgt_ids) already including bos/eos
        self.pairs = [(s, t) for s, t in pairs_ids
                      if len(s) <= max_len and len(t) <= max_len]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        return self.pairs[i]


class BucketBatchSampler(Sampler):
    """Bucket by max(src_len, tgt_len) for approx token batching."""

    def __init__(self, dataset, tokens_per_batch=25000, shuffle=True, seed=0):
        self.dataset = dataset
        self.tokens_per_batch = tokens_per_batch
        self.shuffle = shuffle
        self.seed = seed
        self.batches = self._make_batches()

    def _make_batches(self):
        lens = [(max(len(s), len(t)), i) for i, (s, t) in enumerate(self.dataset.pairs)]
        lens.sort()
        batches, cur, cur_max = [], [], 0
        for L, i in lens:
            cur.append(i)
            cur_max = max(cur_max, L)
            # approx tokens = len(cur) * cur_max * 2 (src+tgt)
            if len(cur) * cur_max * 2 >= self.tokens_per_batch:
                batches.append(cur)
                cur, cur_max = [], 0
        if cur:
            batches.append(cur)
        return batches

    def __iter__(self):
        rng = random.Random(self.seed)
        order = list(range(len(self.batches)))
        if self.shuffle:
            rng.shuffle(order)
        for b in order:
            yield self.batches[b]

    def __len__(self):
        return len(self.batches)


def collate_fn(batch, pad_idx=3):
    srcs = [torch.tensor(s, dtype=torch.long) for s, _ in batch]
    tgts = [torch.tensor(t, dtype=torch.long) for _, t in batch]
    src = torch.nn.utils.rnn.pad_sequence(srcs, batch_first=True, padding_value=pad_idx)
    tgt = torch.nn.utils.rnn.pad_sequence(tgts, batch_first=True, padding_value=pad_idx)
    # decoder input = bos + tgt[:-1]; label = tgt[1:] with tgt including bos..eos
    # Our stored tgt includes [bos, ..., eos]; input = tgt[:, :-1], label = tgt[:, 1:]
    return src, tgt[:, :-1].contiguous(), tgt[:, 1:].contiguous()


def encode_pairs(sp, en_texts, de_texts, max_len=100):
    """Encode en->de pairs with bos/eos. Returns (kept_ids, n_filtered)."""
    ids = []
    n_filt = 0
    for en, de in zip(en_texts, de_texts):
        s = [sp.bos_id()] + sp.encode(en, out_type=int) + [sp.eos_id()]
        t = [sp.bos_id()] + sp.encode(de, out_type=int) + [sp.eos_id()]
        if len(s) > max_len or len(t) > max_len:
            n_filt += 1
            continue
        ids.append((s, t))
    return ids, n_filt
