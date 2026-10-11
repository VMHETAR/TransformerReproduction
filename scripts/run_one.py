"""Train + evaluate one run. Usage:
python -m scripts.run_one --config configs/demo_transformer.yaml --run-name trans_seed0 --data-dir data --ckpt-dir checkpoints --results-dir results
"""
import os
import sys
import json
import argparse
import yaml
import torch
import sentencepiece as spm
import glob

sys.path.insert(0, ".")
from src.data import TranslationDataset
from src.model_transformer import TransformerSeq2Seq, count_parameters
from src.model_rnn import GRUSeq2Seq
from src.train import train_one_config
from src.decode import beam_search


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--ckpt-dir", default="checkpoints")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--pos-encoding", default=None)
    ap.add_argument("--n-heads", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--test-n", type=int, default=1000, help="first N test sentences to decode")
    ap.add_argument("--skip-train", action="store_true")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.pos_encoding is not None:
        cfg["pos_encoding"] = args.pos_encoding
    if args.n_heads is not None:
        cfg["n_heads"] = args.n_heads
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.max_steps is not None:
        cfg["max_steps"] = args.max_steps
    torch.manual_seed(cfg.get("seed", 0))

    device = torch.device("cpu")
    n_threads = int(os.environ.get("TORCH_THREADS", os.cpu_count() or 4))
    torch.set_num_threads(max(1, n_threads))
    print(f"device={device} threads={torch.get_num_threads()}", flush=True)

    # tokenizer: find model file
    models = sorted(glob.glob(os.path.join(args.data_dir, "spm*.model")))
    assert models, f"no SPM model in {args.data_dir}; run prepare_data first"
    sp = spm.SentencePieceProcessor(model_file=models[0])
    cfg["pad_idx"] = sp.pad_id()
    vocab = sp.vocab_size()
    print(f"tokenizer={models[0]} vocab={vocab}", flush=True)

    train_ids = torch.load(os.path.join(args.data_dir, "train_ids.pt"), weights_only=False)
    val = torch.load(os.path.join(args.data_dir, "validation_ids.pt"), weights_only=False)
    test = torch.load(os.path.join(args.data_dir, "test_ids.pt"), weights_only=False)
    train_set = TranslationDataset(train_ids)
    val_set = TranslationDataset(val["ids"], max_len=512)
    print(f"train pairs kept: {len(train_set)}, val: {len(val_set)}, test: {len(test['ids'])}", flush=True)

    if cfg["model"] == "transformer":
        model = TransformerSeq2Seq(vocab, d_model=cfg["d_model"], n_heads=cfg.get("n_heads", 8),
                                   n_layers=cfg.get("n_layers", 6), d_ff=cfg.get("d_ff", 2048),
                                   dropout=cfg.get("dropout", 0.1), pos_encoding=cfg.get("pos_encoding", "sinusoidal"),
                                   pad_idx=sp.pad_id())
    else:
        model = GRUSeq2Seq(vocab, d_model=cfg.get("d_model", 512),
                           enc_layers=cfg.get("enc_layers", 2), dec_layers=cfg.get("dec_layers", 2),
                           enc_hidden=cfg.get("enc_hidden", 512), dec_hidden=cfg.get("dec_hidden", 512),
                           attn_dim=cfg.get("attn_dim", 512), dropout=cfg.get("dropout", 0.1),
                           pad_idx=sp.pad_id())
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params: {n_params/1e6:.2f}M", flush=True)
    model.to(device)

    log_path = os.path.join(args.results_dir, "logs", args.run_name + ".json")
    os.makedirs(os.path.join(args.results_dir, "logs"), exist_ok=True)
    avg_path = os.path.join(args.ckpt_dir, args.run_name + "_avg5.pt")

    if not args.skip_train:
        avg_path, _ = train_one_config(model, train_set, val_set, cfg, device, log_path,
                                       args.ckpt_dir, args.run_name)
    # load averaged checkpoint for eval
    if os.path.exists(avg_path):
        sd = torch.load(avg_path, map_location="cpu")
        model.load_state_dict(sd.get("model", sd), strict=True)
    model.eval()

    # decode test subset
    import sacrebleu
    N = min(args.test_n, len(test["ids"]))
    print(f"Decoding {N} test sentences (beam=4, alpha=0.6) ...", flush=True)
    hyps, refs = [], []
    for i in range(N):
        s_ids, _ = test["ids"][i]
        # strip bos/eos for encoder input? encode raw subword ids incl bos/eos
        hyp = beam_search(model, s_ids, sp, beam_size=4, alpha=0.6, device="cpu")
        hyps.append(hyp)
        refs.append(test["de"][i])
    bleu = sacrebleu.corpus_bleu(hyps, [refs])
    sig = bleu.format()
    print(f"BLEU: {bleu.score:.2f} {sig}", flush=True)
    os.makedirs(os.path.join(args.results_dir, "bleu"), exist_ok=True)
    with open(os.path.join(args.results_dir, "bleu", args.run_name + ".txt"), "w") as f:
        f.write(f"BLEU: {bleu.score:.4f}\n{sig}\nsacrebleu-version: {sacrebleu.__version__}\nN: {N}\n")
    with open(os.path.join(args.results_dir, "bleu", args.run_name + ".hyps.txt"), "w") as f:
        for h in hyps:
            f.write(h + "\n")
    meta = {"run": args.run_name, "params": n_params, "bleu": bleu.score,
            "signature": sig, "sacrebleu_version": sacrebleu.__version__,
            "n_test": N, "cfg": cfg}
    with open(os.path.join(args.results_dir, "bleu", args.run_name + ".json"), "w") as f:
        json.dump(meta, f, indent=1)


if __name__ == "__main__":
    main()
