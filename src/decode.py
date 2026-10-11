"""Beam search decoding with length penalty (beam=4, alpha=0.6 as in the paper)."""
import torch
import torch.nn.functional as F


@torch.no_grad()
def beam_search(model, src_ids, sp, beam_size=4, alpha=0.6, max_len=128, device="cpu"):
    model.eval()
    # CPU safeguard: cap generation length (documented deviation); newstest avg tgt ~30 subwords.
    max_len = min(max_len, len(src_ids) + 30, 64)
    is_transformer = hasattr(model, "encode") and hasattr(model, "greedy_or_beam_step") \
        and model.__class__.__name__ == "TransformerSeq2Seq"
    src = torch.tensor([src_ids], dtype=torch.long, device=device)
    src_pad = (src == sp.pad_id())

    if is_transformer:
        memory = model.encode(src, src_pad)
        # beams: list of (tokens, logprob)
        beams = [([sp.bos_id()], 0.0)]
        for _ in range(max_len):
            candidates = []
            # batch all open beams
            open_idx = [i for i, (toks, _) in enumerate(beams) if toks[-1] != sp.eos_id()]
            if not open_idx:
                break
            batch = torch.tensor([beams[i][0] for i in open_idx], dtype=torch.long, device=device)
            mem = memory.expand(len(open_idx), -1, -1)
            spm_ = torch.zeros(len(open_idx), src.size(1), dtype=torch.bool, device=device)
            logits = model.greedy_or_beam_step(batch, mem, spm_ if False else src_pad.expand(len(open_idx), -1))
            logp = F.log_softmax(logits[:, -1, :], dim=-1)
            for j, i in enumerate(open_idx):
                toks, score = beams[i]
                topk_p, topk_i = logp[j].topk(beam_size)
                for k in range(beam_size):
                    L = len(toks) + 1
                    lp_norm = ((5.0 + 1) ** alpha) / ((5.0 + L) ** alpha)
                    # store unnormalized; normalize at selection
                    candidates.append((toks + [int(topk_i[k])], score + float(topk_p[k])))
            # keep all finished + top open by length-normed score
            finished = [(t, s) for t, s in beams if t[-1] == sp.eos_id]
            open_c = [(t, s) for t, s in candidates if t[-1] != sp.eos_id]
            open_c.sort(key=lambda x: x[1] / (((5.0 + len(x[0])) ** alpha)), reverse=True)
            beams = finished + open_c
            beams.sort(key=lambda x: x[1] / (((5.0 + len(x[0])) ** alpha)), reverse=True)
            beams = beams[:beam_size]
        best = beams[0][0]
    else:
        # GRU incremental decode
        enc_out, h = model.encode(src)
        beams = [([sp.bos_id()], 0.0, h, torch.zeros(1, model.enc_dim, device=device))]
        for _ in range(max_len):
            open_idx = [i for i in range(len(beams)) if beams[i][0][-1] != sp.eos_id()]
            if not open_idx:
                break
            candidates = []
            for i in open_idx:
                toks, score, hi, ctx = beams[i]
                tok = torch.tensor([[toks[-1]]], device=device)
                emb = model.embedding(tok)
                inp = torch.cat([emb.squeeze(1), ctx], dim=-1).unsqueeze(1)
                out, hi2 = model.decoder(inp, hi)
                dec_state = out.squeeze(1)
                ctx2, _ = model.attention(dec_state, enc_out, src_pad)
                logits = F.linear(model.out_norm(dec_state), model.embedding.weight)
                logp = F.log_softmax(logits, dim=-1)
                topk_p, topk_i = logp[0].topk(beam_size)
                for k in range(beam_size):
                    candidates.append((toks + [int(topk_i[k])], score + float(topk_p[k]), hi2, ctx2))
            finished = [b for b in beams if b[0][-1] == sp.eos_id()]
            candidates.sort(key=lambda x: x[1] / (((5.0 + len(x[0])) ** alpha)), reverse=True)
            beams = (finished + candidates)[:beam_size]
        best = beams[0][0]
    # strip bos/eos
    seq = [t for t in best if t not in (sp.bos_id(), sp.eos_id(), sp.pad_id())]
    return sp.decode(seq)
