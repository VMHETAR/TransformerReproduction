"""GRU encoder-decoder with Bahdanau (additive) attention baseline.

Sized to ~60-65M params to match the base Transformer. Uses the same shared
embedding table + tied output projection as the Transformer for a fair comparison.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class BahdanauAttention(nn.Module):
    def __init__(self, enc_dim, dec_dim, attn_dim):
        super().__init__()
        self.W_enc = nn.Linear(enc_dim, attn_dim, bias=False)
        self.W_dec = nn.Linear(dec_dim, attn_dim, bias=False)
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, dec_state, enc_out, src_pad_mask=None):
        # dec_state: (B, dec_dim); enc_out: (B, S, enc_dim)
        # returns context (B, enc_dim), weights (B, S)
        B, S, _ = enc_out.shape
        e = self.v(torch.tanh(self.W_enc(enc_out) + self.W_dec(dec_state).unsqueeze(1))).squeeze(-1)
        if src_pad_mask is not None:
            e = e.masked_fill(src_pad_mask, float("-inf"))
        w = torch.softmax(e, dim=-1)
        ctx = torch.bmm(w.unsqueeze(1), enc_out).squeeze(1)
        return ctx, w


class GRUSeq2Seq(nn.Module):
    def __init__(self, vocab_size, d_model=512, enc_layers=2, dec_layers=2,
                 enc_hidden=512, dec_hidden=512, attn_dim=512, dropout=0.1, pad_idx=0):
        super().__init__()
        self.pad_idx = pad_idx
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.enc_layers = enc_layers
        self.dec_layers = dec_layers
        self.enc_hidden = enc_hidden
        self.dec_hidden = dec_hidden
        # Bidirectional encoder: enc_hidden per direction -> enc_dim = 2*enc_hidden
        self.encoder = nn.GRU(d_model, enc_hidden, num_layers=enc_layers, batch_first=True,
                              bidirectional=True, dropout=dropout if enc_layers > 1 else 0.0)
        self.enc_dim = 2 * enc_hidden
        # Bridge encoder final -> decoder init
        self.bridge = nn.Linear(self.enc_dim, dec_hidden)
        # Decoder input = embed + context
        self.decoder = nn.GRU(d_model + self.enc_dim, dec_hidden, num_layers=dec_layers,
                              batch_first=True, dropout=dropout if dec_layers > 1 else 0.0)
        self.attention = BahdanauAttention(self.enc_dim, dec_hidden, attn_dim)
        self.out_norm = nn.LayerNorm(dec_hidden)
        self.dropout = nn.Dropout(dropout)
        # Tied output projection (same as Transformer)
        self._init_params()

    def _init_params(self):
        for n, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src):
        emb = self.dropout(self.embedding(src))
        enc_out, h_n = self.encoder(emb)  # h_n: (2*enc_layers, B, enc_hidden)
        # Take last layer fwd+bwd
        fwd = h_n[-2]
        bwd = h_n[-1]
        h_cat = torch.cat([fwd, bwd], dim=-1)  # (B, enc_dim)
        dec_init = torch.tanh(self.bridge(h_cat))  # (B, dec_hidden)
        h = dec_init.unsqueeze(0).repeat(self.dec_layers, 1, 1)
        return enc_out, h

    def forward(self, src, tgt_input):
        # teacher forcing over full target
        src_pad = (src == self.pad_idx)
        enc_out, h = self.encode(src)
        emb = self.dropout(self.embedding(tgt_input))
        B, T, _ = emb.shape
        outputs = []
        ctx = torch.zeros(B, self.enc_dim, device=src.device)
        for t in range(T):
            inp = torch.cat([emb[:, t], ctx], dim=-1).unsqueeze(1)
            out, h = self.decoder(inp, h)
            dec_state = out.squeeze(1)  # top layer state (B, dec_hidden)
            ctx, _ = self.attention(dec_state, enc_out, src_pad)
            feat = self.out_norm(dec_state + self.attention.W_dec.bias if False else dec_state)
            outputs.append(feat)
        feats = torch.stack(outputs, dim=1)  # (B, T, dec_hidden)
        # project dec_hidden -> d_model? No: tie requires dec_hidden == d_model.
        # We enforce dec_hidden == d_model so embedding.weight.T works directly.
        assert feats.size(-1) == self.embedding.weight.size(1), \
            f"dec_hidden {feats.size(-1)} must equal d_model {self.embedding.weight.size(1)} for tying"
        logits = F.linear(feats, self.embedding.weight)
        return logits


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def gru_baseline_configs(vocab_size=32000, target_params=62_000_000):
    """Candidate sizing. Default below is ~64.7M with vocab 32k (verified).

    enc_hidden=768 x4-layer bidirectional encoder + 4-layer 512-dim decoder
    + 512-dim Bahdanau attention + shared 512-dim embeddings with tying.
    """
    return dict(vocab_size=vocab_size, d_model=512, enc_layers=4, dec_layers=4,
                enc_hidden=768, dec_hidden=512, attn_dim=512, dropout=0.1)
