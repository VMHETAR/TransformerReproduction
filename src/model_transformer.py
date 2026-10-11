import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def naive_scaled_dot_product_attention(q, k, v, mask=None, dropout_p=0.0):
    """Naive reference SDPA. q,k,v: (..., T_q, d_k). mask: bool, True = keep, False = mask out."""
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    if dropout_p > 0.0:
        attn = F.dropout(attn, p=dropout_p, training=True)
    return torch.matmul(attn, v), attn


def verify_sdpa_match(n_trials=20, tol=1e-5, seed=0):
    """Verify F.scaled_dot_product_attention matches naive impl to within tol."""
    g = torch.Generator().manual_seed(seed)
    max_diff = 0.0
    for _ in range(n_trials):
        B, H, Tq, Tk, D = 2, 4, 8, 8, 16
        q = torch.randn(B, H, Tq, D, generator=g)
        k = torch.randn(B, H, Tk, D, generator=g)
        v = torch.randn(B, H, Tk, D, generator=g)
        # causal-ish random mask
        mask = torch.ones(B, 1, Tq, Tk, dtype=torch.bool)
        # random padding mask on keys
        key_pad = (torch.rand(B, 1, 1, Tk, generator=g) > 0.2)
        mask = mask & key_pad
        # causal
        causal = torch.tril(torch.ones(Tq, Tk, dtype=torch.bool)).view(1, 1, Tq, Tk)
        mask = mask & causal
        with torch.no_grad():
            # F.sdpa expects attn_mask bool where True = keep (same convention) when using bool mask
            out_fast = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False)
            out_naive, _ = naive_scaled_dot_product_attention(q, k, v, mask=mask)
            diff = (out_fast - out_naive).abs().max().item()
            max_diff = max(max_diff, diff)
            if diff > tol:
                return False, diff
    return True, max_diff


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1, use_fast_sdpa=True):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.use_fast_sdpa = use_fast_sdpa

    def forward(self, q, k, v, mask=None):
        # q,k,v: (B, T, d_model); mask: (B, 1, Tq, Tk) bool, True=keep
        B = q.size(0)
        Q = self.w_q(q).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(k).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(v).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)
        if self.use_fast_sdpa:
            out = F.scaled_dot_product_attention(
                Q, K, V, attn_mask=mask, dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=False)
        else:
            out, _ = naive_scaled_dot_product_attention(
                Q, K, V, mask=mask,
                dropout_p=self.dropout.p if self.training else 0.0)
        out = out.transpose(1, 2).contiguous().view(B, -1, self.d_model)
        return self.w_o(out)


class PositionwiseFFN(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w2(self.dropout(F.relu(self.w1(x))))


class EncoderLayer(nn.Module):
    """Post-LN encoder layer as in Vaswani et al. (2017)."""

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, use_fast_sdpa=True):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout, use_fast_sdpa)
        self.ffn = PositionwiseFFN(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask=None):
        # Post-LN: x + dropout(sublayer(x)) then LayerNorm
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, src_mask)))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x


class DecoderLayer(nn.Module):
    """Post-LN decoder layer."""

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, use_fast_sdpa=True):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout, use_fast_sdpa)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout, use_fast_sdpa)
        self.ffn = PositionwiseFFN(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory, tgt_mask=None, memory_mask=None):
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, tgt_mask)))
        x = self.norm2(x + self.dropout(self.cross_attn(x, memory, memory, memory_mask)))
        x = self.norm3(x + self.dropout(self.ffn(x)))
        return x


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


class TransformerSeq2Seq(nn.Module):
    """Base Transformer (Vaswani et al. 2017) with Post-LN.

    - shared source/target embeddings (single embedding table)
    - weight tying between target embedding and output projection
    - sinusoidal / learned / none positional encoding (ablation switch)
    """

    def __init__(self, vocab_size, d_model=512, n_heads=8, n_layers=6, d_ff=2048,
                 dropout=0.1, max_len=512, pos_encoding="sinusoidal",
                 use_fast_sdpa=True, pad_idx=0):
        super().__init__()
        self.d_model = d_model
        self.pad_idx = pad_idx
        self.pos_encoding = pos_encoding
        # Shared source/target embedding
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        if pos_encoding == "sinusoidal":
            self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len, dropout)
            self.pos_emb = None
        elif pos_encoding == "learned":
            self.pos_enc = None
            self.pos_emb = nn.Embedding(max_len, d_model)
            self.pos_drop = nn.Dropout(dropout)
        elif pos_encoding == "none":
            self.pos_enc = None
            self.pos_emb = None
            self.pos_drop = nn.Dropout(dropout)
        else:
            raise ValueError(pos_encoding)
        self.max_len = max_len
        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(d_model, n_heads, d_ff, dropout, use_fast_sdpa) for _ in range(n_layers)])
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(d_model, n_heads, d_ff, dropout, use_fast_sdpa) for _ in range(n_layers)])
        # Tied output projection: logits = hidden @ embedding.weight.T * scale
        self.n_heads = n_heads
        self._init_params()

    def _init_params(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def add_pos(self, x, start=0):
        if self.pos_encoding == "sinusoidal":
            # Sinusoidal module adds from position 0; for start>0 slice accordingly
            if start == 0:
                return self.pos_enc(x)
            else:
                pe = self.pos_enc.pe[:, start:start + x.size(1)]
                return self.pos_enc.dropout(x + pe)
        elif self.pos_encoding == "learned":
            B, T, _ = x.shape
            pos = torch.arange(start, start + T, device=x.device).unsqueeze(0).expand(B, T)
            return self.pos_drop(x + self.pos_emb(pos))
        else:
            return self.pos_drop(x)

    def encode(self, src, src_pad_mask=None):
        # src: (B, S) ids
        x = self.embedding(src) * math.sqrt(self.d_model)
        x = self.add_pos(x, start=0)
        B, S, _ = x.shape
        # src self-attn mask: (B, 1, S, S), True=keep
        if src_pad_mask is not None:
            # src_pad_mask: (B, S) True=pad
            keep = (~src_pad_mask).unsqueeze(1).unsqueeze(2)  # (B,1,1,S)
            src_mask = keep.expand(B, 1, S, S)
        else:
            src_mask = None
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        return x

    def decode_step(self, tgt, memory, src_pad_mask=None):
        # full-sequence teacher-forced decode (used in training)
        x = self.embedding(tgt) * math.sqrt(self.d_model)
        x = self.add_pos(x, start=0)
        B, T, _ = x.shape
        S = memory.size(1)
        causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=tgt.device)).view(1, 1, T, T)
        if tgt.device.type == "cpu":
            pass
        tgt_pad = (tgt != self.pad_idx).unsqueeze(1).unsqueeze(2)  # (B,1,1,T) True=keep
        tgt_mask = causal & tgt_pad.expand(B, 1, T, T)
        if src_pad_mask is not None:
            mem_mask = (~src_pad_mask).unsqueeze(1).unsqueeze(2).expand(B, 1, T, S)
        else:
            mem_mask = None
        for layer in self.decoder_layers:
            x = layer(x, memory, tgt_mask, mem_mask)
        # tied projection
        logits = F.linear(x, self.embedding.weight)
        return logits

    def forward(self, src, tgt_input):
        src_pad = (src == self.pad_idx)
        memory = self.encode(src, src_pad)
        logits = self.decode_step(tgt_input, memory, src_pad)
        return logits

    def greedy_or_beam_step(self, tgt_ids, memory, src_pad_mask):
        """Single-step logits for incremental decoding (tgt_ids: (B, t))."""
        x = self.embedding(tgt_ids) * math.sqrt(self.d_model)
        x = self.add_pos(x, start=0)
        B, T, _ = x.shape
        S = memory.size(1)
        causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=tgt_ids.device)).view(1, 1, T, T)
        tgt_pad = (tgt_ids != self.pad_idx).unsqueeze(1).unsqueeze(2)
        tgt_mask = causal & tgt_pad.expand(B, 1, T, T)
        if src_pad_mask is not None:
            mem_mask = (~src_pad_mask).unsqueeze(1).unsqueeze(2).expand(B, 1, T, S)
        else:
            mem_mask = None
        for layer in self.decoder_layers:
            x = layer(x, memory, tgt_mask, mem_mask)
        return F.linear(x, self.embedding.weight)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
