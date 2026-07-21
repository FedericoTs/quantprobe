"""Real optimizer-state deltas: train a tiny GPT on enwik8 with AdamW, snapshot the
REAL Adam state (exp_avg=m, exp_avg_sq=v) at several steps, and compress the deltas.

This removes the synthetic-gradient caveat from optim_state.py: gradients here are real
backprop on real text, so m's compressibility is no longer understated.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from weights.optim_state import bf16_delta_save, fp32_delta_save, standalone_save  # noqa: E402

torch.manual_seed(0)
BLOCK, BATCH, D, NLAYER, NHEAD = 128, 32, 192, 2, 6


class Blk(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, h, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):
        t = x.size(1)
        mask = torch.triu(torch.full((t, t), float("-inf")), 1)
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + a
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, vocab=256):
        super().__init__()
        self.tok = nn.Embedding(vocab, D)
        self.pos = nn.Embedding(BLOCK, D)
        self.blocks = nn.ModuleList([Blk(D, NHEAD) for _ in range(NLAYER)])
        self.lnf = nn.LayerNorm(D)
        self.head = nn.Linear(D, vocab)

    def forward(self, idx):
        t = idx.size(1)
        x = self.tok(idx) + self.pos(torch.arange(t))
        for b in self.blocks:
            x = b(x)
        return self.head(self.lnf(x))


def snapshot(model, opt):
    W, M, V = [], [], []
    for p in model.parameters():
        st = opt.state[p]
        W.append(p.detach().float().flatten().numpy())
        M.append(st["exp_avg"].float().flatten().numpy())
        V.append(st["exp_avg_sq"].float().flatten().numpy())
    return np.concatenate(W), np.concatenate(M), np.concatenate(V)


def main():
    data = np.frombuffer(open(os.path.join(_ROOT, "data/corpora/generic-text/enwik8_1mb"),
                              "rb").read(), np.uint8).astype(np.int64)
    model = GPT()
    nparam = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    print(f"tiny GPT: {nparam:,} params; training on enwik8_1mb (real gradients)")

    def get_batch():
        ix = np.random.randint(0, len(data) - BLOCK - 1, BATCH)
        x = torch.from_numpy(np.stack([data[i:i + BLOCK] for i in ix]))
        y = torch.from_numpy(np.stack([data[i + 1:i + BLOCK + 1] for i in ix]))
        return x, y

    snaps = {}
    SNAP_AT = (25, 50, 75, 100, 125, 150)
    for step in range(1, 151):
        x, y = get_batch()
        loss = F.cross_entropy(model(x).reshape(-1, 256), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step in SNAP_AT:
            snaps[step] = snapshot(model, opt)
        if step % 30 == 0:
            print(f"  step {step:3d}  loss {loss.item():.3f}", flush=True)

    steps = sorted(snaps)
    print(f"\nREAL optimizer-state delta compression ({nparam:,} params, bf16)")
    print(f"{'gap':<10}{'W':>9}{'m':>9}{'v':>9}{'optim(m+v)':>13}{'full':>9}")
    print("-" * 52)
    for a, b in ((steps[-2], steps[-1]), (steps[0], steps[-1])):
        (W1, M1, V1), (W0, M0, V0) = snaps[b], snaps[a]
        sW, bW = bf16_delta_save(W1, W0)
        sm, bm = bf16_delta_save(M1, M0)
        sv, bv = bf16_delta_save(V1, V0)
        raw = W1.size * 2
        s_opt = (1 - (bm + bv) / (2 * raw)) * 100
        s_full = (1 - (bW + bm + bv) / (3 * raw)) * 100
        print(f"{f'{b-a}-step':<10}{sW:>8.1f}%{sm:>8.1f}%{sv:>8.1f}%{s_opt:>12.1f}%{s_full:>8.1f}%")

    # --- whole-RUN storage: checkpoint_0 standalone + chained deltas (weights+optimizer) ---
    import numpy as _np  # noqa

    def sa_bytes(arr):
        from weights import wcodec as wc
        return len(wc.cd.SplitSmartCodec("zstd", 19).compress(
            wc._to_bf16(arr.astype(_np.float32)).tobytes(), "bf16"))

    n = len(steps)
    raw_run = sum(W1.size * 2 * 3 for _ in steps)  # K checkpoints x (W+m+v) bf16
    sa_run = 0
    chain_wonly = chain_full = 0
    for i, s in enumerate(steps):
        W, M, V = snaps[s]
        sa_run += sa_bytes(W) + sa_bytes(M) + sa_bytes(V)
        if i == 0:
            chain_full += sa_bytes(W) + sa_bytes(M) + sa_bytes(V)
            chain_wonly += sa_bytes(W)
        else:
            Wp, Mp, Vp = snaps[steps[i - 1]]
            chain_full += bf16_delta_save(W, Wp)[1] + bf16_delta_save(M, Mp)[1] + bf16_delta_save(V, Vp)[1]
            chain_wonly += bf16_delta_save(W, Wp)[1]
    raw_w = sum(W1.size * 2 for _ in steps)
    print(f"\nStoring the {n}-checkpoint run (each = weights + Adam m + v, bf16):")
    print(f"  raw:                 {raw_run/1e6:>7.1f} MB")
    print(f"  each standalone:     {sa_run/1e6:>7.1f} MB  (save {(1-sa_run/raw_run)*100:4.1f}%)")
    print(f"  delta-chain (full):  {chain_full/1e6:>7.1f} MB  (save {(1-chain_full/raw_run)*100:4.1f}%)")
    print(f"  (weights-only run: chain {chain_wonly/1e6:.1f} MB vs raw {raw_w/1e6:.1f} MB "
          f"= {(1-chain_wonly/raw_w)*100:.1f}%)")


if __name__ == "__main__":
    main()
