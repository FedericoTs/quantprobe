"""ROUND 19 -- QAT (straight-through-estimator fine-tuning): the one lever flagged to break
the training-free PTQ floor. Rounds 13-18 showed rotated scalar ECVQ+entropy is at the PTQ
frontier (~4.48 ppl @ 3.13b). QAT changes regime: ADAPT the weights to the quantizer.

Clean test: plain per-group 3-bit RTN (NO rotation) is gibberish as PTQ (~70 ppl). With STE
fine-tuning on calibration text, does it reach/beat PTQ-ECVQ's 4.483 at the SAME ~3.13 bits?
A yes proves the floor is training-free, not fundamental.

Memory-careful for 6GB: frozen tied embeddings (136M params), SGD (no optimizer state),
gradient checkpointing, seq 256 / batch 1, fp32.

Run:  python -m weights.quant_qat
"""
from __future__ import annotations

import os
import time

import torch
import torch.nn as nn
import torch.nn.utils.parametrize as P
from transformers import AutoTokenizer

from weights.quant_dataaware import DEV, gpu_model, load_fp16_gpu, ppl_gpu
from weights.quant_fast import HAD, SIGNS
from weights.quant_lab import CFG, WPATH, quant_keys

G = 128


def nf_levels(bits, device=DEV):
    """Normal-float levels: quantiles of N(0,1) at (i+0.5)/2^b, normalized to [-1,1].
    A non-uniform grid matched to the ~Gaussian rotated weights (better than uniform)."""
    n = 1 << bits
    probs = (torch.arange(n, dtype=torch.float32) + 0.5) / n
    lv = torch.distributions.Normal(0.0, 1.0).icdf(probs)
    lv = lv / lv.abs().max()
    return lv.to(device)


class FakeQuant(nn.Module):
    """Per-(row, group) symmetric quantizer with a straight-through estimator.
    rot=True  -> per-group Hadamard incoherence rotation inside the quantizer (undone after).
    nf=True   -> non-uniform normal-float grid instead of uniform; same bit count."""
    def __init__(self, bits=3, g=G, rot=False, nf=False):
        super().__init__()
        self.qmax = (1 << (bits - 1)) - 1
        self.g = g
        self.rot = rot
        self.nf = nf
        self.lv = nf_levels(bits) if nf else None

    def forward(self, W):
        out, inn = W.shape
        g = self.g
        ng = (inn + g - 1) // g
        pad = ng * g - inn
        A = torch.cat([W, W.new_zeros(out, pad)], 1) if pad else W
        Ar = A.reshape(out, ng, g)
        if self.rot:
            Ar = (Ar * SIGNS) @ HAD                      # incoherence rotation (per group)
        amax = Ar.abs().amax(dim=2, keepdim=True).clamp_min(1e-8)
        if self.nf:
            lv, K = self.lv, self.lv.numel()
            x = Ar / amax
            ci = torch.bucketize(x, lv).clamp(1, K - 1)
            left = (x - lv[ci - 1]).abs() <= (x - lv[ci]).abs()
            ci = torch.where(left, ci - 1, ci)
            q = lv[ci] * amax
        else:
            scale = amax / self.qmax
            q = torch.clamp(torch.round(Ar / scale), -self.qmax, self.qmax) * scale
        q = Ar + (q - Ar).detach()                       # STE
        if self.rot:
            q = (q @ HAD.T) * SIGNS                       # undo rotation
        return q.reshape(out, ng * g)[:, :inn]


def bits_per_weight(bits=3, g=G):
    return bits + 16.0 / g                               # index bits + fp16 scale per group


def attach_quant(model, bits, g=G, rot=False, nf=False):
    targets = []
    with __import__("safetensors").safe_open(WPATH, framework="pt") as f:
        qk = quant_keys(f)
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and (name + ".weight") in qk:
            P.register_parametrization(mod, "weight", FakeQuant(bits, g, rot, nf))
            targets.append(name)
    return targets


def calib_batches(tok, seq=256, n=2000):
    # Large calib slice from enwik8 AFTER the 256k file (eval lives at bytes 120000:128000
    # of the 256k slice) -> disjoint. ~1MB -> ~250k tokens -> ~1000 windows (no overfit).
    full = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "data/corpora/generic-text/enwik8")
    raw = open(full, "rb").read()[256000:1256000]
    ids = tok(raw.decode("latin-1"), return_tensors="pt").input_ids[0]
    out = []
    for s in range(0, len(ids) - seq, seq):
        out.append(ids[s:s + seq])
        if len(out) >= n:
            break
    return out


def run(bits=3, steps=500, lr=5e-4, rot=False, nf=False):
    tok = AutoTokenizer.from_pretrained(CFG)
    model = gpu_model()
    load_fp16_gpu(model)
    fp16 = ppl_gpu(model, tok)

    attach_quant(model, bits, rot=rot, nf=nf)
    bpw = bits_per_weight(bits)
    ppl0 = ppl_gpu(model, tok)                            # PTQ (pre-training) at this quant
    tagq = f"{bits}-bit {'ROT' if rot else 'RTN'}{'+NF' if nf else ''}"
    print(f"fp16 {fp16:.3f} | {tagq} PTQ (pre-QAT) {ppl0:.3f} @ {bpw:.3f}b   "
          f"[PTQ-ECVQ ref 4.483@3.13b]\n", flush=True)

    # freeze tied embeddings/lm_head (not quantized); train only quantized linears
    for n_, p_ in model.named_parameters():
        if "embed_tokens" in n_ or "lm_head" in n_:
            p_.requires_grad_(False)
    # use_reentrant=False so gradients flow even though the (frozen-embedding) inputs
    # to the checkpointed layers don't require grad -- reentrant checkpointing would
    # silently zero the gradients of the quantized weights.
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(trainable, lr=lr, momentum=0.9)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)

    batches = calib_batches(tok)
    print(f"  calib windows: {len(batches)}  lr {lr} mom 0.9 cosine  steps {steps}\n", flush=True)
    model.train()
    t0 = time.time()
    best = ppl0
    for i in range(steps):
        ids = batches[i % len(batches)].unsqueeze(0).to(DEV)
        loss = model(ids, labels=ids).loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        sched.step()
        if i % 25 == 0 or i == steps - 1:
            model.eval()
            pe = ppl_gpu(model, tok)
            model.train()
            best = min(best, pe)
            tag = "  <== BEATS PTQ-ECVQ 4.483" if pe < 4.483 else ""
            print(f"  step {i:4d}  loss {float(loss):.3f}  held-out ppl {pe:.3f}  best {best:.3f}  "
                  f"({time.time()-t0:.0f}s){tag}", flush=True)
    model.eval()
    pf = ppl_gpu(model, tok)
    print(f"\n{tagq}: PTQ {ppl0:.3f} -> QAT final {pf:.3f} (best {best:.3f})  @ {bpw:.3f}b   "
          f"(PTQ-ECVQ frontier 4.483@3.13b, fp16 {fp16:.3f})", flush=True)
    return ppl0, pf, bpw


if __name__ == "__main__":
    import sys
    b = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    rot = "rot" in sys.argv
    nf = "nf" in sys.argv
    run(bits=b, rot=rot, nf=nf)
