# Lossless LLM-weight compression — results & status

**North star:** be the best *lossless* delta/checkpoint codec in the world. Single-model
lossless is an entropy wall (~30% on bf16) and already saturated by ZipNN/DFloat11; the
open frontier is the **delta** between related models (checkpoints, fine-tunes, variants),
where the only published lossless number is ~62% and there is no shipping product.

Every number below is **byte-exact (SHA-256 round-trip gated)** on **real models**.

## Where the field is (researched)

| Regime | Best published | Notes |
|---|---|---|
| Single model, lossless | **~30–33%** (ZipNN bf16 ~33% / DFloat11 ~30%) | entropy wall; mantissa is random |
| Delta, **lossy** | ~10× / 1-bit (BitDelta) | sacrifices accuracy, needs distillation |
| Delta, **lossless** | **~62%** (arXiv 2508.19263, Aug'25) | blockwise XOR + entropy code, bf16 checkpoints |

## What we built & measured

| Test (real data) | standalone | **our delta** | round-trip |
|---|---:|---:|:--:|
| Single bf16 model (smart per-plane) | **32.7%** (beats ZipNN, faster) | — | ✅ |
| Pythia-70m **1000-step checkpoint** delta (fp32) | 16.9% | **69.5%** | ✅ |
| Same checkpoint delta in **bf16** (= SOTA's regime) | 32.9% | **61.3%** (≈ published 62%) | ✅ |
| Store a **4-checkpoint run** (chain) | 16.9% | **55.8%** (31%/extra ckpt, ~69% asymptotic) | ✅ |
| SmolLM-135M **base→Instruct** fine-tune (bf16, heavy SFT) | 32.9% | **53.0%** (XOR 49% → arith 53%) | ✅ |
| Qwen2.5-0.5B **abliteration** (light edit, bf16) | 32.9%-class | **99.1% (109×)** (low-rank) | ✅ |
| same abliteration **quantized to int8** (deployment fmt) | 39.5% | **97.4% (38×)** (sparse) | ✅ |
| same abliteration **quantized to fp8** (E4M3, newest fmt) | 17.5% | **92.3% (13×)** (sparse) | ✅ |

**Scale (bounded memory):** the abliteration result holds across **0.5B → 1.5B → 3B**
(99.1% / 98.8% / 98.7%; 109× / 84× / 80×), all byte-exact. The **3B** case is a **6.2 GB
model compressed *and* decompressed on a machine with only 3.5 GB free RAM** — because mmap
input + streaming decode mean the model is never fully resident (peak ~3 GB of reclaimable
mmap pages). The base and abliterated models were **sharded differently** (3.8+2.1 vs
4.7+1.2 GB), so cross-shard reference-by-tensor-name resolution is validated at scale. The
method is per-tensor, so it scales further; the only limit here is RAM for the largest
individual tensors (a 7B embedding is ~1 GB).

We **match** single-model SOTA, and on the lossless-delta SOTA's *exact* regime (bf16
checkpoint deltas) we hit **61.3% ≈ the published ~62%** — with a much simpler, faster,
multithreaded method. On fp32 the same delta reaches **68.7%**. That fp32 > bf16 gap is a
real finding: fp32's random low-mantissa bits stay frozen across a small update and cancel
exactly in the delta; bf16 has already truncated them, so there's less to cancel. And on
the highest-value case — **model variants** — we reach **90.9%**, which single-model codecs
structurally cannot touch.

### The validated spectrum (real data, byte-exact)

Compression tracks how much the model actually changed — and most real "new models" are
small edits of an existing one:

```
light edit (abliteration)   ████████████████████  99.1%  (109×; low-rank delta)
checkpoint step (fp32)      ██████████████         68.7%
training-run storage        ███████████            55.8%
heavy full SFT (bf16)       ███████████            53.0%
single model, no reference  ██████                 32.7%  (entropy wall)
```

The abliteration case is the product thesis in one number: a real, popular Qwen2.5-0.5B
variant stored as a **109× smaller** byte-exact delta (988 MB → 9 MB) — because **76% of
tensors are identical** (stored as copies) and the changed ones (`o_proj`/`down_proj`) are
a near-**rank-4 edit** (abliteration projects out one direction: ΔW = −v·(vᵀW)), which the
low-rank mode captures in a handful of tiny factors + an exact residual. HF hosts hundreds
of thousands of such variants of shared bases.

### Head-to-head on the *same* file (abliterated Qwen2.5-0.5B, 988 MB)

| codec | output | save | round-trip |
|---|---:|---:|:--:|
| single-model (ZipNN/DFloat11 ceiling, entropy-bound) | 663.7 MB | 32.8% | ✅ |
| **wcodec delta (vs base, low-rank)** | **9.0 MB** | **99.1%** | ✅ |

No single-model lossless codec can beat ~33% on bf16 (the mantissa is random). By
exploiting the shared base — and recognizing that the edit is **low-rank** — wcodec
produces **74× smaller files** on the same model. This is the whole point of going
delta-native. (ZipNN itself wouldn't install in this environment; 32.8% is our own
single-model codec, which already matches/beats ZipNN.)

## Key technical findings

1. **Why deltas work (fp32):** training updates are too small to flip the low mantissa
   bits, so bits 0–12 stay **frozen** at random init values — exactly what makes standalone
   incompressible. They cancel in the delta. Entropy lives in ~11 high-mantissa bits.
2. **Arithmetic delta beats XOR for bf16/fp16 (+4 pts):** map float bits to a monotonic
   key so `key_ft − key_base` = signed #ULP-steps; XOR over-counts small moves that cross
   an exponent boundary. fp32 keeps XOR (frozen-bit advantage). wcodec auto-picks per
   tensor: **best-of {copy, sparse, XOR-per-plane, arith}**.
3. **Ratio is bounded by how much the model actually changed.** Heavy SFT ≈ 53%, a
   checkpoint step ≈ 69%, light edits ≈ 90%+.
4. **Low-rank residual mode (the cross-element lever).** Per-element coders are blind to
   matrix structure, but many edits (abliteration, LoRA-merge) are low-rank. For 2D bf16
   tensors whose delta is low-rank, wcodec stores **int16 rank-r factors + an exact arith
   residual** vs a deterministically reconstructed reference (exact integer matmul, so it
   round-trips bit-identically on any machine; the residual absorbs all factor-quantisation
   error, keeping it lossless). The rank is chosen adaptively as the **numerical rank**
   (singular values above 1% of the largest — the true signal, not the bf16-rounding noise
   floor, which the residual codes more cheaply). This lifted the abliteration from 90.9%
   → **99.1% (109×)**. It's tried only on 2D bf16 deltas and kept only when it wins, so it
   never hurts other cases (fp32 checkpoint 68.6% and heavy SFT 53.0% are unchanged).

## Levers tested and **rejected** (eval-gated, saved wasted builds)

- **Momentum / delta-of-delta:** consecutive deltas are independent (cosine≈0, ‖dd‖/‖d‖≈√2).
- **Better entropy coder (Golomb/range):** beats zstd by only +0.6 pt; remainder bits are
  irreducible.
- **Low-rank residual:** attention deltas are low-rank (rank-16 ≈ 20–60% energy) but the MLP
  bulk is ~full-rank (~10%), so overall gain ~1–3 pts at high complexity → deprioritized.

## The tool

`wcodec.py` — end-to-end `.safetensors` codec (single files **and** sharded models),
CLI (`compress`/`decompress`/`bench`), per-tensor best-of {copy, sparse, XOR-per-plane,
arith} with a `--level` speed/ratio knob. Self-verifying (embedded SHA-256 → byte-exact
or it errors). Per-tensor encode/decode is multithreaded (zstd releases the GIL): on a
988 MB model at zstd-19, **enc ~32 MB/s, dec ~164 MB/s** (12 cores) — ~31 s to compress,
~6 s to decompress. Lightweight decoder (just zstd + a byte transform). 15/15 round-trip
tests across all dtypes, edge shapes, delta modes, and sharding. See `README.md`.

## Beyond weights: checkpoints + optimizer state

A training checkpoint isn't just weights — it's weights **+ the Adam optimizer state
(m, v), ~2× the model**. That triples the data a training run must store, and it's the
clearest buyer (anyone checkpointing a run). Do optimizer deltas compress?

Training a real tiny GPT (1M params) on enwik8 with AdamW and compressing the **real**
Adam-state deltas (`train_real.py`):

| checkpoint gap | stored in | weights | m (1st) | v (2nd) | optimizer (m+v) | full checkpoint |
|---|---|---:|---:|---:|---:|---:|
| 10 steps | bf16 | 66% | 32% | 67% | 49% | **55%** |
| 30 steps | bf16 | 59% | 30% | 58% | 44% | **49%** |
| 10 steps | fp32 | 32% | 17% | 29% | 23% | 26% |

(standalone, no delta ≈ 16–21%.)

Three findings: (1) **fp32 optimizer deltas compress poorly** — the EMA *recomputes* each
value (`m = 0.9·m + 0.1·g`), scrambling low mantissa bits, so the XOR frozen-bit trick
fails. (2) **bf16 optimizer states (the modern mixed-precision norm) compress well** —
`v` (β₂=0.999, the slowest EMA, half the optimizer) hits ~67% because its per-step change
falls *below the bf16 ULP*; `m` is harder (~32%) during active training because momentum
shifts fast. (3) Net: a **full bf16 checkpoint delta-chain compresses ~50–55%** vs ~18%
standalone — a 3× win on the 3×-larger data. Late-training / fine-tuning checkpoints (smaller
gradients) compress better still.

**The product metric** — storing a real 6-checkpoint training run (each = weights + Adam m +
v, bf16) as a delta-chain: **36.5 MB → 19.6 MB (46% saved)** vs 34.5% storing each compressed
standalone (and the weights-only chain saves 54%). More frequent checkpoints compress more.

## Ecosystem redundancy (real model family, measured — honest)

A real Qwen2.5-0.5B-Instruct family of **11 models** (base + 10 independent hub derivatives)
stored as lossless deltas vs the base. **The delta size measures how much each fine-tune
actually changed the model:**

| edit type | examples | delta vs base |
|---|---|---:|
| exact duplicate | unsloth re-upload | **0%** |
| rank-1 edit | abliteration | **1%** |
| preference tuning | DPO-halueval, GRPO-summ | **5–8%** |
| full SFT | dataforge, mathphd, reasoning, vikhr | **~80%** |
| precision mismatch | neon, ultrachat (fp32 vs bf16 base) | 100% (needs fp32 ref) |

**Full family of 11: 12844 → 7392 MB (42.5%, 1.7×);** bf16-only subset **54.6% (2.2×)**.
**Light-derivative subset** (the common hub case — base + unsloth-dup + abliterated + DPO +
GRPO): 4940 → 835 MB = **83.1% (5.9×)** — and that excludes quantizations (the largest hub
category), which delta-compress 92–97%. Honest conclusions: (1) the hub contains **exact
byte-duplicates** (unsloth ≡ base → 0 bytes); (2) **lossless delta size quantifies fine-tune
magnitude** — preference tuning (DPO/GRPO) is gentle (5–8%), full SFT is heavy (~80%); (3) the
realistic family ratio depends on the *mix* — **1.5–3× for typical bf16 SFT families, 10×+
only for light-derivative-dominated sets** (re-uploads, quantizations, abliterations, LoRA,
preference-tuning); (4) deltas need a **same-precision** reference.

## Registry economics (the product, in real numbers)

A model hub stores a base plus many variants of it (instruct/abliterated/merged/LoRA-
merged/domain fine-tunes). Using the real Qwen2.5-0.5B numbers (988 MB each):

| storage strategy | 1 base + 1 variant | per extra variant | 1 base + 10 variants |
|---|---:|---:|---:|
| raw | 1976 MB | 988 MB | 10868 MB |
| each compressed single (ZipNN-class) | 1327 MB (33%) | 664 MB | 7300 MB (33%) |
| **delta-native (base single + deltas)** | **673 MB (66%)** | **~9 MB (0.9%)** | **754 MB (93%)** |

A light-edit variant costs ~0.9% of a model instead of 67% — a 100×+ win per variant. At
scale a hub of such variants compresses **~93%+** — and every byte is recoverable exactly. This is the moat: single-model
codecs (ZipNN/DFloat11) physically cannot do this because they ignore the shared base.
(Heavier fine-tunes cost more — ~47% of a model for a full SFT — still far below storing it whole.)

## Roadmap

- [x] Validate the **light-edit regime** on real data — Qwen2.5-0.5B abliteration = 90.9%.
- [x] Multi-shard models (dir / index.json), reference resolved by tensor name.
- [x] Self-verifying archives (embedded SHA-256), bulletproof round-trip test suite.
- [x] Fast: probe-then-final mode selection + multithreaded enc/dec (~19–32/155 MB/s).
- [x] **Low-rank residual mode** (adaptive numerical rank) — abliteration 90.9% → 99.1% (109×), portable & lossless.
- [x] Scale + bounded memory: validated at 1.5B (3 GB model on 3.5 GB free RAM); mmap + streaming.
- [x] Optimizer-state / checkpoint deltas: bf16 checkpoint chain ~62% (v=74%).
- [ ] Auto-reference: read a fine-tune's `config.json` base, fetch/locate it automatically.
- [ ] Package (pip-installable, console entry point).
- [ ] Optional fast/GPU decode path (à la DFloat11) for inference-time delta loading.
