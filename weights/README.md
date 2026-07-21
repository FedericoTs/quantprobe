# wcodec — lossless compression for LLM weight files

A lightweight, **byte-exact** codec for `.safetensors` weight files, plus the eval
harness used to derive it. Two modes:

| mode | what it does | real result (Pythia-70m, 281.7 MB) |
|------|--------------|-----------------------------------|
| **single** | per-tensor: compress the float exponent plane, store the (random) mantissa raw | 16.9% saved on fp32 / **32.7% on bf16** (beats ZipNN) |
| **delta** | per-tensor: per-plane XOR vs the same tensor in a reference file (base model / previous checkpoint) | **68.6% saved** (3.19×) vs a 1000-step training delta |

Every result is gated by a SHA-256 round-trip (`decompress(compress(x)) == x`). The
archive stores the original's SHA-256, so **decompression is self-verifying**: it
reproduces the exact original bytes or raises (e.g. if you supply the wrong reference) —
never silent corruption.

## Why it works

LLM weights are near-incompressible standalone: the low mantissa bits are essentially
random (entropy wall ≈ 17% for fp32, 32% for bf16). But the **change** between two
related models is not random. We found (bit-level diagnostic on real Pythia training
checkpoints) that training updates are too small to flip the low mantissa bits, so for
fp32 **bits 0–12 stay frozen** across checkpoints — exactly the bits that make
standalone incompressible. They XOR to zero in the delta and vanish; the delta's
entropy lives in ~11 high-mantissa bits. Compressing each byte-plane of the XOR
separately (`per-plane`) sits near the order-0 entropy floor.

Two things we **tested and rejected** (eval-gated, not assumed):
- *Momentum / delta-of-delta*: consecutive training deltas are statistically
  independent (cosine ≈ 0, ‖dd‖/‖d‖ ≈ √2), so a 2nd-order arithmetic coder would not
  help. Dense training deltas are at the floor.
- *Bit-plane coding*: loses to per-plane on real data.

For **sparse** deltas (LoRA / pruned fine-tunes, <50% of elements changed) the codec
auto-switches to a bitmap + raw-values path.

## Usage

```bash
# single-model compression
python -m weights.wcodec compress model.safetensors -o model.wc
python -m weights.wcodec decompress model.wc -o restored.safetensors

# delta vs a reference (base model or previous checkpoint)
python -m weights.wcodec compress finetune.safetensors --ref base.safetensors -o ft.wc
python -m weights.wcodec decompress ft.wc --ref base.safetensors -o restored.safetensors

# benchmark (round-trip gated)
python -m weights.wcodec bench model.safetensors --ref base.safetensors --level 3

# sharded models (real LLMs): pass a directory or its index.json -- the delta reference
# is resolved by tensor name across any sharding (read on demand, bounded memory)
python -m weights.wcodec compress  ./finetune_dir --ref ./base_dir -o ft.wc
python -m weights.wcodec decompress ft.wc --ref ./base_dir -o ./restored_dir
python -m weights.wcodec bench ./model_dir --ref ./base_dir
```

`--level` is the zstd level (1–22, default 19). Speed/ratio trade-off on the real
282 MB delta: **zstd-3 → 66.0% @ 46 MB/s enc**, zstd-19 → 68.6% @ 1 MB/s. Decompression
is 100–150 MB/s at any level (the read path stays lightweight). Use a low level for
interactive work, a high level for write-once archival.

## Storing a whole training run

Store checkpoint 0 standalone, every later checkpoint as a delta vs the previous one
(the chain is exact). On a real 4-checkpoint Pythia run: **1126.8 MB → 498.3 MB
(55.8% saved)**; each extra checkpoint costs only ~31% of a model. Asymptotically a
long run compresses ~69%.

## Eval harness (how the codec was derived)

- `codecs.py` — `WeightCodec` interface + baselines (raw, ZipNN-style byte-split,
  smart per-plane).
- `evaluate.py` — size-weighted save% / ratio / MB-s per dtype, SHA-256 round-trip
  gate, entropy diagnostics.
- `delta.py` — `DeltaCodec` (XOR vs reference) + synthetic fine-tune benchmark.
- `delta_real.py` / `delta_diag.py` / `delta_seq.py` — validation on **real** Pythia
  checkpoints: per-pair delta, bit-level structure, and trajectory/momentum analysis.
- `track.py` + `results/leaderboard.json` + `EVOLUTION_LOG.md` — append-only tracking.
- `wcodec.py` — the end-to-end file codec + CLI (this tool).
