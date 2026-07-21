# Weight-Codec Evolution Log

Every codec variant's KPIs, newest at the bottom. Headline KPI = size-weighted **save%** on the real-model dataset with byte-exact round-trip; **bf16** is the priority dtype. Speeds are MB/s of input.

Baselines to beat: raw zstd/lzma, and **zipnn-zstd19** (byte-split + zstd, ~33% on real bf16). Our edge: a context model on the exponent plane + (creatively) predictive/structural coding to break the mantissa wall.

---

### 2026-06-04 15:57:11 | raw-gzip9

- **overall: save 12.76%** (ratio 1.1462), enc 4.9 / dec 56.4 MB/s, round-trip OK (40 tensors)
- **bf16: save 20.44%** (ratio 1.2569, dec 52.6 MB/s)
- by dtype: bf16 20.44%, fp16 7.69%, fp32 8.46%
- config: `{"type": "raw", "backend": "gzip", "level": 9}`
- note: baseline

### 2026-06-04 15:57:38 | raw-lzma9

- **overall: save 16.04%** (ratio 1.191), enc 1.2 / dec 6.5 MB/s, round-trip OK (40 tensors)
- **bf16: save 27.62%** (ratio 1.3816, dec 7.1 MB/s)
- by dtype: bf16 27.62%, fp16 9.56%, fp32 9.35%
- config: `{"type": "raw", "backend": "lzma", "level": 9}`
- note: baseline

### 2026-06-04 15:57:52 | raw-zstd19

- **overall: save 13.51%** (ratio 1.1562), enc 2.0 / dec 187.7 MB/s, round-trip OK (40 tensors)
- **bf16: save 22.08%** (ratio 1.2834, dec 143.7 MB/s)
- by dtype: bf16 22.08%, fp16 7.96%, fp32 8.7%
- config: `{"type": "raw", "backend": "zstd", "level": 19}`
- note: baseline

### 2026-06-04 15:58:12 | zipnn-zstd19

- **overall: save 21.81%** (ratio 1.279), enc 1.4 / dec 131.5 MB/s, round-trip OK (40 tensors)
- **bf16: save 30.96%** (ratio 1.4484, dec 105.2 MB/s)
- by dtype: bf16 30.96%, fp16 15.34%, fp32 16.78%
- config: `{"type": "byte-split", "backend": "zstd", "level": 19}`
- note: baseline

### 2026-06-04 15:58:40 | split-lzma9

- **overall: save 22.04%** (ratio 1.2827), enc 1.1 / dec 26.0 MB/s, round-trip OK (40 tensors)
- **bf16: save 31.18%** (ratio 1.4531, dec 23.4 MB/s)
- by dtype: bf16 31.18%, fp16 14.82%, fp32 17.14%
- config: `{"type": "byte-split", "backend": "lzma", "level": 9}`
- note: baseline

### 2026-06-04 16:00:08 | split-brotli11

- **overall: save 23.12%** (ratio 1.3007), enc 0.3 / dec 58.0 MB/s, round-trip OK (40 tensors)
- **bf16: save 32.79%** (ratio 1.4879, dec 51.0 MB/s)
- by dtype: bf16 32.79%, fp16 15.5%, fp32 17.92%
- config: `{"type": "byte-split", "backend": "brotli", "level": 11}`
- note: baseline

### 2026-06-04 16:00:29 | perplane-zstd-lzma

- **overall: save 21.94%** (ratio 1.2811), enc 1.4 / dec 32.7 MB/s, round-trip OK (40 tensors)
- **bf16: save 31.04%** (ratio 1.4501, dec 28.7 MB/s)
- by dtype: bf16 31.04%, fp16 14.83%, fp32 17.05%
- config: `{"type": "per-plane", "plane_backends": ["zstd", "lzma"], "level": 9}`
- note: baseline

---
## KEY FINDING (eval-driven): single-model is near floor; DELTA is the breakthrough

Diagnostics on real weights: mantissa is genuinely random (h0~8.0; XOR-delta of
consecutive elems does NOT help -> weights are not a smooth signal). Entropy floor:
bf16 ~32%, fp32 ~17%, fp16 ~16%. Baselines already at the wall (split-brotli bf16 32.79%).

PROVEN (delta_poc.py, pythia-70m bf16 base): compressing the XOR-delta vs a reference
(base model / previous checkpoint), exact-reversible given the reference:
  - sparse  1% changed: delta = 3% of standalone  (~97% save)
  - sparse 10% changed: delta = 18% of standalone (~94% save)
  - full fine-tune eps=1%: delta = 49% of standalone (~67% save)
vs standalone single-model = 33% save.

=> Direction: (1) lightweight single-tensor codec = split + tiny-CM exponent + raw
mantissa (match ~33% bf16, fast/light, beats zipnn-zstd). (2) REFERENCE/DELTA mode =
XOR vs base + the same coder on the delta (50-97% for related models). The delta mode
is the differentiator (fine-tunes / LoRAs / checkpoints / model versions).

### 2026-06-04 16:06:46 | raw-gzip9

- **overall: save 12.76%** (ratio 1.1462), enc 4.6 / dec 56.8 MB/s, round-trip OK (40 tensors)
- **bf16: save 20.44%** (ratio 1.2569, dec 52.8 MB/s)
- by dtype: bf16 20.44%, fp16 7.69%, fp32 8.46%
- config: `{"type": "raw", "backend": "gzip", "level": 9}`
- note: baseline

### 2026-06-04 16:07:09 | raw-lzma9

- **overall: save 16.04%** (ratio 1.191), enc 1.4 / dec 7.0 MB/s, round-trip OK (40 tensors)
- **bf16: save 27.62%** (ratio 1.3816, dec 7.6 MB/s)
- by dtype: bf16 27.62%, fp16 9.56%, fp32 9.35%
- config: `{"type": "raw", "backend": "lzma", "level": 9}`
- note: baseline

### 2026-06-04 16:07:22 | raw-zstd19

- **overall: save 13.51%** (ratio 1.1562), enc 2.3 / dec 191.4 MB/s, round-trip OK (40 tensors)
- **bf16: save 22.08%** (ratio 1.2834, dec 144.9 MB/s)
- by dtype: bf16 22.08%, fp16 7.96%, fp32 8.7%
- config: `{"type": "raw", "backend": "zstd", "level": 19}`
- note: baseline

### 2026-06-04 16:07:40 | zipnn-zstd19

- **overall: save 21.81%** (ratio 1.279), enc 1.6 / dec 136.7 MB/s, round-trip OK (40 tensors)
- **bf16: save 30.96%** (ratio 1.4484, dec 111.6 MB/s)
- by dtype: bf16 30.96%, fp16 15.34%, fp32 16.78%
- config: `{"type": "byte-split", "backend": "zstd", "level": 19}`
- note: baseline

### 2026-06-04 16:08:06 | split-lzma9

- **overall: save 22.04%** (ratio 1.2827), enc 1.1 / dec 28.0 MB/s, round-trip OK (40 tensors)
- **bf16: save 31.18%** (ratio 1.4531, dec 25.2 MB/s)
- by dtype: bf16 31.18%, fp16 14.82%, fp32 17.14%
- config: `{"type": "byte-split", "backend": "lzma", "level": 9}`
- note: baseline

### 2026-06-04 16:09:27 | split-brotli11

- **overall: save 23.12%** (ratio 1.3007), enc 0.3 / dec 62.2 MB/s, round-trip OK (40 tensors)
- **bf16: save 32.79%** (ratio 1.4879, dec 51.2 MB/s)
- by dtype: bf16 32.79%, fp16 15.5%, fp32 17.92%
- config: `{"type": "byte-split", "backend": "brotli", "level": 11}`
- note: baseline

### 2026-06-04 16:09:46 | perplane-zstd-lzma

- **overall: save 21.94%** (ratio 1.2811), enc 1.5 / dec 33.8 MB/s, round-trip OK (40 tensors)
- **bf16: save 31.04%** (ratio 1.4501, dec 30.0 MB/s)
- by dtype: bf16 31.04%, fp16 14.83%, fp32 17.05%
- config: `{"type": "per-plane", "plane_backends": ["zstd", "lzma"], "level": 9}`
- note: baseline

### 2026-06-04 16:09:59 | smart-zstd19

- **overall: save 22.22%** (ratio 1.2856), enc 2.3 / dec 263.8 MB/s, round-trip OK (40 tensors)
- **bf16: save 32.71%** (ratio 1.4862, dec 252.1 MB/s)
- by dtype: bf16 32.71%, fp16 15.37%, fp32 16.33%
- config: `{"type": "smart-split", "exp_backend": "zstd", "level": 19, "mantissa": "stored-raw"}`
- note: baseline

### 2026-06-04 16:10:37 | smart-brotli11

- **overall: save 22.27%** (ratio 1.2865), enc 0.7 / dec 125.3 MB/s, round-trip OK (40 tensors)
- **bf16: save 32.79%** (ratio 1.4878, dec 101.9 MB/s)
- by dtype: bf16 32.79%, fp16 15.51%, fp32 16.35%
- config: `{"type": "smart-split", "exp_backend": "brotli", "level": 11, "mantissa": "stored-raw"}`
- note: baseline

### 2026-06-04 16:10:54 | smart-lzma9

- **overall: save 21.12%** (ratio 1.2677), enc 1.8 / dec 34.6 MB/s, round-trip OK (40 tensors)
- **bf16: save 31.04%** (ratio 1.4502, dec 30.2 MB/s)
- by dtype: bf16 31.04%, fp16 14.84%, fp32 15.51%
- config: `{"type": "smart-split", "exp_backend": "lzma", "level": 9, "mantissa": "stored-raw"}`
- note: baseline

### 2026-06-04 16:12:37 | delta-brotli11@full eps=1%

- **overall: save 66.56%** (ratio 2.9905), enc 0.0 / dec 71.3 MB/s, round-trip OK (3 tensors)
- **bf16: save 66.56%** (ratio 2.9905, dec 71.3 MB/s)
- by dtype: bf16 66.56%
- config: `{"type": "delta", "inner": {"type": "byte-split", "backend": "brotli", "level": 11}, "scenario": "full eps=1%"}`
- note: delta vs standalone 50%

### 2026-06-04 16:12:45 | delta-zstd19@full eps=1%

- **overall: save 66.22%** (ratio 2.9603), enc 0.0 / dec 136.3 MB/s, round-trip OK (3 tensors)
- **bf16: save 66.22%** (ratio 2.9603, dec 136.3 MB/s)
- by dtype: bf16 66.22%
- config: `{"type": "delta", "inner": {"type": "byte-split", "backend": "zstd", "level": 19}, "scenario": "full eps=1%"}`
- note: delta vs standalone 50%

### 2026-06-04 16:13:08 | delta-brotli11@full eps=5%

- **overall: save 51.17%** (ratio 2.0477), enc 0.0 / dec 57.2 MB/s, round-trip OK (3 tensors)
- **bf16: save 51.17%** (ratio 2.0477, dec 57.2 MB/s)
- by dtype: bf16 51.17%
- config: `{"type": "delta", "inner": {"type": "byte-split", "backend": "brotli", "level": 11}, "scenario": "full eps=5%"}`
- note: delta vs standalone 73%

### 2026-06-04 16:13:15 | delta-zstd19@full eps=5%

- **overall: save 51.15%** (ratio 2.0471), enc 0.0 / dec 119.0 MB/s, round-trip OK (3 tensors)
- **bf16: save 51.15%** (ratio 2.0471, dec 119.0 MB/s)
- by dtype: bf16 51.15%
- config: `{"type": "delta", "inner": {"type": "byte-split", "backend": "zstd", "level": 19}, "scenario": "full eps=5%"}`
- note: delta vs standalone 73%

### 2026-06-04 16:13:45 | delta-brotli11@sparse 10%

- **overall: save 87.93%** (ratio 8.2851), enc 0.0 / dec 68.6 MB/s, round-trip OK (3 tensors)
- **bf16: save 87.93%** (ratio 8.2851, dec 68.6 MB/s)
- by dtype: bf16 87.93%
- config: `{"type": "delta", "inner": {"type": "byte-split", "backend": "brotli", "level": 11}, "scenario": "sparse 10%"}`
- note: delta vs standalone 18%

### 2026-06-04 16:13:53 | delta-zstd19@sparse 10%

- **overall: save 87.57%** (ratio 8.0437), enc 0.0 / dec 123.4 MB/s, round-trip OK (3 tensors)
- **bf16: save 87.57%** (ratio 8.0437, dec 123.4 MB/s)
- by dtype: bf16 87.57%
- config: `{"type": "delta", "inner": {"type": "byte-split", "backend": "zstd", "level": 19}, "scenario": "sparse 10%"}`
- note: delta vs standalone 19%

### 2026-06-04 16:14:17 | delta-brotli11@sparse 1%

- **overall: save 98.23%** (ratio 56.45), enc 0.0 / dec 102.3 MB/s, round-trip OK (3 tensors)
- **bf16: save 98.23%** (ratio 56.45, dec 102.3 MB/s)
- by dtype: bf16 98.23%
- config: `{"type": "delta", "inner": {"type": "byte-split", "backend": "brotli", "level": 11}, "scenario": "sparse 1%"}`
- note: delta vs standalone 3%

### 2026-06-04 16:14:24 | delta-zstd19@sparse 1%

- **overall: save 97.98%** (ratio 49.5965), enc 0.0 / dec 156.2 MB/s, round-trip OK (3 tensors)
- **bf16: save 97.98%** (ratio 49.5965, dec 156.2 MB/s)
- by dtype: bf16 97.98%
- config: `{"type": "delta", "inner": {"type": "byte-split", "backend": "zstd", "level": 19}, "scenario": "sparse 1%"}`
- note: delta vs standalone 3%

### 2026-06-04 16:44:39 | delta-real@1000steps

- **overall: save 68.07%** (ratio 3.1321), enc 0.0 / dec 0.0 MB/s, round-trip OK (76 tensors)
- **bf16: save -%** (ratio -, dec - MB/s)
- by dtype: fp32 68.07%
- config: `{"type": "delta-real", "repo": "EleutherAI/pythia-70m", "gap_steps": 1000, "tensors": 76, "dtype": "fp32"}`
- note: REAL checkpoint delta, 1000 training steps apart; standalone=16.9%

### 2026-06-04 17:08:00 | delta-real@1000steps

- **overall: save 68.07%** (ratio 3.1899), enc 0.0 / dec 0.0 MB/s, round-trip OK (76 tensors)
- **bf16: save -%** (ratio -, dec - MB/s)
- by dtype: fp32 68.07%
- config: `{"type": "delta-real", "repo": "EleutherAI/pythia-70m", "gap_steps": 1000, "tensors": 76, "dtype": "fp32"}`
- note: REAL checkpoint delta, 1000 training steps apart; standalone=16.9%

### 2026-06-04 18:42:53 | finetune-delta@SmolLM-135M

- **overall: save 48.97%** (ratio 1.9597), enc 0.0 / dec 0.0 MB/s, round-trip OK (272 tensors)
- **bf16: save 48.97%** (ratio 1.9597, dec 0.0 MB/s)
- by dtype: bf16 48.97%
- config: `{"type": "finetune-delta", "base": "SmolLM-135M", "ft": "SmolLM-135M-Instruct", "ref_dtype": "bf16"}`
- note: REAL fine-tune delta; standalone=32.9%; 97% elems changed

### 2026-06-04 19:36:32 | wcodec-delta@Qwen2.5-0.5B-abliterated

- **overall: save 90.9%** (ratio 10.99), enc 3.0 / dec 104.0 MB/s, round-trip OK (290 tensors)
- **bf16: save 90.9%** (ratio 10.99, dec 104.0 MB/s)
- by dtype: bf16 90.9%
- config: `{"type": "delta-real", "base": "Qwen2.5-0.5B-Instruct", "variant": "abliterated", "dtype": "bf16"}`
- note: REAL light edit (abliteration); 76% tensors identical; TARGET HIT 90.9%

### 2026-06-04 20:29:19 | wcodec-delta@pythia-bf16-1000step

- **overall: save 61.3%** (ratio 2.59), enc 2.0 / dec 77.0 MB/s, round-trip OK (77 tensors)
- **bf16: save 61.3%** (ratio 2.59, dec 77.0 MB/s)
- by dtype: bf16 61.3%
- config: `{"type": "delta-real", "model": "pythia-70m", "dtype": "bf16", "gap": 1000}`
- note: bf16 checkpoint delta = SOTA regime; 61.3% ~= published 62%; fp32 same delta 68.6%

### 2026-06-04 20:53:11 | wcodec-lowrank@Qwen2.5-0.5B-abliterated

- **overall: save 97.6%** (ratio 42.22), enc 19.0 / dec 155.0 MB/s, round-trip OK (290 tensors)
- **bf16: save 97.6%** (ratio 42.22, dec 155.0 MB/s)
- by dtype: bf16 97.6%
- config: `{"type": "delta-lowrank", "base": "Qwen2.5-0.5B-Instruct", "variant": "abliterated", "dtype": "bf16"}`
- note: LOW-RANK mode; abliteration rank-<=4 delta; 90.9->97.6% (42x)

### 2026-06-04 21:03:29 | wcodec-lowrank-adaptive@Qwen2.5-0.5B-abliterated

- **overall: save 99.1%** (ratio 109.27), enc 19.0 / dec 150.0 MB/s, round-trip OK (290 tensors)
- **bf16: save 99.1%** (ratio 109.27, dec 150.0 MB/s)
- by dtype: bf16 99.1%
- config: `{"type": "delta-lowrank", "base": "Qwen2.5-0.5B-Instruct", "variant": "abliterated", "dtype": "bf16", "rank": "numerical"}`
- note: adaptive numerical-rank low-rank; abliteration 99.1% / 109x

### 2026-06-04 22:42:06 | wcodec-lowrank@Qwen2.5-1.5B-abliterated

- **overall: save 98.8%** (ratio 84.19), enc 14.0 / dec 82.0 MB/s, round-trip OK (338 tensors)
- **bf16: save 98.8%** (ratio 84.19, dec 82.0 MB/s)
- by dtype: bf16 98.8%
- config: `{"type": "delta-lowrank", "base": "Qwen2.5-1.5B-Instruct", "variant": "abliterated", "dtype": "bf16", "scale": "1.5B"}`
- note: SCALE 1.5B (3GB model, 3.5GB free RAM via mmap+streaming); abliteration 98.8%/84x

### 2026-06-05 03:18:25 | wcodec-lowrank@Qwen2.5-3B-abliterated

- **overall: save 98.7%** (ratio 79.56), enc 9.0 / dec 123.0 MB/s, round-trip OK (434 tensors)
- **bf16: save 98.7%** (ratio 79.56, dec 123.0 MB/s)
- by dtype: bf16 98.7%
- config: `{"type": "delta-lowrank", "base": "Qwen2.5-3B-Instruct", "variant": "abliterated", "dtype": "bf16", "scale": "3B", "cross_sharded": true}`
- note: SCALE 3B (6.2GB model, 3.5GB free RAM, cross-sharded base/ablit); abliteration 98.7%/80x

### 2026-06-05 03:31:45 | wcodec-delta@Qwen0.5B-abliterated-int8

- **overall: save 97.4%** (ratio 38.9), enc 0.0 / dec 0.0 MB/s, round-trip OK (290 tensors)
- **bf16: save -%** (ratio -, dec - MB/s)
- by dtype: int8 97.4%
- config: `{"type": "delta-quantized", "dtype": "int8", "base": "Qwen2.5-0.5B-Instruct", "variant": "abliterated"}`
- note: QUANTIZED int8 variant delta survives quantization; 97.4% (38x); 244/290 tensors identical
