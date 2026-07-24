# Worked examples — real commands, real outputs, measured

## Zero-config: one file, nothing else

```text
$ quantprobe plan --gguf Qwen3-Coder-30B-A3B-Instruct-Q2_K_L.gguf
[quantprobe] read from GGUF: 30.5B total, 3.4B active, 2.97 effective bits, KV 96 KB/pos
[quantprobe] no hardware flags: auto-detected this machine
  *   17.5 tok/s  hybrid: attention->VRAM, experts->RAM
```

Measured right after: **18.32 ± 0.17 tok/s** (prediction +4.7% conservative).

## What the optimizer is worth — a measured A/B (same model, same box)

| path | file | promised | measured |
|---|---|---|---|
| bits guessed at the grid, boundary invisible | Q3_K_M (13.7 GB) | "17.6" | **3.38 ± 2.66** (RAM-boundary thrash) |
| `--gguf` autospec (2.97 bits) → boundary-aware pick | Q2_K_L (11.3 GB) | 17.5 | **18.32 ± 0.17** |

**×5.4 realized** from correct specification + boundary routing alone — the equation is free; correct inputs and
boundary/gate knowledge are the product. Raw log: [`weights/data/optimizer_ab.log`](../weights/data/optimizer_ab.log).
A second measured gate: quantized K-cache at 16k depth on Pascal-class = 2.72 vs 16.12 tok/s — `optimize` refuses it for you.

## Probe, then quantize (30 minutes, any GGUF)

```bash
quantprobe probe --gguf your-model-f16.gguf --eval wiki.test.raw
```

Quantizes one FFN band to Q2_K at a time, measures perplexity per band, and prints the fragility curve **plus the ready-to-run depth-aware recipe**. Stock llama.cpp, no code changes, no calibration data. Example (Gemma 4 12B — the byte-identical winner):

```bash
llama-quantize \
  --tensor-type "blk\.([0-9]|[12][0-9]|3[0-5])\.ffn_.*=q2_k" \
  --tensor-type "blk\.(3[6-9]|4[0-7])\.ffn_.*=q4_k" \
  --tensor-type "attn_.*=q4_k" --token-embedding-type q4_k \
  gemma-4-12B-f16.gguf out-depthaware.gguf Q2_K 8
```

More recipes + the full fragility atlas: **[weights/GGUF_DEPTH_RECIPE.md](../weights/GGUF_DEPTH_RECIPE.md)**.


## What to expect on first run

`quantprobe probe` on a 12B takes ~30 min and prints a curve like this — the spike is the fragile band, and the recipe follows automatically:

```
quantprobe probe: gemma-4-12B-f16.gguf | 48 layers -> 4 bands
[2/3] band probe (one band's FFNs -> Q2_K at a time)
  layers 0-11 : PPL 9.51  (delta +2.14)
  layers 12-23: PPL 10.59 (delta +3.22)
  layers 24-35: PPL 10.53 (delta +3.16)
  layers 36-47: PPL 15.35 (delta +7.98)   <- fragile band
[3/3] recipe: protect layers 36-47 at Q4_K
  llama-quantize --tensor-type "blk\.(3[6-9]|4[0-7])\.ffn_.*=q4_k" ...
```

`quantprobe plan`/`target`/`run` are instant (they compute from the law). `quantprobe bench` runs a real llama-bench and prints predicted-vs-measured. Validated on **llama.cpp b9596+** (needs `--tensor-type` regex support).


## Troubleshooting

Every row here is a bug I actually hit and diagnosed — the table is the scar tissue.

| symptom | cause | fix |
|---|---|---|
| `llama-quantize: failed to quantize` from a Q6/Q8 source | requantizing an already-quantized GGUF | add `--allow-requantize` (quantprobe does this automatically) |
| hybrid MoE placement *slower* than pure CPU | full-file `mmap` + CUDA staging thrash a tight RAM box | use `--no-mmap` (quantprobe's `run` emits it for hybrids) |
| bench numbers wildly unstable (±3 on a 30B) | benching two >8 GB models back-to-back, or a cold page cache | warm-up pass first, then measure; don't bench big models back-to-back |
| post-reboot benches read low for ~10 min | antivirus first-read scan + cold cache | run once to warm, discard it, then measure |
| `ModuleNotFoundError: sentencepiece` on conversion | some tokenizers need it and it isn't a hard dep | `pip install sentencepiece` |
| perplexity step OOMs on a big model | too many GPU layers for 6 GB | lower `--ngl` (e.g. `--ngl 0` for pure CPU) |
| the GPU makes a MoE *slower*, not faster | Pascal-class low-bit decode collapses (η≈0.04 at 2-bit) | serve experts from CPU: `-ot "exps=CPU"` — often +54% |

