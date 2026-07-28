# Worked examples — real commands, real outputs, measured

## Zero-config: one file, nothing else

```text
$ quantprobe plan --gguf Qwen3-Coder-30B-A3B-Instruct-Q2_K_L.gguf
[quantprobe] read from GGUF: 30.5B total, 3.4B active, 2.97 effective bits, KV 96 KB/pos
[quantprobe] no hardware flags: auto-detected this machine (vram 6GB@192 | ram 16GB@48 | disk 0.5 GB/s). Pass --machine/flags to estimate a different box.
[quantprobe] calibration applied [ram 24.3 GB/s measured; disk 3.13 GB/s measured] (2026-07-28)
[quantprobe] anchored: CPU x1.18, GPU x0.60 from your calibrate anchor runs [tier ratios; format x0.56; --no-anchors disables]

quantprobe plan - custom model @ 2.97-bit on THIS machine [auto-detected - run `quantprobe hw` for details]
  model 11.3 GB | active 1.77 GB/token | est. quality cost x1.05 (depth-aware recipe)

  *   17.1 tok/s  split experts: 30%->VRAM, rest->RAM   [pins 7GB of 12GB RAM (CUDA host memory) - fails under memory pressure; if it does, drop -ot and let auto-placement decide]
      15.6 tok/s  hybrid: attention->VRAM, experts->RAM   [RAM boundary - needs --no-mmap; can be unstable; pins 10GB of 12GB RAM (CUDA host memory) - fails under memory pressure; if it does, drop -ot and let auto-placement decide]
      11.4 tok/s  pure CPU (GPU idle)   [RAM boundary - expect bimodal speed]

  speculation: pays ONLY when output copies its context (edits, refactors, RAG quoting)
  - on novel generation the ngram drafter produces 0 drafts and changes nothing (D-10,
  independently replicated on an RTX 3090). Details below.

  run it:  llama-server -m model.gguf -ngl 99 -ot "blk\.(14|15|16|17|18|19|20|21|22|23|24|25|26|27|28|29|30|31|32|33|34|35|36|37|38|39|40|41|42|43|44|45|46|47)\.ffn_.*_exps\.=CPU" --no-mmap -b 1024 -ub 1024 --threads 4
           (--threads 4 = this machine's LOGICAL cores; llama.cpp's own auto-detect may pick
           physical-only and cost 2x on CPU-bound decode - verify with your fork if unsure)

  [... phase, workload, KV-cache, batching and speculation-tuning advice - trimmed]
```

Captured from shipped v1.20.2 on the reference box. The command is still just a file — the two
`[quantprobe]` provenance lines appear because this machine was measured once with
`quantprobe calibrate` (next section), so the predictions are anchored to its own benchmark
runs by default.

Two measurements against those rows, on the same box:

- **hybrid, 17.5 predicted → 18.32 ± 0.17 measured** (+4.7% conservative). This was the original
  validation. It no longer reproduces byte-for-byte — today's code prints 15.6 for this row —
  and that is deliberate: default-on anchoring (v1.20) and the v1.20.1/1.20.2 constant fixes
  moved the printed predictions. The current validation record is the original-case retest
  (pre-registrations #60/#61) and [MACHINE_LADDER.md](../MACHINE_LADDER.md).
- **split experts** — still the `*` row. The v1.8 stake was 19.6 predicted → **19.22 ± 0.7
  measured** (−2%). The current record at healthy clocks: **20.4–22.2 tok/s measured** across
  the cold-boot retest and the one-session ladder (pre-registrations #60/#61 and #65), against
  an anchored prediction of 22.0 (the plan output captured in pre-registration #60). The 17.1 printed above carries the documented gguf-arm
  under-promise (−18.8% in the ladder's v1.20.2 self-consistent column) — when this tool
  misses, it misses low.

The `*` moved from hybrid to split experts back in v1.8 because the tool learned a better
placement, not because the hybrid prediction moved. Both rows are checked in the release gate.

## Measure your machine once: `quantprobe calibrate`

The anchored predictions above come from a one-time command:

```bash
quantprobe calibrate --model your-model.gguf   # --skip-bench to measure hardware only
```

It measures what this box actually delivers instead of what the spec sheet promises: RAM stream
(a real read — this box delivers ~24–26 of its nominal 48 GB/s), disk on your own file, and GPU
sustained clocks, which catches the stuck-boost state that silently costs 25–30% and that only a
reboot clears (pre-registrations #60/#61). Passing `--model` adds short anchor benchmark runs on
your own GGUF. Everything persists to `~/.quantprobe/calibration.json`, and `plan`, `bench`,
`optimize` and `auto` consume it automatically, tagged `[calibrated]`.

With the anchor runs, predictions are anchored by default — the law scaled through your own two
measured arms. That shipped default-on only after passing the gate it pre-registered before any
number existed (pre-registration #64: leave-one-out across 5 arms, anchored median error 5.8%
vs the plain law's 19.0%). Across the full ladder the shipped v1.20.2 accuracy is ~12% median,
misses erring low ([MACHINE_LADDER.md](../MACHINE_LADDER.md)). `--no-anchors` restores the
plain law.

## What the optimizer is worth — a measured A/B (same model, same box)

| path | file | promised | measured |
|---|---|---|---|
| bits guessed at the grid, boundary invisible | Q3_K_M (13.7 GB) | "17.6" | **3.38 ± 2.66** (RAM-boundary thrash) |
| `--gguf` autospec (2.97 bits) → boundary-aware pick | Q2_K_L (11.3 GB) | 17.5 | **18.32 ± 0.17** |

**×5.4 realized** from correct specification + boundary routing alone — the equation is free; correct inputs and
boundary/gate knowledge are the product. Raw log: [`weights/data/optimizer_ab.log`](../weights/data/optimizer_ab.log).
A second measured gate: quantized K-cache at 16k depth on Pascal-class = 2.72 vs 16.12 tok/s — `optimize` refuses it for you.

## Probe, then quantize (any GGUF — it estimates its own runtime first)

```bash
quantprobe probe --gguf your-model-f16.gguf --eval wiki.test.raw
```

Quantizes one FFN band to Q2_K at a time, measures perplexity per band, and prints the fragility curve **plus the ready-to-run depth-aware recipe**. Stock llama.cpp, no code changes.

**On runtime:** this scales hard with model size — roughly 50 minutes for a 7B, but about 6 hours for a 35 GB source, dominated by whether the working file fits your RAM. The command prints its own estimate before starting, shows a live ETA from your machine's measured pace, and asks before committing you to anything over two hours. (This document previously said "30 minutes" flat, which was wrong by an order of magnitude on large models.)

**Already measured by someone else?** `quantprobe recipes` lists known fragility bands — the band is a property of the model, not your hardware, so it transfers. `quantize --recipe <key>` skips the probe entirely.

Example (Gemma 4 12B — the byte-identical winner):

```bash
# --apply builds it; --dry prints the exact llama-quantize command first if you want to inspect it.
quantprobe probe --gguf gemma-4-12B-f16.gguf --eval wiki.test.raw --apply --imatrix auto
```

The generated command protects more than the fragile band. **Always-active tensors are protected
too** — the shared expert (`ffn_*_shexp`, which fires on every token, unlike routed experts),
attention, recurrent state (`ssm_*` on hybrid architectures), and the embedding.

This document previously hand-wrote that command, and the transcription went stale: it omitted
the SSM and shared-expert protections added in v1.6.4 and v1.7.0. Anyone who copied it built a
measurably worse model — on one 35B-class model those two omissions cost 24% and 3% perplexity
respectively. The command is now generated by the tool so it cannot drift from the recipe again.

More recipes + the full fragility atlas: **[weights/GGUF_DEPTH_RECIPE.md](../weights/GGUF_DEPTH_RECIPE.md)**.


## What to expect on first run

`quantprobe probe` on a 12B is an hour-class run (it prints its own estimate and a live ETA before committing you — see the runtime note above) and produces a curve like this — the spike is the fragile band, and the recipe follows automatically:

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
| the GPU makes a MoE *slower*, not faster | the experts don't fit, so the GPU is thrashing rather than serving | serve experts from CPU: `-ot "exps=CPU"` — often +54% |

