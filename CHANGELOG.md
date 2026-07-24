# Changelog

## 1.3.0 — 2026-07-24

**Any hardware combination.** Prompted by a wild 744B rig (72 GB VRAM + 128 GB RAM + RAID-0 Gen5
NVMe at 3.6 tok/s) that the two-tier model under-predicted.

- Three-tier expert cache: new ADDITIVE placement row "stream from disk (VRAM+RAM expert cache)" —
  models what expert-caching runtimes (ktransformers/colibri-class) achieve; the stock-llama.cpp
  rows are untouched (validated: retrodicts the 3.6 tok/s rig at 2.9, within the law's +/-25%).
- Multi-device inputs: comma lists aggregate — `--vram 24,24,24 --vram-bw 936,936,936` (x0.85 TP
  efficiency [est]) and `--disk-bw 14,14` (x0.75 stripe [est from the RAID-0 eta 0.66 datapoint]).
- Simulator carries the same three-tier row (CLI parity).
- Validation matrix green: every measured anchor identical to the digit (30B hybrid 18.9, ctx-16k
  15.4, 110B 0.2, Laguna 0.3) with the new rows strictly additive. 31 smoke tests.

## 1.2.0 — 2026-07-24

**Zero-configuration.** The minimal command is now `quantprobe plan --gguf model.gguf` — nothing else.

- New `quantprobe hw`: detects RAM (sticks + configured MT/s -> peak GB/s), GPU(s) (nvidia-smi +
  name->bandwidth/eta table; multi-GPU aggregated at 0.85 TP efficiency), Apple unified memory.
  Every value tagged [os]/[table]/[default]; nothing leaves the machine. `--measure FILE` adds a real
  sequential-read disk measurement.
- GGUF autospec: `--gguf` alone yields total/active params (tensor sums + expert metadata), TRUE
  effective bits (file size), EXACT KV bytes/pos (MLA-aware). Explicit flags always override.
- Auto-detection engages only when no `--machine` and no hardware flags are given — presets/flags
  are unchanged and remain the way to estimate a machine you are not running on.
- `--bits` freed to continuous values (e.g. 2.88) + nearest-key quality lookup.
- Verified: auto-detected reference box reproduces the hand-measured `2016-xmp` preset exactly
  (17.6 == 17.6 tok/s on the same GGUF). 28 smoke tests green.
- Pre-registration #7 HIT: Laguna S 2.1 (118B) on the 2016 desktop — staked 0.2-0.4 tok/s before
  the download, measured 0.38 +/- 0.17 (llama-bench, mainline b10098, no draft).

## 1.1.0 — 2026-07-23

**Law 4 v2: the context term.** Prompted by u/RogerAI--fyi's observation that the decode law omitted
per-token KV reads; measured same-day on the reference box (tg32 clean 20.02 ± 0.02 → 16.12 ± 0.06
at depth 16384, −19.5% vs pre-registered −8…−15% — a published near-miss; η_kv ≈ 0.70 single-point
calibration).

- `--ctx N` on `plan` / `target` / `run` / `bench`: adds per-token KV reads (served from the tier KV
  lives on) **and** KV memory to the fit check — large contexts can flip the winning placement.
- `bench --depth N`: measure the context term on your box (llama-bench `-d`); prediction follows depth.
- Per-model KV bytes/pos in presets (MLA ≈10× smaller: DeepSeek 31 KB vs Qwen3-30B 98 KB; SWA [est]).
- `--kv-per-pos KB` override for custom models; `run --ctx` launches llama.cpp with `-c` set.
- Simulator: context-depth input, same math, CLI-parity verified.
- Chart I (`weights/data/x_chart_I_kvdepth.png`): measured KV-depth slope vs the law.
- New pre-registered prediction: 30B-A3B Q4 pure-CPU on DDR4-45 at 16k = −29% (8.0 → 5.7 tok/s).
- 6 new smoke tests (19 total): ctx=0 identity, monotonicity, placement-dependence, calibration
  anchor, fit-flip, `bench --depth --dry`.

## 1.0.0 — 2026-07-22

Initial public release: four placement laws, 8-command CLI (plan / target / fetch / quantize / probe /
run / bench / dashboard), depth-aware GGUF compression verified end-to-end, browser calculator,
opt-in community datapoint loop, validation bundle for the 19 tok/s claim.
