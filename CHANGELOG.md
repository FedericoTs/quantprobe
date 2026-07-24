# Changelog

## 1.6.1 — 2026-07-25

- **`python -m quantprobe` now works** — the PATH-proof entry point. On Windows, `pip install`
  frequently lands `quantprobe.exe` in a user-site Scripts folder that is not on PATH ("'quantprobe'
  is not recognized"); `python -m quantprobe <anything>` is identical and always works. Found when
  it happened on our own machine. 46 tests.

## 1.6.0 — 2026-07-25

The full-customization pipeline, delivered as one decision-making command:

- **`quantprobe auto` with no arguments is now interactive**: detects the machine, asks for the
  model (preset or any HF GGUF repo), asks one question — best standard quant (skip
  quantization), full custom probe-and-build, or a speed target — and offers to launch when
  ready. Clean one-line refusal when there is no terminal to ask.
- **`--custom` is now machine-gated by the laws.** If the optimizer wants ≥3.5 bits on your
  hardware, the surgery doesn't pay (Laws 1–2: the fragile-band fix matters below ~3 bits) —
  auto says so and fetches the optimal standard quant instead. `--force-custom` overrides.
  The same command on a 6 GB/16 GB box still builds the depth-aware file, because there it wins.
- 4 new tests (45 total, recounted: the historical "41" was itself off by 4 — real ladder 37 → 41 → 45): gate, force-override, wizard with piped answers, wizard EOF.

## 1.5.2 — 2026-07-25

- **Unknown preset names now fail loudly.** `plan/target/optimize --model <unknown>` used to fall
  back silently to a 13B default and produce plausible-looking wrong numbers (found while
  answering a real user question). Now: clean error listing the presets, plus the two escape
  hatches (`--total/--active` to describe any model, `--gguf` to read the exact spec from the
  file). Same for `--machine`. 4 new tests (41 total).
- Docs: the one-command pipeline (`auto <model> --custom --run`) is now the first thing in
  QUICKSTART; bit-level selection clarified (quantize = fixed validated recipe, `auto`/`fetch`
  = standard quants).

## 1.5.1 — 2026-07-25

**`auto --custom` — the personalized recipe, now truly one command.** Fetches the best
requantizable source from the repo (prefers Q8-class over f16: half the download, identical PTQ
quality), auto-fetches the WikiText-2 eval corpus (1.3 MB, once), probes YOUR model's fragile
band (~30-60 min), and builds the depth-aware GGUF - personalized to the model, sized by the
optimizer. The fast path (closest community quant) remains the default; every fast-path run
advertises the upgrade. 41 tests.

## 1.5.0 — 2026-07-25

**One command from empty machine to running model — and the law, watchable.**

- `quantprobe auto <model> [--tps N] [--run]`: machine auto-detected -> optimizer picks the
  effective bits -> the HF repo's file list is scanned and the closest quant matched BY SIZE
  (bits = size x 8 / params; format-agnostic) -> resumable fetch -> run command (or --run
  launches). First live run picked the exact file independently measured at 18.32 tok/s.
  The custom probe path (better quality at the same bytes) is advertised on every run.
- Dashboard v2.1: single-viewport app (fixed sidebar, internal chat scroll), streaming replies
  with VISIBLE thinking, a thinking TOGGLE + per-reply anatomy (TTFT / thinking / answer), and
  the NEURON GALAXY - every expert of every layer as a dot, lit per generated token, colored by
  its memory tier. Honesty printed on the panel: uniform sampling is the statistically exact
  picture under the measured flat-routing law; stock llama.cpp exposes no router telemetry.
- Hardening from real use: completion-probe readiness (llama-server reports healthy before
  weights load), exclusive port bind (Windows silently allowed double-binds), RTX 50-series in
  the detect table, per-card multi-GPU bandwidth aggregation.
- 40 smoke tests. Eleven commands.

## 1.4.0 — 2026-07-24

**`quantprobe optimize` — the cheapest path to a target speed.** A pure search layer over the
frozen law (no physics touched; anchors untouchable by construction): bits ladder x placement x
KV levers x hardware deltas, Pareto-ranked by quality cost then euros, with realize-commands.

- Backtested: blind on the reference box it rediscovers the measured-best config (2.5-bit
  depth-aware hybrid, 18.9 predicted / 19.30-20.02 measured) on the frontier.
- Boundary-aware: on a 16 GB card with a just-over file it picks the bits-shave that crosses into
  all-in-VRAM (x4+), the pre-registration #8 lesson operationalized.
- Measured lever gates: KV-q8 blocked on weak-decode GPUs (measured -83% at 16k on Pascal,
  2026-07-24); REAP-class pruning never ranked without --allow-prune (+39% OOD ppl measured).
- Realizable-by-default: only stock-llama.cpp placements unless --any-runtime.
- 37 smoke tests. Ten commands.

## 1.3.1 — 2026-07-24

**Tier-boundary advisor.** Corollary of Law 4 made explicit: decode speed is a step function of
placement, so the marginal value of a gigabyte is ~zero mid-tier and enormous at a boundary. When a
config sits within 30% over a tier boundary, `plan` now names the gap and prices the promotion
("1.6 GB over the VRAM boundary - shave it -> ~67.6 tok/s (x4.3)"). Works for any shave lever:
quant step, tighter probed band, pruned variant, KV quantization. Validated on the pre-reg #8 REAP
pair: fires on the 14.7 GB parent, silent on the promoted 11.5 GB prune. 32 smoke tests.

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
