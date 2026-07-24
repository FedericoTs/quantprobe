# evo-compress

A 2-week **feasibility spike**: can an AlphaEvolve-style evolutionary search discover
**lossless** compression *pipelines* that beat strong baselines on a target data
domain — reproducibly, and without ever losing a byte?

The evolvable artifact is **not** a codec. It is a **pipeline**: an ordered list of
reversible preprocessing transforms (delta, zigzag, byte-transpose, BWT, …) followed
by a backend entropy coder/codec (zstd / brotli / lzma / bz2 / zlib) and its level.
Backends are called as libraries — we are testing the *search*, not re-implementing
zstd in C.

> **North star: the Hutter Prize (the "Wikipedia compression prize").**
> It is alive and well-funded — €500,000 pool, only ~€30k paid out as of late 2025.
> Target: `enwik9` (first 10⁹ bytes of a specific English Wikipedia dump). Current
> record: `fx2-cmix` (Orav & Knoll, 2024) → 110,793,128 bytes (~9.03×).
> See [the honest framing](#about-the-hutter-prize) below for why this spike is a
> *stepping stone*, not a prize entry: every winner is a context-mixing model, a
> different league from pipeline-over-zstd.

## The 5 guiding principles (enforced in code)

1. **Lossless only.** The scorer's hard gate is byte-exact round-trip
   `decode(encode(x)) == x` for every file, double-checked by SHA-256. Any candidate
   that fails is invalid and scored `-inf`. This makes the objective **un-gameable**
   ([`evaluator.py`](evocompress/evaluator.py)).
2. **Don't rewrite zstd.** Backends are libraries; the search evolves the *pipeline*.
3. **Avoid overfitting the corpus.** Files are split into **TRAIN** (search optimizes
   here) and a disjoint **HELD-OUT TEST** of the same domain. The headline number is
   held-out ratio. We report both.
4. **Runs day one with no API key.** Default engine is a classical **genetic
   algorithm** over pipelines. OpenEvolve + LLM is an optional later mode
   ([`openevolve_adapter.py`](evocompress/openevolve_adapter.py)).
5. **Deterministic & seeded.** Every run is reproducible; everything is logged.

## Quickstart

```bash
make setup            # create .venv and install pinned deps
make smoke            # quick end-to-end spike on the default time-series domain
```

### Windows / PowerShell (no `make` required)

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt

# quick smoke
.venv\Scripts\python.exe -m experiments.run_spike --domain time-series --engine ga `
    --population 40 --generations 20 --seed 0 --objective max_ratio

# run the tests
.venv\Scripts\python.exe -m pytest -q
```

### The CLI

```bash
python -m experiments.run_spike --domain time-series --engine ga \
    --population 80 --generations 40 --seed 0 --objective max_ratio
```

It loads the TRAIN/TEST split, evolves on TRAIN, selects the champion by TRAIN
fitness, evaluates it (and the baselines `store, gzip-9, bz2-9, lzma-9, zstd-19,
brotli-11`) on the **held-out** TEST split with round-trip confirmed, prints a
comparison table, writes [`results/results.json`](results/) + a Pareto PNG, appends
to [`EXPERIMENT_LOG.md`](experiments/EXPERIMENT_LOG.md), and prints a GO/NO-GO verdict.

Useful flags: `--objective {max_ratio,ratio_at_speed,pareto}`, `--speed-floor MBps`,
`--max-len N`, `--islands N`, `--train-frac 0.7`, `--download` (fetch enwik8/Silesia),
`--max-bytes N` (truncate large files for a fast smoke), `--seed`.

## Domains (`--domain`)

| domain           | data                                                        | offline? |
|------------------|-------------------------------------------------------------|----------|
| `time-series`    | correlated float32 telemetry channels (**default**)         | ✅ generated |
| `server-logs`    | synthetic Apache-combined access logs                       | ✅ generated |
| `genomic-fastq`  | synthetic FASTQ reads (ACGT + Phred quality)                | ✅ generated |
| `ml-weights`     | float32 tensors with low-rank structure                     | ✅ generated |
| `generic-text`   | **enwik8** (`--download`) or a Zipf-text fallback           | ⬇️ / fallback |
| `generic-binary` | **Silesia** corpus (`--download`) or structured-binary fallback | ⬇️ / fallback |

All synthetic corpora are deterministic (seeded numpy), so they regenerate
byte-for-byte. The default smoke needs **no network**.

## Repo layout

```
evocompress/
  transforms.py   reversible transforms; each has forward()/inverse() + a property test
  backends.py     store / zlib / gzip / bz2 / lzma / zstd / brotli wrappers (graceful degrade)
  pipeline.py     Pipeline = transforms + backend + level; self-describing header; exact decode
  evaluator.py    score(): ratio + enc/dec MB/s + roundtrip_ok (the hard gate)
  genome.py       Genome <-> Pipeline; SearchSpace (catalog + param samplers)
  engine.py       classical GA: tournament, crossover, mutation, elitism, islands, HoF, early stop
  fitness.py      pluggable objective + speed-floor & complexity penalties
  report.py       champion vs baselines on held-out -> table + results.json + Pareto PNG
  openevolve_adapter.py   OPTIONAL evaluate(spec)->metrics stub for LLM-driven search
data/fetch_data.py        generate/fetch corpora; TRAIN/TEST split helper
experiments/run_spike.py  the end-to-end driver (the CLI above)
tests/                    round-trip + pipeline + evaluator + engine tests
```

## How the pipeline stays exactly reversible

`encode` applies the transforms in order, then backend-compresses. The output blob is
**self-describing**:

```
MAGIC(4) | VERSION(1) | HEADER_LEN(uint32) | HEADER_JSON | PAYLOAD
```

`HEADER_JSON` records the transform sequence (names + params), backend and level, so
`Pipeline.decode_blob(blob)` reconstructs and inverts the pipeline **from the bytes
alone**. Length-changing transforms (RLE, BWT, bitpack, LZ77) additionally embed their
own small headers, so each inverse needs nothing external.

## GO / NO-GO criteria

On **held-out** domain data, this spike is a **GO** (worth continuing past 2 weeks) if:

- the evolved pipeline beats the best general baseline (`zstd-19`/`brotli-11`)
  **ratio by ≥ 5%** with byte-exact round-trip, **or**
- it **matches** the best baseline ratio at **≥ 2× decode throughput**, **or**
- on at least one real domain corpus, it beats the domain-standard tool.

It is a **NO-GO** if, after a reasonable search, the evolved pipelines never beat plain
`zstd-19` on held-out data — that means the approach isn't adding value on this domain.
**A clean negative result is a valid outcome and is reported honestly either way.**

## About the Hutter Prize

The prize is the long-term motivation, but be clear-eyed:

- **What wins it:** context-mixing models (PAQ/cmix-class) that learn a per-symbol
  probability model. The current record compresses enwik9 ~9×.
- **What this spike does:** searches *preprocessing pipelines* in front of
  general-purpose LZ/entropy backends, which top out around ~3.5–4× on English text.
- **Therefore:** evo-compress **cannot win the Hutter Prize as built.** The realistic,
  testable question for the spike is principle-aligned: *does evolutionary search add
  value over strong baselines on a chosen domain?*
- **The bridge:** the prize would require adding a *learned context-mixing backend* as
  a pluggable codec and evolving its hyper-parameters / mixing structure (CPU-only,
  ≤10 GB RAM, open source). That is a far larger effort and the natural "if GO, then…"
  next phase. The pipeline/evaluator architecture here is designed to host exactly that.

### Prize track: `cmcore` (context-mixing codec, evolved as Rust source)

This bridge is now under construction in [`cmcore/`](cmcore/). The evolvable artifact
is the **Rust source of a bitwise context-mixing arithmetic coder** — compiled and
scored each round (AlphaEvolve-style). The model learns online, so **no weights are
stored** (the decompressor is just code), which matches the prize's
`size = |compressed| + |decompressor|` rule. The arithmetic coder and I/O are fixed;
the LLM evolves the `Predictor` inside a marked `EVOLVE-BLOCK`. Round-trip is
structural (encoder/decoder share the update), so every candidate is exactly lossless.

**30 LLM evolution steps** took cmcore from 2.20 to **1.775 bpc** on a 1 MB enwik8
slice (full trajectory in `experiments/HUTTER_NOTES.md` (internal notes, not published)).
Key wins: match models, word/prev-word/capitalization models, SSE/APM chains, a
**nonstationary bit-history (indirect) model**, **hash-collision checksums**, a
**mixture-of-experts + online neural (MLP) mixer**, an **XML/wiki structure model**,
**indirect byte-prediction models**, a **dictionary pre-pass (the fx2-cmix lever)**,
and an **echo-state reservoir** for cheap temporal memory. Several swings were
*reverted* after the round-trip+bpc gate showed no gain (2-byte dictionary codes,
an 8-mixer MoE, an Adam optimizer).

On enwik8 (round-trip byte-exact), cmcore reaches **1.6749 bpc on a 16 MB slice**
(ratio 4.78×; full 100 MB projects to ~1.61) — beating every general-purpose
compressor: lzma-9 1.9892 (−16%), zstd-19 2.1550 (−22%), brotli-11 2.1636 (−22%),
gzip-9 2.9181 (−43%). The research frontier (cmix ~1.0–1.1, the enwik9 record
**0.886**) is ~1.6–1.9× away — closing it needs a large trained **LSTM**, the one
lever beyond afternoon-scale tweaks. Run it with
`python -m experiments.run_hutter --slice enwik8_1mb`.

<!-- SUMMARY:START -->
## Summary of the spike run

**Domain:** `time-series` (synthetic, 18 devices × 5 correlated float32 channels with
trend/seasonality/noise). **Search:** classical GA, `--population 80 --generations 40
--seed 0 --objective max_ratio`, 597 evaluations, 13 TRAIN / 5 HELD-OUT files.
*(The lighter `make smoke` 40×20 run finds the **same** champion — robust, not seed-luck.)*

### Held-out test results (the headline)

| method                  | ratio | decode MB/s | out bytes | round-trip |
|-------------------------|------:|------------:|----------:|:----------:|
| **evolved** `float_split(f4) → lzma-4` | **1.6045** | ~14 | 255,283 | ✅ byte-exact |
| lzma-9                  | 1.5077 | ~13 | 271,669 | ✅ |
| brotli-11               | 1.4112 | ~59 | 290,246 | ✅ |
| zstd-19                 | 1.3296 | ~278 | 308,055 | ✅ |
| bz2-9                   | 1.2828 | ~16 | 319,310 | ✅ |
| gzip-9                  | 1.2749 | ~89 | 321,276 | ✅ |
| store                   | 0.9993 | — | 409,870 | ✅ |

*(Decode MB/s varies run-to-run with machine load; ratios are deterministic.)*

### Answering the spike's questions

- **Did the evolved pipeline beat the best baseline ratio on held-out data?** **Yes.**
  It beats the best *general* baseline `zstd-19` by **+20.7%**, beats `brotli-11` by
  **+13.7%**, and beats the strongest baseline of all, `lzma-9`, by **+6.4%** — with
  byte-exact round-trip on every held-out file. → **Verdict: GO.**
- **At what speed?** Slowly: decode ~14 MB/s (LZMA-class), ~20× slower than `zstd-19`.
  On the ratio↔speed Pareto frontier it is the **ratio-optimal** point, not the
  speed-optimal one. Switch to `--objective ratio_at_speed --speed-floor 100` to make
  the GA prefer zstd-backed pipelines instead.
- **What transforms did it pick?** A single preprocessing pass: **`float_split(f4)`**
  (split each float32 into a high-half *sign+exponent+high-mantissa* plane and a
  low-half *noisy-mantissa* plane) followed by **LZMA at preset 4**. The insight the
  search found: separating the structured exponent bytes from the near-random low
  mantissa gives LZMA exploitable runs — so even LZMA *preset 4* on the preprocessed
  stream beats LZMA *preset 9* on the raw bytes. **This is the value-add of pipeline
  search: the preprocessing, not the codec, is what wins.**

### Honest caveats

- The synthetic corpus has genuine per-sample mantissa noise, which caps the
  achievable ratio (~1.6×). Real telemetry (quantized sensors, repeated states) would
  compress much more, and the *relative* baseline gap is the meaningful signal.
- The champion is a 1-transform pipeline; deeper `delta`/`transpose` stacks didn't win
  here, partly because the catalog's `transpose` strides don't align with the 20-byte
  (5-channel) record. A channel-aware stride is an obvious next experiment.
- This is a **GO on the time-series domain**, not a universal claim. Re-run other
  domains (`--domain generic-text --download`, etc.) before generalizing.

See [`EXPERIMENT_LOG.md`](experiments/EXPERIMENT_LOG.md), [`results/results.json`](results/results.json),
[`results/champion.json`](results/champion.json), and [`results/pareto.png`](results/pareto.png).
<!-- SUMMARY:END -->

## License

MIT (see `LICENSE`).
