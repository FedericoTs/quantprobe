# Experiment Log

Each `run_spike` invocation appends an entry below (newest at the bottom).  The
headline number is always the **held-out** compression ratio of the evolved
champion vs the best general-purpose baseline.

Template / fields per entry:

- **config**: population, generations, objective, max pipeline length, #train/#test files, total evaluations
- **champion**: the evolved pipeline (`transforms => backend-level`)
- **HELD-OUT ratio**: evolved vs the strongest general baseline (and the overall best baseline)
- **decode MB/s**: evolved vs that baseline
- **verdict**: GO / MARGINAL / NO-GO and the reason

---
<!-- run_spike appends entries here -->

### 2026-06-03 01:09:02 | domain=time-series engine=ga seed=0

- config: pop=40 gen=20 objective=max_ratio max_len=5 train=13 test=5 files, evals=331
- champion: `[float_split(dtype=f4)] => lzma-4`
- HELD-OUT ratio: evolved=1.6045 vs zstd-19=1.3296 (best baseline lzma-9=1.5077)
- decode MB/s: evolved=5.8 vs zstd-19=174.7
- **verdict: GO** -- evolved ratio 1.6045 beats zstd-19 1.3296 by >=5%

### 2026-06-03 01:16:36 | domain=time-series engine=ga seed=0

- config: pop=80 gen=40 objective=max_ratio max_len=5 train=13 test=5 files, evals=597
- champion: `[float_split(dtype=f4)] => lzma-4`
- HELD-OUT ratio: evolved=1.6045 vs zstd-19=1.3296 (best baseline lzma-9=1.5077)
- decode MB/s: evolved=13.7 vs zstd-19=278.4
- **verdict: GO** -- evolved ratio 1.6045 beats zstd-19 1.3296 by >=5%

### 2026-06-03 01:46:20 | HUTTER cmcore | slice=enwik8_1mb

- cmcore bpc=2.0812 (rt=ok), best baseline brotli-11=2.2481
- cmcore 2.0812 bpc beats best baseline brotli-11 2.2481 by 7.4%

### 2026-06-03 01:48:00 | HUTTER cmcore | slice=enwik8_1mb

- cmcore bpc=2.0812 (rt=ok), best baseline brotli-11=2.2481
- cmcore 2.0812 bpc beats best baseline brotli-11 2.2481 by 7.4%

### 2026-06-03 | HUTTER cmcore FULL enwik8 (100MB)
- match-only build: 1.9996 bpc (24,995,483 B), round-trip OK -- tie with lzma-9
- **word+SSE build: 1.8987 bpc (23,733,222 B), round-trip OK -- beats lzma-9 (1.9892) by 4.5%**
- beats all general tools: zstd-19 2.1550, brotli-11 2.1636, gzip-9 2.9181
- frontier remains: cmix ~1.0-1.1, enwik9 record 0.886 bpc (~2x away)

### 2026-06-03 | HUTTER cmcore FULL enwik8 (100MB) -- evolved stack (step 12)
- **1.7207 bpc (21,509,089 B), ratio 4.65x, round-trip OK**
- beats lzma-9 1.9892 (-13.5%), zstd-19 2.1550 (-20%), brotli-11 2.1636 (-20%)
- trajectory at full scale: step3 1.8987 -> step6 1.8042 -> step12 1.7207
- slice trajectory (1MB): 2.1987 -> 1.8400 over 12 LLM evolution steps
- frontier: cmix ~1.0-1.1, enwik9 record 0.886 bpc (~1.9x away)
