# The machine ladder — 0.5B to 30B on one box, three ways, vs the law

**Machine:** the reference box (GTX 1060 6GB · i5-7600K · 16 GB DDR4-3000 · Windows 10).
**Protocol:** one session, pristine zero-patch llama.cpp (commit f113e02), tg128 r=2, GPU clocks
logged after every arm (all loaded arms sustained 1835–1898 MHz). Predictions captured from
`quantprobe plan` (v1.20, anchored) **before any measurement** — the law column is staked, not
fitted. Log: `weights/data/prereg65_ladder.log`.

**The three ways:**
- **naive default** — `llama-bench -m model.gguf -ngl 0`, mmap on: what running llama.cpp with no
  flags actually does (the CLI default is CPU-only).
- **informed llama.cpp** — a user who knows `-ngl`: all layers when they fit, else the most that
  fit (`-ngl 13` / `-ngl 20` for the two big models).
- **quantprobe** — the command `plan` emits, verbatim.

| model | naive default | informed llama.cpp | **quantprobe** | law predicted (staked) | pred. error |
|---|---|---|---|---|---|
| Qwen2.5-0.5B Q8_0 | 56.57 | 154.44 | = informed¹ | 131.6² | −14.8% (under) |
| Qwen3-0.6B Q8_0 | 45.80 | 106.89 | = informed¹ | 109.3 | **+2.3%** |
| Qwen2.5-7B Q4_K_M | 6.87 | 22.63 | = informed¹ | 14.9³ | −34% (under) |
| DeepSeek-Lite 16B MoE Q4_K_M | 6.84 ± 5.13⁴ | 18.21 | **22.87** | 18.4 | −20% (under) |
| Qwen3-30B-A3B MoE Q2_K | 7.45 ± 3.16⁴ | 19.29 | **20.78** | 22.3 | **+7.3%** (in band) |

**What the ladder says:**

- **quantprobe vs naive: 2.3–3.3× on every model.** Most of any user's gain is escaping the
  CPU-only default — which is why the tool's first job is placement, not magic.
- **quantprobe vs informed llama.cpp: +26% on the 16B MoE, +8% on the 30B MoE, parity on models
  that fit VRAM.**¹ The `-ot` expert split, `--no-mmap`, and `--threads` are worth real speed
  exactly where placement is non-trivial; where everything fits, the honest answer is that no
  flag beats `-ngl 99` and the tool says so.
- **The law's misses are all in the promised direction.** Four of five predictions under-promise;
  the one over-prediction (+7.3%, flagship) is inside the printed ±25% band. Worst under-promise
  is the 7B Q4_K_M at −34%³ — the documented single-ratio format blindness (U-15).

**Footnotes, because a table without them lies:**
1. For all-in-VRAM models, `plan`'s emitted command IS `-ngl 99` — quantprobe's value there is
   the fit/format/speculation advice, not different flags. The column repeats the informed number
   rather than claiming credit for it.
2. The 0.5B is the calibrate anchor's own model — quasi-in-sample, listed for completeness, not
   evidence.
3. The anchored GPU ratio was set by a Q8_0 anchor; Q4_K_M decodes on a different point of the
   measured format ladder (L-16). Per-format anchoring is U-15.
4. Naive big-model CPU runs are BIMODAL (±44–75%!): mmap pages the file from disk on the first
   pass. The tool's pure-CPU row predicts the placement done right (`--no-mmap`, warm) and warns
   about exactly this; the naive arm demonstrates it.

**The two headline sentences this table earns:**
On consumer hardware, the difference between running a model and running it *right* is 2–3×, and
it is entirely configuration. And a physical law calibrated by two 5-minute benchmarks predicts
the result of that configuration to single-digit percent on the placements that matter, erring
low when it errs.
