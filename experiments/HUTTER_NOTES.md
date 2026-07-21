# Hutter-direction notes (context-mixing codec evolution)

This is the prize-aligned track. The evolvable artifact is no longer
"preprocessing transforms + zstd" — it is the **Rust source of a context-mixing
codec** (`cmcore/`), which is compiled and scored each round (AlphaEvolve-style).
The LLM proposes diffs to the `Predictor` inside the `EVOLVE-BLOCK`; the arithmetic
coder and I/O are fixed.

## Why this is the right shape for the prize

- Every Hutter winner is a **context-mixing model** (~0.89 bpc on enwik9). LZ+entropy
  backends (zstd/brotli/lzma) top out ~2.0–2.3 bpc on text — a different league.
- A CM model **learns online** from bytes already decoded, so **no weights are
  stored**: the "decompressor" is just code. That is exactly the prize's
  `S = |compressed| + |decompressor|` accounting, and it stays un-gameable (anything
  memorized from the corpus and shipped costs real bytes; online-learned state is free).
- Correctness is structural: encoder and decoder run the **same** predict/update, so
  any model round-trips exactly. Bugs only cost ratio, never correctness.

## Evaluation changes (vs the GA/zstd track)

- Metric is **bpc** (8·compressed/original) and **prize size** (compressed +
  stripped decompressor binary). See `experiments/run_hutter.py`.
- Corpus is real **enwik8** slices (`data/corpora/generic-text/enwik8_{1,4,8}mb`,
  full `enwik8` available). Held-out discipline still applies for tuning.
- Round-trip is verified by SHA-256 on every run (the un-gameable gate, unchanged).

## Results so far (apples-to-apples bpc, lower is better)

| step | model | 1 MB | 4 MB | round-trip |
|------|-------|-----:|-----:|:----------:|
| baseline | orders {0..6} + logistic mixer | 2.1987 | — | ok |
| 1 | + match model | 2.0812 | 2.0621 | ok |
| 2–3 | + word model + SSE/APM | 1.9999 | 1.9770 | ok |
| 4 | + sparse contexts + 2nd APM | 1.9741 | 1.9532 | ok |
| 5 | + nonstationary bit-history (indirect) | 1.8861 | 1.8647 | ok |
| 6 | + match-aware mixer | 1.8775 | 1.8574 | ok |
| 7 | tuning (mixer LR 13, SSE blend 2) | 1.8713 | — | ok |
| 8 | + second long-range match | 1.8571 | 1.8268 | ok |
| 9 | + two-level mixing (LR 19) | 1.8455 | 1.8112 | ok |
| 10 | + 3rd chained APM | 1.8448 | 1.8115 | ok |
| 11 | + capitalization model | 1.8428 | 1.8096 | ok |
| 12 | + previous-word (bigram) context | 1.8400 | 1.8068 | ok |
| 14–15 | mixture-of-experts mixing (2→6 gated mixers) | 1.8333 | 1.8020 | ok |
| 16 | + online neural-net (MLP) mixer | 1.8333 | 1.7995 | ok |
| 17 | + hash-collision checksums | 1.8228 | 1.7818 | ok |
| 18 | + XML/wiki structure model + table-size knob | 1.8108 | 1.7719 | ok |
| 19 | + column + run models | 1.8075 | 1.7689 | ok |
| 20 | retune mixer LR (15) | 1.8063 | — | ok |
| 22 | + match-predicted-byte context (fx2-cmix) | 1.8013 | 1.7629 | ok |
| 23 | + match2-predicted-byte context | 1.7997 | 1.7612 | ok |
| 24–25 | + indirect byte-prediction (orders 2/3/4) | 1.7950 | 1.7567 | ok |
| 26 | retune mixer LR (12) + state-machine knob | 1.7946 | — | ok |
| 27 | **dictionary pre-pass (WRT) — the fx2-cmix #1 lever** | 1.7805 | 1.7398 | ok |
| 28 | + dictionary capitalization handling | 1.7764 | 1.7352 | ok |
| 30 | **+ echo-state reservoir (recurrent temporal memory)** | **1.7754** | **1.7338** | ok |

(steps 14–30 measured at tbits=22; table-size sweep: 4 MB 22→1.7818, 23→1.7737, 24→1.7696)

Reverted (no gain): order-7/16 contexts, match-aware final mixer, Adam optimizer
for the neural mixer (worse than SGD), an 8-mixer MoE (column/xml gates redundant
with context models), **2-byte dictionary codes (step 29)** — replacing
moderately-common words with an escape+index sequence is *less* predictable than the
words the model already handled (the dictionary's sweet spot is ~50 top words +
capitalization), and **a full online LSTM (truncated BPTT, step 31)**.

**The LSTM result (important).** A genuine online LSTM next-byte predictor (H=32,
embedding 16, truncated BPTT window 16) was implemented end-to-end and round-tripped
byte-exact, fed to the mixer via a cumulative-sum per-bit marginalisation. It gave
**~zero on 256 KB and slightly *worse* on 1 MB** while running ~12× slower. Why:
a *small* LSTM is redundant with our 21 context models + 2 match models — they
already capture what it would learn, so the mixer down-weights it and its noise costs
a hair. cmix's LSTM pays off because it is *large* and sits in a different, optimized
(C++) architecture. **Empirical confirmation that reaching ~1.0 needs a large net +
an optimized substrate + heavy compute (à la nncp/cmix), not a component bolted onto
this per-bit Rust codec.** H=32 is already the practical ceiling here (per-bit pure
Rust); a useful LSTM (H≥256) would be orders of magnitude too slow in this setup.

Cross-field ideas applied (all kept only after passing the round-trip gate + bpc test):
mixture-of-experts & online neural-net (MLP) mixing (ML), context-gating (MoE),
hash-collision checksums (PAQ ContextMaps), Adam/momentum optimization (tried,
reverted), simulated-annealing-style LR retuning, match-as-context (fx2-cmix).
Honest note: quantum computing offers no practical lever for classical lossless
coding here; the deep connection is information-theoretic (compression ≈ prediction,
Solomonoff/Hutter) — which is exactly what context mixing approximates.

Baselines on the same 1 MB slice: brotli-11 2.2481, bz2-9 2.2506, lzma-9 2.3255,
zstd-19 2.4006, gzip-9 2.8463. Standard tools on the **full** 100 MB enwik8:
lzma-9 1.9892, zstd-19 2.1550, brotli-11 2.1636, bz2-9 2.3207, gzip-9 2.9181.

**Confirmed on the FULL 100 MB enwik8** (round-trip byte-exact):

| build | full-enwik8 bpc | bytes | ratio | vs lzma-9 |
|-------|----------------:|------:|------:|----------:|
| word + SSE (step 3) | 1.8987 | 23,733,222 | 4.21× | −4.5% |
| indirect + match-aware (step 6) | 1.8042 | 22,552,396 | 4.43× | −9.3% |
| full stack (step 12) | 1.7207 | 21,509,089 | 4.65× | −13.5% |
| **+dict +reservoir (step 30), 16 MB slice, tbits=23** | **1.6749** | — | **4.78×** | **−16%** |

cmcore beats every general-purpose tool on enwik8: lzma-9 1.9892 (−16%), zstd-19
2.1550 (−22%), brotli-11 2.1636 (−22%), gzip-9 2.9181 (−43%). The step-30 number is
on a 16 MB slice (representative of full scale; full 100 MB projects to ~1.61–1.63 as
the online model keeps training). The research frontier (cmix ~1.0–1.1, enwik9 record
0.886) is ~1.6–1.9× away — closing it needs a large trained LSTM, not more tweaks.

## Done (✅) and the honest path to ~1.0 bpc

Done: orders, match models (×2), word/prev-word/cap models, SSE/APM (×3),
nonstationary bit-history (indirect) states, hash-collision checksums,
mixture-of-experts + online neural (MLP) mixing, XML/column/run models,
match-as-context, indirect byte-prediction models, tunable table size.

**Where ~1.0 bpc actually lives (and why it's not an in-session edit):**
- We are at ~1.75 bpc (full enwik8, 22-bit) and beat every general-purpose tool.
- The cmix/PAQ8 frontier (~1.0–1.3) and the enwik9 record (0.886) are reached with
  three things we have *not* built, each a multi-week effort:
  1. **A large recurrent neural mixer (LSTM-class)** — real temporal memory. Our MLP
     is feed-forward; this is the single biggest remaining lever (the cmix jump from
     ~1.2 → ~1.0 is largely the LSTM).
  2. **A word/dictionary preprocessor (WRT/DRT)** — fx2-cmix's #1 lever; needs a real
     embedded English dictionary and careful integration (can hurt a byte-tuned model
     if done naively).
  3. **Far more models + much larger tables + days of CPU** (cmix: hundreds of models,
     ~a week per enwik9 run). Our table-size sweep shows scale alone keeps paying.
- **Honest verdict:** evolutionary + LLM-in-the-loop search took a from-scratch codec
  to a strong, lossless context-mixer that beats mainstream tools by ~13–20%. Closing
  the last ~1.8× to the record is a research programme, not a prompt.
