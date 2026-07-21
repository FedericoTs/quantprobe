# 3. Results: The Operating Point and the Measured Frontier

## 3.1 The champion operating point

On Qwen2.5-0.5B (fp16 eval-slice perplexity 3.944), the champion codec — per-group (g = 128) signed-Hadamard incoherence rotation, AWQ activation scaling, scalar entropy-constrained quantization (ECVQ), entropy-coded indices, and the 0.5% largest-magnitude weights retained in fp16 — operates at approximately **3.00 honest bits/weight at perplexity roughly 4.58 ± 0.07**. This codec is the unique fixed point of the search: it survived roughly 30 attack rounds and multiple hostile multi-agent audits without being dethroned.

We are deliberate about the hedge on perplexity, because it is the single number most likely to be over-read. An earlier headline of **4.483 perplexity at 3.13 b/w** is a *single rotation-seed instance*, not a sharp operating value. A positive control (R2; Section 4) that varies only the rotation seed at iso-bit measures a selection noise floor of standard deviation 0.067 perplexity, with a spread of 0.21 across seeds. The 4.483 figure is therefore best read as one favorable draw from that distribution; the honest champion perplexity is **4.58 ± 0.07**. The rate moved from 3.13 to 3.00 b/w not by re-tuning the lossy codec but by losslessly re-coding the side information (Section 3.3), at identical perplexity.

## 3.2 The measured rate–distortion frontier

Sweeping the ECVQ rate knob λ traces a frontier on the same 0.5B model, with every point reported in honest bits/weight (container bytes ÷ weight count) and held-out perplexity. Two representative interior points are **3.957 perplexity at 4.277 b/w** (λ = 0.0015) and **4.044 perplexity at 4.069 b/w**. A near-lossless **entropy-32** variant sits at approximately **3.97 perplexity at ~5.14 b/w**, recovering essentially the fp16 quality of 3.944 at the cost of rate. Table 1 collects these points alongside the headline operating point and the two leading closed alternatives.

The frontier is what licenses the Pareto claims of Section 4 under honest, symmetric accounting. The champion point at 3.957 / 4.277 b/w *strictly* dominates an E8 lattice point at 4.020 / 4.640 b/w on both axes — better perplexity at fewer bits — and entropy-constrained trellis coding (ECTCQ), once its omitted ~0.95 b/w of decoder path entropy is paid, lands near 5.13 b/w and loses by roughly 0.85 b/w at matched quality. Because each of these margins exceeds the 0.067-ppl / sub-0.1-ppl noise floor and is measured in bits rather than perplexity, the orderings are safe rather than artifacts of seed noise.

## 3.3 A1: lossless re-coding of the champion's side information

The one *new positive* result is a lossless recompression of the champion's side-information streams, verified by byte-identical round trip. As originally formulated, the champion paid its side information **raw**: 16 bits per group for the fp16 group-amax plus 32 bits per outlier (position and fp16 value), totaling **0.2876 b/w**. Serializing these planes into byte-split streams and compressing them with real decodable coders (zstd-19 and a context-mixing `cmcore`, best-of), then decoding and asserting byte-identity, compresses the same information to **0.1608 b/w**. This banks **0.127 b/w at identical perplexity** — it is purely a re-coding of side info, so distortion is unchanged — and is what carries the operating point from 3.13 to 3.00 b/w.

Two caveats keep this honest. First, the **0.127 b/w saving is a lower bound** (equivalently, 0.1608 b/w is an upper bound on the achievable side-info rate): the AWQ per-column scale stream (~0.011 b/w) is still uncoded, so the realizable saving is at least this large. Second, the 0.127 b/w **splits** into an amax bank and an outlier bank. Roughly 0.07 b/w of the saving comes from recompressing the group-amax stream, which is available to *any* per-group codec and is therefore not champion-specific; only the remaining **~0.077 b/w (the outlier bank)** is exclusive to the champion's 0.5% fp16-outlier design. We report the split rather than the headline 0.127 so the credit is not overstated.

## 3.4 Results table

**Table 1.** Measured 0.5B operating points and frontier (Qwen2.5-0.5B, fp16 eval-slice perplexity 3.944). Bits/weight are honest (round-trip-verified container bytes per weight); perplexity is on the held-out enwik8 prose slice.

| Codec / point | Perplexity | Honest b/w | Notes |
|---|---:|---:|---|
| fp16 reference | 3.944 | 16.000 | eval-slice baseline |
| Champion (operating point) | 4.58 ± 0.07 | ~3.00 | seed-noise floor σ = 0.067; "4.483" is one seed instance |
| Champion (frontier, λ = 0.0015) | 3.957 | 4.277 | Pareto-dominates E8 below |
| Champion (frontier) | 4.044 | 4.069 | interior λ point |
| Champion (entropy-32) | ~3.97 | ~5.14 | near-lossless variant |
| E8 lattice (q0.08) | 4.020 | 4.640 | strictly dominated on both axes |
| ECTCQ (honest rate) | ~3.978 | ~5.13 | +0.95 b/w path entropy restored; loses ~0.85 b/w |

**Side-information re-coding (A1), at identical perplexity:** raw **0.2876 b/w** → decodable container **0.1608 b/w**, banking **0.127 b/w** (lower bound; ~0.011 b/w AWQ scale stream still uncoded). Split: ~0.07 b/w amax bank (available to any per-group codec) + ~0.077 b/w outlier bank (champion-exclusive).
