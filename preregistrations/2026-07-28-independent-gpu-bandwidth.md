# Pre-registration #44: is llama.cpp the limit, or is the card? — the first GPU measurement NOT through llama.cpp

**Author:** Federico Sciuca · **Date staked:** 2026-07-28, BEFORE the measurement. **Status: STAKED.**

## The epistemic problem this closes

Every GPU number in this project came out of llama.cpp, so "the hardware is the wall" and
"llama.cpp is the wall" have been indistinguishable on that side. On the CPU side they are NOT
indistinguishable, and that asymmetry is the point:

| side | independent measurement | llama.cpp-derived | verdict |
|---|---|---|---|
| CPU / DRAM | numpy stream: **26.1 GB/s** read, 30.4 copy (#27) | dense kernel effective **28.4–29.7 GB/s** (#31) | llama.cpp is AT hardware — corroborated |
| **GPU / VRAM** | **none, ever** | η **0.32–0.56** of the 192 GB/s spec (C-02) | **unresolved** |

C-02 has been open all project: models that fit entirely in VRAM run 0.91×–1.84× our prediction,
efficiency 0.32–0.56, six candidate explanations refuted. If the GTX 1060 really delivers ~192
GB/s to a well-written kernel, then llama.cpp's CUDA path is leaving **~2×** on the table and the
"no software lever" conclusion is wrong for the GPU half. If the card itself only delivers ~80
GB/s in practice, C-02 dissolves into hardware and the conclusion stands.

## Method — deliberately NOT llama.cpp

CuPy (bundles its own CUDA runtime; independent of ggml entirely):
1. **Pure read bandwidth**: `cupy.sum` over a large device array, r=5, best-of.
2. **Copy bandwidth**: device-to-device `copy`, counting read+write.
3. **GEMV** — the decode-shaped operation: (1 × N) @ (N × M), the matrix streamed once. This is
   what a decode step actually is, and its effective GB/s is directly comparable to llama.cpp's η.
All timed with CUDA events / explicit sync, warm-up discarded, GPU state logged.

## Stakes

- **P-1 (the card is real).** Independent pure-read bandwidth ≥ **150 GB/s** (78% of the 192 spec).
  A GTX 1060's memory controller should reach this; if it does not, the spec is the fiction and
  C-02 is explained by hardware.
- **P-2 (THE DECISIVE ONE).** Independent **GEMV** effective bandwidth ≥ **120 GB/s** — i.e.
  η ≥ 0.63 for an operation shaped like decode. That would be **above llama.cpp's measured
  0.32–0.56**, proving the runtime, not the card, owns the shortfall.
- **P-3 (scale check).** GEMV bandwidth is within ±25% of pure-read bandwidth. A large gap means
  GEMV is compute- or launch-bound rather than bandwidth-bound, and the comparison to η is
  invalid — which would itself explain C-02.

## What each outcome means, decided in advance

- **P-2 HIT** → llama.cpp's CUDA decode path is the limit, C-02 is a software gap worth ~2×, and
  "no software lever left" is **refuted for the GPU half**. That reopens the fork/PR question with
  a measured prize, on the side we never examined.
- **P-2 MISS with P-1 HIT** → the card streams fine but decode-shaped work cannot, so the limit is
  the operation, not the runtime. C-02 becomes a property of GEMV on Pascal.
- **P-1 MISS** → the 192 GB/s spec is fiction like DDR4-3000's 48 was, every VRAM-tier prediction
  in the tool is built on a wrong constant, and that is the headline.

## What ships

The first independently-grounded VRAM number in this project, and whichever of the three
conclusions above the data supports. If P-2 hits, a follow-up pre-registration targets the CUDA
decode path directly.

---

## Scored (2026-07-28, log: `weights/data/prereg44_independent_gpu.log`)

**Verdict: P-1 HIT, P-2 HIT DECISIVELY, P-3 HIT. The GTX 1060 delivers η 0.84 on a decode-shaped
operation. llama.cpp achieves 0.32–0.56. "No software lever left" is REFUTED for the GPU half.**

| measurement (CuPy — no ggml anywhere in the path) | GB/s | η vs 192 spec |
|---|---|---|
| pure read, 256 MB sum | **165.6** | 0.86 |
| copy (read+write) | 150.1 | 0.78 |
| **GEMV fp32 (1×4096)@(4096×14336) — decode-shaped** | **161.3** | **0.84** |
| GEMV fp16, same shape | 108.9 | 0.57 |

- **P-1 (read ≥150 GB/s): HIT** at 165.6. The 192 GB/s spec is *not* fiction, unlike DDR4-3000's
  48 which measured 26.1. The VRAM controller is real.
- **P-2 (GEMV ≥120 GB/s, above llama.cpp's band): HIT** at 161.3 — **1.5× to 2.6× what llama.cpp
  extracts** from the same card.
- **P-3 (GEMV within ±25% of read): HIT** at 2.6% apart. GEMV is bandwidth-bound, so the
  comparison to η is valid — the operation is not launch- or compute-limited at fp32.

### What this establishes, and what it does NOT

**Establishes:** the hardware is not the constraint on the GPU side. A well-written kernel reaches
0.84 where llama.cpp reaches 0.32–0.56. C-02 — open since the start of this project, six
explanations refuted — is a **software** gap of roughly **1.5–2.6×**, and it was invisible because
every GPU number we ever took came out of llama.cpp. The epistemic critique that prompted this was
correct.

**Does NOT establish that the gap is trivially recoverable.** cuBLAS fp32 GEMV and llama.cpp's
quantized decode are not the same operation: llama.cpp streams Q2_K/Q4_K blocks and dequantises
them in-kernel, which is work cuBLAS never does. The fp16 row is the warning — 108.9 GB/s, well
below fp32's 161.3, because Pascal's fp16 ALU throughput is 1/64 rate. **On this architecture the
arithmetic format can bind before bandwidth does**, and quantized dequant may do the same.

So the honest statement is: **the ceiling is 0.84, not 0.35–0.56; how much of that 1.5–2.6× a
quantized kernel can actually reach is the next measurement, not this one.**

### The discriminating follow-up, specified now

Measure llama.cpp's all-in-VRAM η against **format**, holding model and size fixed, on a model
large enough that fixed overheads do not dominate — Q8_0 (near-trivial dequant) vs Q4_K_M vs Q2_K.
If η rises toward 0.84 as dequant gets cheaper, the gap is **arithmetic** and Pascal owns it. If η
stays ~0.4 at Q8_0, the gap is the **kernel** and it is addressable. C-02's existing data hints at
the awkward answer — Q8_0 measured the LOWEST η (0.354) — but those were 0.5–0.6B models where
per-token overhead dominates, so the test must be re-run at 7B+.

**Wired into:** `findings/REGISTER.json:C-02` (promoted from "unexplained" to "software gap, size
bounded, mechanism open") · `findings/REGISTER.json:L-14` (the independent VRAM ceiling).

### Addendum, same session: the discriminator ran, and it NARROWS the claim above

The format ladder (same model, all-in-VRAM, one session, r=3) that the "next measurement" section
specified — it needed no CUDA toolkit, only the shipped binary:

| format | bits | tg32 | GB/s effective | η |
|---|---|---|---|---|
| Q2_K | 2.8 | 21.26 | 59.7 | 0.311 |
| IQ3_XS | 3.4 | 20.44 | 63.7 | 0.332 |
| IQ3_M | 3.6 | 19.76 | 65.8 | 0.343 |
| **Q4_K_M** | 4.8 | 22.60 | **98.6** | **0.513** |

**η climbs monotonically with bit-width** — 0.311 → 0.513 — exactly as the dequantisation
hypothesis predicts: fewer bits means more unpacking work per delivered byte, so the byte saving
is partly cancelled. This is **C-05's sixth instance** ("a quantized byte is not a byte"), now on
the VRAM tier. Note the practical inversion it produces: Q4_K_M carries **55% more bytes** than
Q2_K and is nonetheless **faster** (22.60 vs 21.26) — which the tool already tells users, from #16.

**And it forces me to narrow the headline I wrote above.** I compared llama.cpp against the
**fp32** GEMV ceiling (161.3 GB/s) and called the gap 1.5–2.6×. But the independent **fp16** GEMV
measured **108.9 GB/s** on this card, because Pascal's fp16 ALU runs at 1/64 rate — and
llama.cpp's Q4_K_M reaches **98.6 GB/s, which is 91% of that fp16 ceiling.**

So the size of the software gap depends entirely on which ceiling is the right comparand, and that
depends on the arithmetic llama.cpp's quantized CUDA kernels actually use — which I have not
determined:

| if the kernel's binding arithmetic is… | llama.cpp Q4_K_M sits at | remaining software headroom |
|---|---|---|
| fp32 (161.3 GB/s) | 61% | up to 1.64× |
| **fp16 (108.9 GB/s)** | **91%** | **~1.1×, i.e. nearly none** |

**Corrected conclusion.** The card is definitively not limited to η 0.32–0.56 — that part stands,
and it is why C-02 is reclassified. But "1.5–2.6× of software headroom" was an overclaim resting
on an unexamined choice of comparand. The honest range is **1.1× to 1.64× at Q4_K_M**, and closing
it requires reading which arithmetic path ggml's `dequantize_mul_mat_vec` / MMQ kernels take on
sm_61 — a source question, answerable without hardware.

**Wired into:** `findings/REGISTER.json:L-14` (ceiling qualified by arithmetic) · `C-02`
(headroom range corrected) · `C-05` (sixth instance).
