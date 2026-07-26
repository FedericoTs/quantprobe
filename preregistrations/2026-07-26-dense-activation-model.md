# Pre-registration #17: the dense activation model prices every parameter as protected

**Author:** Federico Sciuca · **Date staked:** 2026-07-26, BEFORE the discriminating measurement.
**Status: STAKED.**

## How this surfaced

A user reading the published browser calculator noticed an implausible pair on the same hardware:

| model | on DGX Spark | |
|---|---|---|
| **Gemma 4 12B** (dense, 11.9B active) | **28.0 tok/s** | |
| **GLM-4.5-Air 106B** (MoE, 12B active) | **42.4 tok/s** | at 2.5 bits |

A 12B dense predicted **slower** than a 106B MoE with the *same* active parameter count. That is
not obviously wrong — MoE reads fewer bytes per token than its total suggests — but it inverts
the intuition badly enough to be worth checking. It is a real defect.

## The mechanism

Law 4 prices the always-active parameters at the depth-aware recipe's protected precision:

```python
ab = max(bits, 4.5)                    # attention held at ~4-bit by the recipe
act = ne * ab/8 + (a - ne) * bits/8    # protected part, then the rest
```

For a **MoE** this is right: `ne` is attention + shared experts, genuinely the protected set, and
the routed experts scale with `bits`. For a **dense** model the tables set `ne = t` — every
parameter is always-active, which is true for *activation* and false for *quantization*. The
consequence is that the whole dense model is priced at ≥4.5 bits, so:

> **A dense model's predicted speed does not respond to quantization at all.** Gemma 4 12B is
> predicted at 7.70 GB/token — and therefore the same tok/s — at 2.5 bits and at 4.5 bits.

That is indefensible independently of any measurement, and it is why a 106B MoE overtakes it.

The recipe protects `attn_.*` and `ssm_.*` (and `shexp`/`nextn` on MoE) — **not** embeddings and
not the FFN. Measured from real GGUF tensor shapes, the attention share of a dense model is
**10.8% (Qwen2.5-7B), 20.2% (gemma4-12B), 25.1% (Qwen3.5-4B), 29.6% (Qwen3-0.6B)**.

## The proposed correction

For dense models only, protect a fraction of the parameters rather than all of them:

```python
prot = t * DENSE_PROTECTED_SHARE       # 0.214, the mean measured attention share
act  = (prot * ab/8 + (a - prot) * bits/8) * 1.15
```

MoE is untouched. Scored against the six dense all-in-VRAM points already measured on the
reference box, mean |error| falls **18% → 10%**, with the sub-4.5-bit points carrying the gain
(7B Q2_K −29% → +1%; 7B IQ3_XS −25% → −5%). Reading each model's *true* attention share from its
GGUF instead of using the constant scores 11% — no better — so the constant ships and the extra
machinery does not.

## The discriminating stake

Those six points are what the constant was calibrated on, so they cannot validate it.
**`Qwen2.5-7B-Instruct-IQ3_M`** is on disk, has never been benchmarked, and is not in that set.
3.57 GB → **3.75 effective bits**, all in VRAM, tg128, r=3.

| | prediction |
|---|---|
| current model | **13.6 tok/s** |
| corrected model | **15.7 tok/s** |

- **P-1 (direction).** Measured lands **above 15.7** — both models under-predict, consistent with
  the known one-directional in-VRAM pessimism (pre-registration #15, unresolved).
- **P-2 (the stake).** The corrected model is **closer to measurement than the current one**:
  `|15.7 − m| < |13.6 − m|`. This is the claim that decides whether the correction ships.
- **P-3 (bracketing).** Measured lands in **17.5–20.5**, bracketed by its own siblings on the same
  card: Q2_K (2.8 bits) 19.17, IQ3_XS (3.3 bits) 18.11, Q4_K_M (4.5 bits) 20.03.
- **P-4 (no anchor moves).** All four published anchors are MoE and must be **bit-identical**.
  If any moves, the change is wrong and does not ship.

## Refuted if

P-2 fails — i.e. the measurement lands below ~14.6, where the current model is closer. That would
mean dense decode really is insensitive to bit-width in a way the byte model accidentally
captured, and the structural argument would need rethinking rather than patching.

Note that P-2 can hold while the prediction is still poor: the in-VRAM regime is known pessimistic
by 2–67% and this correction does not claim to fix that. It claims only that a dense model's speed
should respond to its quantization at all.

---

## Scored (2026-07-26, log: `weights/data/prereg17_dense_activation.log`)

**Verdict: P-1 HIT, P-2 HIT decisively, P-3 MISS. The correction ships.**

`Qwen2.5-7B-Instruct-IQ3_M`, 3.75 effective bits, all in VRAM, tg128. Measured three times in
different thermal states, which turned out to matter:

| run | GPU entry state | measured |
|---|---|---|
| 1 | 810 MHz, 47 °C, Chrome running | 17.53 ± 0.02 |
| 2 | 1873 MHz, 57 °C, Chrome closed | 17.03 ± 0.02 |
| 3 | settled, 72 °C, r=5 | **16.89 ± 0.15** |

Taking the settled run as representative:

- **P-1 (above 15.7): HIT.** 16.89 — both models under-predict, as the known in-VRAM pessimism
  predicts.
- **P-2 (corrected is closer): HIT, decisively.** |15.7 − 16.89| = **1.19** against
  |13.6 − 16.89| = **3.29**. Error more than halved, −22% → −7%. This is the stake that decides
  shipping, and it holds under every one of the three runs.
- **P-3 (lands in 17.5–20.5): MISS.** 16.89 is **below** the band I staked from its own siblings
  (Q2_K 19.17, IQ3_XS 18.11, Q4_K_M 20.03). On the first, coldest run it scraped inside at 17.53;
  on the properly settled run it does not. Scored against the settled run, so: a miss.
- **P-4 (no anchor moves): HIT.** All four published anchors are MoE and retrodict bit-identically;
  `ne` already names the protected set exactly there, so the MoE path is untouched by construction.

### The methodological finding inside the miss

P-3 missing is more interesting than P-2 hitting. The band came from sibling measurements taken
earlier in the session — and this model measured **17.53 → 17.03 → 16.89 as the card warmed to
72 °C**, a 3.8% decay across thermal states on the same file, with error bars of ±0.02 that
never overlap. So the siblings the band was built from were very likely measured on a cooler
card than this one, and part of the "miss" is my own measurement protocol, not the model.

That is the **third** GPU-state effect this project has been bitten by, after orphaned-process
contention (a retracted Law 5 finding) and boost-clock inflation (a 28% verify anomaly). The
convention gains a third clause: **log entry clock AND temperature, and prefer a thermally
settled run.** The existing VRAM_GAPS anchors should be re-measured warm before any of them is
used to justify a constant.

### What ships

`DENSE_PROTECTED_SHARE = 0.214` in `plan.py`, applied to dense models only. Mean |error| over the
six calibration points falls **18% → 10%**, and the held-out point moves −22% → −7%. The defect it
removes is structural rather than numerical: a dense model's predicted speed now responds to its
quantization at all, which it previously did not.

**Wired into:** `quantprobe/plan.py:DENSE_PROTECTED_SHARE` · `tests/smoke.py:t_dense_speed_responds_to_bits` · `docs/index.html` — the same correction is applied to the published simulator, which reimplements the law separately.
