# Pre-registration #79: price each tier with the format it actually holds

**Author:** Federico Sciuca · **Date staked:** 2026-07-30, BEFORE implementation. **STAKED.**
**Baseline:** the state-locked ladder `a19aeee4` (`weights/data/ladder_state_locked.json`),
median |err| 8.8%, MoE-IQ +9.1% mean, MoE-K -7.0% mean, dense AIV -1.5%.

## The mechanism, measured from the ladder's own files

On a MoE split the GPU holds ATTENTION plus resident experts while the CPU holds offloaded
EXPERTS - and those two sets decode at very different speeds:

| file | attention ebw | expert ebw | file-wide (what we use for BOTH) |
|---|---|---|---|
| Qwen3-30B-A3B | 82.4 | 62.2 | 63.6 |
| Qwen3-Coder-30B | 89.6 | 62.2 | 64.2 |
| Qwen3.5-35B APEX-Mini | 89.6 | 69.2 | 71.4 |
| Qwen3.6-35B Q2_K_XL | 105.9 | 57.6 | 64.5 |
| DS-Lite IQ2_XS | 73.0 | **83.6** | 82.7 |

The file-wide value tracks the EXPERTS (they dominate bytes), so the GPU tier - which is mostly
attention - is priced with the slow number and **under-predicts**. Where the inequality inverts
(DS-Lite: attention slower than experts) the same rule **over-predicts**. Same for the CPU
codebook penalty: Qwen3.6-Q2_K_XL's experts are 97% codebook while the file is 83%, so the tier
that pays the tax is under-charged.

## The change

`spec.from_gguf` additionally returns `fmt_bw_attn`, `fmt_bw_exp`, `codebook_share_exp`.
`evaluate` prices MoE rows per tier: the GPU eta scales by the byte-weighted blend of
(attention at `fmt_bw_attn`, resident experts at `fmt_bw_exp`) over the file-wide value it
currently uses; the CPU penalty uses `codebook_share_exp` instead of the file-wide share.

## Stakes (against the a19aeee4 baseline)

- **P-1.** MoE-K mean error moves from **-7.0%** to inside **±4%**.
- **P-2.** MoE-IQ mean error moves from **+9.1%** to inside **±5%**.
- **P-3 (THE CONTAINMENT TEST).** All six dense all-in-VRAM arms move by **< 1 point** — they
  have no expert tensors, so per-tier statistics are identical to file-wide and a correct
  implementation cannot touch them.
- **P-4 (stated so it cannot be claimed later).** DS-Lite Q4_K_M (-19.1%) is NOT predicted to be
  fixed by this: its attention/expert gap is small (105.6 vs 110.5) and it is the only MLA
  architecture in the set. If it improves anyway, that is luck, not evidence.

## KILL RULE

**If P-3 fails the change is reverted** regardless of P-1/P-2 — touching arms with nothing to fix
means the implementation is wrong, not the theory. If P-1 and P-2 both fail while P-3 holds, the
per-tier hypothesis is refuted as the dominant term and U-28 is scored refuted; the residual then
belongs to something else and the ladder keeps its honest 8.8%.

**Wired into:** pending; `spec.from_gguf` + `plan.evaluate` MoE rows.

---

## SCORED — 2026-07-30. **KILL RULE FIRED; THE CHANGE IS REVERTED.**

| arm | baseline | with per-tier | moved |
|---|---|---|---|
| Qwen3-30B-A3B | −10.2% | **−1.4%** | +8.8pt |
| Qwen3-Coder-30B | −10.0% | **+1.1%** | +11.1pt |
| DS-Lite IQ2_XS | +22.7% | **+17.5%** | −5.2pt |
| Qwen3.5-35B APEX-Mini | −3.8% | +9.0% | +12.8pt |
| **Qwen3.6-35B Q2_K_XL** | +11.3% | **+38.4%** | **+27.1pt** |
| **Qwen3.6-35B APEX-MTP** | +8.5% | **+35.4%** | **+26.9pt** |
| DS-Lite Q4_K_M | −19.1% | −19.5% | −0.4pt (P-4 held: unchanged, as stated) |

- **P-3 HIT, exactly as designed.** All six dense all-in-VRAM arms moved by **≤ 0.04 points** —
  no expert tensors, no per-tier difference, no movement. The implementation is contained.
- **P-1 MISS** (MoE-K −7.0% → **+4.6%**, overshooting past zero, outside ±4%).
- **P-2 MISS, badly** (MoE-IQ +9.1% → **+20.6%**, target ±5%).
- Median |error| unchanged at 8.8%. Per the kill rule as written — "if P-1 and P-2 both fail
  while P-3 holds, the per-tier hypothesis is refuted as the dominant term" — **the change is
  reverted and U-28 is scored refuted-as-formulated.**

**What the split evidence actually says, recorded rather than acted on.** The blend is
spectacularly right for the two K-quant flagships (−10% → −1.4% and +1.1%) and spectacularly
wrong for the two Qwen3.6 files (+27 points worse). Those are exactly the files with the widest
attention-vs-expert gap (105.9 vs 57.6), so the blend's weight — `act_ne` vs `f·act_ex`, i.e. how
many bytes we believe are attention versus experts — is what breaks. **The per-tier PRICE looks
right; the per-tier WEIGHTS come from the `ne` heuristic, and on newer architectures that
heuristic is unverified** (already queued as the gpt-oss-class shared-expert question). Fixing
the weights is the prerequisite; re-running #79 on top of a corrected `ne` is the follow-up, and
it must be a fresh prereg, not a revival of this one.

**It would have been easy to keep this** — two flagship arms inside ±1.5% is a seductive result.
The stakes were set in advance precisely so that a change which helps the models we care about
and wrecks the ones we care about less does not get to ship on selective reading.
