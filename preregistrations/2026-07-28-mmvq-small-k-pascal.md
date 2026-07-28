# Pre-registration #55: is llama.cpp's Pascal carve-out for Q2_K multi-row blocking wrong?

**Author:** Federico Sciuca · **Date staked:** 2026-07-28, BEFORE the measurement. **STAKED.**

## What was found in source

`ggml/src/ggml-cuda/mmvq.cu`, `should_use_small_k`:

```c
constexpr std::array<ggml_type, 3> slow_pascal = { GGML_TYPE_IQ3_S, GGML_TYPE_Q2_K, GGML_TYPE_Q3_K };
...
} else if ((... iq_slow_other ...) ||
        (is_nvidia_pascal_older && std::find(slow_pascal.begin(), slow_pascal.end(), type) != slow_pascal.end()) ||
        GGML_CUDA_CC_IS_RDNA(cc)) {
    use = false;
}
```

`small_k = true` sets `rows_per_cuda_block = nwarps` instead of 1 — i.e. one CUDA block computes
several output rows and can amortise row-invariant per-activation work. **Q2_K and Q3_K are
explicitly excluded from it on Pascal-and-older.** Q4_0 is not.

## Why that is suspicious rather than obviously right

Measured on this card, all this session:

| | GW/s | fraction of its own kernel ceiling |
|---|---|---|
| Q4_0, real 7B, end-to-end | 204.7 | **88.5%** |
| Q2_K, real 7B, end-to-end | 165.1 | **46.8%** |

And `vec_dot_q2_K_q8_1_impl_mmvq` spends **8 dp4a per 16 weights where 4 suffice**, because its
min term is computed as `dp4a(m_broadcast, u[i])` = `m × sum(activations)` — a quantity that does
not depend on the output row. Multi-row blocking is exactly the mechanism that would amortise it.
So the format that most needs multi-row blocking is the one excluded from it, on the hardware where
ALU is scarcest.

**The honest alternative:** whoever added `slow_pascal` measured a regression. The name says
"slow", which is evidence they did. This test is therefore as likely to confirm their call as to
overturn it — and confirming it would kill the whole line of inquiry, which is why it is worth one
run.

## Method

`GGML_MMVQ_SMALL_K=1` forces multi-row ON, `=0` forces OFF, unset = stock. Single build, runtime
switch, so no rebuild sits between the arms. A one-time stderr line reports the actual
`nwarps` and `rows/block` chosen, because **if `nwarps == 1` then forcing `small_k` changes
nothing** and a null would be meaningless — that check has to happen before any result is read.

Arms interleaved with position control, Qwen2.5-7B Q2_K all-in-VRAM, tg128.

## Stakes

- **P-0 (validity gate, checked first).** The instrumentation reports `rows/block 1->N` with
  **N ≥ 2** for Q2_K. If N == 1, the experiment did not run and no verdict may be recorded.
- **P-1 (THE CLAIM).** Forcing multi-row ON makes Q2_K **≥ 8% faster** than stock.
- **P-2 (specificity).** The effect does not appear on Q4_0, which stock already allows multi-row
  where beneficial — so this must be a Q2_K-path result, not a global blocking win.
- **P-3 (correctness).** Output identical between arms at temp 0, same seed. Blocking changes the
  order of accumulation across rows, not within a row, so identity is expected here — unlike the
  E8 scheduler change, where it was not.

## KILL RULE

**If P-1 fails — forcing multi-row ON is neutral or slower — then the `slow_pascal` carve-out is
correct, upstream measured it right, and the 46.8%-of-ceiling gap is NOT reachable by blocking.**
The remaining candidate would then be the extra dp4a itself, which cannot be removed without either
a finer-grained activation sum in `block_q8_1` or a different kernel structure entirely — and I
would report the gap as diagnosed but unclaimable at this level, rather than keep proposing fixes.

Recorded expectation before the run: **I genuinely do not know.** The instruction count says the
headroom is there; the existence of a deliberate, named carve-out says someone already looked. This
is the tenth mechanism in this project and nine of the previous nine moved under a control.

**Wired into:** pending P-1.

---

## Scored (2026-07-28, log: `weights/data/prereg55_small_k.log`)

**Verdict: P-0 PASSES, P-1 FAILS decisively. THE KILL RULE FIRES. Upstream's `slow_pascal`
carve-out is CORRECT and my hypothesis is refuted — twice over, the second time more informatively
than the first.**

### P-0 validity gate — the experiment did run

```
stock      [qp] mmvq type=10 ncols_dst=1 nwarps=4 small_k 0->0 rows/block 1->1
forced ON  [qp] mmvq type=10 ncols_dst=1 nwarps=4 small_k 0->1 rows/block 1->4
```

`nwarps = 4`, so forcing `small_k` genuinely moved the blocking from 1 to 4 rows per CUDA block.
The gate was worth having: a null with `nwarps == 1` would have been meaningless.

### P-1 — Qwen2.5-7B Q2_K, all-in-VRAM, tg128, r=2

| arm | tg128 |
|---|---|
| stock (multi-row disabled by `slow_pascal`) | **22.25 ± 0.07** |
| forced multi-row ON (4 rows/block) | **17.33 ± 0.03** |
| stock, last position (control) | **22.06 ± 0.03** |

**Multi-row blocking is 22% SLOWER.** Staked ≥ +8% faster; measured −22%. Position control confirms
no drift (22.25 vs 22.06 across the session).

Whoever added `slow_pascal` measured this correctly. The array is named "slow" because it is.

### The second refutation, which matters more

**Q4_0 — the format that reaches 88.5% of its kernel ceiling — also runs at 1 row per block:**

```
[qp] mmvq type=2 ncols_dst=1 nwarps=4 small_k 0->0 rows/block 1->1
```

So blocking never distinguished the healthy format from the sick one. My framing in
`findings/Q2K_MIN_TERM.md` — "llama.cpp forfeits row reuse, which costs asymmetric formats a dp4a
per row" — was wrong in its causal half: **both** formats forfeit that reuse, and only one is slow.

### What the two nulls leave standing, and it is now a complete account

The difference between Q4_0 and Q2_K is what the source said all along and nothing more exotic:
**Q2_K issues 8 dp4a per 16 weights, Q4_0 issues 4.** Twice the ALU, on a card where ALU is the
binding constraint, for ~half the throughput. That is the whole explanation, and it needs no
blocking story.

And the two results together close the loop: the *only* way to remove Q2_K's extra dp4a is to
amortise the row-invariant `m × sum(x)` across rows, which requires multi-row blocking — which is
measurably worse on this hardware for unrelated reasons (a quarter of the blocks on a 10-SM card,
and strided weight reads). **The two available fixes are mutually exclusive.** Upstream chose the
better one.

### Honouring the kill rule as written

I pre-committed: *"I would report the gap as diagnosed but unclaimable at this level, rather than
keep proposing fixes."* So:

**The Q2_K decode deficit on Pascal is diagnosed and NOT claimable.** It is the format's intrinsic
ALU cost under llama.cpp's kernel structure, and llama.cpp's structure is already the better of the
two options available to it. The 2.07× headroom measured in #53 is real as a *ceiling*, and is not
reachable by any change I have identified.

One untested candidate remains, and it is recorded rather than pursued: storing **finer-grained
activation sums in `block_q8_1`** (it already carries a sum over 32; Q2_K needs sums over 4) would
let the min term become a multiply without any blocking change. That touches a struct shared by
every quantization type and every backend, which is a far larger and riskier change than anything
in this project so far — and it is exactly the kind of thing I should not start on the strength of
two refuted hypotheses.

**Wired into:** `findings/REGISTER.json:D-18` (multi-row blocking refuted, −22%, and upstream's
carve-out independently validated) · `D-19` (the "forfeited row reuse" framing refuted — Q4_0
forfeits it too) · `U-13` (finer-grained q8_1 sums: untested, high cost, not started) ·
`C-02` (closed at this level: Q2_K's deficit is ALU count, structural, not a bug).
