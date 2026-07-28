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
