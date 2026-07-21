# ROADMAP_DECODE.md — The batch-1 GEMV decode kernel: floor, swings, and the one oracle to run next

Consolidated 2026-06-12 from 3 scout briefs + 24 hostile-verified hypotheses, against the
GTX 1060 (sm_61, GP106) physics and the project's standing facts (D_idx=0 iid indices;
FWHT leaves the weight kernel; dp4a is the only sm_61 compute asset; fp16-arith banned).

**The question this file answers.** Given the per-output-row codec
`y_j = sum_i level[idx_ij] * (R^T x)_i` with a per-tensor codebook of ONLY 9-12 levels
(max 12 across the 168 tensors of 0.5B), what kernel/format physically BEATS llama.cpp
Q4_K_M = 21.8 tok/s @ 4.36GB on this card — and what cheap, ideally CUDA-free, oracle
decides it next.

> **Scope note vs ROADMAP_SPEED.md.** SPEED fact #2 ("decode-COMPUTE-bound") is about the
> *full runtime-rANS* decode path (80-130 Gops/token of variable-length entropy decode).
> This file is about the **fixed-low-bit RESIDENT** path the campaign pivoted to (no
> runtime rANS; indices stored at fixed 3-4 bits, decoded by a register LUT). On THAT path
> there is no rANS tax, the inner loop is `streamed index -> register int8 codebook -> dp4a`,
> and the binding constraint reverts to **bytes-moved per token** (bandwidth-bound),
> exactly as all three scouts argue. Both framings are kept explicit below; the FLOOR and
> SWINGS are scored under the fixed-low-bit (bandwidth-bound) model.

---

## 0. Decode-cost decomposition (where the tok/s actually goes at batch-1)

At batch-1 token generation, one token = one GEMV per weight matrix. The arithmetic
intensity is ~1 FLOP/byte: every weight byte is read exactly once and consumed by ~0.5
int-op (dp4a does 4 MACs/instr). Decompose the per-token cost on the 1060:

| Cost term | Magnitude (0.5B-equiv per token) | Binding? |
|---|---|---|
| **Weight/index DRAM traffic** | `resident_bytes` @ 192 GB/s | **YES — the binding term** |
| dp4a contraction (compute) | ~1 GFLOP-class; <1 ms vs ~23 ms of load | No (20-25x slack) |
| Register-LUT codebook lookup (12 int8 in regs/cmem) | ~1 ALU op/weight, fully overlapped | No |
| Activation-side FWHT-128 + RMSNorm + Q8_1 | once/token, fp32, microseconds (fact #5) | No |
| Per-row warp-shuffle reduction + scale epilogue | once/row | No |

**The roofline identity (the only number that moves tok/s on the fixed-low-bit path):**

```
tok/s  ≈  VRAM_bandwidth × util  /  resident_bytes_per_token
       =  192e9 × util / resident_bytes
```

- Q4_K_M: 4.36 GB resident, measured 21.8 tok/s ⇒ **util ≈ 0.50** (192/4.36 ≈ 44 tok/s
  pure-bandwidth ceiling; 21.8/44 ≈ 0.50, consistent with the 54% bw figure).
- **Two levers, and only two:** (a) push `resident_bytes` below 4.36 GB at iso-fidelity,
  or (b) raise `util` above ~0.50. Everything else (bucket-GEMV, bit-serial, partial-sum
  LUTs, affine-snap, pairwise tables) optimizes **compute or ALU slack that is not scarce**
  and buys ~0 tok/s.

**Consequences that kill whole families before any kernel (paper-kills):**

1. **Compute-side cleverness is free-but-useless.** The "12 multiplies vs 6.5B multiplies"
   reframe is true but answers the wrong question — the multiplies were never the cost.
   Kills lever-1 bucket-GEMV, lever-3 bit-serial, lever-4 partial-sum LUTs *as speedups*.
2. **The 12-level codebook is a GPU liability, not an asset** — UNLESS it is dropped into a
   register/cmem int8 codebook (zero per-weight reconstruction, no gather). Its only
   kernel-level payoff is that 12 ≤ 16 fits a 4-bit index feeding dp4a. Codebook *gather*
   (strided/random shared-mem) stalls the Pascal load pipe (CodeGEMM/SBVR/FLUTE unanimous).
3. **Input-activation sparsity is dead by construction.** `R^T x` is signed-Hadamard-spread
   to maximal density — that is the *point* of the rotation. No sparse basis coexists with
   the incoherence the codec relies on.
4. **Index entropy is closed.** D_idx=0 is triple-verified (`ecvq_idx_oracle.py`): indices
   are iid post-rotation, order-0 is the true rate, real coders do *worse* than order-0
   (zstd 3.638 vs H0 3.607). Kills the entire index-context-coding family (RLE/run-skip,
   pair-coding, palette/dedup, product factorization) on the *rate* axis.

---

## 1. The de-risked FLOOR — guaranteed parity + memory win (build this first)

### F0. Fixed-4-bit branch-free register-LUT MMVQ  [the FLOOR; ~1 day microbench]

- **Format:** all 12 levels covered EXACTLY by a 4-bit index (12 ≤ 16). Packed 2 idx/byte,
  coalesced. **4.45 b/w resident, exact, no escape, no second pass.** Per-tensor codebook =
  12 int8 values (IQ4_NL kvalues trick) resident in registers/constant memory.
- **Kernel:** llama.cpp MMVQ shape verbatim — 1-2 rows/threadblock, register-direct weight
  streaming, NO smem weight staging, NO scatter/atomics, idx → register int8 codebook →
  dp4a vs Q8_1 activations, fp32 accumulate, single warp-shuffle reduction. Activation-side
  signed-FWHT-128 + Q8_1 done once/token upstream.
- **Why-physical:** pure sm_61 (CUDA cores + dp4a + register codebook; no TC, no fp16-arith,
  no smem gather, no atomics). Byte-identical to a K-quant MMVQ kernel — the configuration
  most likely to actually HOLD the ~50% util Q4_K_M holds, because there is zero branch and
  zero exception machinery to stall the Pascal load pipe.
- **Bit-exact:** trivially — every level maps 1:1 to a 4-bit code; int32-accumulator-exact
  vs the reference dequant.
- **Cheap oracle:** microbench the branch-free fixed-4-bit decode's sustained bandwidth
  fraction on one 7B FFN matrix (3584×18944) on the 1060. Gate: ≥ 50% of 192 GB/s.
- **Kill criterion:** sustained util < 50% ⇒ even iso-footprint loses; investigate occupancy
  / rows-per-warp before any sub-4-bit work.
- **Expected:** 4.45 b/w ≈ 4.36 GB-class (iso-footprint with Q4_K_M) ⇒ **parity tok/s (~22)
  with equal-or-slightly-less VRAM**, at *exact* reconstruction. This is the guaranteed
  deliverable — a parity-speed, bit-exact, K-quant-class kernel. It does NOT beat Q4_K_M on
  bandwidth alone; it beats it only via fidelity/footprint at the same speed.
- **Prior art:** llama.cpp MMVQ / IQ4_NL kvalues affine decode; SqueezeLLM non-uniform
  codebook dequant. Zero novelty — this is the safe substrate every swing builds on.

> **The FLOOR's role:** it converts the speed target from "invent a faster multiply"
> (impossible — compute isn't the bottleneck) into "ship the parity kernel, then shave
> resident bytes." Build F0 FIRST; microbench its util; only then attempt the swings.

---

## 2. The creative SWINGS — what could actually beat 21.8 tok/s

Ranked by expected-speedup-per-engineering-week. Only swings that reduce `resident_bytes`
at iso-fidelity OR raise `util` are live; all others are recorded as kills in §3.

### S1. 3-bit + structured outlier sidecar (SpQR-style two-stream)  [TOP SWING; score 4.5]

- **Mechanism:** assign the 8 most-frequent of the per-tensor 9-12 levels to a 3-bit index
  (packed, coalesced, branch-free hot path = F0 shape but 3-bit). The remaining rare levels
  go to a separate compact (position, level) exception stream decoded in a SECOND dense pass
  into a per-row override before the dp4a contraction. Hot GEMV is byte-for-byte MMVQ; the
  exception pass is a tiny dense scatter amortized once per row, never per-weight-branchy.
- **Why-physical:** 3-bit ⇒ ~3.36 GB resident (CSR sidecar) vs 4.36 GB; bandwidth ceiling
  rises 44 → 56 tok/s. At iso-util (0.50) that is **~28 tok/s, a clean ~1.3x beat**. sm_61-legal
  (coalesced + dp4a + register codebook); escapes batched into a dense pass keep the hot
  stream branch-free and dp4a-packable (SpQR/T-MAC: keep the hot path branch-free).
- **Bit-exact:** YES — top-8/rest is a lossless re-encoding of the same 12-level index;
  int32-accumulator-exact vs reference dequant. No re-quantization.
- **Cheap oracle (CUDA-free, ½ day):** per-tensor histogram the 12 ECVQ levels; compute the
  exception rate if only top-8 stay in-plane; net resident b/w = 3·1 + exception_bytes;
  projected tok/s = 192e9·0.50/net_bytes. **This oracle was partially run already** (see §4):
  it lands at **3.356 b/w (CSR sidecar)** but **FIRES the pre-registered per-tensor kill**
  (61/168 tensors exceed the 3%-exception threshold; global 2.76%, max 5.21%, p90 3.63%).
- **Kill criterion:** exception rate > 3% per tensor (net b/w erodes past ~3.7) OR a
  branch-free 3-bit decode microbenches < 45% util (then it cannot beat Q4_K_M even at fewer
  bytes). **The per-tensor exception gate currently FAILS on 61/168 tensors** — so S1 as
  written is a marginal/conditional swing, NOT a clean beat. The util risk concentrates
  entirely in the data-dependent exception scatter (the thing most likely to drop Pascal util
  from 0.50 toward IQ3's measured 0.36).
- **Expected:** floor ~parity-to-1.15x (memory win even if util sags to IQ3's 0.36 → ~25
  tok/s); ceiling ~1.7x (38 tok/s) IF 0.50 util holds, which the exception scatter makes
  unlikely. **Realistic landing: 24-28 tok/s = 1.1-1.3x = a memory win with a small speed win,
  NOT a decisive beat.** This is the strongest *honest* swing precisely because it is the only
  one that reduces resident bytes at provable bit-exactness — but its headline "guaranteed beat"
  framing is over-claimed by the exception gate.
- **Prior art:** SpQR (2306.03078, ICLR'24) — this IS SpQR with the sidecar holding rare
  *levels* rather than high-magnitude weights, composed with the FWHT+ECVQ codec. The memory
  already lists "3-bit+escape ~3.50 b/w" as a derived option. Non-novel; it is the engineering
  instantiation of an already-chosen format.

### S2. Geometric 12→8-level codebook for escape-free 3-bit (QAT-nudge)  [score 2; conditional]

- **Mechanism:** if a light QAT nudge can collapse the per-tensor book to **8 levels** on a
  3-signed-plane / uniform 3-bit grid with NO escape stream, resident drops to a flat 3.0 b/w
  — beating S1's 3.36 with zero sidecar and zero scatter-stall risk.
- **Why-physical:** flat 3-bit, branch-free, register codebook, dp4a. ~2.9 GB ⇒ ceiling
  ~66 tok/s ⇒ ~33 tok/s at 0.50 util = ~1.5x, the strongest *footprint* bet IF fidelity holds.
- **Bit-exact:** NO vs the 12-level reference — it is exact only vs a re-trained 8-level model.
  3 signed planes span 8 levels, not 12; forcing 12→8 is a fidelity cut, and the project
  measured sub-4-bit/2-bit collapse as a ppl cliff (R25).
- **Cheap oracle:** offline re-fit the codebook to 8 uniform/3-plane levels; re-measure
  held-out ppl on 0.5B/1.5B. Gate: ppl delta within the project's fidelity band (≤ +0.05,
  noting the ~0.07 seed-noise floor). **AND** a physics pre-gate: microbench a standalone
  8-level 3-plane GEMV decode on the 1060 with random data; if util < 45-50%, the bet is dead
  before any QAT.
- **Kill criterion:** 12→8 re-quant raises held-out loss past the gate (the rare levels carry
  signal — likely, since ECVQ learns entropy-optimal non-uniform levels), OR the 3-bit decode
  microbench can't clear 45% util.
- **Expected:** ~1.05-1.5x IF fidelity holds; realistically lands near ~1.05x because the
  project measured 34-36% util on Pascal sub-4-bit kernels (the IQ3 tax), and 12→8 likely
  trips the ppl cliff. The trap: physics is fine and the kernel builds — it fails on *quality*,
  not hardware.
- **Prior art:** BCQ (Xu 2018) + LUT-GEMM (2206.09557) + AnyBCQ (2510.10467). No codec novelty.

### S3. Two-rows-per-load index packing (util lever, iso-footprint)  [score 2; near-certain tie]

- **Mechanism:** persistent token loop stages the single FWHT'd activation once; interleave
  consecutive rows' indices for the same K-block so one 128-bit load feeds 2-4 dp4a
  accumulators (different rows, SAME resident activation). Classic batch-1 GEMV reuse trick.
- **Why-physical:** maximizes useful bytes/sector and keeps many dp4a in flight (Little's law
  on the load pipe) to lift util toward the ~60-65% ceiling. iso-footprint (no byte change).
- **Bit-exact:** YES (pure layout permutation; dp4a untouched).
- **Cheap oracle:** a no-op weight-streaming microbench at 1 vs 2 vs 4 rows/threadblock,
  measure achieved GB/s vs the Q4_K_M bar. **Run it on llama.cpp's existing mmvq FIRST** with
  Nsight (gld_efficiency, achieved_occupancy) — if mmvq already row-blocks and hits ~0.50, the
  headroom is gone.
- **Kill criterion:** no-op stream already ≥ 50% at 1 row/block (no headroom over llama.cpp),
  OR multi-row packing < 2% util gain.
- **Expected:** ~1.0x. **llama.cpp mmvq already does this** — the gain over the real baseline
  is ~0%. Useful only as the activation-staging substrate INSIDE another kernel, not standalone.
- **Prior art:** llama.cpp mmvq.cu row-blocking (the existing baseline).

### S4. Fused RMSNorm + signed-FWHT + Q8_1 activation kernel  [GUARDRAIL; score 3]

- **Mechanism:** collapse the per-token RMSNorm, the per-128 FWHT, and the Q8_1 quant into
  ONE activation-side kernel so the rotation adds zero extra global-memory traffic (rides the
  norm's load; FWHT-128 = 7 butterfly stages in 512B smem).
- **Why-physical / bit-exact:** sm_61-trivial; signed-Hadamard is exact in fp32; Q8_1 matches
  llama.cpp reference; fused output == unfused output.
- **Cheap oracle:** profile-count global bytes for (3-pass) vs (fused) on the current evoq
  forward. Gate: if standalone rotation+quant is < 0.5% of per-token time, don't bother fusing.
- **Kill criterion (= the gate):** activation-side traffic is ~0.05-0.1% of the 4.36 GB weight
  stream ⇒ **the gate FIRES: keep it simple, unfused.** This is a guardrail against a strawman
  3-kernel implementation, not a speedup lever (~<0.1 tok/s protected, not the claimed 1-2).
- **Prior art:** QuaRot (2404.00456) fuses exactly this; AMD MXFP4 online-rotation. Textbook.

---

## 3. The KILL LIST — families that cannot move tok/s on sm_61 (do not re-propose)

All paper-killed by the §0 decomposition; most also have an internal measured negative.

| Killed idea | Why it cannot beat 21.8 tok/s | Status |
|---|---|---|
| **Bucket-GEMV / scatter-add (lever 1)** | Reads the SAME index bytes (zero bw win); 3584 smem atomicAdds vs 896 coalesced dp4a = 4x slots uncontended, ~20x at 78-cyc contended; 12 random buckets is the worst case. CodeGEMM names scatter-add as the A100 bottleneck — GP106 strictly worse. Multiplies were never the cost. | KILL (compute lever) |
| **Bit-serial / early-exit (lever 3)** | ~4 add-equiv/weight vs dp4a's 0.5; warp-divergent, breaks coalescing; SIMT-32 runs to max bit-depth anyway (no batch-1 early exit); early-exit is lossy (fails bit-exact). | KILL |
| **Partial-sum LUT-of-input-groups (lever 4, T-MAC/LUT-GEMM)** | Needs binary planes; 12-level book ⇒ 4 BCQ planes = 4-bit-equiv work. T-MAC's win is a pshufb/tbl 16-way gather sm_61 LACKS. smem LUT is bank-conflict-throttled (FLUTE duplicated the LUT just to survive on A100). Realizable g tiny (12^g: g=2→144, g=3→1728). | KILL (no sm_61 analog) |
| **Input-activation sparsity / sparse basis (lever 2)** | `R^T x` is Hadamard-spread to maximal density by design. No sparse basis coexists with incoherence. Output-side 128-block sparsity is a SEPARATE live lever (B1 in ROADMAP_SPEED), not this. | KILL (by construction) |
| **Affine 2-tap "matmul IS the dequant"** | Bit-exact fails: oracle found 11/168 snapped books lack a small-field 2-plane affine. Even where it fits, iso-footprint ⇒ no bw win; the gather it removes was never the Pascal bottleneck (register-LUT already has zero gather stall). | KILL (util lever, no headroom) |
| **Product factorization / asymmetric split-plane (2+1, 2+2)** | H(a)+H(b) ≥ H(joint) (entropy sub-additivity) — can never beat coding the 12-symbol joint. 2+1 sumset = 8 < 9-12 levels (impossible to cover exactly); 2+2 = 4 b/w = iso with F0. | KILL (info-theory self-refuting) |
| **Pair / RLE / palette / dedup index coding** | D_idx=0 triple-verified: indices iid post-rotation; `pair_tans_gate.py` measured pair == product to +0.0002 b/w; `ecvq_idx_oracle.py` all context banks negative. No sub-order-0 rate exists. Also: storage lever on a bandwidth/compute-bound runtime. | KILL (rate closed) |
| **Zero-level structural skip / 3-level outlier collapse** | Oracle on real ECVQ levels: 0/10 tensors have a level that int8-snaps to 0; forcing it costs 7.66% Frobenius error. iFWHT mixes all 128 coeffs ⇒ no weight-space position is structurally zero for all tokens. 3-level = the 2-bit ppl cliff. | KILL (no zero exists; lossy) |
| **Activation-conditioned tile pruning** | Same Hadamard-density wall as lever 2; energy is spread uniformly post-rotation, ~0% tiles skippable. Lossy (thresholding). | KILL |
| **Dual-codebook pairwise int8 dp4a-density** | dp4a lanes ARE the contraction; pre-decoding two indices still consumes two lanes against two distinct activations — no MAC fusion. Saves a shift/mask (non-bottleneck ALU). Divergent cmem gather risks serialization. iso-footprint. | KILL (mechanism void) |

---

## 4. Oracle status — what is already measured (don't re-run)

- **D_idx = 0** (triple-verified, `ecvq_idx_oracle.py` + `ecvq_cmcore.py`): rotated ECVQ
  indices are iid; order-0 entropy is the true rate; real coders do worse than order-0. ⇒
  Closes the entire index-entropy axis (S-family pair/RLE/palette/product all dead on rate).
- **Pair-tANS** (`pair_tans_gate.py` on real qwen05b.evoq post-rotation streams): pair-rate ==
  product-distribution to **+0.0002 b/w** (MI ≈ 0). ⇒ Closes pair coding.
- **S1 top-8 exception oracle (run, CUDA-free):** net **3.356 b/w (CSR sidecar)** / 4.055 b/w
  (dense bitmap) on the real 168-tensor 0.5B container (mean 10.9 levels). **FIRES the
  per-tensor 3% kill on 61/168 tensors** (global 2.76%, max 5.21%, p90 3.63%) while passing the
  global <3.7 b/w gate — the two pre-registered criteria contradict, so S1 is under-specified
  and, by the strict per-tensor reading, a conditional/marginal swing.
- **Affine 2-plane oracle (run):** 11/168 snapped books lack a small-field 2-plane affine ⇒
  not bit-exact. KILL.
- **Zero-level oracle (run, real ECVQ levels at λ=0.008):** 0/10 tensors snap a level to 0;
  forcing it = 7.66% Frobenius error. KILL.

---

## 5. THE SINGLE NEXT ORACLE TO RUN

**Build F0 (fixed-4-bit branch-free register-LUT MMVQ) on ONE 7B FFN matrix and microbench
its sustained bandwidth fraction on the 1060.** (~1 day; this is the cheapest *decisive* test
that is NOT already settled CUDA-free.)

**Why this one, and why now:**

1. **It is the load-bearing unknown for EVERY swing.** The entire roofline argument
   (`tok/s ≈ 192e9·util/bytes`) hinges on whether a branch-free fixed-bit decode actually
   HOLDS ~0.50 util on Pascal. Q4_K_M's 0.50 is measured for *its* kernel; our F0 util is
   *assumed*. If F0 holds ≥ 0.50, the FLOOR is real (parity + memory win banked) AND S1's
   3-bit ceiling (~28 tok/s) becomes reachable. If F0 sags toward IQ3's 0.36, every sub-4-bit
   swing collapses to ≤ parity and the whole campaign retreats to the IQ3-domination claim.
2. **The CUDA-free oracles are exhausted in the kill direction.** D_idx=0, pair-tANS, affine,
   zero-level, and the S1 top-8 histogram are all already run. The remaining decisive question
   is a *hardware util* number, which only a kernel microbench can produce — and F0 is the
   minimal, zero-novelty, bit-exact kernel that yields it.
3. **It gates the next decision cleanly.** F0 util ≥ 0.50 ⇒ build S1's branch-free 3-bit hot
   path and measure whether the exception scatter holds util (the real risk per §4). F0 util
   < 0.50 ⇒ stop; the gap is occupancy/issue-bound, not format-bound, and no resident-byte
   reduction converts to tok/s until that is fixed.

**Pre-registered gate:** F0 sustained ≥ 50% of 192 GB/s on the 7B FFN matrix ⇒ FLOOR banked,
greenlight S1. 45-50% ⇒ parity-only, S1 conditional on its own exception-scatter microbench.
< 45% ⇒ profile occupancy before any sub-4-bit work; the bandwidth lever is not yet accessible.

> Secondary CUDA-free oracle if a kernel cannot be built this week: run the S1 top-8 exception
> oracle with the **per-tensor gate as the binding criterion** (not the global average) across
> all 168 tensors at the actual λ-class operating point, and decide whether a 9-level (4-bit,
> escape-free) variant dominates the 8-level-plus-CSR variant on net b/w. This resolves the
> contradiction in S1's pre-registration without any CUDA.
