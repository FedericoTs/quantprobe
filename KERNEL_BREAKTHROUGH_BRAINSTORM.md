# Kernel Brief Analysis + Creative Measurement Program

*Source: `GROK_KERNEL_BRIEF.md` (2026-07-28) + `tools/kernelprobe/{bench.cu,quality.py}`  
Goal: raise **novel generation** tok/s on consumer hardware (batch-1), iso-quality.  
Method: adversarial reading of the brief, then proposals that survive its five retractions, each with a kill test.*

---

## 0. What the kernel session actually proved

### 0.1 The new physics sentence

> **The decode wall on Pascal is unpack instruction cost, not VRAM bandwidth.**

Controlled evidence (same buffer, same access, only the instruction sequence changes):

| Path | GB/s | vs stream |
|---|---|---|
| Pure stream | 161.0 | 1.00 |
| fp16 matvec, **no unpack** | 152.5 | **0.95** |
| 4.5-bit naive float unpack | 67.8 | 0.42 |
| 4.5-bit → `__dp4a` | 128.7–132.1 | **0.80–0.82** |

**1.90×** from swapping the unpack alone. Layout, smem staging, and reduction are essentially free. That kills a generation of “smarter GEMV geometry” proposals on this card (retraction #4: multi-row blocking **−22%**).

### 0.2 Law 4 must be rewritten

```
OLD:  tok/s = η(tier) × BW / active_bytes
NEW:  tok/s = GW(format, kernel) / N_weights_active
      with  GB/s = GW × (bits/8)   falling as bits fall when ALU-bound
```

Same card, same model, same session: Q4_0 η **0.619** vs Q4_K_M η **0.553** — pure format/unpack, forbidding a tier-only η. This is the mechanism behind six prior “quantized byte ≠ byte” sightings.

### 0.3 Where the prize is (and is not)

| Regime | Headroom | Status |
|---|---|---|
| **4-bit, all-in-VRAM, e2e** | **≤ ~1.11× kernel**, ≤ ~1.28× to no-unpack matvec | llama.cpp Q4_0 already **87%** of best hand kernel — **effectively closed** |
| **2–2.6-bit dp4a-native microbench** | GW/s **1.52–1.63×** vs 4.5-bit | **Open** — bits buy weights/s if unpack stays dp4a-native |
| **Q2_K e2e vs Q2_K-cost-model kernel** | ~**2.07×** GW/s gap | Diagnosed (extra min-term `dp4a`s in llama path); multi-row fix **fails** on Pascal; **not claimable as a casual upstream tweak** |
| **New asymmetric 2-bit format (Q2_A)** | Hoped 1.70× | **Killed**: Q2_K cost model is **25% faster** than Q2_A in the same harness |
| **MoE expert gather / scatter** | Hoped 1.8× tax | **Killed**: penalty tracks block starvation, not scatter; real MoE launches enough blocks |
| **Graph fragmentation (split placement)** | Hoped big η | **Killed**: −85% submit time, only **+6.5%** e2e |

### 0.4 Link to the novel-gen goal (hybrid 30B MoE ~21 tok/s)

Kernelprobe is **all-in-VRAM matvec physics**. Flagship novel gen is **hybrid**: GPU attention + partial experts, host DRAM experts. Map carefully:

| Kernel finding | Hybrid novel-gen implication |
|---|---|
| 4-bit GPU path nearly optimal | Do **not** spend months on GPU Q4 kernels for the flagship |
| 2-bit GW/s high in microbench | GPU-resident **expert** bytes at 2-bit are ALU-cheap *per weight* but e2e Q2_K is sick in llama — format+kernel co-design still matters for **VRAM half** |
| Unpack is the wall | Prefer formats whose metadata is **byte-aligned / hoisted**, not nibble soap |
| Gather free | Expert *index* scatter is not the hybrid tax; **DRAM bytes + CPU path + idle bubble** still are |
| Fragmentation free | Don’t chase CUDA graph surgery for split η |

**Effective strategy:** treat kernelprobe as the instrument for **format × unpack** science; treat hybrid novel-gen as a **multi-tier** problem where GPU kernel wins only move the **GPU share** of the token (often a minority of milliseconds). Kernel breakthroughs that matter most are those that either:

1. **Raise GW/s at ≤2.5 bits** enough to fit more of the model in VRAM (change placement, cut DRAM term), or  
2. **Make a 2-bit path as healthy as Q4_0** so “smaller file” stops meaning “slower decode,” or  
3. **Export a technique to the CPU expert path** (same math, AVX2) where most novel-gen ms live.

---

## 1. Critical re-read: is the “unclaimable” 2.07× really unclaimable?

The brief asserts: Q2_K’s extra `dp4a(m_broadcast, u)` is row-invariant only if amortized across **output rows**, multi-row blocking is slower, therefore headroom is unclaimable.

**But the same repo’s `k_matvec_q2k_equiv` already does something else:**

```text
sum((q*s - m)*x) = s*dp4a(q,x) - m*sum(x)
sum(x) hoisted to smem once per block  →  4 dp4a + scalar m·sumx per group
```

That is **not** multi-row weight blocking. It is **activation group-sum hoist**. It lands at **356.6 GW/s**. llama.cpp e2e Q2_K sits at **165.1 GW/s**.

**Creative hypothesis H-REOPEN (not yet framed as such in the brief):**

> The 2.07× is unclaimable *inside llama.cpp’s current 1-row MMVQ with min-via-dp4a*, but **claimable** in a kernel that (a) hoists `sumx` once, (b) applies min as a scalar FMA, (c) uses warp-per-row like L1h, (d) keeps Q2_K’s **exact** bit layout for quality bit-identity.

**Falsification (highest-value kernel experiment remaining):**

1. Implement `k_matvec_true_q2k_gguf` that consumes **real Q2_K bytes** (not a repacked cost model) with hoisted `sumx` + 4 `dp4a`.  
2. Microbench GW/s vs L1g.  
3. Wire into a **single FFN** of a 7B Q2_K model (or a tiny custom loop) and measure tg.  
4. **Kill** if real Q2_K layout cannot reach ≥80% of L1g (layout packing / scale nibble cost dominates).  
5. **Claim** if e2e 7B Q2_K moves from ~21.7 toward Q4_0’s ~26.9 **without** quality change (same file).

If this hits, the brief’s “do not file upstream” becomes “file a **different** PR”: not multi-row blocking, but **min-term scalarization + sumx hoist** on Pascal MMVQ.

This is the single most important *kernel-shaped* bet for the project.

---

## 2. Direct answers to the brief’s Q1–Q5 (with creative extensions)

### Q1 — Where does GW/s peak as bits → 0?

**Known:** 4.5 → 2.625 → 2.5 bit still rising (234.9 → 356.6 → 382.7 GW/s) with ~constant 4 `dp4a` / 16 weights.

**Creative layouts to measure (none in the ladder yet):**

| ID | Layout | bits/w | Unpack idea | Why it might peak higher |
|---|---|---|---|---|
| **K1a** | 2-bit × 2 groups packed in one `uint32`, dual-issue | 2.0 | two masks, 2× dp4a | fewer scale loads |
| **K1b** | **1.5-bit ternary** {−1,0,+1} in 2 bits with illegal state as run-length escape | ~1.2–1.5 | LOP3 extract | sparsity in weights free |
| **K1c** | **1-bit signs + shared scale per 32** | 1.03 | popc / dp4a on 0/1 | max GW/s candidate |
| **K1d** | **Bit-plane progressive**: plane0 (MSB) first | 1 then 2 then … | optional early-out | see K3 |
| **K1e** | **0-bit**: all weights = scale (rank-1) | ~0.05 | no unpack | quality kill expected; measures pure scale path |

**Turnover theory (predict before measure):**

```
GW/s rises while:  t_unpack + t_dp4a  <  t_bytes / BW_eff
GW/s falls when:   scale/metadata load + reduction + launch  dominate,
                   or occupancy dies from register pressure,
                   or you issue more than 1 ALU-op/weight.
```

**Protocol:** extend `bench.cu` L1 ladder to 2.0 / 1.5 / 1.0 / 0.5 with **forced** 4-dp4a-per-16 structure; plot GW/s and GB/s; kill a format if quality.py RMSE > 1.05× Q2_K on residual-writer tensors.

**Novelty:** almost nobody publishes Pascal GW/s ladders below 2.5 with dp4a-native symmetry; the *turnover point* is a publishable constant.

---

### Q2 — Unpack cheaper than shift/mask/dp4a?

Gap: **0.69 → 0.79** of spec (~**1.15×**).

| ID | Idea | Mechanism | Kill test |
|---|---|---|---|
| **K2a** | `LOP3.LUT` fused mask | one inst vs shift+and | disasm + bench; kill if <2% |
| **K2b** | `__byte_perm` / `PRMT` | align nibbles without shift chain | same |
| **K2c** | **`uint4` 16 B vector load** | cut LDG count | kill if GB/s unchanged (already LDG bound) |
| **K2d** | **Pre-expanded int8 tile in registers from previous layer’s prefetch** | unpack once, reuse for gate+up+down | MoE fuse; kill if VRAM traffic rises more than ALU saves |
| **K2e** | **Weight-as-int8 resident for GPU-only tensors** | delete unpack entirely on attention | fits? measure resident; should hit ~0.79 path |
| **K2f** | **Texture path / linear texture** | old Pascal free interpolation units — usually worse | kill if < stream |

**Creative long-shot K2g — “unpack in the quantizer, not the kernel”:**  
At GGUF build time, store **already lane-permuted int8 codes in separate planes** so the kernel is `ld.global.u32 → dp4a` with **zero** shift. Pay +0.1–0.3 b/w metadata alignment. The brief says pre-permute helps “for layouts we tried”; a **full plane-separated codebook** may close more of the 1.15×.

---

### Q3 — Why no-unpack matvec is 0.79 not 0.84?

Least-understood 6%. Creative isolations:

| ID | Experiment | If gap shrinks, cause was… |
|---|---|---|
| **K3a** | fp32 weights, no convert | fp16→fp32 convert |
| **K3b** | activations in registers only (no smem) | smem bank/throughput |
| **K3c** | skip warp reduction; write partials | reduction |
| **K3d** | 1 output element only, no grid | tail / launch |
| **K3e** | `ld.global.cg` vs `ca` cache hints | cache policy |
| **K3f** | WDDM vs TCC / Linux | OS interference (cross-link B7) |

**Why it matters:** if 0.79 is convert tax, **int8 activation path everywhere** (already natural with dp4a) is the only way to the 0.84 rail. If it is reduction, **persistent segmented reduction** is the lever.

---

### Q4 — Ampere+ inversion

Already well-posed. Add:

- Same ladder on one 30xx/40xx → does Q4_K_M beat Q4_0 again?  
- Does Q2_K’s extra min `dp4a` **hide** when ALU/BW ratio flips?  
If yes, **format advice in quantprobe must be GPU-generation-conditional** (C-05 productized).

---

### Q5 — Fundamentally different structure

The brief lists: shfl codebook, activation-conditional skip, pre-permute, persistent kernels, fuse MoE projections. Below: **stronger / less tested** variants.

---

## 3. Creative structures nobody has measured here

### Cluster A — Change *what* is multiplied (not how fast unpack is)

#### A1. Warp-shfl **vector** codebook (not scalar levels)

**Idea:** Each 8-bit (or 6-bit) index selects a **length-4 or length-8 int8 vector** from a 256-entry table in smem; one index → one `dp4a` chain. AQLM/PQ spirit, but **register/smem gather**, not DRAM.

**Why new here:** brief only mentions shfl codebook for non-uniform *scalar* quant. Vector PQ can hit RD better at same index rate (your own VQ work said outliers steal VQ’s lunch *on whitened PTQ* — experts at RD floor may still prefer scalar; **measure on expert tensors only**).

**Kill:** quality.py-style RMSE on `ffn_*(exps)` vs Q2_K; microbench GW/s vs L1e. Kill if RMSE worse and GW/s not ≥1.2×.

#### A2. **Product of two 4-bit codes** → effective ~8-level without 3-bit unpack

`w ≈ s · (a + 4*b)` with a,b ∈ 0..3 — two tiny dp4a streams. Might match Q3 quality at 2×2-bit unpack cost (known cheap).

**Kill:** Pareto vs Q3_K_M ppl and vs L1d speed.

#### A3. **Sign-Plane + Magnitude-LUT**

1-bit sign plane (popc/xor path) + 3-bit magnitude codebook. Two streams; magnitude may be sparser / more compressible for cold experts on CPU.

**Kill:** joint GW/s and ppl.

---

### Cluster B — Skip weight traffic without token speculation (D-10-safe)

#### B1. **Bit-plane progressive matvec with residual energy early-out** ⭐

**Idea:** Store weights as bit-planes (MSB→LSB). Compute partial matvec after plane 0..p; if \(\|y^{(p)} - y^{(p-1)}\|\) or activation-weighted energy is below ε, **stop loading further planes**.

This is **activation-conditional weight skipping** at plane granularity — not TEAL neuron sparsity, not token draft.

**Arithmetic:** if average planes used = 1.2 of 2 → ~40% less weight traffic on GPU path; GW/s on completed planes stays dp4a-native.

**Kill (offline first):** for each token, count planes needed for logit KL < 1e-3 vs full. Kill if mean planes > 1.85 on novel prose (no savings).

**Novelty:** progressive coding is old in video; **for MoE expert matvec on Pascal with a KL kill** it is untested in this project.

#### B2. **Activation-top-k channel keep on *down_proj* only**

down_proj is residual-writer and tall. If only top-α% of \|h\| channels matter for this token, skip those **columns** of weights (DRAM/VRAM). Block-align to 32/128 for coalescing.

**Kill:** same as historical sparsity oracles; bind on ≥32-group matrices. Project history is hostile to unstructured sparsity — **block-aligned channel keep** is the only variant still live.

#### B3. **Shared base + expert delta (Δ-experts)**

All experts in a layer share a mean expert \(\bar W\); store \(\Delta_i = W_i - \bar W\) at 1–2 bit; reconstruct \(W_i = \bar W + \Delta_i\). If experts are similar (routing-flat world still allows similar *weights*), Δ is compressible.

**Kill:** measure \(\mathrm{RMSE}(\Delta)\) vs RMSE(W); kill if bits(Δ)+bits(base) ≥ bits(Q2_K) at same RMSE.

**Novelty:** adjacent to your lossless **delta codec** thread — applied *inside* a layer’s expert set for *lossy* decode, not checkpoints.

---

### Cluster C — Fuse and persist (structure, not format)

#### C1. **Fused MoE triple (gate, up, down) with one weight stream schedule**

Load activation once; for each expert id, stream gate/up/down tiles with **shared unpack state** and shared `sumx`. Cuts activation re-read and repeated scale unpack setup.

**Kill:** microbench 3 separate matvecs vs fused; kill if <8% on 1060 (activation is small — may already be free).

#### C2. **Persistent kernel / grid-resident decode**

One grid lives for the whole token/layer; work queue of matvecs. Attacks launch + WDDM submit (retraction #1 said submit was only ~2 ms — so **kill bar is strict: must beat +3% e2e**).

#### C3. **Software-pipelined unpack: warp A unpacks tile t+1 while warp B dots tile t**

Classic GPU double-buffer. On Pascal may recover part of 0.69→0.79 if unpack and LDG dual-issue.

**Kill:** if SM occupancy forces fewer blocks and net −%.

#### C4. **CPU export of the same dp4a math (AVX2 VNNI-less)**

Novel-gen ms are on **host experts**. Port L1e/L1g ideas to AVX2: unpack-once, hoist sumx, Q2_K scalar min. Target: move MoE path from ~16 GB/s effective toward stream 26 GB/s.

**This is the highest expected-value “kernel” work for the actual 21 tok/s goal**, even though it is not CUDA.

**Kill:** single-expert GEMV GB/s; then full split placement novel tg. Claim if ≥ +15% novel tok/s.

---

### Cluster D — Placement × format co-design (kernelprobe enables it)

#### D1. **“Q4 brain, Q2 limbs” dual-format model**

Attention + first N layers: **Q4_0** (healthy 87%-of-ceiling path).  
Cold experts: **Q2_K or sym2**.  

Kernelprobe says Q2 is not “more BW efficient” in tok/s today; but if limbs live in **DRAM**, bytes dominate, and CPU dequant tax differs from GPU.

**Kill:** dual-file or custom loader; novel tg + ppl vs uniform Q2 and uniform Q4.

#### D2. **VRAM = only unpack-free int8 for active set**

Use 6 GB to hold **int8 (or fp16) expansions of the working expert set**, refilled by async CPU unpack from Q2 store in RAM. GPU kernel becomes L1b/L1c (0.78–0.95 stream). CPU pays unpack once per expert **touch**, amortized if router locality were high — **but locality was refuted for long windows**.  

**Revised variant:** expand only **attention + shared + always-on**, keep routed experts packed. Matches Law 2 density but protects the GPU-bound healthy path.

**Kill:** if expand bandwidth (PCIe 12.2 or CPU) > savings.

#### D3. **Bits that maximize tok/s under a VRAM card, not under a file-size card**

Objective flip:  
\(\max \mathrm{GW}(b)/\mathrm{quality\_cost}(b)\) s.t. resident ≤ 5.5 GB.  

Kernelprobe’s GW ladder makes this a **numeric program**, not a vibe. Possibly optimal is **all Q4_0 of a smaller active set** vs **Q2 of a larger set** — on 7B, Q4_0 already beat Q2_K absolute tok/s.

For 30B MoE: search **N_gpu_layers × format** on the GW-aware law.

---

### Cluster E — High-risk scientific long-shots (measure cheaply first)

| ID | Idea | Field analogy | 1-day oracle |
|---|---|---|---|
| **E1** | **Stochastic 1-bit with noise averaged over speculative verify batch** | dithering | ppl with same seed noise |
| **E2** | **Hadamard on activation + 1-bit weights** (binary net + rotation) | your Law 1 | RMSE after rotation |
| **E3** | **Kernel writes directly into next layer’s Q8 act buffer** (no global store) | fused pipeline | e2e 1 layer pair |
| **E4** | **Approximate dp4a with 2-bit × 2-bit → 4-bit product LUT in regs** | bit-serial rebirth | only if L1e turnover fails |
| **E5** | **Expert weights as low-rank + diag on GPU, residual on CPU** | split precision by spectrum | SVD energy; kill if rank > 64 needed |
| **E6** | **Mem-mapped “execute weights”**: embed unpack in page-fault handler | OS | overhead per fault |
| **E7** | **Reversible residual stream** so some layers can run in INT8 end-to-end | revnets | quality |

Only promote E* if A–D kill.

---

## 4. Ranked program for *this* project’s goal

Priority = P(impact on novel tok/s) × honesty after retractions ÷ effort.

### Tier 0 — This week (instruments already exist)

| # | Experiment | Cost | Success looks like |
|---|---|---|---|
| 1 | **H-REOPEN**: true Q2_K bytes + sumx hoist + scalar min (no multi-row) | 1–2 days | microbench ≥0.8× L1g; 7B Q2_K tg → toward Q4_0 |
| 2 | **Q1 ladder**: 2.0 / 1.5 / 1.0 bit dp4a-native GW/s peak | 1 day | plot + turnover bit-width |
| 3 | **K3a–c**: dissect 0.79 vs 0.84 | hours | named tax |
| 4 | **K2a–c**: LOP3 / PRMT / uint4 on L1h | hours | +≥3% or kill unpack-micro path |
| 5 | **C4 prototype**: AVX2 hoist-sumx Q2 expert GEMV | 2–3 days | host expert GB/s ↑; novel split tg ↑ |

### Tier 1 — If Tier 0 says bits still buy GW/s

| # | Experiment | Notes |
|---|---|---|
| 6 | **B1** bit-plane early-out oracle (offline KL) | zero CUDA if done in torch |
| 7 | **B3** Δ-expert shared base | numpy on one layer |
| 8 | **D1** dual-format hybrid load | product-shaped |
| 9 | **A1** vector codebook on experts only | quality first |
| 10 | **C1** fused gate-up-down | after H-REOPEN |

### Tier 2 — Only if hybrid still stuck at ~21

| # | Experiment |
|---|---|
| 11 | Ghost-FFN residual speculation (from `BREAKTHROUGH_ANALYSIS.md`) — *not* a kernel unpack problem |
| 12 | C-09 20 ms attribution — may dwarf all of §0 |
| 13 | Linux hugepage DRAM ceiling |

---

## 5. What not to brainstorm further (brief already executed)

| Dead | Evidence |
|---|---|
| New asymmetric 2-bit **format** as the speed win | Q2_A < Q2_K cost model |
| Multi-row blocking to save Q2_K min term on Pascal | −22% e2e |
| Expert gather optimization | free at real block counts |
| Graph merge as main split-placement lever | +6.5% only |
| Chasing 4-bit e2e past ~1.1× | Q4_0 at 87% of ceiling |
| Claiming llama.cpp “doesn’t use dp4a” | 61 call sites |

---

## 6. Scientific measurement standards (kernelprobe-native)

1. **Same harness, same buffer, same session** for any unpack claim.  
2. **Correctness vs double host reference** before timing (existing rule).  
3. **GW/s primary**, GB/s secondary, tok/s only with full model.  
4. **quality.py RMSE** on residual-writers (`down_proj`, `o_proj`) + attention, not global average only.  
5. **Position-control** for any e2e (±10% thermal).  
6. **Retraction log**: if a creative idea dies, append to the brief’s table — that *is* the product.

---

## 7. One-paragraph strategy

The kernel session moved the problem from “llama.cpp is slow” to “**unpack arithmetic defines η(format)**,” and showed **4-bit is finished** on this GPU while **sub-3-bit GW/s is still climbing**. The creative frontier is not another MMVQ tiling paper; it is (1) **reclaim Q2_K health without multi-row** via sumx/scalar min on real bytes, (2) **find the GW/s peak below 2.5 bits** with quality gates, (3) **export that unpack philosophy to the CPU expert path** where novel generation actually spends its life, and (4) **skip planes/channels/deltas** so fewer weights are touched at all — with offline KL kills before any heroic CUDA. Anything that ignores hybrid placement and only polishes 4-bit VRAM matvec is entertaining physics and the wrong objective function.

---

## 8. Cross-links

| Doc | Role |
|---|---|
| `GROK_KERNEL_BRIEF.md` | Measured kernel state |
| `tools/kernelprobe/bench.cu` | Ladder L0–L1h |
| `tools/kernelprobe/quality.py` | Format RMSE vs Q2_K |
| `BREAKTHROUGH_ANALYSIS.md` | System-level novel-gen axioms A/B/C |
| `UNTESTED_STRATEGIES.md` | Broad remainder catalog |
| `BREAKTHROUGH_BRIEF.json` L-11, L-14, C-05, D-10 | Binding constraints |

---

### Immediate recommendation

**Run H-REOPEN next.** It is the only idea that simultaneously:

- respects every retraction,  
- uses the existing kernelprobe harness,  
- could convert an “unclaimable” 2.07× into a real Q2_K e2e win,  
- and if it fails, cleanly confirms the brief’s ceiling so effort pivots to **CPU export (C4)** and **byte-skipping (B1/B3)** without guilt.


---

# ENRICHMENT (2026-07-28, post-review): corrections, new ideas, and kills-by-arithmetic

*Everything below follows the house rule: an idea is only listed with either a falsifiable
prediction + kill test, or the arithmetic that already kills it. Killed ideas are kept — the
retraction log IS the product (§6.6).*

## E-0. Corrections to the document above, from source and from the ladder

**E-0a. H-REOPEN is bigger than stated — it is a K-QUANT FAMILY defect, not a Q2_K defect.**
Read from `vecdotq.cuh:518` (`vec_dot_q4_K_q8_1_impl_vmmq`):

```c
const int dot2 = dp4a(0x01010101, u[2*i+1], dp4a(0x01010101, u[2*i+0], 0)); // sum of u
```

**Q4_K spends 2 of its 4 dp4a per iteration computing `sum(u)`** — row-invariant, recomputed for
every one of 768+ output rows. Same pattern in Q5_K and Q2_K (Q2_K via `dp4a(m_broadcast, u)`).
So the prize is not "make Q2_K healthy" — it is "remove the min-term recomputation from the
most-downloaded format in the ecosystem (Q4_K_M) on every ALU-weak GPU." Exactly half of Q4_K's
dp4a budget is sums. Measured e2e gap consistent with this: Q4_K_M η 0.553 vs Q4_0 η 0.619.

**E-0b. The fix is NOT blocking, NOT `block_q8_1` surgery — it is a per-token side buffer.**
#55 refuted multi-row blocking (−22%) and my "mutually exclusive fixes" claim missed the third
option: precompute per-4 (or per-8) activation sums ONCE PER TOKEN into a side array
(K/4 ints = 2 KB — L2-resident, every block reads the same values). CUDA-backend-local;
precedent exists (`quantize_mmq_q8_1` already uses a path-specific activation layout).
Mechanism bonus: replaces ALU-port dp4a with LD/ST-port cached loads — on Pascal those
dual-issue, so the win can exceed the naive instruction count.

**E-0c. H-REOPEN's ingredient (c) "warp-per-row like L1h" is a distraction.** L1h measured
+2.6% at 4 bits. The essential ingredients are (a) hoisted/cached sums and (b) scalar min FMA.
Test at llama.cpp's own 1-row-per-block geometry or the result will not transfer (the L1g
356 GW/s number used 768-rows-per-block, which #55 proved is NOT available on this card).

**E-0d. §0.3's "≤ ~1.11× kernel headroom at 4-bit" row should carry its decomposition:**
0.62→0.69 kernel engineering, 0.69→0.79 unpack floor (no known instruction recovers it),
0.79→0.84 matvec-vs-stream (cause unknown — K3 is the experiment that names it).

## E-1. New ideas (N-series), each with its verdict path

**N1 — Per-token activation-sum side buffer for ALL asymmetric K-quants** ⭐ *(the upgraded
H-REOPEN; supersedes §1's version)*
Oracle: kernelprobe, 1-row-per-block mmvq geometry, three arms at 4.5-bit asymmetric cost model:
(i) min-via-dp4a (llama.cpp exact), (ii) min-via-cached-sums, (iii) symmetric control (no min).
Prediction to stake: (ii) recovers ≥ 60% of the (i)→(iii) gap. If it holds at 4-bit, patch
`vec_dot_q4_K` in-tree, A/B on the real 7B Q4_K_M, then upstream — a mainstream-format PR, far
stronger than a Q2_K-only story. **KILL** if (ii)−(i) < 5%: cached loads do not dual-issue as
hoped, and the K-quant tax is unremovable at this geometry.

**N2 — fp16 attention plane in an otherwise-quantized file: KILLED BY ARITHMETIC.**
Per-weight decode time: Q4_0 = 4.5/(8·119) ≈ 4.7 ps/w; fp16 = 16/(8·152) ≈ 13.2 ps/w. fp16 is
~3× slower *per weight* despite the healthier kernel — the bytes always win at these ratios.
Only revisit on a card where the quantized path is < 0.35 of spec (nothing we measured is).

**N3 — Q8_0-everything for the unpack-free path: KILLED BY EXISTING DATA.**
L1c int8-dp4a = 117–120 GW/s vs Q4_0-dp4a 227–235 GW/s. Halving the ALU per byte does not pay
for doubling the bytes. Already measured; recorded so nobody re-derives it.

**N4 — "Use both buses": stream compressed experts over PCIe to the GPU while the CPU computes
its own experts concurrently: KILLED BY A SHARED RESOURCE.** PCIe DMA sources from the same
host DRAM the CPU expert path already saturates (28 GB/s measured vs ~30 GB/s ceiling). The two
"buses" share one memory controller; the parallelism is an illusion. Residual variant (worth one
oracle only if desperate): measure total DRAM throughput under concurrent DMA+stream — if the
controller sustains > 34 GB/s combined, ~+20% exists. Predicted: it does not.

**N5 — The sharpest open number: split-GPU η 0.15 vs all-in-VRAM Q2_K η 0.34.** ⭐
L-15 now explains *half* of the old C-02 split mystery: the flagship IS Q2_K, and Q2_K's format
tax alone prices η ~0.34. The remaining ×2.3 (0.34→0.15) is the true residual, and no current
hypothesis covers it. Candidates: per-boundary activation quant/dequant roundtrips (33/token),
attention kernels at batch-1 occupancy, non-matvec ops. Experiment: extend E6 with per-op CUDA
event attribution on the split graph (the instrument exists; one day). This is the most
e2e-relevant unknown for the actual 21-tok/s flagship goal — kernel polish is second to it.

**N6 — Odd-level symmetric 2-bit {−3,−1,+1,+3}: quality oracle costs 60 seconds.**
quality.py showed 4-level symmetric-with-zero {−2,−1,0,+1} loses 15% RMSE to asymmetric.
Odd levels drop the zero but double the top level's reach and stay dp4a-native (int lanes).
Run the numpy oracle before any CUDA. **KILL** if RMSE > asym-g16's 0.995× parity — likely,
but it is the only remaining symmetric candidate and symmetric = no min term = no N1 needed.

**N7 — L2-prefetch of next layer's weights: KILLED BY CAPACITY.** GTX 1060 L2 = 1.5 MB;
one flagship layer's GPU share ≈ 15–21 MB. Nothing fits; nothing to schedule.

**N8 — T-MAC-style activation-product LUT decode (GPU): recorded with its kill math.**
Store 4×2-bit weights as one byte index; per token build LUTs of partial dot products; inner
loop = byte load → smem lookup → add. Zero unpack, and the codebook may be NON-uniform for free.
Kill math: full-quad LUT needs 256 entries × K/4 quads × 4 B = 512 KB smem (Pascal has 96 KB/SM)
— dead. Bit-plane variant (16-entry LUTs) fits in 32 KB but costs 8 lookups+adds per 16 weights
vs dp4a's 4 ops — op-count negative, and smem gather is not faster than dp4a issue. On CPU
(PSHUFB) the same trick targets a tier already at its memory ceiling (L-11), so there is nothing
to buy there either. **Dead on this hardware generation; revisit only where smem ≥ 256 KB/SM.**

**N9 — Verify-round batch as the legitimate multi-row amortizer.** #55 killed multi-row at
batch 1, but the mmvq batch path (ncols_dst 2–8) already runs 2 rows/block — and speculation's
verify round IS a batch. The min-term sums amortize across rows there naturally. No new kernel:
this is an argument for re-scoring speculation economics on K-quant models specifically —
the verify round is cheaper per token on Q4_K_M than batch-1 decode is, by more than the
generic batching argument predicts. Oracle: batch sweep, Q4_0 vs Q4_K_M — if the K-quant batch
speedup exceeds the Q4_0 batch speedup, the effect is real and `speculation_advice()` should
know the format.

**N10 — Sub-2.5-bit dp4a ladder (extends Q1/K1): the turnover is one afternoon.**
2.0-bit (no superblock scale), 1.585-bit ternary-in-8 (5 trits/byte via base-3 LUT — LUT
unpack, watch N8's lesson), 1.0-bit sign+scale (dp4a on ±1 lanes). Stake the GW/s curve shape
BEFORE running; the turnover bit-width is a publishable constant either way. Quality gate:
the RMSE ladder from quality.py decides which points are even eligible.

## E-2. Revised Tier 0 (supersedes §4, with what we now know)

| # | Experiment | Cost | Why first |
|---|---|---|---|
| 1 | **N1 side-buffer oracle at mmvq geometry** (prereg #56) | hours | mainstream-format upstream shot; corrects #55's false closure |
| 2 | **N6 odd-level quality oracle** (numpy) | minutes | decides if any symmetric 2-bit survives |
| 3 | **K3a–c: name the 0.79 tax** | hours | decides if int8 activations are mandatory everywhere |
| 4 | **N10 bit ladder** | afternoon | GW/s turnover constant |
| 5 | **K2a–c: LOP3/PRMT/uint4** | hours | closes or kills the last 1.15× unpack gap |
| 6 | **N5 split-η per-op attribution** | 1 day | the flagship's actual wall; everything above is VRAM-side |

Dropped from Tier 0: H-REOPEN-as-Q2_K-only (superseded by N1), C4 AVX2 export (the CPU tier is
at its memory ceiling per L-11 — re-check only if N1's mechanism suggests the CPU path also
wastes ALU, which E3's measured 28 GB/s says it does not).
