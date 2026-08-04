# What to run, and how fast

Every cell is Law 4 through quantprobe's shipped `evaluate()` - the same function
`quantprobe plan` calls. The letter after each number is **which resource binds**:
`V` VRAM bandwidth, `R` RAM bandwidth, `D` disk, `C` CPU compute. A speed without its
binding constraint is not actionable - *3 tok/s, disk-bound* means buy RAM, *3 tok/s,
bandwidth-bound* means do not bother.

**One row here has been measured** (the 2016 desktop: 14 models, median 8.4% absolute
error). Every other row is the law applied to spec-sheet bandwidth. The all-in-VRAM
placement is a documented **floor** - measured speed came in at or above the printed
number on 13 of 13 benchmarks, typically 1.1-1.8x higher (C-02). Read unmeasured rows
as lower bounds.

| machine | Qwen3-0.6B Q8_0<br><sub>1 GB</sub> | Qwen2.5-7B Q4_K_M<br><sub>5 GB</sub> | Qwen2.5-14B Q4_K_M<br><sub>9 GB</sub> | Qwen3-30B-A3B Q2_K<br><sub>10 GB</sub> | Qwen3.5-35B-A3B Q4_K_M<br><sub>21 GB</sub> | gpt-oss-120B Q4_K_M<br><sub>73 GB</sub> | Qwen3-235B-A22B Q2_K<br><sub>79 GB</sub> | DeepSeek V4-Flash Q2_K<br><sub>96 GB</sub> | GLM-5.2 753B Q2_K<br><sub>254 GB</sub> | Kimi-K2.6 1058B Q2_K<br><sub>357 GB</sub> | Qwen3.8-Max Q2_K<br><sub>810 GB</sub> |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2016 desktop - GTX 1060 6GB + 16GB DDR4 *(MEASURED)* | **131** V | **19** V | **2** R | **11** R | 0.60 d | 0.24 d | *0.09* d | 0.15 d | *0.05* d | *0.05* d | *0.02* d |
| laptop - iGPU + 16GB DDR4 | **20** R | **3** R | **2** R | **6** R | **3** d | **1** d | 0.50 d | 0.83 d | 0.32 d | 0.31 d | 0.11 d |
| laptop - iGPU + 64GB DDR4 (ProBook class) | **43** R | **6** R | **3** R | **13** R | **9** R | **3** d | **1** d | **2** d | 0.41 d | 0.37 d | 0.13 d |
| RTX 3060 12GB + 32GB DDR4 | **246** V | **36** V | **19** V | **118** V | **13** R | **2** d | 0.70 d | **1** d | 0.38 d | 0.35 d | 0.13 d |
| RTX 4090 24GB + 64GB DDR5 | **852** V | **125** V | **65** V | **408** V | **293** V | **6** R | **2** R | **3** d | 0.79 d | 0.73 d | 0.25 d |
| RTX 5090 32GB + 128GB DDR5 | **1515** V | **222** V | **116** V | **726** V | **520** V | **14** R | **6** R | **9** R | 0.96 d | 0.83 d | 0.26 d |
| 2x RTX 4090 48GB + 128GB DDR5 | **1450** V | **212** V | **111** V | **694** V | **498** V | **14** R | **6** R | **9** R | 0.96 d | 0.83 d | 0.26 d |
| DGX Spark - 128GB unified @ 273 *(checked vs reports)* | **231** R | **34** R | **18** R | **107** V | **77** V | **50** V | **16** V | **28** V | **1** d | 0.93 d | 0.30 d |
| 2x DGX Spark - 256GB via RPC *(UPPER BOUND - RPC)* | **231** R | **34** R | **18** R | **107** V | **77** V | **50** V | **16** V | **28** V | **3** d | **1** d | 0.35 d |
| 4x DGX Spark - 512GB via RPC *(UPPER BOUND - RPC)* | **231** R | **34** R | **18** R | **107** V | **77** V | **50** V | **16** V | **28** V | **12** V | **12** V | 0.49 d |
| EPYC server - 512GB DDR5 8ch, no GPU | **169** R | **25** R | **13** R | **50** R | **36** R | **23** R | **8** R | **13** R | **6** R | **6** R | 0.50 d |

*Italic* = under 1 tok/s, i.e. a capacity demo rather than usable inference.

## Scored against third-party DGX Spark reports

The Spark rows are the only ones anyone else has published numbers for, so they are the
closest thing we have to out-of-sample validation. On a single unit:

| model | our prediction | third-party report | ratio |
|---|---|---|---|
| 32B dense Q4 | 8.4 | 10.7 | **1.27x low** |
| 30B-A3B MoE Q4 | 81.7 | 89.0 | **1.09x low** |
| Gemma-4-26B A4B Q4 | 67.4 (depth-blind) | 51.6 | *resolved below* |

The first two land inside C-02's floor band (real speed 1.1-1.8x above the printed
number). The third looked like a violation for two days and is now **resolved with the
model's own GGUF header** (read remotely, 2026-08-04): our active-parameter figure was
fine (header 3.82B vs the ~4.0B we used) - but gemma4 carries **480 KB of KV per
position, 5x Qwen-class**. Our 67.4 was a zero-depth floor; priced at the reporter's
plausible 1-2k context the same floor gives 58.0-49.2 tok/s, bracketing the 51.6
report inside the C-02 band. **No C-02 exception exists.** The lesson is now a rule:
third-party reports get scored at their stated context, with kvp read from the header,
or they do not get scored.

## Adding Sparks buys capacity, not speed

Each unit is 128 GB at 273 GB/s, and linking them does not raise per-unit bandwidth - a
token still traverses every layer in sequence. So 1x, 2x and 4x give identical tok/s on
anything that already fits one unit. What 4x buys is the first configuration where
GLM-5.2 (753B) and Kimi-K2.6 (1058B) become usable at ~12 tok/s instead of ~1.

**The 2x/4x rows are upper bounds.** They model unified memory; real multi-node
llama.cpp uses RPC. The one public datapoint - GLM-5.2 UD-IQ1_S on 2x Spark at 256K
context - reports 8 tok/s where unified-memory arithmetic gives 23.7. We are **3x
optimistic** there. One datapoint is not a coefficient, so nothing has been fitted to
it; the gap is disclosed and the rows are labelled.

## A number going around that cannot be true

"DGX Spark runs 70B Q4 at 35-45 tok/s" appears in several write-ups. A 70B dense model
at Q4 moves **42.5 GB per token**. At 273 GB/s the ceiling - perfect efficiency, eta =
1.0, no overhead of any kind - is **6.4 tok/s**. Reaching 35-45 would need 1,488-1,914
GB/s, i.e. **5.5-7x the bandwidth the hardware physically has**.

Whatever those benchmarks measured, it was not single-stream decode of a 70B dense
model: most likely batched throughput, prompt processing, or an MoE mislabelled as
dense. This is the sort of claim Law 4 is *for* - you do not need the machine to know
the number is impossible, only the bandwidth and the bytes.

## And one that was reported honestly, then relayed wrong

[kimi-k3-in-c](https://github.com/FareedKhan-dev/kimi-k3-in-c) reached us as "running
Kimi at 10 tok/s". Its README says no such thing - it reports **seconds per token**:
`~32 s/token` on the laptop preset, `~19-21 s/token` on the server, and a run line
reading *8 tokens in 261.5 s*. That is 0.03-0.05 tok/s. The claim as it travelled was
the reciprocal, off by 200-320x. We made the same slip on first read.

Its four presets are a genuine out-of-sample test. **We first scored them with the
wrong byte model, and the correction is published here at the same size** (2026-08-04):
our original note assumed only routed experts move (23.8 GB/token). The repo's own
`docs/data` states the engine re-reads the **trunk in full every token - 108.81 GB -**
plus ~25.8 GB of touched experts: **134.6 GB/token, trunk-dominated.** Under the
corrected two-tier arithmetic (trunk cached up to RSS, remainder + experts from NVMe):

| preset | RSS | measured | Law 4, corrected bytes |
|---|---|---|---|
| laptop | 8.2 GB | 32.69 s/tok | **32.6 - within 0.3%** |
| desktop | 31.9 GB | ~29.5 s/tok | 27.9 |
| workstation | 95.5 GB | ~24 s/tok | 15.2 - we over-predict the gain |
| server | ~128 GB | ~20 s/tok | 11.9 - we over-predict the gain |

The low-RAM end lands to 0.3% with ordinary bandwidths (4 GB/s NVMe, 20 GB/s DDR).
The high-RAM end runs ~1.7x slower than full-trunk-caching predicts - *consistent
with* their ladder's stated hard cgroup caps, which throttle page cache; that residual
is a property of their harness we cannot verify from here, and we say so rather than
fit it.

What survives untouched: the naive rival - *it is slow because it does not fit, add
RAM* - predicts speed tracks resident set, **15.6x** from laptop to server. Measured:
**1.63x**. What we corrected: our original "Law 4 predicts ~1x because experts cannot
be cached" was also wrong - with the true bytes the tiered model predicts ~3.8x by the
server preset. Law 4 wins the anchored end exactly and over-predicts the capped end;
a smaller, honest win, not the clean sweep we first published.

And the relayed "10 tok/s" is now further away, not closer: 134.6 GB/token needs
**1,346 GB/s sustained**. Even a trunk fully resident in HBM leaves 25.8 GB/token of
expert traffic. No single box gets there.

