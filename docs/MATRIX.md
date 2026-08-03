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
| DGX Spark - 128GB unified @ 273 | **231** R | **34** R | **18** R | **107** V | **77** V | **50** V | **16** V | **28** V | **1** d | 0.93 d | 0.30 d |
| 2x DGX Spark - 256GB unified @ 273 *(capacity, not speed)* | **231** R | **34** R | **18** R | **107** V | **77** V | **50** V | **16** V | **28** V | **3** d | **1** d | 0.35 d |
| 4x DGX Spark - 512GB unified @ 273 *(capacity, not speed)* | **231** R | **34** R | **18** R | **107** V | **77** V | **50** V | **16** V | **28** V | **12** V | **12** V | 0.49 d |
| EPYC server - 512GB DDR5 8ch, no GPU | **169** R | **25** R | **13** R | **50** R | **36** R | **23** R | **8** R | **13** R | **6** R | **6** R | 0.50 d |

*Italic* = under 1 tok/s, i.e. a capacity demo rather than usable inference.

## The DGX Spark rows are the interesting ones

Adding Sparks multiplies **capacity**, not speed. Each unit is 128 GB at 273 GB/s, and
linking them does not raise per-unit bandwidth - a token still traverses every layer in
sequence. So 4x Spark lets you *hold* a 2.4T model that one unit cannot, at roughly the
same tok/s as one unit running something that fits. That is Law 4 stated as a purchase
decision: buy Sparks to fit a bigger model, not to run the same model faster.

