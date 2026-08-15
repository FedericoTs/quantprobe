# The evaluation table — one 2016 machine, every strategy

Rows are benchmarks, columns are model x inference strategy, **every cell is a
committed measurement** (`weights/data/grid_*.json`; the generator renders a dash
for anything it cannot cite). Machine: GTX 1060 6GB + 16GB DDR4, llama.cpp b10098,
placements planned by quantprobe. `lanes` = 16 sampled candidates in parallel server
slots, winner picked by BASE tests only, scored on the hidden plus set (selection
never sees the exam).

| benchmark | Qwen3-0.6B<br>single | Qwen3-0.6B<br>lanes16 | Qwen3.5-4B<br>single | Qwen3.5-4B<br>lanes16 | Qwen2.5-7B<br>single | Qwen2.5-7B<br>lanes16 | Qwen3-Coder-30B<br>single |
|---|---|---|---|---|---|---|---|
| MBPP+ (371, plus tests) | 36.7 | 56.9 | 66.0 | 75.2 | 68.5 | **75.7** | 75.5 |
| HumanEval+ (164, plus tests) | 24.4 | 56.1 | 76.8 | **88.4** | 72.0 | 84.8 | 87.8 |

Median wall-clock per task (seconds), same cells:

| benchmark | 0.6B single | 0.6B lanes | 4B single | 4B lanes | 7B single | 7B lanes | 30B single |
|---|---|---|---|---|---|---|---|
| MBPP+ | 0.4 | 3.3 | 2.1 | 16.6 | 2.7 | 19.0 | 3.4 |
| HumanEval+ | 1.4 | 8.1 | 7.5 | 37.1 | 7.9 | 44.8 | 9.5 |

Notes:
1. The bold cell per row is the best score **on this machine** - on both benches it
   is a lanes column, not the biggest model. Verified test-time compute beats
   parameter count here (Phase A verdict, prereg 2026-08-05).
2. Lanes assume executable tests exist (the verification regime); wall-clock is
   charged honestly - all 16 candidates plus selection execution.
3. The 30B lanes column is absent by staked prior evidence (U-39: MoE expert-offload
   batching caps ~2x), not omission.
4. Columns arriving with the program: k=32 arms, early-exit lanes (P0b), and the
   Phase C/D tuned models - every phase adds a column, and misses stay on the table.
5. The depth-aware-quant campaign has LANDED and has its own page - benchmark
   scores across quant variants (naive vs recipe vs original) on MATH-500/GSM8K/
   IFEval, with the size-dependence law and the hybrid-generalization result:
   [does depth-aware quantization preserve capability?](QUANT_QUALITY.md).
