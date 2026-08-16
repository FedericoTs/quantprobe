# DESIGN: Prereg #95 stage 1 — Morris screening of llama.cpp flags

**Serves:** `preregistrations/2026-08-07-doe-flag-screening.md` (STAKED 2026-08-07). This
document designs the measurement; it never edits the stake. Every deviation the design was
forced into is listed in section 5 as a ready-to-append amendment, which the OPERATOR appends
to the prereg before the run. The harness never touches the prereg.

**Design date:** 2026-08-16, before any screening data exists. Scorer spec (section 4) is
precommitted now, kill-rule discipline: the scoring code must exist and be frozen before the
first CSV row.

**Box state at design time:** GTX 1060 6GB (6143 MiB, CC 6.1), Intel i5-7600K (4C/4T),
llama.cpp build `0278d8362` / b10098 at
`C:/Users/Federico/Documents/evo-compress/tools/llamacpp-b10098/llama-bench.exe`.
Source: `llama-bench --help` startup banner and the CSV `build_commit,build_number,cpu_info`
fields from the flag probes run 2026-08-16 (quoted below).

Models, exact paths, sizes verified on disk 2026-08-16:

| tag | file | bytes | regime |
|---|---|---|---|
| 7B  | `D:/evo-compress-data/gguf/Qwen2.5-7B-Instruct-Q4_K_M.gguf` | 4,683,074,240 (4.68 GB) | all-in-VRAM |
| 30B | `D:/evo-compress-data/gguf/Qwen3-30B-A3B-Q2_K.gguf` | 11,258,610,240 (11.26 GB) | CPU-expert split |

**DEVIATION (known before design):** the prereg names Qwen3-30B-A3B **Q2_K_L**; the file we
hold is plain **Q2_K**. Recorded in section 5. NEVER substitute Qwen3-Coder-30B — different
finetune; the cross-model join failure is exactly what prereg #102's verify pass caught.

---

## 1. Factor exercisability audit

The prereg stakes 8 factors. For each: the exact flag on OUR build (quoted from
`llama-bench --help`, build 0278d8362/b10098), the levels per model, and whether llama-bench
tg128 can exercise it at all. Verdicts were probed live on 2026-08-16 against
`Qwen3-0.6B-Q8_0.gguf` (cheap, seconds per probe); probe outputs are quoted verbatim.

### 1.1 `-ngl` — EXERCISABLE

Help line: `-ngl, --n-gpu-layers <n>                    (default: -1)`

Default -1 offloads everything; we always pass an explicit value so the factor is what we
set, not what the build decides.

- 7B levels: `{0, 9, 19, 99}`. 99 = everything, matching the shipped emit (`-ngl 99` in
  `weights/data/ladder_state_locked.json`). Intermediate levels assume Qwen2.5-7B has 28
  blocks [est — harness asserts `block_count == 28` from the GGUF header at startup and
  aborts if wrong, because the level meaning would drift].
- 30B levels: `{0, 16, 32, 99}`. Qwen3-30B-A3B has 48 blocks (source: the shipped ladder
  `-ot` pattern enumerates blk 15..47 of 0..47).
- Note: at ngl=0 on the 30B, the `-ot` factor below is inert (nothing is on the GPU to
  override). That is a real interaction; Morris reports it as sigma, not a design bug.

### 1.2 `-ub` — EXERCISABLE

Help line: `-ub, --ubatch-size <n>                      (default: 512)`

- Both models: `{128, 512, 1024, 2048}`. `-b` is pinned at its default 2048 for every run
  (so ub <= b always holds; `-b` is not a staked factor).
- Caveat, stated before data: the +73%/-39% ub effects the prereg cites (prereg #19) were
  measured on the serving path. The stage-1 response is tg128 decode only; a small or null
  ub main effect here is a legitimate screening outcome, not a harness failure. P-4 stakes
  ub's **sigma**, which tg128 can show.

### 1.3 `-t` — EXERCISABLE

Help line: `-t, --threads <n>                           (default: 4)`

- Both models: `{1, 2, 3, 4}`. The box CPU is `Intel(R) Core(TM) i5-7600K CPU @ 3.80GHz`
  (probe CSV `cpu_info`), 4 cores / 4 threads, no SMT. t > 4 is oversubscription of a 4-lane
  part and is excluded from the plausible tuning range.

### 1.4 KV cache type — EXERCISABLE AS `-ctk` ONLY (deviation)

Help lines: `-ctk, --cache-type-k <t>                    (default: f16)` and
`-ctv, --cache-type-v <t>                    (default: f16)`

Probe (2026-08-16): `-ctk q8_0 -ctv q8_0 -fa 0` fails —

    llama_bench: error: failed to create context with model 'D:/evo-compress-data/gguf/Qwen3-0.6B-Q8_0.gguf'

V-cache quantization requires flash attention on this build, so a KV factor that sets both
`-ctk` and `-ctv` **cannot vary independently of `-fa`**: the cell (KV=q8_0, fa=off) is
infeasible and would deterministically DNF every trajectory that visits it. Morris is a
hypercube method; a structurally infeasible cell breaks it.

Resolution: the KV factor exercises **`-ctk` only**, levels `{f16, q8_0}`, with `-ctv`
pinned f16. Probed independent: `-ctk q8_0 -fa 0` runs fine (probe CSV shows
`type_k=q8_0, type_v=f16, flash_attn=0`, 78.56 tok/s on the 0.6B). Deviation recorded in
section 5: U-01 measured both K and V; stage 1 screens K only.

### 1.5 `-ot` expert-offload fraction — EXERCISABLE ON 30B ONLY (dense 7B: inert)

Help line: `-ot --override-tensor <tensor name pattern>=<buffer type>;...` /
`                                            (default: disabled)`

Probe (2026-08-16): `-ot "exps=CPU"` on a DENSE model loads and runs with the override
recorded but matching zero tensors (probe CSV `tensor_buft_overrides=exps=CPU`, tok/s within
noise of baseline). It does not error — it is **silently inert**. A factor that provably
cannot move the response has no place in a screening design; including it would spend
R = 10 runs measuring zeros.

- 7B: **dropped** (dense model, no `_exps` tensors). k drops to 6. Deviation in section 5.
- 30B levels, as fraction f of the 48 expert layers whose experts live on CPU:
  `{0.75, 0.833, 0.917, 1.0}` = last `{36, 40, 44, 48}` blocks. Generated patterns:
  - f = 1.0: `exps=CPU`
  - f < 1.0, CPU set = blk (48-N)..47, e.g. f = 0.75 (N = 36):
    `blk\.(12|13|...|47)\.ffn_.*_exps\.=CPU` (same direction as the shipped recipe, which
    keeps EARLY blocks' experts on GPU: ladder emit holds blk 15..47 on CPU).
  - The range floor 0.75 is a VRAM feasibility bound: the worst corner (ngl 99, f 0.75,
    ub 2048) puts 12 expert layers plus all attention on the 6 GB card. The ladder row ran
    15 expert layers on GPU at ub 1024 without OOM (ladder_state_locked.json), so 12 at
    ub 2048 should fit [est — the pre-flight corner check in section 3 proves it before the
    night starts]. Fractions below 0.75 OOM the max-ngl corner and are excluded.
- Build alternative NOT used: `-ncmoe, --n-cpu-moe <n>                    (default: 0)`
  offloads the FIRST n layers' experts — the opposite end of the stack from the shipped
  recipe. We exercise the staked flag (`-ot`) in the shipped direction instead.

### 1.6 `--no-mmap` — EXERCISABLE AS `-mmp 0`

Help line: `-mmp, --mmap <0|1>                          (default: 1)`

llama-bench dialect: `--no-mmap` (server spelling) = `-mmp 0` here, the same translation
`full_ladder_v124.py::flags_from_emit` and `autotune_sweep.py::bench_args` already use.
Both models: `{1, 0}`.

### 1.7 `-np` — **NOT EXERCISABLE. DEVIATION, declared, not silently dropped.**

`-np` does not appear anywhere in this build's `llama-bench --help`. Probe (2026-08-16):

    error: invalid parameter for argument: -np

`-np` is a llama-server concurrency flag (swept in U-05 via the server harness, see the
`p0_server_*_np2/np4.log` files). llama-bench tg128 is a single-stream benchmark and cannot
exercise it. Stage 1 therefore screens **7 of the 8 staked factors**; `-np`'s screening is
deferred to a server-harness arm, and the amendment (section 5) says so explicitly.

### 1.8 `-fa` — EXERCISABLE

Help line: `-fa, --flash-attn <on|off|auto>             (default: auto)`

Probe (2026-08-16): numeric `-fa 1` is accepted and recorded (probe CSV `flash_attn=1`);
`-fa 0` likewise (`flash_attn=0`); the default `auto` serializes as `-1`. Levels, both
models: `{0, 1}`. **`auto` is never passed** — auto lets the build pick per-config, which
hides the factor inside the response. Pascal (CC 6.1) flash-attention kernels may be a
regression rather than a win on this card; that is precisely what screening measures.

### Audit summary

| factor | flag on this build | 7B levels | 30B levels |
|---|---|---|---|
| gpu layers | `-ngl` | 0 / 9 / 19 / 99 | 0 / 16 / 32 / 99 |
| microbatch | `-ub` | 128 / 512 / 1024 / 2048 | same |
| threads | `-t` | 1 / 2 / 3 / 4 | same |
| KV type (K only) | `-ctk` | f16 / q8_0 | same |
| expert offload | `-ot` (generated pattern) | DROPPED (inert on dense) | f = 0.75 / 0.833 / 0.917 / 1.0 |
| mmap | `-mmp` | 1 / 0 | same |
| flash attn | `-fa` | 0 / 1 | same |
| concurrency | `-np` | NOT EXERCISABLE (llama-bench rejects; server-only) | same |

k = 6 (7B), k = 7 (30B). Pinned for every run: `-b 2048`, `-ctv f16`, `-n 128 -p 0 -r 3`,
`-o json`, warmup ON (default; `--no-warmup` exists and is not used).

---

## 2. Morris design

- **k:** 6 (7B), 7 (30B) — after the audit above.
- **R = 10 trajectories** per model. Standard practice for k <= 10 screening; matches the
  25-125-run budget band the prereg quotes for the upstream tool.
- **p = 4 levels**, grid `{0, 1/3, 2/3, 1}` per factor, **delta = p/(2(p-1)) = 2/3** — the
  canonical Morris choice; every elementary effect crosses the midpoint, so 2-level factors
  (ctk, mmp, fa: level = 0 if grid < 0.5 else 1) flip on every delta step and are never
  stranded.
- **Numeric mapping:** grid index round(g*3) selects from the level lists in section 1.
- **Response:** tg128 tok/s = `avg_ts` from llama-bench JSON, per-rep samples read from
  `samples_ns` (read, not inferred — the unattended_serial lesson).

### Deterministic trajectory generation (no clocks, no unseeded randomness)

    rng = random.Random("prereg95:" + model_tag + ":20260807")   # model_tag in {"7B","30B"}

For each trajectory r in 0..R-1, drawn strictly in this order from that one stream:

1. base: for each factor i, start s_i drawn uniform from {0, 1/3} (the grid points from
   which +delta stays in [0,1]);
2. direction d_i drawn uniform from {+1, -1}; if d_i = -1 the factor starts at s_i + 2/3
   and steps down by delta instead of up;
3. order: a permutation of the k factors via rng.shuffle.

Trajectory = k+1 configs: the base, then one factor stepped at a time in permutation order.
Same seed => same design, forever; the design is regenerable from this paragraph alone.

### Run count and runtime budget

| block | runs R*(k+1) | per-run wall | settle | block total |
|---|---|---|---|---|
| pre-flight corners | 4 + 4 (not in CSV) | ~40 s [est] | none | ~10 min [est] |
| 7B | 10*(6+1) = **70** | mean 60 s [est] | 45 s [est] | ~2.0 h |
| 30B | 10*(7+1) = **80** | mean 90 s [est] | 45 s [est] | ~3.0 h |
| **total** | **150 designed runs** | | | **~5.2 h nominal [est]** |

Per-run cost traces:

- 7B: `ladder_state_locked.json` row "Qwen2.5-7B Q4_K_M": `bench_s: 17` for tg64 r=2
  including load (bench cmd `-n 64 -p 0 -r 2`, `full_ladder_v124.py::bench`), measured
  22.73 tok/s. tg128 r=3 adds (4*128 - 3*64) = 320 gen tokens (warmup counted) ~= +14 s
  => ~31 s typical; slow corners (ngl 0, t 1: pure-CPU 7B [est 4-6 tok/s]) => 90-130 s.
  Mean across the design [est] 60 s.
- 30B: same file, row "Qwen3-30B-A3B Q2_K": `bench_s: 36` for tg64 r=2 **including the
  11.26 GB no-mmap load**, measured 21.71 tok/s. tg128 r=3 => ~51 s typical; mmap=1 rows
  reload from warm page cache and save most of the ~20 s load [est]; slow corners (ngl 0,
  t 1) [est] 120-200 s. Mean across the design [est] 90 s.
- Settle: thermal settle between runs per the llama-optimize practice the prereg credits
  (and our own #60/#61 stuck-boost scar: 28% and it looked like nothing): poll every 5 s
  until `temperature.gpu <= 52 C`, minimum 30 s, cap 180 s. Idle temp measured 36 C at
  design time; 52 C threshold is [est]. Typical settle [est] 45 s.

Fit to the overnight window (~10 h): nominal ~5.2 h — 1.9x headroom. Absolute worst case
(every run DNF at its timeout cap: 70*(240+45) + 80*(360+45) s) = ~14.7 h, which cannot
happen short of a broken build, but the design makes overrun harmless anyway:

- **deadline guard:** the harness takes `--deadline-hours` (default 9.5) and checks it
  before launching each run; on breach it stops cleanly, releases the lock, and logs
  `DEADLINE reached - resume tomorrow night`;
- **resume safety:** completed run_ids are skipped on relaunch (section 3), so a second
  night finishes exactly the runs the first night did not. The scorer refuses partial data
  (section 4), so nothing can be scored in between.

Block order: 7B first (shorter block, all-in-VRAM, no RAM pressure), then 30B.

---

## 3. Harness spec — `weights/doe_morris.py`

Serial runs only: one measurement owns the box (C-14). No parallelism anywhere.

1. **Lock + orphan kill, via `weights/runner.py` — REUSED, not copied.**
   - Register `".doe_lock"` in `runner.LOCK_NAMES` (that tuple edit is the whole
     registration step, per runner.py's own docstring;
     `t_every_runner_guards_against_every_other_lock` fails if forgotten).
   - `with runner.owns_the_box(".doe_lock", DATA):` refuses while ANY other lock exists
     and releases on every exit path.
   - `runner.kill_orphans("llama-server.exe", "llama-bench.exe")` before the first run
     (orphans squat VRAM and corrupt the next number).
2. **Unique log:** `weights/data/doe_morris_<YYYYmmdd_HHMMSS>.log` via `runner.make_log`.
   Never a shared filename.
3. **Startup assertions:** model files exist with the exact byte sizes above; GGUF
   `block_count` is 28 (7B) and 48 (30B), else abort — level meanings drift otherwise;
   if the CSV already exists, its header line's sha256 must equal the pinned hash in
   section 4, else refuse to resume into a drifted schema.
4. **Pre-flight corner check** (before any designed run): per model, run the 4 extreme
   corners (all-grid-0; all-grid-1; max-VRAM corner = max ngl + max ub + min f; max-CPU
   corner = ngl 0 + t 1 + mmp 0) at `-n 8 -p 0 -r 1`. Any OOM/nonzero exit => ABORT the
   night with the failing corner named. Ten minutes spent so a structural infeasibility
   cannot eat eight hours as serial DNFs. Pre-flight results go to the log, never the CSV.
5. **Per-run sequence:**
   1. deadline check (stop cleanly if past `--deadline-hours`);
   2. skip if run_id already in CSV (resume);
   3. pre-state: free RAM (`Get-CimInstance Win32_OperatingSystem` FreePhysicalMemory via
      PowerShell) and GPU state — exact query, verified 2026-08-16:

          nvidia-smi --query-gpu=timestamp,temperature.gpu,clocks.sm,clocks.mem,memory.used,power.draw --format=csv,noheader

      sample output: `2026/08/16 15:34:15.706, 36, 1506 MHz, 4006 MHz, 400 MiB, 29.35 W`;
   4. launch llama-bench with `subprocess.run(..., timeout=CAP)`, CAP = **240 s (7B)** /
      **360 s (30B)** — sized ~2x the slowest [est] corner;
   5. on `TimeoutExpired`: `taskkill /F /IM llama-bench.exe`, record `status=dnf_timeout`
      with empty tok_s. **A DNF is a recorded row, never a retry and never a hole** — a
      config that hangs must not eat the night (180 s of it is the worst single loss);
      nonzero exit with `out of memory` in stderr => `status=dnf_oom`; unparseable JSON
      => `status=fail_parse`;
   6. post-state: same nvidia-smi query;
   7. append the CSV row, flush + fsync;
   8. thermal settle loop (section 2 parameters), settle seconds recorded in the row.
6. **CSV** `weights/data/doe_morris_stage1.csv` — append-only, one row per designed run,
   header written once. Columns (exact, order is load-bearing — the scorer hash-pins it):

       run_id,model,traj,pos,changed_factor,ngl,ub,t,ctk,mmp,fa,moe_cpu_frac,status,tok_s,stddev_ts,reps_tok_s,wall_s,settle_s,free_ram_gb_pre,temp_pre,sm_mhz_pre,mem_mhz_pre,vram_mib_pre,power_w_pre,temp_post,sm_mhz_post,mem_mhz_post,vram_mib_post,power_w_post,ts_utc,cmd

   - `run_id` = sha256(`f"{model}|{traj}|{pos}|{canonical_config_json}"`)[:16] — stable
     across relaunches, which is what makes resume skipping sound;
   - `moe_cpu_frac` empty for the 7B; `reps_tok_s` is a JSON list in one quoted field
     (per-rep values from `samples_ns`); `changed_factor` = `base` at pos 0;
   - written with the csv module, QUOTE_MINIMAL.
7. **Command templates** (pinned args in every run):

       7B:  llama-bench.exe -m D:/evo-compress-data/gguf/Qwen2.5-7B-Instruct-Q4_K_M.gguf
            -ngl {0|9|19|99} -ub {128|512|1024|2048} -t {1|2|3|4} -ctk {f16|q8_0}
            -ctv f16 -b 2048 -mmp {0|1} -fa {0|1} -n 128 -p 0 -r 3 -o json

       30B: llama-bench.exe -m D:/evo-compress-data/gguf/Qwen3-30B-A3B-Q2_K.gguf
            -ngl {0|16|32|99} -ub {128|512|1024|2048} -t {1|2|3|4} -ctk {f16|q8_0}
            -ctv f16 -b 2048 -mmp {0|1} -fa {0|1}
            -ot "<generated pattern for f, section 1.5>" -n 128 -p 0 -r 3 -o json

8. The harness writes measurements and logs ONLY. It never writes the prereg, the
   amendment, or any verdict.

---

## 4. Scorer spec — `weights/prereg95_score.py` (PRECOMMITTED)

Frozen before any data exists. Reads `weights/data/doe_morris_stage1.csv`, writes
`weights/data/prereg95_verdict.json` and prints the table. It scores ONLY the stage-1
stakes; it refuses everything else.

**Refusals (exact messages, checked in this order):**

1. CSV absent:

       REFUSED: weights/data/doe_morris_stage1.csv not found. Stage 1 has not produced data; the scorer never invents a verdict.

2. Header drift — sha256 of the header line (exact bytes, no trailing newline) must equal
   the pinned design hash
   `47ed63b0f4ecfa3c3a7e6a140a79eb38b0713841ee5109c8f49a3c8e27d0c624`
   (computed 2026-08-16 from the section-3 header string):

       REFUSED: CSV header hash <found> != design hash 47ed63b0f4ecfa3c3a7e6a140a79eb38b0713841ee5109c8f49a3c8e27d0c624. This file was not written by the staked harness; scoring it would score a different experiment.

3. Partial design — the CSV must contain **all 150 designed run_ids** (70 + 80,
   regenerated from the seeded design, statuses included; a declared DNF row counts as
   present, a missing row does not):

       REFUSED: incomplete design: <n_present> of 150 designed runs present (<n_ok> ok, <n_dnf> declared DNF). Resume weights/doe_morris.py; a partial night is not the staked design.

**Computation (rows with status=ok only):**

- Normalize each factor to its [0,1] grid. For each trajectory step where factor i changed:
  `EE_i = (y_after - y_before) / (g_after - g_before)` — signed, in tok/s per unit
  normalized range, so mu* is comparable across factors.
- A step whose either endpoint is non-ok yields no EE (the DNF poisons exactly the two
  adjacent effects, nothing more).
- `mu_star_i` = mean(|EE_i|), `sigma_i` = stddev(EE_i, ddof=1), per model.
- **Validity floor:** a factor with fewer than 6 valid EEs (of R = 10) in a model is
  UNSCOREABLE there; any stake touching an unscoreable factor is reported VOID for that
  model, never guessed.

**Stakes scored — stage 1 only:**

- **P-1 (concentration):** PASS iff, on BOTH models, sum of the top-3 mu_star >= 0.70 *
  sum of all mu_star.
- **P-2 (regimes separate):** PASS iff argmax-mu_star factor differs between 7B and 30B.
- **P-4 (interaction warning):** PASS iff sigma(-ub) > median(all sigma) on BOTH models —
  the strict both-models reading, pinned here before data exists so a favourable reading
  cannot be picked afterwards. Per-model values reported either way.
- **P-3:** printed verbatim as `P-3: deferred to stage 2 (Sobol) - not scored by this
  tool.` The scorer must never fake a Sobol index from Morris data.

Output JSON: per-model mu_star/sigma tables, EE counts per factor, DNF list, the three
verdicts + P-3 deferral line, csv sha256, scored_utc. The scorer edits no prereg and no
README; humans wire verdicts into documents.

---

## 5. Amendment draft (operator appends to the prereg BEFORE the run)

> ### Pre-data amendment — 2026-08-16, before any stage-1 run
> Design doc: `docs/DESIGN_DOE_MORRIS.md`. Deviations found while making the staked
> protocol executable on this box, declared before data:
>
> 1. **Model substitution:** Qwen3-30B-A3B **Q2_K** (11,258,610,240 bytes on disk), not
>    the staked Q2_K_L — the Q2_K_L we hold is the Coder finetune, which prereg #102's
>    verify pass established must never be joined across. Same base model, same regime.
> 2. **`-np` is not exercisable in stage 1:** llama-bench b10098 rejects it
>    (`error: invalid parameter for argument: -np`) — it is a llama-server concurrency
>    flag. Stage 1 screens 7 of the 8 staked factors; `-np` screening is deferred to a
>    server-harness arm (U-05 lineage), not silently dropped.
> 3. **`-ot` dropped for the 7B only:** dense model, no `_exps` tensors; probed inert
>    (override recorded, zero tensors matched, tok/s unmoved). 7B design is k = 6.
> 4. **KV factor narrowed to `-ctk`:** `-ctv q8_0` requires `-fa` on (probed:
>    context-creation failure with fa off), which would confound the KV and fa factors on
>    a hypercube. `-ctv` pinned f16. U-01 measured K+V jointly; stage 1 screens K only.
> 5. **`-fa` levels are {0, 1}, never `auto`** (the build default): auto lets the build
>    decide per-config and hides the factor.
> 6. **`-t` range is {1..4}:** i5-7600K is 4C/4T; the staked range implied oversubscription
>    levels that do not exist on this part.
> 7. **30B expert-offload fraction restricted to [0.75, 1.0] CPU-side** ({36,40,44,48} of
>    48 layers), exercised via generated `-ot` patterns in the shipped direction (early
>    blocks' experts stay on GPU). Lower fractions OOM the 6 GB card at the max-ngl
>    corner. Build's `-ncmoe` not used: it offloads the FIRST n layers — a different
>    layer set than the shipped recipe.
> 8. **Chosen design constants:** Morris R = 10, p = 4, delta = 2/3, seeds
>    `"prereg95:{7B|30B}:20260807"`; response tg128 r = 3 (per-rep samples read from
>    JSON); `-b` pinned 2048; runs 70 + 80 = 150; timeouts 240 s / 360 s with DNF rows;
>    thermal settle to <= 52 C (min 30 s, cap 180 s) between runs.
>
> The harness (`weights/doe_morris.py`) cannot write this prereg; this amendment was
> appended by the operator before the first designed run.

---

*Design sources: llama-bench --help + live flag probes (build 0278d8362/b10098, 2026-08-16);
`weights/data/ladder_state_locked.json` (bench_s and tok/s for both exact models);
`weights/runner.py` (lock discipline); `weights/full_ladder_v124.py` (bench_s semantics);
`weights/unattended_serial.py` (per-rep JSON parsing, DNF-as-result); prereg #60/#61
(thermal settle), #19 (ub regime flip), #102 (no cross-finetune joins). Numbers without a
source are labeled [est].*
