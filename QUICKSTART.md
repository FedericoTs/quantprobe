# quantprobe — 60-second start

**If you only remember one command** — hardware auto-detected, nothing else to fill in:

```bash
quantprobe auto
```

It detects your machine, asks for the model, decides — by the laws — whether your hardware
even needs custom quantization (and says so), then builds or fetches and launches. The
non-interactive form of the same thing:

```bash
quantprobe auto <model-or-HF-repo> --custom --run
```

That is the ENTIRE pipeline: pick a high-precision source → measure YOUR model's fragile
band → build its personalized depth-aware GGUF → predict the placement for THIS machine →
launch chat. Swap `--custom` for `--tps 15` to skip the ~30-60 min probe and just fetch the
best community quant for a speed target. Everything below is the same machinery with more
control.

Three levels. Pick where you want to stop.

## Level 0 — nothing to install (10 seconds)

Open the calculator in your browser and type in your machine:
**→ https://federicots.github.io/quantprobe/**

It predicts decode speed, memory fit, quality cost, and your cheapest upgrade for any model. No install, no account.

---

## Level 1 — the Python tool, zero external tools (1 minute)

```bash
pip install quantprobe
```

> **Windows: "'quantprobe' is not recognized"?** pip put the .exe in a Scripts folder that
> isn't on your PATH. Use `python -m quantprobe ...` instead — identical, always works.

**Zero-config on your own machine.** `quantprobe hw` shows what it detected (GPU, RAM speed, every value tagged with its source); any command with no hardware flags uses it automatically. And `--gguf model.gguf` reads the model's parameters from the file itself (total/active params, true effective bits, exact KV bytes). The minimal commands are now:

```bash
quantprobe auto qwen3-coder --tps 15 --run     # ONE command: bits chosen, quant fetched, chat running
quantprobe hw                                  # what the tool sees (override anything with flags)
quantprobe plan --gguf your-model.gguf         # THIS machine + THAT file: nothing else to type
quantprobe bench --gguf your-model.gguf        # predicted vs measured, zero configuration
```

Presets and explicit flags still work exactly as before — use them to estimate a machine you're NOT running on.

Now these work immediately — they're pure calculators, no model download, no llama.cpp:

```bash
# "How fast will Qwen3-30B run on my machine, and how should I place it?"
quantprobe plan --model qwen3-30b --machine 2016-xmp

# "I need at least 5 tok/s — what's the smartest model I can run?"
quantprobe target --tps 5 --machine gaming --ladder

# planning long-context work (coding agents, RAG)? add --ctx: it prices the KV reads AND the KV memory
quantprobe plan --model qwen3-30b --machine 2016-xmp --ctx 16384
```

Don't have a preset for your machine? Pass raw numbers:

```bash
quantprobe plan --model qwen3-30b --vram 8 --vram-bw 300 --ram 32 --ram-bw 50 --disk-bw 2
```

**Machine presets** (`--machine`): `2016-xmp` `2016` `rtx-3060` `rtx-3090` `rtx-4090` `rtx-5090` `laptop-8gb` `mac-m2-max` `mac-m3-max` `mac-m4-max` `mac-m2-ultra` `mac-m3-ultra` `ddr5` `colibri` `epyc-256` `dgx-spark`
**Model presets** (`--model`): `qwen3-30b` `deepseek-16b` `gemma-12b` `mistral-7b` `glm-air` `glm-744b`

> Presets marked `[est]` are the law's predictions on hardware I haven't personally measured (Mac numbers especially — I've never run one). They're falsifiable: run `quantprobe bench` on your box and [open an issue](https://github.com/FedericoTs/quantprobe/issues) with predicted-vs-measured. Only `2016*` are measured by me; `dgx-spark` is validated against published benchmarks.

### Don't know your numbers? (for `--vram-bw` / `--ram-bw`)

You rarely need these — pick the closest preset above. But if you go custom, here are the common ones (GB/s):

| your hardware | memory bandwidth |
|---|---|
| DDR4-2400 / 3200 dual-channel | ~38 / ~51 |
| DDR5-4800 / 6400 dual-channel | ~77 / ~102 |
| GTX 1060 / RTX 3060 / 3090 / 4090 / 5090 (VRAM) | 192 / 360 / 936 / 1008 / 1792 |
| Apple M2/M3 Max · M2 Ultra · M3/M4 Ultra (unified) | ~400 · 800 · ~819 |
| SATA SSD / NVMe Gen3 / Gen4 (disk) | ~0.5 / ~3.5 / ~7 |

### Multi-GPU / multi-device?

Native since v1.3 — pass comma lists and quantprobe aggregates them: `--vram 24,24 --vram-bw 936,936` (×0.85 tensor-parallel efficiency [est]; the law held on a 4× DGX Spark cluster this way), `--disk-bw 14,14` for RAID (×0.75 stripe [est]). Rigs with big VRAM *and* disk streaming also get the **three-tier expert-cache row** — what ktransformers/colibri-class runtimes achieve; stock llama.cpp performs at the RAM-cache row.

---

> ### What quantprobe decides for you — and what you set up once
>
> **Automated (the hard part):** *which bits go where* (depth-aware compression) and *which memory tier serves what* (the placement that turns a model that merely fits into one that runs fast). `quantize`/`probe` pick the bits; `run` picks the placement and launches — you never hand-tune a flag.
>
> **You do these once (plumbing, not strategy):**
> 1. **Install llama.cpp** — download the [prebuilt binaries](https://github.com/ggml-org/llama.cpp/releases) once; point quantprobe at them (`--llama-dir`, `QUANTPROBE_LLAMA_DIR`, or `PATH`). *(Not needed for `plan`/`target`/the web calculator.)*
> 2. **Convert a HuggingFace model to GGUF** — only if you're compressing a model that has no community GGUF. Run llama.cpp's `convert_hf_to_gguf.py` once, then feed the `.gguf` to `quantize`. Models with an existing GGUF skip this entirely.
> 3. ~~Tell it your hardware~~ **auto-detected since v1.2** (`quantprobe hw` shows what it sees; flags/`--machine` only needed to estimate a different machine).
>
> So: **the memory-speed strategy is applied autonomously; the one-time setup is on you.** And the hands-off pipeline shipped: `quantprobe auto <model>` (fetch the best community quant and run) and `auto --custom` (probe YOUR model, build its personalized GGUF).


---

## Level 2 — actually quantize & run a model (needs llama.cpp)

The commands that touch real weights (`probe`, `run`, `bench`, `dashboard`) drive **llama.cpp**. Get it once:

- **Download prebuilt binaries:** https://github.com/ggml-org/llama.cpp/releases (grab the release for your OS, unzip)
- Then either add that folder to your `PATH`, set `QUANTPROBE_LLAMA_DIR=/path/to/llama.cpp`, or pass `--llama-dir /path/to/llama.cpp` on each command.

If quantprobe can't find llama.cpp it tells you exactly this — it never fails silently.

Then, from model to chatting (llama.cpp installed above):

```bash
# 1. download a known-good model (robust, resumes if interrupted)
quantprobe fetch qwen3-30b ./models

# 2. launch chat with the placement the law picked for your machine (zero config - it reads the file + your hardware)
quantprobe run --gguf ./models/Qwen3-30B-A3B-Q2_K.gguf

# check the law on your own hardware (predicted vs measured):
quantprobe bench --gguf ./models/Qwen3-30B-A3B-Q2_K.gguf
```

### Make your own compressed model

The one-command version — picks a requantizable source from the repo, fetches the eval corpus,
probes the fragile band (~30-60 min), builds the personalized depth-aware GGUF:

```bash
quantprobe auto qwen3-coder --custom
```

Manual control over each step (same machinery):

```bash
# A. compress directly — protect the last 12 layers (good default for most late-fragile models)
quantprobe quantize --gguf your-model-f16.gguf --out your-model-2bit.gguf
#    -> produces a depth-aware ~2-bit GGUF you can run immediately

# B. or probe first (measure YOUR model's fragile band, ~30 min) then build it in one step
quantprobe probe --gguf your-model-f16.gguf --eval wiki.test.raw --apply --out your-model-2bit.gguf

# C. then run it
quantprobe run --gguf your-model-2bit.gguf --model <preset> --machine <preset>
```

`quantize` and `probe --apply` **actually build the file** (they run llama.cpp's quantizer for you) — you don't copy-paste anything. `--dry` shows the exact command first if you want to inspect it.

**"How do I pick the bits?"** — `quantize` has no bits flag *on purpose*: it always builds the one validated recipe (body ~2-bit, fragile band at 4-bit ≈ 2.5 effective bits). Its knobs control *where* the protection goes: `--protect-late N` (default 12) or `--protect LO-HI`, and `probe` measures your model's real fragile band instead of assuming. Want a **standard** quant at a specific level instead (Q4_K_M, Q5, Q8)? That's `auto <model> --tps <target>` (the optimizer picks the bits for your speed target) or `fetch` (you pick the file) — at 4-bit and up, community quants are fine as-is; the depth-aware recipe earns its keep below ~2.5 bits, where uniform quantization falls off the quality cliff. Full manual control: `--dry` prints the llama.cpp command — edit the format token and run it yourself.

**Starting from a HuggingFace model (safetensors)?** Convert it to a high-precision GGUF once with llama.cpp's `convert_hf_to_gguf.py` (ships in the [llama.cpp repo](https://github.com/ggml-org/llama.cpp)), then feed that `.gguf` to `quantize`/`probe`. A one-command `quantprobe convert` wrapper is on the roadmap.

---

### Already using Ollama? Your models are GGUFs already

Ollama stores every model as a standard GGUF blob — quantprobe can point straight at it, no re-download:

```bash
# 1. find the blob path of a model you already pulled
ollama show qwen2.5-coder:7b --modelfile     # the FROM line shows ...blobs/sha256-<hash>
# 2. bench it (works even though the file has no .gguf extension - verified)
quantprobe bench --gguf ~/.ollama/models/blobs/sha256-<hash> --total 7.2 --active 7.2 --bits 4.5 --vram 0 --ram 32 --ram-bw 45 --disk-bw 2
```

(Windows: `C:\Users\<you>\.ollama\models\blobs\`.) Two Ollama gotchas the law keeps exposing in the wild: Ollama's **default context window (~4k) is smaller than coding-agent payloads** — Continue/Cline overflow it, truncation breaks prompt-cache reuse, and every request re-prefills from zero (minutes on CPU). Either raise it (`PARAMETER num_ctx 16384` in a Modelfile / `OLLAMA_CONTEXT_LENGTH`) or serve with `quantprobe run --serve --extra "-c 16384"` instead. And on CPU-only boxes, prefer MoE models — dense 14B decodes ~3 tok/s where a 30B-A3B does ~8 with more intelligence (`plan` shows this per-box).

## Recipes — getting the most out of it

```bash
# YOUR OWN FINE-TUNE (the most custom thing there is): convert once, then probe ITS fragile band
#   python convert_hf_to_gguf.py ./my-finetune --outfile my-finetune-f16.gguf   (ships with llama.cpp)
quantprobe probe --gguf my-finetune-f16.gguf --eval wiki.test.raw --apply --out my-finetune-2bit.gguf
#   -> a ~2-bit model carrying YOUR fine-tune's own protection map, ~3x smaller than f16 per bit class

# LOCAL CODING AGENT (Cline / Continue / Aider): serve the coding MoE with agent-sized context
quantprobe auto qwen3-coder --tps 15 --run --serve --extra "-c 16384"
#   -> OpenAI-compatible endpoint at localhost:8080/v1 (any key); MoE beats dense ~3x on CPU decode

# BEFORE BUYING HARDWARE: price the candidate box (flags = the machine you're NOT on)
quantprobe optimize --tps 20 --vram 16 --vram-bw 448 --ram 64 --ram-bw 85 --disk-bw 7
#   -> the cheapest bits/placement/hardware combination that meets the target, Pareto-ranked

# LONG-CONTEXT HONESTY (agents dump 10k+ tokens): the KV term prices it before you feel it
quantprobe plan --gguf model.gguf --ctx 16384
```

## Help grow the law (optional, opt-in)

Every `bench` is a test of the tiered decode law on hardware I may never have touched. If you want to contribute your point:

```bash
quantprobe bench --gguf model.gguf --model qwen3-30b --machine mac-m2-ultra --contribute
```

It prints **exactly** what would be shared — your hardware label, model, predicted vs measured tok/s — and a pre-filled GitHub issue link. You review it, edit if you like, and submit. **Nothing is ever sent automatically; no system scan, no IP, no hidden fields.** Contributed points get plotted on the law chart (orange triangles) — and points that miss the prediction are the *most* valuable, because they sharpen the law. That's the whole feedback loop: measure → review → submit → the law improves.

## A note on llama.cpp versions

quantprobe drives **stock llama.cpp** and emits its flags. llama.cpp occasionally renames flags between releases — while building this I hit four (`--allow-requantize`, `--no-mmap`, `--draft-max`→`--spec-draft-n-max`, `--no-cnv` vs `-no-cnv`). quantprobe targets stable, widely-supported flags, but:

- **Validated on llama.cpp build b9596+** (needs `--tensor-type` regex support in `llama-quantize`).
- If a `run`/`bench`/`quantize` command errors with *"invalid/unknown argument"*, your llama.cpp is a different vintage — check that binary's `--help` for the current flag name. Use `--dry` to see the exact command quantprobe would run before it runs it.
- For exact reproduction of the numbers in this repo, use b9596.
- **Scripting single-shot generations? Use `llama-server` + one HTTP request, not `llama-cli`.** Models
  that ship a chat template can silently force llama-cli into interactive conversation mode (overriding
  `--no-conversation` on some builds) — with no terminal attached it spins forever printing prompts.
  This cost me three takes on a 118B run; the server path has no interactive surface and returns
  timing telemetry in the JSON.

## What measures what (the three verbs people mix up)

| command | what it does | measured or computed? |
|---|---|---|
| `plan` / `--machine` | **describes your hardware** — preset or raw `--vram/--ram/...` numbers; prediction is *computed* from the decode law | computed (no run, no cache) |
| `probe` | **measures your model** — which layers break under low-bit quantization (quality, not speed) | measured (~30 min, llama.cpp) |
| `bench` | **measures your machine** — real tok/s vs the law's prediction, side by side | measured (llama-bench) |

`--machine` is never learned from `probe` and nothing is cached between them. The only dynamic input: passing `--gguf` calibrates bytes-per-token to your actual file size on disk.

## What each command does

| command | needs llama.cpp? | what it does |
|---|---|---|
| `hw` | no | detect THIS machine (RAM/GPU/disk, source-tagged); used automatically when you pass no flags |
| `plan` | no | predict decode speed + best placement for a model on your machine |
| `target` | no | inverse: give a tok/s target, get the smartest model + a speed↔intelligence ladder |
| `optimize` | no | cheapest path to a target speed: bits × placement × KV × hardware searched over the law, measured lever gates |
| `auto` | network | ONE command: closest community quant fetched and ready (`--run` launches); `--custom` probes YOUR model and builds its personalized depth-aware GGUF |
| `fetch` | no (network) | robust model download (resumes, retries) |
| `quantize` | **yes** | **compress**: build a depth-aware ~2-bit GGUF (protect the fragile band, rest 2-bit) |
| `probe` | **yes** | measure a model's fragility curve → emit (or `--apply` to build) the depth-aware GGUF |
| `run` | **yes** | plan the placement, then launch llama.cpp chat with those flags |
| `bench` | **yes** | measure real tok/s and print predicted-vs-measured |
| `dashboard` | **yes** | a local web page: chat while every reply is scored against the prediction |

Every command has `--help`. The [full README](README.md) has the science; [LAWS.md](LAWS.md) has the four laws.
