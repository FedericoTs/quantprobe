# quantprobe — 60-second start

Three levels. Pick where you want to stop.

> ### What quantprobe decides for you — and what you set up once
>
> **Automated (the hard part):** *which bits go where* (depth-aware compression) and *which memory tier serves what* (the placement that turns a model that merely fits into one that runs fast). `quantize`/`probe` pick the bits; `run` picks the placement and launches — you never hand-tune a flag.
>
> **You do these once (plumbing, not strategy):**
> 1. **Install llama.cpp** — download the [prebuilt binaries](https://github.com/ggml-org/llama.cpp/releases) once; point quantprobe at them (`--llama-dir`, `QUANTPROBE_LLAMA_DIR`, or `PATH`). *(Not needed for `plan`/`target`/the web calculator.)*
> 2. **Convert a HuggingFace model to GGUF** — only if you're compressing a model that has no community GGUF. Run llama.cpp's `convert_hf_to_gguf.py` once, then feed the `.gguf` to `quantize`. Models with an existing GGUF skip this entirely.
> 3. **Tell it your hardware** — pick a `--machine` preset or pass `--vram/--ram/...`. quantprobe does **not** auto-detect your specs yet.
>
> So: **the memory-speed strategy is applied autonomously; the one-time setup is on you.** A single hands-off `quantprobe auto <hf-model>` (auto-detect hardware → convert → compress → run) is on the roadmap.


---

## Level 0 — nothing to install (10 seconds)

Open the calculator in your browser and type in your machine:
**→ https://federicots.github.io/quantprobe/**

It predicts decode speed, memory fit, quality cost, and your cheapest upgrade for any model. No install, no account.

---

## Level 1 — the Python tool, zero external tools (1 minute)

```bash
pip install git+https://github.com/FedericoTs/quantprobe
```

Now these work immediately — they're pure calculators, no model download, no llama.cpp:

```bash
# "How fast will Qwen3-30B run on my machine, and how should I place it?"
quantprobe plan --model qwen3-30b --machine 2016-xmp

# "I need at least 5 tok/s — what's the smartest model I can run?"
quantprobe target --tps 5 --machine gaming --ladder
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

quantprobe models a single device. For tensor-parallel rigs (e.g. 2× 3090, or a DGX cluster), approximate: set `--vram` to the **summed** VRAM and `--vram-bw` to the **aggregate** bandwidth (the law held on a 4× DGX Spark cluster this way — its published number lands in the eta bands).

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

# 2. launch chat with the placement the law picked for your machine
quantprobe run --gguf ./models/Qwen3-30B-A3B-Q2_K.gguf --model qwen3-30b --machine 2016-xmp

# check the law on your own hardware (predicted vs measured):
quantprobe bench --gguf ./models/Qwen3-30B-A3B-Q2_K.gguf --model qwen3-30b --machine 2016-xmp
```

### Make your own compressed model (the full pipeline)

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

**Starting from a HuggingFace model (safetensors)?** Convert it to a high-precision GGUF once with llama.cpp's `convert_hf_to_gguf.py` (ships in the [llama.cpp repo](https://github.com/ggml-org/llama.cpp)), then feed that `.gguf` to `quantize`/`probe`. A one-command `quantprobe convert` wrapper is on the roadmap.

---

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
| `plan` | no | predict decode speed + best placement for a model on your machine |
| `target` | no | inverse: give a tok/s target, get the smartest model + a speed↔intelligence ladder |
| `fetch` | no (network) | robust model download (resumes, retries) |
| `quantize` | **yes** | **compress**: build a depth-aware ~2-bit GGUF (protect the fragile band, rest 2-bit) |
| `probe` | **yes** | measure a model's fragility curve → emit (or `--apply` to build) the depth-aware GGUF |
| `run` | **yes** | plan the placement, then launch llama.cpp chat with those flags |
| `bench` | **yes** | measure real tok/s and print predicted-vs-measured |
| `dashboard` | **yes** | a local web page: chat while every reply is scored against the prediction |

Every command has `--help`. The [full README](README.md) has the science; [LAWS.md](LAWS.md) has the four laws.
