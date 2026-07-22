# quantprobe — 60-second start

Three levels. Pick where you want to stop.

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

Then, zero to chatting:

```bash
# 1. download a known-good model (robust, resumes if interrupted)
quantprobe fetch qwen3-30b ./models

# 2. launch chat with the placement the law picked for your machine
quantprobe run --gguf ./models/Qwen3-30B-A3B-Q2_K.gguf --model qwen3-30b --machine 2016-xmp

# check the law on your own hardware (predicted vs measured):
quantprobe bench --gguf ./models/Qwen3-30B-A3B-Q2_K.gguf --model qwen3-30b --machine 2016-xmp
```

Want a better quant than the community default? Probe your model's fragile layers (~30 min) and get a copy-paste recipe:

```bash
quantprobe probe --gguf your-model-f16.gguf --eval wiki.test.raw
```

---

## What each command does

| command | needs llama.cpp? | what it does |
|---|---|---|
| `plan` | no | predict decode speed + best placement for a model on your machine |
| `target` | no | inverse: give a tok/s target, get the smartest model + a speed↔intelligence ladder |
| `fetch` | no (network) | robust model download (resumes, retries) |
| `probe` | **yes** | measure a model's fragility curve → emit the depth-aware quant recipe |
| `run` | **yes** | plan the placement, then launch llama.cpp chat with those flags |
| `bench` | **yes** | measure real tok/s and print predicted-vs-measured |
| `dashboard` | **yes** | a local web page: chat while every reply is scored against the prediction |

Every command has `--help`. The [full README](README.md) has the science; [LAWS.md](LAWS.md) has the four laws.
