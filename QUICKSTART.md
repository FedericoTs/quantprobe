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

**Machine presets:** `2016-xmp` `2016` `gaming` `ddr5` `colibri`
**Model presets:** `qwen3-30b` `deepseek-16b` `gemma-12b` `mistral-7b` `glm-air` `glm-744b` (or use `--total/--active/--always-active` for any model)

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
