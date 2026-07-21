# weights/data — Re-download Manifest

`weights/data/` (~67 GB) is **gitignored** and was **deleted** to make the project portable.
This file records exactly what was there and how to re-download it. Most experiments only
need a small subset (see "Essential" below).

Re-download tool (recommended):
```bash
pip install -U "huggingface_hub[cli]"
hf download <repo_id> <file_or_glob> --local-dir weights/data/<target>
```
or plain `curl.exe -L -o <out> "https://huggingface.co/<repo>/resolve/main/<file>"`.

---

## ESSENTIAL for the current quantization work (~1 GB)
The codec discovery loop (`weights/quant_*.py`, `weights/codec_zoo.py`) needs only:

| target | what | source |
|---|---|---|
| `qwen_cfg/` | config.json + tokenizer (tokenizer.json, vocab.json, merges.txt, generation_config.json) | `Qwen/Qwen2.5-0.5B-Instruct` |
| `qwen/base.safetensors` | the 0.5B base weights (bf16, 942 MB) | `Qwen/Qwen2.5-0.5B-Instruct` → `model.safetensors` |
| eval corpus | `../data/corpora/generic-text/enwik8_256k` (NOT in weights/data — separate ~120 MB `data/` dir) | enwik8 (public); already in repo unless gitignored |

```bash
hf download Qwen/Qwen2.5-0.5B-Instruct --local-dir weights/data/qwen_cfg \
  --include "config.json" "tokenizer*" "vocab.json" "merges.txt" "generation_config.json"
curl.exe -L -o weights/data/qwen/base.safetensors \
  "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/main/model.safetensors"
```

## ESSENTIAL for abliteration-detector work (~2 GB)
| target | source |
|---|---|
| `qwen/ablit.safetensors` | `huihui-ai/Qwen2.5-0.5B-Instruct-abliterated` → `model.safetensors` |
| `real_models/qwen05_base.safetensors` | `Qwen/Qwen2.5-0.5B-Instruct` → `model.safetensors` |
| `real_models/qwen05_ablit.safetensors` | `huihui-ai/Qwen2.5-0.5B-Instruct-abliterated` → `model.safetensors` |
| `real_models/qwen15_base.safetensors` | `Qwen/Qwen2.5-1.5B-Instruct` → `model.safetensors` |
| `real_models/qwen15_ablit.safetensors` | `huihui-ai/Qwen2.5-1.5B-Instruct-abliterated` → `model.safetensors` |

---

## FULL historical dataset (the deleted 67 GB)

| dir | size | contents → source |
|---|---|---|
| `qwen/` | 1.8 GB | base + ablit (0.5B) → `Qwen/Qwen2.5-0.5B-Instruct`, `huihui-ai/Qwen2.5-0.5B-Instruct-abliterated` |
| `qwen1.5b/` | 5.8 GB | base + ablit → `Qwen/Qwen2.5-1.5B-Instruct`, `huihui-ai/Qwen2.5-1.5B-Instruct-abliterated` |
| `qwen3b/` | 11.5 GB | base + ablit (2 shards each) → `Qwen/Qwen2.5-3B-Instruct`, `huihui-ai/Qwen2.5-3B-Instruct-abliterated` |
| `qwen7b/` | 28.4 GB | base + ablit (4 shards each) → `Qwen/Qwen2.5-7B-Instruct`, `huihui-ai/Qwen2.5-7B-Instruct-abliterated` |
| `real_models/` | 7.6 GB | qwen05/qwen15 base+ablit (see ESSENTIAL above) |
| `qwen_cfg/` | 0.01 GB | config + tokenizer → `Qwen/Qwen2.5-0.5B-Instruct` |
| `qwen_gen/` | 0.9 GB | `qwen2-0.5b.safetensors` → `Qwen/Qwen2-0.5B` (Qwen2, for cross-generation test) |
| `smollm/` | 1.0 GB | base, base_bf16, instruct → `HuggingFaceTB/SmolLM-135M`, `HuggingFaceTB/SmolLM-135M-Instruct` |
| `pythia_bf16/` | 0.26 GB | s142k, s143k → `EleutherAI/pythia-70m` @ revisions `step142000`, `step143000` (converted to bf16) |
| `qwen_family/` | 10.1 GB | 9 Qwen2.5-0.5B fine-tunes (merge experiments — OPTIONAL, now de-prioritized). Files: dataforge-sft, dpo-halueval, grpo-summ, mathphd, reasoning, unsloth, vikhr, neon-sft, ultrachat-sft. `unsloth` = `unsloth/Qwen2.5-0.5B-Instruct`; others are community fine-tunes of Qwen2.5-0.5B (search HF by name; exact repo IDs not all recorded — only needed to reproduce the abandoned model-merge work). |
| `0000.bin`..`0039.bin` + `manifest.json` | small | MiniLM/pythia/bert-tiny tensor slices + synthetic. **Regenerate:** `python -m weights.fetch_weights` |
| `gsm8k_test.jsonl` | small | GSM8K test split (1319 problems) → `openai/gsm8k` (test) |

**Note:** huihui-ai abliterated repos are single-file `model.safetensors` for ≤1.5B, sharded for 3B/7B. The 7B was postponed (RAM-bound on the laptop) — re-download on the GPU desktop to run scale tests.
