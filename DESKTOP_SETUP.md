# Desktop setup — restore the full project + continue the conversation

This repo contains **everything except the 67 GB of model weights** (deleted; re-downloadable
per `weights/DATA_MANIFEST.md`). The **Claude conversation + memory** backup
(`_session_archive.zip`) was moved out-of-band during the original transfer and was never
committed — steps 3 and 6 below are the historical record of that one-time move and cannot be
run from a fresh clone. Follow the remaining steps on the desktop to pick up where we left off.

## 1. Clone the repo (code + full git history + test results)
```powershell
gh repo clone FedericoTs/quantprobe "lossless compression/evo-compress"
cd "lossless compression/evo-compress"
```
> Put it under a folder named `lossless compression` (the parent), because Claude Code keys its
> data off the project path — matching the path makes the conversation auto-resume (see step 3).

## 2. Recreate the Python environment (do NOT copy the old `.venv`)
```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -U pip
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m pip install -e .
```
(Needs Python 3.13. Key deps: torch, transformers, safetensors, numpy, zstandard.) The last line
installs the `quantprobe` CLI itself — requirements.txt alone does not: it pulls `gguf` and
`requests` per pyproject.toml and exposes the `quantprobe` entry point.

## 3. (historical — one-time transfer) Restore the Claude conversation + memory
The zip this step restores was transferred out-of-band and is not in the repo; kept for the record.
The conversation/memory lives **outside** the project, in `~/.claude/projects/<path-hash>/`.
The folder name encodes the project's absolute path, with every `:`, `\`, and space replaced by `-`.

- **If the desktop user is also `Samsung`** and the project sits at
  `C:\Users\Samsung\Documents\Projects\lossless compression` → the hash is
  `C--Users-Samsung-Documents-Projects-lossless-compression` (unchanged). Just unzip:
  ```powershell
  $dest = "$env:USERPROFILE\.claude\projects\C--Users-Samsung-Documents-Projects-lossless-compression"
  New-Item -ItemType Directory -Force $dest
  Expand-Archive _session_archive.zip -DestinationPath $dest -Force
  ```
- **If the desktop user differs** (e.g. `Fede`): build the hash from the NEW full path
  (replace `:` `\` and spaces with `-`), e.g.
  `C--Users-Fede-Documents-Projects-lossless-compression`, and unzip there instead.

The zip contains the chat transcript (`*.jsonl`), the `memory/` folder (project knowledge I
maintain across sessions), and the sub-agent/workflow history.

## 4. Re-download the model data you need (see `weights/DATA_MANIFEST.md`)
Minimum for the quantization work (~1 GB):
```powershell
pip install -U "huggingface_hub[cli]"
hf download Qwen/Qwen2.5-0.5B-Instruct --local-dir weights/data/qwen_cfg `
  --include "config.json" "tokenizer*" "vocab.json" "merges.txt" "generation_config.json"
mkdir weights/data/qwen
curl.exe -L -o weights/data/qwen/base.safetensors `
  "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/main/model.safetensors"
```
For the abliteration detector and scale tests, see the ESSENTIAL/FULL tables in `DATA_MANIFEST.md`.
On the GPU desktop you can finally pull the 7B (`Qwen/Qwen2.5-7B-Instruct` + huihui abliterated).

## 5. Open Claude Code in the project folder and continue
Launch Claude Code with the working directory at `…/lossless compression/evo-compress` (or its
parent). With step 3 done, the prior conversation + memory are available — say "continue" and
we pick up from the codec discovery loop and the GPU scale plan.

## 6. (historical — nothing to do) Clean up
The transfer artifact was never committed to the public repo; from a fresh clone there is
nothing to remove. The original one-time cleanup was:
```powershell
git rm _session_archive.zip && git commit -m "remove transfer artifact"
```
