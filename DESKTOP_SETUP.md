# Desktop setup — restore the full project + continue the conversation

This repo contains **everything except the 67 GB of model weights** (deleted; re-downloadable
per `weights/DATA_MANIFEST.md`) and includes a backup of the **Claude conversation + memory**
(`_session_archive.zip`). Follow these steps on the desktop to pick up exactly where we left off.

## 1. Clone the repo (code + full git history + test results)
```powershell
gh repo clone FedericoTs/<REPO_NAME> "lossless compression/evo-compress"
cd "lossless compression/evo-compress"
```
> Put it under a folder named `lossless compression` (the parent), because Claude Code keys its
> data off the project path — matching the path makes the conversation auto-resume (see step 3).

## 2. Recreate the Python environment (do NOT copy the old `.venv`)
```powershell
python -m venv .venv
.venv\Scripts\pip install -U pip
.venv\Scripts\pip install -r requirements.txt
```
(Needs Python 3.13. Key deps: torch, transformers, safetensors, numpy, zstandard.)

## 3. Restore the Claude conversation + memory  ← this is what lets you CONTINUE the chat
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

## 6. (optional) Clean up
After confirming everything works you can remove the one-time transfer artifact:
```powershell
git rm _session_archive.zip && git commit -m "remove transfer artifact"
```
