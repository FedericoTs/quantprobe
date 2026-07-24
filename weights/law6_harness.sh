#!/usr/bin/env bash
# Law 6 pilot harness — WRITTEN 2026-07-25, TO BE EXECUTED during launch week (week of 2026-07-27),
# scoring preregistrations/2026-07-24-law6-speculation-economics.md. NO speculative flag has been
# run on this box; this file is preparation only. Conventions per LAW5_PROTOCOL + prereg #10.
set -u
T="C:/Users/Federico/Documents/evo-compress/tools/llamacpp-b10098"
G="/d/evo-compress-data/gguf"
W="C:/Users/Federico/Documents/evo-compress/.claude/worktrees/law5-prefill/weights/data"
SC="${LAW6_SCRATCH:?set LAW6_SCRATCH to a scratch dir}"
L="$W/law6_pilot.log"   # unique log, never reused
PORT=8095
ck(){ [ -s "$1" ] || { echo "CHECKPOINT FAIL: $2" | tee -a "$L"; exit 1; }; echo "CHECKPOINT OK: $2" | tee -a "$L"; }
gpu(){ echo "[gpu-state] $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)" | tee -a "$L"; }

# STEP 0 — orphans + GPU state (binding convention)
taskkill //F //IM llama-server.exe 2>/dev/null; taskkill //F //IM llama-bench.exe 2>/dev/null; sleep 2
echo "=== LAW6 PILOT $(date '+%F %H:%M') ===" > "$L"; gpu

# STEP 1 — payloads: W-code (real repo file + localized-edit instruction), W-prose (wiki continuation)
python - "$SC" <<'PY'
import json, sys
sc = sys.argv[1]
code = open(r"C:\Users\Federico\Documents\evo-compress\quantprobe\plan.py", encoding="utf-8").read()[:8000]
wcode = code + "\n\n# Task: rename the function qual_of to quality_of everywhere above, and show the full corrected file.\n"
prose = open(r"D:\evo-compress-data\eval\wiki.test.raw", encoding="utf-8", errors="ignore").read()[:8000]
for name, prompt in [("wcode", wcode), ("wprose", prose)]:
    json.dump({"prompt": prompt, "n_predict": 256, "cache_prompt": False, "temperature": 0},
              open(f"{sc}/law6_{name}.json", "w"))
print("payloads ok")
PY
ck "$SC/law6_wcode.json" "payloads"

# STEP 2 — flag pre-flight (parse-only): exact spec flags in this build
"$T/llama-server.exe" --help 2>&1 | grep -iE "spec-type|model-draft|draft-m|spec-ngram|draft " | tee -a "$L"

serve(){ # serve <model> <extra flags...>
  "$T/llama-server.exe" -m "$1" "${@:2}" -c 4096 -np 1 --port $PORT > "$SC/law6_server.log" 2>&1 &
  for i in $(seq 1 90); do R=$(curl -s -m 8 http://127.0.0.1:$PORT/completion -d '{"prompt":"hi","n_predict":1}'); echo "$R" | grep -q '"timings"' && return 0; sleep 5; done; return 1
}
cell(){ # cell <label> <payload>  -> logs decode tok/s
  curl -s -m 900 http://127.0.0.1:$PORT/completion --data-binary @"$2" | python -c "
import json,sys; t=json.load(sys.stdin)['timings']
print('CELL $1 tg tok/s:', round(t['predicted_per_second'],2), ' n:', t['predicted_n'])" | tee -a "$L"
}
stop(){ taskkill //F //IM llama-server.exe 2>/dev/null; sleep 2; }

# STEP 3 — baselines (no spec), then S-a, S-b, S-c. r2: each cell issued twice, second counts (warm).
gpu
serve "$G/qwen7b-Q4_K_M.gguf" -ngl 99                       && { cell base-dense-wcode "$SC/law6_wcode.json"; cell base-dense-wcode "$SC/law6_wcode.json"; cell base-dense-wprose "$SC/law6_wprose.json"; }; stop
serve "$G/qwen7b-Q4_K_M.gguf" -ngl 99 --spec-type ngram-simple && { cell Sa-ngram-wcode "$SC/law6_wcode.json"; cell Sa-ngram-wcode "$SC/law6_wcode.json"; cell Sa-ngram-wprose "$SC/law6_wprose.json"; }; stop
gpu
serve "$G/Qwen3-30B-A3B-Q2_K.gguf" -ngl 99 -ot "exps=CPU"      && { cell base-moe-wcode "$SC/law6_wcode.json"; cell base-moe-wcode "$SC/law6_wcode.json"; }; stop
serve "$G/Qwen3-30B-A3B-Q2_K.gguf" -ngl 99 -ot "exps=CPU" --spec-type ngram-simple && { cell Sb-moe-ngram-wcode "$SC/law6_wcode.json"; cell Sb-moe-ngram-wcode "$SC/law6_wcode.json"; }; stop
gpu
serve "$G/Qwen2.5-7B-Instruct-Q4_K_M.gguf" -ngl 99             && { cell base-inst-wcode "$SC/law6_wcode.json"; cell base-inst-wcode "$SC/law6_wcode.json"; }; stop
serve "$G/Qwen2.5-7B-Instruct-Q4_K_M.gguf" -ngl 99 --spec-type draft-simple -md "$G/Qwen2.5-0.5B-Instruct-Q8_0.gguf" --n-gpu-layers-draft 99 && { cell Sc-draft-wcode "$SC/law6_wcode.json"; cell Sc-draft-wcode "$SC/law6_wcode.json"; }; stop
gpu
echo "=== LAW6 PILOT DONE $(date '+%F %H:%M') ===" | tee -a "$L"
# Scoring: S-a = Sa/base per workload vs x1.25-2.0 (code) and <=x1.15 (prose);
# S-b: (S_moe-1) <= 0.75*(S_dense-1); S-c: Sc/base-inst vs x1.2-1.8. Misses publish.
