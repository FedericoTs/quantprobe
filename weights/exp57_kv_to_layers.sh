#!/bin/bash
# Task #57 / prereg #93 — does freed KV VRAM convert into weight-tier promotion?
#
# STAKED BEFORE RUNNING:
#   q8_0 KV halves cache bytes (L-24: quality ratio 1.00031 +/- 0.019, no measurable cost).
#   At depth on a 6GB card that freed VRAM admits MORE MODEL LAYERS. Both halves are
#   measured separately; the CONVERSION has never been timed.
#
#   arm A  -ngl 21, f16 KV   (the shipped config)          -> baseline
#   arm B  -ngl 21, q8_0 KV  (KV effect alone)             -> prereg #25 says ~+37% at d16384
#   arm C  -ngl 27, q8_0 KV  (KV effect + promoted layers) -> the thing being tested
#
#   P1 KILL RULE: arm C must beat arm A by MORE than arm B does. If C <= B, the promotion
#   added nothing and only the KV effect is real - say so and withdraw the compound claim.
#   P2: arm C must not OOM. If it does, the promotion is infeasible at this depth: report
#   that, do NOT quietly reduce -ngl and re-run (that voids the staking).
#
#   CANNOT-VARY GUARD: at shallow depth the KV term does not bind and all three arms return
#   ~the same number. Depth is 16384, not the default. If A and B agree within 2%, the run
#   is UNINFORMATIVE, not a null - the KV effect itself failed to appear.
set -u
cd "C:/Users/Federico/Documents/evo-compress/.claude/worktrees/law5-prefill" || exit 1
B="/c/Users/Federico/Documents/evo-compress/tools/llamacpp-b10098/llama-bench.exe"
M="D:/evo-compress-data/gguf/Qwen2.5-14B-Instruct-Q4_K_M.gguf"
OUT="weights/data/exp57_kv_to_layers"
D=16384

g=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
[ "$g" -gt 1500 ] && { echo "ABORT: GPU busy ${g}MiB"; exit 2; }
echo "=== #57 staked: C must beat A by MORE than B does, else promotion added nothing ==="

run() { # $1=tag $2=ngl $3...=kv flags
  tag=$1; ngl=$2; shift 2
  echo "--- arm $tag: -ngl $ngl $* at d$D ---"
  "$B" -m "$M" -ngl "$ngl" -p 0 -n 64 -d $D -r 2 -fa 1 "$@" > "${OUT}_${tag}.log" 2>&1
  grep -E "tg64|d$D" "${OUT}_${tag}.log" | tail -1 || echo "  FAILED/OOM (recorded as the result)"
}

run A 21
run B 21 -ctk q8_0 -ctv q8_0
run C 27 -ctk q8_0 -ctv q8_0
echo ""
echo "logs -> ${OUT}_{A,B,C}.log   score A vs B vs C by hand against the staked rule"
