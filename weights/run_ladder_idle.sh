#!/bin/bash
# Idle-state ladder rerun. New guard: CPU idleness gate (the 2026-07-31 ladder was
# contaminated by ambient CPU load - every CPU-participating row fell 10-20% while
# GPU-only rows held; we gated GPU idleness but never CPU).
cd "C:/Users/Federico/Documents/evo-compress/.claude/worktrees/law5-prefill" || exit 1
TS=$(date +%Y%m%d_%H%M)
LOG="weights/data/ladder_idle_${TS}.log"

taskkill //IM llama-server.exe //F 2>/dev/null; taskkill //IM llama-bench.exe //F 2>/dev/null
sleep 3

# GPU gate
g=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
[ "$g" -gt 1500 ] && { echo "ABORT: GPU busy ${g}MiB" | tee "$LOG"; exit 2; }

# CPU gate: WAIT for 3 consecutive samples under 25%, sampling every 60s, up to ~40 min.
# Missing counter data still counts as busy (fail-safe).
ok=0; tries=0
while [ $ok -lt 3 ]; do
  c=$(powershell -NoProfile -Command "(Get-CimInstance Win32_Processor | Measure-Object -Property LoadPercentage -Average).Average" 2>/dev/null | grep -oE '^[0-9]+' | head -1)
  tries=$((tries+1))
  if [ "${c:-100}" -le 25 ]; then ok=$((ok+1)); else ok=0; fi
  echo "cpu sample $tries: ${c:-unreadable}%  (idle streak $ok/3)" | tee -a "$LOG"
  [ $tries -ge 40 ] && { echo "GAVE UP after 40 samples - machine never idle" | tee -a "$LOG"; exit 3; }
  [ $ok -lt 3 ] && sleep 60
done
echo "gates passed: GPU ${g}MiB, CPU idle. Fresh calibrate + full ladder." | tee -a "$LOG"

python weights/full_ladder_v124.py --bench >> "$LOG" 2>&1
rc=$?
echo "=== ladder exit $rc ===" | tee -a "$LOG"
tail -30 "$LOG"
