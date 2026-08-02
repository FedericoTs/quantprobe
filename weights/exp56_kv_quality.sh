#!/bin/bash
# Prereg #91: KV-quant quality A/B. Guards per the staked prereg - see the .md.
# Round 2: probe the deepest -c that f16 KV fits (8192 OOMs, 4096 fits), then run
# both arms with 12 chunks so the error bar can actually resolve the 2% gate.
set -u
BIN="<repo>/tools/llamacpp-b10098/llama-perplexity.exe"
M="D:/evo-compress-data/gguf/Qwen3-30B-A3B-Q2_K.gguf"
F="weights/data/wikitext2_test.txt"
OT='blk\.(16|17|18|19|20|21|22|23|24|25|26|27|28|29|30|31)\.ffn_.*_exps\.=CPU'
OUT="weights/data/exp56_kv_quality"
CHUNKS=12

busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
[ "$busy" -gt 1500 ] && { echo "ABORT: GPU busy (${busy} MiB)"; exit 2; }

probe() { # 1-chunk f16 probe at depth $1 -> 0 if it fits
  "$BIN" -m "$M" -f "$F" -c "$1" --chunks 1 -ngl 99 -ot "$OT" --no-mmap -fa 1 \
    > "${OUT}_probe_$1.log" 2>&1
  grep -qi "Final estimate" "${OUT}_probe_$1.log"
}

C=0
for d in 7168 6144 5120; do
  echo "probing f16 at -c $d ..."
  if probe "$d"; then C=$d; echo "  fits"; break; else echo "  OOM/fail"; fi
done
[ "$C" -eq 0 ] && C=4096 && echo "falling back to known-good 4096"
echo "=== deepest f16-viable depth: $C | chunks: $CHUNKS ==="

run() {
  tag=$1; shift
  echo "=== arm $tag (c=$C x $CHUNKS) ==="
  "$BIN" -m "$M" -f "$F" -c "$C" --chunks $CHUNKS -ngl 99 -ot "$OT" --no-mmap -fa 1 "$@" \
    > "${OUT}_${tag}.log" 2>&1
  grep -i "Final estimate" "${OUT}_${tag}.log" | tail -1
}

run f16
run q8  -ctk q8_0 -ctv q8_0

pf=$(grep -i -oE "PPL = [0-9.]+" "${OUT}_f16.log" | tail -1 | grep -oE "[0-9.]+")
pq=$(grep -i -oE "PPL = [0-9.]+" "${OUT}_q8.log"  | tail -1 | grep -oE "[0-9.]+")
ef=$(grep -i -oE "\+/- [0-9.]+" "${OUT}_f16.log" | tail -1 | grep -oE "[0-9.]+")
eq=$(grep -i -oE "\+/- [0-9.]+" "${OUT}_q8.log"  | tail -1 | grep -oE "[0-9.]+")
echo ""
echo "depth=$C chunks=$CHUNKS  f16=$pf +/-$ef  q8_0=$pq +/-$eq"
python - <<PYEOF
pf, pq, ef, eq = $pf, $pq, $ef, $eq
r = pq / pf
# error on the ratio, first-order
err = r * ((ef/pf)**2 + (eq/pq)**2) ** 0.5
print(f"ratio {r:.5f} +/- {err:.5f}   gate <=1.02")
if abs(pq - pf) < 5e-5:
    print("VERDICT: UNINFORMATIVE (cannot-vary signature)"); raise SystemExit(3)
if err > 0.02:
    print(f"RESOLUTION WARNING: ratio error {err:.4f} exceeds the 0.02 gate width -")
    print("a pass here is 'no large cost', NOT a certified <=2%. Say so when scoring.")
print("VERDICT:", "P1 PASS" if r <= 1.02 else "P1 KILL - disclosed option only")
PYEOF
