"""Evolution ARENA: scores every codec in codec_zoo.CODECS via the held-out verifier
(perplexity + bits) + decode-cost class, maintains a leaderboard / MAP-Elites archive,
and persists to results/quant_arena.json. CRASH-RESILIENT + RESUMABLE: each codec is
evaluated in isolation (try/except + gc) and checkpointed to disk immediately, so an
OOM only loses the current codec; re-running skips finished ones.

The LLM (Claude) is the mutation operator: edit codec_zoo.py to add codecs, run this.
"""
from __future__ import annotations

import gc
import hashlib
import inspect
import json
import os

from transformers import AutoTokenizer

from weights import codec_zoo
from weights.quant_lab import CFG, build_model, calibrate, load_fp16, load_quant, ppl

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(_ROOT, "weights", "results", "quant_arena.json")
WALL = 4.934


def chash(fn):
    return hashlib.md5(inspect.getsource(fn).encode()).hexdigest()[:8]


def main():
    os.makedirs(os.path.dirname(RES), exist_ok=True)
    db = json.load(open(RES)) if os.path.exists(RES) else {}

    tok = AutoTokenizer.from_pretrained(CFG)
    model = build_model()
    load_fp16(model)
    print("calibrating ...", flush=True)
    calib = calibrate(model, tok)
    fp16 = ppl(model, tok)
    db["_fp16"] = fp16
    print(f"fp16 held-out ppl = {fp16:.3f}   (wall to beat: {WALL})\n", flush=True)

    for name, (fn, cost) in codec_zoo.CODECS.items():
        h = chash(fn)
        if name in db and db[name].get("hash") == h and "ppl" in db[name]:
            print(f"  [cached] {name:<24} ppl {db[name]['ppl']:.3f}", flush=True)
            continue
        try:
            bpw = load_quant(model, lambda a, k: fn(a, k, calib))
            p = ppl(model, tok)
            db[name] = {"hash": h, "bpw": bpw, "ppl": p, "cost": cost}
            print(f"  {name:<24} {bpw:5.3f}b  ppl {p:8.3f}  [{cost}]", flush=True)
        except Exception as e:
            db[name] = {"hash": h, "err": type(e).__name__}
            print(f"  {name:<24} FAILED: {type(e).__name__}", flush=True)
        json.dump(db, open(RES, "w"), indent=1)
        gc.collect()

    rows = [(n, d) for n, d in db.items()
            if isinstance(d, dict) and d.get("ppl") is not None]
    rows.sort(key=lambda x: x[1]["ppl"])
    print("\nLEADERBOARD (held-out ppl, lower=better):")
    for n, d in rows:
        flag = "  <== NEW BEST (broke wall)" if d["ppl"] < WALL - 0.02 and "champ" not in n else ""
        print(f"  {d['ppl']:8.3f} @ {d['bpw']:.3f}b [{d['cost']:>4}]  {n}{flag}", flush=True)
    if rows:
        best = rows[0]
        print(f"\n  fp16 {fp16:.3f} | wall {WALL} | best {best[1]['ppl']:.3f}  ({best[0]})")
        print("  => DISCOVERY: new operator broke the wall." if best[1]["ppl"] < WALL - 0.02
              else "  => wall holds; mutate further.")


if __name__ == "__main__":
    main()
