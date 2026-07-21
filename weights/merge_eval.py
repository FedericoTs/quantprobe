"""Model recombination experiment, step 1: does merging trade off cleanly + is there a
sweet spot? We interpolate a math specialist into the base (task arithmetic) and measure
perplexity on math (GSM8K) and general (enwik8) text. A good merge should keep most of the
math gain while recovering general competence -- a point better than the naive endpoints.
Fast: perplexity only, no generation.
"""

from __future__ import annotations

import json
import os
import sys

import torch
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = os.path.join(_ROOT, "weights", "data", "qwen_cfg")
D = os.path.join(_ROOT, "weights", "data")
torch.set_num_threads(os.cpu_count() or 4)


def load_st(path):
    out = {}
    with safe_open(path, framework="pt") as f:
        for k in f.keys():
            out[k] = f.get_tensor(k).float()
    return out


def main(spec_name="mathphd"):
    print(f"loading config/tokenizer + base + {spec_name} ...", flush=True)
    cfg = AutoConfig.from_pretrained(CFG)
    tok = AutoTokenizer.from_pretrained(CFG)
    model = AutoModelForCausalLM.from_config(cfg).eval()
    base = load_st(os.path.join(D, "qwen", "base.safetensors"))
    spec = load_st(os.path.join(D, "qwen_family", f"{spec_name}.safetensors"))
    tau = {k: (spec[k] - base[k]) for k in base if k in spec and spec[k].shape == base[k].shape}
    del spec
    print(f"  {len(tau)}/{len(base)} tensors form the task vector", flush=True)

    def set_alpha(a):
        sd = {k: (base[k] + a * tau[k] if k in tau else base[k]) for k in base}
        model.load_state_dict(sd, strict=False)
        model.tie_weights()

    gen_text = open(os.path.join(_ROOT, "data/corpora/generic-text/enwik8_256k"),
                    "rb").read()[:60000].decode("latin-1")
    gsm = [json.loads(l) for l in open(os.path.join(D, "gsm8k_test.jsonl"))][:120]
    math_text = "\n\n".join(f"Question: {x['question']}\nAnswer: {x['answer']}" for x in gsm)

    @torch.no_grad()
    def ppl(text, max_tokens=3500):
        ids = tok(text, return_tensors="pt").input_ids[:, :max_tokens]
        return float(torch.exp(model(ids, labels=ids).loss))

    print(f"\n{'alpha':<8}{'math PPL':>11}{'general PPL':>13}", flush=True)
    print("-" * 32)
    rows = []
    for a in (0.0, 0.25, 0.5, 0.75, 1.0, 1.25):
        set_alpha(a)
        m, g = ppl(math_text), ppl(gen_text)
        tag = " (base)" if a == 0 else (f" ({spec_name})" if a == 1 else "")
        print(f"{a:<8}{m:>11.2f}{g:>13.2f}{tag}", flush=True)
        rows.append((a, m, g))

    base_m, base_g = rows[0][1], rows[0][2]
    spec_m, spec_g = [r for r in rows if r[0] == 1.0][0][1:]
    # a merge "wins" if it beats base at math AND beats the specialist at general
    wins = [r for r in rows if 0 < r[0] < 1 and r[1] < base_m and r[2] < spec_g]
    print()
    if wins:
        b = min(wins, key=lambda r: r[1] / base_m + r[2] / spec_g)
        print(f"  SWEET SPOT alpha={b[0]}: math {b[1]:.1f} (< base {base_m:.1f}) AND "
              f"general {b[2]:.1f} (< {spec_name} {spec_g:.1f})")
        print("  => merging Pareto-beats both endpoints: better at math than base, better at")
        print("     general than the specialist. The mechanic works; search has room to optimize.")
    else:
        print("  no interpolation point Pareto-beats both endpoints for this specialist.")
        print(f"  (base math {base_m:.1f}/gen {base_g:.1f}; {spec_name} math {spec_m:.1f}/gen {spec_g:.1f})")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "mathphd")
