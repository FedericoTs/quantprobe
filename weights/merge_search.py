"""Model recombination, step 2: does a RICHER recipe search beat the best uniform merge?

Uniform task arithmetic uses one coefficient alpha. Sakana/AlphaEvolve's premise is that
searching a richer recipe space wins. We give each MODULE TYPE (embed, attention, MLP, norm)
its own coefficient and search them (coordinate descent), optimizing a normalized math+general
perplexity score. If the per-module recipe beats the best uniform alpha, search-over-recipes
has real headroom -- the kernel that an AlphaEvolve operator-discovery engine would exploit.
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


def group_of(name):
    if "embed" in name or "lm_head" in name:
        return "embed"
    if "self_attn" in name:
        return "attn"
    if "mlp" in name:
        return "mlp"
    return "norm"


GROUPS = ["embed", "attn", "mlp", "norm"]


def main(spec_name="mathphd"):
    cfg = AutoConfig.from_pretrained(CFG)
    tok = AutoTokenizer.from_pretrained(CFG)
    model = AutoModelForCausalLM.from_config(cfg).eval()
    base = load_st(os.path.join(D, "qwen", "base.safetensors"))
    spec = load_st(os.path.join(D, "qwen_family", f"{spec_name}.safetensors"))
    tau = {k: (spec[k] - base[k]) for k in base if k in spec and spec[k].shape == base[k].shape}
    del spec
    grp = {k: group_of(k) for k in base}

    gen_text = open(os.path.join(_ROOT, "data/corpora/generic-text/enwik8_256k"),
                    "rb").read()[:60000].decode("latin-1")
    gsm = [json.loads(l) for l in open(os.path.join(D, "gsm8k_test.jsonl"))][:120]
    math_text = "\n\n".join(f"Question: {x['question']}\nAnswer: {x['answer']}" for x in gsm)

    @torch.no_grad()
    def ppl(text, mt=1600):
        ids = tok(text, return_tensors="pt").input_ids[:, :mt]
        return float(torch.exp(model(ids, labels=ids).loss))

    def set_recipe(coef):  # coef: dict group->alpha
        sd = {k: (base[k] + coef[grp[k]] * tau[k] if k in tau else base[k]) for k in base}
        model.load_state_dict(sd, strict=False)
        model.tie_weights()

    # base normalizers
    set_recipe({g: 0.0 for g in GROUPS})
    bm, bg = ppl(math_text), ppl(gen_text)

    def score(coef):
        set_recipe(coef)
        return ppl(math_text) / bm + ppl(gen_text) / bg  # lower is better; base = 2.0

    # best UNIFORM alpha (the naive recipe)
    print("uniform-alpha baseline:", flush=True)
    best_u, best_us = None, 9e9
    for a in (0.25, 0.5, 0.75, 1.0):
        s = score({g: a for g in GROUPS})
        print(f"  alpha={a}: score {s:.4f}", flush=True)
        if s < best_us:
            best_us, best_u = s, a

    # per-module COORDINATE DESCENT (the richer recipe)
    print("\nper-module coordinate-descent search:", flush=True)
    coef = {g: best_u for g in GROUPS}
    cur = score(coef)
    for it in range(1):
        for g in GROUPS:
            for a in (0.0, 0.5, 1.0, 1.25):
                trial = dict(coef); trial[g] = a
                s = score(trial)
                if s < cur:
                    cur, coef = s, trial
        print(f"  pass {it+1}: score {cur:.4f}  recipe {coef}", flush=True)

    print(f"\n  best uniform:    score {best_us:.4f} (alpha={best_u})")
    print(f"  best per-module: score {cur:.4f}  {coef}")
    print(f"  improvement over uniform: {(best_us-cur):+.4f} ({(best_us-cur)/best_us*100:+.1f}%)")
    if cur < best_us - 0.005:
        print("  => RICHER RECIPE WINS: per-module search beats the best uniform merge.")
        print("     Search-over-recipes has real headroom -> the AlphaEvolve premise holds here.")
    else:
        print("  => per-module ~= uniform here; richer recipe gave little. Need a bigger recipe")
        print("     space / multiple specialists / learned operators to find headroom.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "mathphd")
