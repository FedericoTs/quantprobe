"""Campaign2-B2 analytic gate -- can a PERMUTATION rescue block-structured masking?

B1 showed natural 32/128-blocks are catastrophic (+0.73/+1.39 at f=0.7 vs FREE unstructured).
A permutation (re-order neurons at encode time) can only help if drop events are CLUSTERABLE:
i.e., per-token dropped neurons are largely the SAME across tokens (quasi-static), not contextual.

Measures, per layer (MLP intermediate, f=0.7 top-|energy| keep):
  - per-neuron drop frequency distribution (bimodal => clusterable; flat ~0.3 => hopeless)
  - fraction of neurons dropped on >95% of tokens (statically prunable mass = the permutation's
    best-case jointly-dead block material)
  - mean Jaccard overlap of dropped sets between random token pairs (contextuality)

ROADMAP KILL: if best-case jointly-dead material < 25%, the down-proj gating/permutation branch
dies analytically -- no GA, no kernel. (Spec-decode then takes the multiplier slot.)

Run:  python -m weights.coactivation_oracle
"""
from __future__ import annotations

import sys

import numpy as np
import torch

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from transformers import AutoTokenizer
from weights.quant_lab import CFG, EVAL_TEXT, build_model, load_fp16

F_KEEP = 0.7


def main():
    tok = AutoTokenizer.from_pretrained(CFG)
    model = build_model()
    load_fp16(model)
    ids = tok(EVAL_TEXT, return_tensors="pt").input_ids[:, :512]

    keeps = {}                                   # layer -> bool [T, I] keep mask

    def mk(li):
        def pre_hook(mod, args):
            (h,) = args
            T, I = h.shape[1], h.shape[2]
            e = (h.float() ** 2)[0]              # [T, I] per-neuron energy
            k = int(round(I * F_KEEP))
            thr = e.topk(k, dim=-1).values[:, -1:]
            keeps[li] = (e >= thr).numpy()
            return None
        return pre_hook

    hooks = [model.model.layers[li].mlp.down_proj.register_forward_pre_hook(mk(li))
             for li in range(len(model.model.layers))]
    with torch.no_grad():
        model(ids)
    for h in hooks:
        h.remove()

    print(f"f_keep={F_KEEP} on MLP intermediate (per-neuron |h|^2 top-k), 512 eval tokens\n")
    print(f"{'layer':>6}{'always-drop%':>13}{'always-keep%':>13}{'drop-Jaccard':>13}")
    print("-" * 48)
    ad_all, jac_all = [], []
    rng = np.random.default_rng(0)
    for li in sorted(keeps):
        K = keeps[li]                            # [T, I] keep
        D = ~K
        dropfreq = D.mean(0)                     # per-neuron drop frequency
        always_drop = float((dropfreq > 0.95).mean())
        always_keep = float((dropfreq < 0.05).mean())
        # contextuality: Jaccard of dropped sets across random token pairs
        T = K.shape[0]
        pairs = rng.integers(0, T, (200, 2))
        j = []
        for a, b in pairs:
            inter = np.logical_and(D[a], D[b]).sum()
            union = np.logical_or(D[a], D[b]).sum()
            j.append(inter / max(union, 1))
        jac = float(np.mean(j))
        ad_all.append(always_drop)
        jac_all.append(jac)
        if li % 6 == 0 or li == len(keeps) - 1:
            print(f"{li:>6}{100*always_drop:>12.1f}%{100*always_keep:>12.1f}%{jac:>13.3f}")

    AD = float(np.mean(ad_all))
    JC = float(np.mean(jac_all))
    print(f"\nMEAN: always-dropped neurons = {100*AD:.1f}% | drop-set Jaccard = {JC:.3f}")
    print(f"(random-overlap baseline for Jaccard at 30% drop ~= 0.176)")
    best_case = AD                               # only quasi-static mass clusters into jointly-dead blocks
    if best_case >= 0.25:
        print(f"=> {100*best_case:.0f}% quasi-static droppable mass: PERMUTATION BRANCH ALIVE "
              f"(>=25% gate) -- proceed to greedy 128-clustering + re-encode test.")
    else:
        print(f"=> only {100*best_case:.0f}% quasi-static (<25% gate) and Jaccard ~random means "
              f"drops are CONTEXTUAL: no permutation can form jointly-dead 128-blocks. "
              f"DOWN-PROJ MASKING BRANCH DEAD ANALYTICALLY -- spec-decode (B3) takes the slot.")


if __name__ == "__main__":
    main()
