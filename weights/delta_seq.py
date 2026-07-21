"""Sequence / trajectory compression: store a whole training RUN cheaply.

The real use case isn't one model -- it's the dozens of checkpoints a training run
produces (or the many fine-tunes of a base). Two questions:

 1. PRACTICAL: if we store checkpoint_0 standalone and every later checkpoint as a
    per-plane XOR-delta vs the previous one, what's the total size for the run vs
    storing every checkpoint standalone?  (chain is exact: XOR is reversible.)

 2. HYPOTHESIS: are consecutive *arithmetic* deltas correlated (Adam momentum ->
    d_t ~= d_{t-1})?  If yes, a second-order (delta-of-delta) arithmetic coder could
    beat the per-checkpoint XOR delta. We measure cosine(d_t, d_{t-1}) and the
    magnitude ratio ||d_t - d_{t-1}|| / ||d_t||. (XOR can't exploit this -- it
    telescopes -- so this only motivates a future arithmetic coder.)
"""

from __future__ import annotations

import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from weights import codecs as cd  # noqa: E402
from weights.delta import DeltaCodec  # noqa: E402
from weights.fetch_weights import parse_safetensors  # noqa: E402

REPO = "EleutherAI/pythia-70m"
# oldest -> newest; we use whichever consecutive run is fully cached
CANDIDATES = ["step140000", "step141000", "step142000", "step143000"]


def try_load(rev):
    from huggingface_hub import hf_hub_download
    try:
        p = hf_hub_download(REPO, "model.safetensors", revision=rev, local_files_only=True)
    except Exception:
        return None
    return {n: (dt, sh, raw) for n, dt, sh, raw in parse_safetensors(p)}


def main():
    loaded = []
    for rev in CANDIDATES:
        T = try_load(rev)
        if T is not None:
            loaded.append((rev, T))
            print(f"  cached: {rev}", flush=True)
    if len(loaded) < 2:
        print("need >=2 consecutive cached checkpoints; have", len(loaded))
        return

    revs = [r for r, _ in loaded]
    Ts = [T for _, T in loaded]
    common = [k for k in Ts[0] if all(k in T for T in Ts)
              and Ts[0][k][0] == "F32"
              and all(len(T[k][2]) == len(Ts[0][k][2]) for T in Ts)]

    smart = cd.SplitSmartCodec("zstd", 19)
    dp = DeltaCodec(cd.SplitPerPlaneCodec(["zstd"], 19, "pp"), "delta-perplane")

    # ---- 1. practical: chain size vs all-standalone ----
    print(f"\nstoring a {len(revs)}-checkpoint run ({' -> '.join(r[4:] for r in revs)}):")
    raw = std_total = chain_total = 0
    first_std = 0
    for k in common:
        nbytes = len(Ts[0][k][2])
        raw += nbytes * len(revs)
        for i, (rev, T) in enumerate(loaded):
            b = T[k][2]
            std_total += len(smart.compress(b, "fp32"))
            if i == 0:
                first_std += len(smart.compress(b, "fp32"))
                chain_total += len(smart.compress(b, "fp32"))
            else:
                prev = loaded[i - 1][1][k][2]
                chain_total += len(dp.compress(b, "fp32", prev))
    n_later = len(revs) - 1
    print(f"  raw:                {raw/1e6:>9.1f} MB")
    print(f"  all standalone:     {std_total/1e6:>9.1f} MB  (save {(1-std_total/raw)*100:4.1f}%)")
    print(f"  chain (delta):      {chain_total/1e6:>9.1f} MB  (save {(1-chain_total/raw)*100:4.1f}%)")
    print(f"  -> checkpoint_0 {first_std/1e6:.1f} MB + {n_later} deltas avg "
          f"{(chain_total-first_std)/n_later/1e6:.1f} MB each "
          f"({(chain_total-first_std)/n_later/(raw/len(revs))*100:.0f}% of a model)")

    # ---- 2. momentum hypothesis: correlation of consecutive arithmetic deltas ----
    if len(loaded) >= 3:
        print("\nmomentum check (consecutive float deltas d_t = w_t - w_{t-1}):")
        # build concatenated float vectors
        W = []
        for _, T in loaded:
            W.append(np.concatenate([np.frombuffer(T[k][2], "<f4").astype(np.float64)
                                     for k in common]))
        D = [W[i] - W[i - 1] for i in range(1, len(W))]
        for i in range(1, len(D)):
            a, b = D[i], D[i - 1]
            cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
            ratio = float(np.linalg.norm(a - b) / (np.linalg.norm(a) + 1e-30))
            print(f"  d[{revs[i+1][4:]}-{revs[i][4:]}] vs d[{revs[i][4:]}-{revs[i-1][4:]}]: "
                  f"cosine={cos:+.3f}  ||dd||/||d||={ratio:.3f}")
        print("  (cosine>>0 => momentum real => 2nd-order arithmetic coder worth building)")
    else:
        print("\nmomentum check: need >=3 consecutive checkpoints (skipped)")


if __name__ == "__main__":
    main()
