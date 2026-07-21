"""Validate delta compression on REAL training deltas.

Pythia publishes training checkpoints. Two checkpoints differ by exactly the real
training updates over N steps -- a genuine, dense, small delta (the honest test, not
a synthetic perturbation). We compress the XOR-delta (exact given the older
checkpoint) vs standalone, per matching tensor, aggregated.
"""

from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from weights import codecs as cd  # noqa: E402
from weights import track  # noqa: E402
from weights.delta import DeltaCodec  # noqa: E402
from weights.fetch_weights import parse_safetensors  # noqa: E402

REPO = "EleutherAI/pythia-70m"
PAIRS = [("step143000", "step142000"), ("step143000", "step139000")]
DT_MAP = {"F16": "fp16", "F32": "fp32", "BF16": "bf16"}
CACHE_ONLY = os.environ.get("DELTA_DOWNLOAD") != "1"


def load(rev):
    from huggingface_hub import hf_hub_download
    print(f"    loading {rev} ...", flush=True)
    path = hf_hub_download(REPO, "model.safetensors", revision=rev,
                           local_files_only=CACHE_ONLY)
    return {n: (dt, sh, raw) for n, dt, sh, raw in parse_safetensors(path)}


def main():
    print(f"validating delta on REAL {REPO} checkpoint pairs\n")
    smart = cd.SplitSmartCodec("zstd", 19)
    dz = DeltaCodec(cd.SplitCodec("zstd", 19), "delta-zstd")
    dp = DeltaCodec(cd.SplitPerPlaneCodec(["zstd"], 19, "pp"), "delta-perplane")
    print(f"{'pair':<12}{'dt':>5}{'raw MB':>9}{'standalone':>12}{'delta-zstd':>12}"
          f"{'delta-perplane':>16}  rt", flush=True)
    print("-" * 70, flush=True)
    for newer, older in PAIRS:
        try:
            Tn = load(newer)
            To = load(older)
        except Exception as exc:
            print(f"{newer} vs {older}: skip ({type(exc).__name__}: {exc})", flush=True)
            continue
        common = [k for k in Tn if k in To and Tn[k][0] in DT_MAP
                  and len(Tn[k][2]) == len(To[k][2]) and len(Tn[k][2]) >= 256]
        traw = tstd = tdz = tpp = 0
        ok = True
        dtype = None
        for k in common:
            dtype = DT_MAP[Tn[k][0]]
            new_b, old_b = Tn[k][2], To[k][2]
            tstd += len(smart.compress(new_b, dtype))
            zb = dz.compress(new_b, dtype, old_b)
            pb = dp.compress(new_b, dtype, old_b)
            if dz.decompress(zb, dtype, old_b) != new_b or dp.decompress(pb, dtype, old_b) != new_b:
                ok = False
            traw += len(new_b)
            tdz += len(zb)
            tpp += len(pb)
        gap = int(newer[4:]) - int(older[4:])
        s_std = (1 - tstd / traw) * 100
        s_dz = (1 - tdz / traw) * 100
        s_pp = (1 - tpp / traw) * 100
        label = f"{gap}step"
        print(f"{label:<12}{dtype:>5}{traw/1e6:>9.1f}{s_std:>11.1f}%{s_dz:>11.1f}%"
              f"{s_pp:>15.1f}%  {'ok' if ok else 'FAIL'}", flush=True)
        tdz = tpp  # record the better (per-plane) result below
        track.record({
            "codec": f"delta-real@{gap}steps",
            "config": {"type": "delta-real", "repo": REPO, "gap_steps": gap,
                       "tensors": len(common), "dtype": dtype},
            "overall": {"in_bytes": traw, "out_bytes": tdz,
                        "ratio": round(traw / tdz, 4), "save_pct": round(s_dz, 2),
                        "enc_MBps": 0.0, "dec_MBps": 0.0, "rt_ok": ok},
            "by_dtype": {dtype: {
                "in_bytes": traw, "out_bytes": tdz, "ratio": round(traw / tdz, 4),
                "save_pct": round(s_dz, 2), "dec_MBps": 0.0, "rt_ok": ok}},
            "n_tensors": len(common),
        }, note=f"REAL checkpoint delta, {gap} training steps apart; standalone={s_std:.1f}%")


if __name__ == "__main__":
    main()
