"""Prereg #98 - does TAIL KL discriminate damage that mean perplexity hides?

WHY. quantprobe reports perplexity, and perplexity is a MEAN. Quantization damage is not
uniformly distributed: it concentrates in rare tokens and confident predictions, exactly where
an average washes it out. L-24 is the standing example - q8_0 KV measured a quality ratio of
1.00031 +/- 0.0188, a noise band 60x the effect, which is a metric telling you nothing while
sounding precise.

THE CHEAP DISCOVERY, made before staking: llama.cpp's --kl-divergence ALREADY reports the full
distribution - Maximum, 99.9%, 99.0%, 95.0%, 90.0%, Median, and the low tail, plus "Same top p"
(the share of tokens where the quantized model still picks the same argmax). We have been
running perplexity and discarding a richer signal that costs the same forward passes. Nothing
needs to be computed; it needs to be READ.

CANNOT-VARY GUARD, run BEFORE staking so the harness is known to work: Q8_0 scored against its
OWN logits returned Maximum KLD 0.000053, Median -0.000000, Same top p 100.000%. The harness
returns ~0 for an identical model, so a non-zero result later is signal and not machinery.

STAKED BEFORE THE DAMAGED ARMS RAN (2026-08-01):
  Reference logits: Qwen2.5-0.5B-Instruct Q8_0. Arms: the SAME model requantized to Q4_K_M
  (mild) and Q2_K (heavy). Requantizing from Q8_0 is lossy-on-lossy and that is fine here -
  the point is graded, real damage, not a release-quality quant.

  P1 (the whole question). Let R_ppl  = ppl(Q2_K)/ppl(Q4_K_M)
                           and R_p99  = KLD_99%(Q2_K)/KLD_99%(Q4_K_M).
      TAIL KL EARNS ITS PLACE iff R_p99 >= 1.5 * R_ppl - i.e. the tail moves at least 50%
      more than the mean does. If the two track each other within 1.5x, the tail is carrying
      no information the mean lacks, and the metric MUST NOT SHIP. That is the null this
      experiment is built to be able to return, and it is a real possibility: ppl and KL are
      not independent quantities.

  P2 Same-top-p degrades monotonically Q8_0 -> Q4_K_M -> Q2_K, giving a directly interpretable
      number ("this quant changes the chosen token on X% of positions") that perplexity cannot
      express at any sample size.

  P3 The heavy arm's damage is visibly ASYMMETRIC across the distribution: KLD_99% / KLD_median
      is materially larger than 1. A symmetric widening would mean the damage is uniform and
      the mean is an adequate summary after all.

  KILL RULES
  K-A If any arm fails to produce a parseable KLD block, report UNRUN. Do not substitute ppl.
  K-B If Q4_K_M and Q2_K return the same ppl within 1%, the damage lever did not pull and the
      run is UNINFORMATIVE regardless of what the tail did - the same failure that killed #96.
  K-C No threshold moves after seeing a number. If P1 lands between 1.0x and 1.5x that is a
      MISS, published as one, and the metric does not ship.

  python weights/exp98_tail_kl.py
"""
from __future__ import annotations
import json, os, re, subprocess, sys, time

# Resolved at RUNTIME. The privacy scrub (61d8068) forbids a hardcoded machine path, and a
# "<repo>" placeholder is not a runnable one - a scrubbed script that silently cannot find its
# binary is a worse outcome than either. Override with QP_LLAMACPP.
B = os.environ.get("QP_LLAMACPP") or os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "tools", "llamacpp-b10098")
PERP = os.path.join(B, "llama-perplexity.exe")
QUANT = os.path.join(B, "llama-quantize.exe")
SRC = "D:/evo-compress-data/gguf/Qwen2.5-0.5B-Instruct-Q8_0.gguf"
TMP = "D:/evo-compress-data/tmp"
REF = os.path.join(TMP, "ref.kld")
CORPUS = "weights/data/wikitext2_test.txt"
OUT = "weights/data/exp98_tail_kl.json"
CHUNKS = 6

PCTS = ["Maximum", "99.9%", "99.0%", "95.0%", "90.0%", "Median"]


def parse(out):
    """Read the KLD block and the top-p agreement. Returns {} if the block is absent (K-A)."""
    r = {}
    for key in PCTS:
        m = re.search(re.escape(key) + r"\s+KLD:\s*(-?[0-9.]+)", out)
        if m:
            r[key] = float(m.group(1))
    m = re.search(r"Same top p:\s*([0-9.]+)", out)
    if m:
        r["same_top_p"] = float(m.group(1))
    m = re.search(r"Final estimate: PPL = ([0-9.]+)", out)
    if m:
        r["ppl"] = float(m.group(1))
    m = re.search(r"Mean\s+KLD:\s*(-?[0-9.]+)", out)
    if m:
        r["mean_kld"] = float(m.group(1))
    return r


def run(model, tag):
    """TWO passes, because --kl-divergence REPLACES the perplexity report rather than adding
    to it. The first attempt asked one run for both and K-A correctly refused to score a
    missing PPL - the kill rule worked, the harness did not."""
    print(f"  [{tag}] KL pass against the Q8_0 reference logits...", flush=True)
    a = subprocess.run([PERP, "-m", model, "-f", CORPUS, "--chunks", str(CHUNKS), "-ngl", "99",
                        "--kl-divergence", "--kl-divergence-base", REF],
                       capture_output=True, text=True, errors="replace")
    print(f"  [{tag}] perplexity pass...", flush=True)
    b = subprocess.run([PERP, "-m", model, "-f", CORPUS, "--chunks", str(CHUNKS), "-ngl", "99"],
                       capture_output=True, text=True, errors="replace")
    out = a.stdout + a.stderr + "\n===== PPL PASS =====\n" + b.stdout + b.stderr
    open(os.path.join("weights", "data", f"exp98_{tag}.log"), "w", encoding="utf-8").write(out)
    return parse(out)


def quantize(kind, tag):
    dst = os.path.join(TMP, f"q05b_{tag}.gguf")
    if os.path.isfile(dst):
        print(f"  [{tag}] reusing {os.path.basename(dst)}")
        return dst
    print(f"  [{tag}] quantizing {kind}...", flush=True)
    p = subprocess.run([QUANT, "--allow-requantize", SRC, dst, kind],
                       capture_output=True, text=True, errors="replace")
    if not os.path.isfile(dst):
        print(p.stdout[-800:] + p.stderr[-800:])
        return None
    return dst


def main():
    res = {"utc": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
           "reference": SRC, "chunks": CHUNKS, "arms": {}}
    if not os.path.isfile(REF):
        print(f"missing reference logits {REF} - generate with --kl-divergence-base first")
        return 2

    for kind, tag in [("Q4_K_M", "q4km"), ("Q2_K", "q2k")]:
        m = quantize(kind, tag)
        if not m:
            res["verdict"] = f"UNRUN - K-A: could not build the {kind} arm."
            json.dump(res, open(OUT, "w"), indent=1); print(res["verdict"]); return 1
        res["arms"][tag] = run(m, tag)
        a = res["arms"][tag]
        print(f"    ppl {a.get('ppl')}  KLD99 {a.get('99.0%')}  median {a.get('Median')}  "
              f"same-top-p {a.get('same_top_p')}")

    A, Bm = res["arms"].get("q4km", {}), res["arms"].get("q2k", {})
    need = ["ppl", "99.0%", "Median", "same_top_p"]
    if any(k not in A or k not in Bm for k in need):
        res["verdict"] = "UNRUN - K-A: a KLD block did not parse. Not substituting perplexity."
        json.dump(res, open(OUT, "w"), indent=1); print("\n" + res["verdict"]); return 1

    R_ppl = Bm["ppl"] / A["ppl"]
    R_p99 = Bm["99.0%"] / A["99.0%"] if A["99.0%"] else None
    res["R_ppl"], res["R_p99"] = round(R_ppl, 4), (round(R_p99, 4) if R_p99 else None)
    res["ratio_of_ratios"] = round(R_p99 / R_ppl, 3) if R_p99 else None
    res["asymmetry_q2k"] = (round(Bm["99.0%"] / Bm["Median"], 2)
                            if Bm.get("Median") else None)
    print(f"\n  R_ppl {R_ppl:.4f}   R_p99 {R_p99:.4f}   tail/mean {R_p99/R_ppl:.2f}x")

    if abs(R_ppl - 1.0) < 0.01:
        res["verdict"] = (f"UNINFORMATIVE - K-B: the damage lever did not pull (ppl ratio "
                          f"{R_ppl:.4f}, within 1%). Nothing about the tail can be claimed.")
    elif R_p99 >= 1.5 * R_ppl:
        res["verdict"] = (f"P1 CONFIRMED - the tail moved {R_p99/R_ppl:.2f}x more than the mean "
                          f"(R_p99 {R_p99:.2f} vs R_ppl {R_ppl:.2f}, threshold 1.5x). Tail KL "
                          f"carries information perplexity does not, and it is free - llama.cpp "
                          f"already computes it on the passes we were already running.")
    else:
        res["verdict"] = (f"P1 MISSED - tail/mean {R_p99/R_ppl:.2f}x is below the staked 1.5x. "
                          f"The tail tracks the mean on this pair, so tail KL adds nothing here "
                          f"and MUST NOT ship on this evidence. Published as a miss.")
    print("\n" + res["verdict"])
    json.dump(res, open(OUT, "w"), indent=1)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
