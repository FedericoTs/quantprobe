"""Cross-model plausibility sweep - every model x machine the public calculator can show.

The five verification layers each check the law against something we chose: tests, anchors, a
real llama.cpp run, our own findings. None of them asks the question a *reader* asks, which is
"do these numbers make sense next to each other?"

That question found a real bug. A user compared two cells on the published simulator and saw a
12B dense predicted SLOWER than a 106B MoE with the same active parameter count. It was right to
be suspicious: the dense activation model priced every parameter as protected, so a dense model's
speed did not respond to quantization at all (pre-registration #17).

So this sweeps the whole surface and asserts RELATIONAL invariants - properties that must hold
between cells, which no single-cell test can see:

  I1  monotone in bits      fewer bits per weight -> never slower, on a bandwidth-bound tier
  I2  monotone in bandwidth faster memory -> never slower, same model and bits
  I3  fewer bytes wins      at matched bits/machine/tier, reading fewer active bytes must not
                            be slower (dominance on both axes where a cache is involved)
  I4  no free lunch         on SINGLE-TIER rows only, the speed gap may not exceed the byte gap
  I5  sane absolute range   nothing exceeds what the tier's raw bandwidth allows at eta = 1.0
  I6  bytes per parameter   bytes-per-ACTIVE-PARAMETER must rise with the share the recipe holds
                            at >=4.5 bits. THIS is the one that catches the reported defect, and
                            the only one that could: I1-I5 compare the law's outputs against each
                            other, so they are blind to an error in how those outputs are
                            computed. I6 reaches outside the law, to declared parameter counts.

Every invariant here was wrong at least once before it was right, and each wrong version flagged
dozens of perfectly sane cells. The binding quantity is tier-dependent (active bytes, not total
size), eta is architecture-dependent (0.38 MoE vs 0.62 dense on CPU), and multi-tier rows have a
second free variable (the split, or the cache hit rate). A checker that cries wolf is worse than
none, so each restriction below is there because a mutation test proved it necessary - and the
mutation test also caught a version of I3 that had been narrowed until it could no longer see the
bug it was written for.

    python weights/plausibility_sweep.py [--verbose]
"""
from __future__ import annotations
import itertools, sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from quantprobe.plan import evaluate, MODELS, MACHINES, DENSE_PROTECTED_SHARE

BITS = [2.0, 2.5, 3.0, 4.5]
SKIP_MACHINES = set()


def top(model_key, machine_key, bits):
    """Winning row for a cell, or None if nothing is feasible."""
    m, M = MODELS[model_key], MACHINES[machine_key]
    hw = dict(vc=M["vc"], vb=M["vb"], rc=M["rc"], rb=M["rb"], db=M["db"],
              geta=M["geta"], gl=M["gl"])
    try:
        _sz, act, cfgs = evaluate(m["t"], m["a"], m["ne"], m["moe"], bits,
                                n_layer=m.get("nl"), **hw)
    except Exception as e:
        return None
    if not cfgs:
        return None
    return dict(tps=cfgs[0][1], place=cfgs[0][0], act=act, size=_sz,
                moe=m["moe"], a=m["a"])


def main():
    verbose = "--verbose" in sys.argv
    machines = [k for k in MACHINES if k not in SKIP_MACHINES]
    cells = 0
    bad = []

    # I1: monotone in bits (more aggressive quantization is never slower)
    for mk, hk in itertools.product(MODELS, machines):
        rows = [(b, top(mk, hk, b)) for b in BITS]
        rows = [(b, r) for b, r in rows if r]
        for (b1, r1), (b2, r2) in zip(rows, rows[1:]):
            cells += 1
            if r1["place"] != r2["place"]:
                continue                    # placement changed; the comparison is not like-for-like
            if r2["tps"] > r1["tps"] * 1.001:
                bad.append(f"I1 {mk} on {hk}: {b1} bits -> {r1['tps']:.1f} but "
                           f"{b2} bits -> {r2['tps']:.1f} (higher bits should not be faster)")

    # I2: monotone in memory bandwidth, comparing machines of the same shape
    pairs = [("2016", "2016-xmp"), ("rtx-3060", "rtx-4090"), ("mac-m2-max", "mac-m4-max")]
    for lo, hi in pairs:
        for mk, b in itertools.product(MODELS, BITS):
            a, c = top(mk, lo, b), top(mk, hi, b)
            if not a or not c or a["place"] != c["place"]:
                continue
            cells += 1
            if c["tps"] < a["tps"] * 0.999:
                bad.append(f"I2 {mk} @{b}: {hi} ({c['tps']:.1f}) slower than {lo} ({a['tps']:.1f})")

    # I3 + I4: at matched bits and machine, fewer ACTIVE parameters must not be slower, and the
    # advantage must not exceed the byte ratio. This is the invariant the user's report broke.
    for hk, b in itertools.product(machines, BITS):
        rows = {mk: top(mk, hk, b) for mk in MODELS}
        rows = {k: v for k, v in rows.items() if v}
        for k1, k2 in itertools.combinations(rows, 2):
            r1, r2 = rows[k1], rows[k2]
            if r1["place"] != r2["place"]:
                continue                    # different tiers are not comparable
            # eta is architecture-dependent: the law applies a MEASURED 0.38 for MoE vs 0.62
            # for dense on a CPU tier, so a MoE reading fewer bytes can legitimately be slower
            # (mistral-7b beats glm-air 2.75x on 1.68x fewer bytes: 1.68 x 0.62/0.38 = 2.74,
            # exactly). Comparing across architectures therefore needs the eta ratio, which makes
            # the check restate the law. Within an architecture class eta cancels and bytes alone
            # decide - so compare like with like, and let I1/I2/I5 cover the rest.
            # Cross-architecture comparison is valid exactly where eta does NOT depend on
            # architecture. On "all in VRAM" both dense and MoE use geta, so bytes alone decide
            # and a dense model may be compared with a MoE directly - which is the comparison a
            # reader actually makes, and the one that exposed the dense-activation bug (a 12B
            # dense shown slower than a 106B MoE with the same active parameters, on a DGX
            # Spark). Everywhere else the law applies a measured per-architecture eta (0.38 MoE
            # vs 0.62 dense on a CPU tier), so a MoE reading fewer bytes can legitimately be
            # slower and the comparison needs the eta ratio - which would just restate the law.
            # Restricting this to same-architecture EVERYWHERE was tried, and made the sweep
            # blind to the very bug it was written for; the mutation test caught that.
            if r1["moe"] != r2["moe"] and r1["place"] != "all in VRAM":
                continue
            cells += 1
            # ACTIVE bytes per token is the binding quantity on every tier - that is the whole
            # content of Law 4, and it is why a 744B MoE streams faster than a dense 70B despite
            # being 10x larger on disk: the MoE reads only its active slice per token, the dense
            # model reads all of itself. Two earlier versions of this check used total size for
            # disk rows and flagged exactly that result as implausible, which would have been the
            # checker contradicting the project's central finding.
            lean, fat = (r1, r2) if r1["act"] < r2["act"] else (r2, r1)
            lk, fk = (k1, k2) if r1["act"] < r2["act"] else (k2, k1)
            streaming = "disk" in lean["place"].lower()
            # On a memory-resident row, fewer active bytes is decisive. On a CACHED STREAMING row
            # it is not: the cache hit rate is a second free variable, so a model with marginally
            # fewer active bytes can still lose if it is far larger overall and caches worse
            # (kimi-k2.6 reads slightly less per token than glm-744b but is 1058B against 753B).
            # There the honest test is DOMINANCE - smaller on BOTH axes must not be slower.
            dominates = lean["act"] < fat["act"] and (not streaming or lean["size"] <= fat["size"])
            if dominates and lean["tps"] < fat["tps"] * 0.999:
                bad.append(f"I3 {hk} @{b} [{lean['place'][:28]}]: {lk} reads {lean['act']:.2f} "
                           f"GB/token and runs {lean['tps']:.2f}, but {fk} reads {fat['act']:.2f} "
                           f"and runs {fat['tps']:.2f} - fewer active bytes must not be slower")
            # I4 (the speed gap may not exceed the byte gap) applies ONLY to memory-resident rows.
            # A streaming row has a SECOND free variable - what fraction of the model fits the
            # VRAM+RAM cache - so two models can differ in speed by more than their byte ratio
            # entirely legitimately. Asserting a single-variable bound on a two-variable row is
            # not a weaker check, it is a wrong one.
            # SINGLE-TIER rows only. "all in VRAM" and "pure CPU" apply one bandwidth to every
            # byte, so the speed ratio cannot exceed the byte ratio. Hybrid and split rows are
            # two-tier weighted sums, so the SPLIT is a second free variable and a model can beat
            # another by more than its byte ratio purely by putting a larger share on the fast
            # tier (gpt-oss-120b holds 35% of its active bytes in VRAM against glm-air's 22.5%,
            # and wins 2.76x on a 2.09x byte advantage - correct, not a defect).
            single_tier = lean["place"] in ("all in VRAM", "pure CPU (GPU idle)")
            if single_tier:
                ratio = lean["tps"] / fat["tps"] if fat["tps"] else 0
                byte_ratio = fat["act"] / lean["act"] if lean["act"] else 0
                if byte_ratio and ratio > byte_ratio * 1.02:
                    bad.append(f"I4 {hk} @{b} [{lean['place'][:28]}]: {lk} beats {fk} by "
                               f"{ratio:.2f}x but only reads {byte_ratio:.2f}x fewer active bytes")

    # I6: bytes per ACTIVE PARAMETER must be monotone in the protected fraction.
    #
    # This is the one that matters, and the first five could not have caught the bug that
    # prompted them. I1-I5 all compare the law's own outputs against each other, so they cannot
    # see an error in how those outputs are COMPUTED - the law was perfectly self-consistent
    # while feeding on a wrong byte count. This check reaches outside the law, to the model's
    # declared parameter counts.
    #
    # At a fixed bit-width, active bytes should be active PARAMETERS times bits/8, adjusted only
    # by the share the recipe holds at >=4.5 bits. So sorting models by that share must sort them
    # by bytes-per-active-parameter too. When dense models were priced as 100% protected while
    # declaring 21.4%, they leapt above every MoE - including deepseek-16b at 54.2% - and the
    # ordering inverted. That is exactly the shape of the reported defect.
    for hk, b in itertools.product(machines, BITS):
        rows = []
        for mk, m in MODELS.items():
            r = top(mk, hk, b)
            if not r:
                continue
            share = (m["ne"] / m["a"]) if m["moe"] else DENSE_PROTECTED_SHARE
            rows.append((share, r["act"] / m["a"], mk))
        rows.sort()
        for (s1, v1, k1), (s2, v2, k2) in zip(rows, rows[1:]):
            cells += 1
            if v2 < v1 * 0.98:
                bad.append(f"I6 {hk} @{b}: {k1} declares {s1:.1%} protected and reads "
                           f"{v1:.3f} GB per B-active, but {k2} declares MORE ({s2:.1%}) and reads "
                           f"LESS ({v2:.3f}) - bytes/active-param must rise with the protected share")

    # I5: nothing may exceed the tier's raw bandwidth at eta = 1.0
    for mk, hk, b in itertools.product(MODELS, machines, BITS):
        r = top(mk, hk, b)
        if not r:
            continue
        cells += 1
        M = MACHINES[hk]
        ceiling = max(M["vb"], M["rb"]) / max(r["act"], 1e-9)
        if r["tps"] > ceiling * 1.001:
            bad.append(f"I5 {mk} on {hk} @{b}: {r['tps']:.1f} exceeds the {ceiling:.1f} "
                       f"physical ceiling at eta=1.0")

    print(f"\n  swept {len(MODELS)} models x {len(machines)} machines x {len(BITS)} bit-widths")
    print(f"  {cells} relational comparisons checked (I1 bits, I2 bandwidth, I3 active-params, "
          f"I4 byte-ratio, I5 physical ceiling)")
    if bad:
        print(f"\n  {len(bad)} IMPLAUSIBLE:")
        for b in bad[:40]:
            print("    " + b)
        if len(bad) > 40:
            print(f"    ... and {len(bad) - 40} more")
        sys.exit(1)
    print("  no cell contradicts another - the surface is internally coherent")

    if verbose:
        print("\n  headline cell per machine (2.5-bit):")
        for hk in machines:
            best = max(((mk, top(mk, hk, 2.5)) for mk in MODELS),
                       key=lambda x: x[1]["tps"] if x[1] else 0)
            if best[1]:
                print(f"    {hk:<14} {best[0]:<14} {best[1]['tps']:>7.1f} tok/s  {best[1]['place'][:38]}")


if __name__ == "__main__":
    main()
