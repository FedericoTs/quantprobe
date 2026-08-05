"""Generate docs/HARDWARE.md + docs/hardware.json from quantprobe.detect's tables.

  python weights/make_hardware_table.py

Single source of truth is THE CODE (detect.GPU_TABLE / detect.MAC_BW) - this script renders
it; tests/smoke.py asserts the rendered doc matches the code, so they cannot drift. Every row
carries a status: 'measured' (this project's reference box), 'external' (a contributor's
scored datapoint, cited), or 'spec' (spec-sheet peak, unvalidated). The table doubles as the
validation Atlas: a `bench --contribute` submission on a spec row is what upgrades it.
"""
from __future__ import annotations
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from quantprobe.detect import GPU_TABLE, MAC_BW  # noqa: E402

DOCS = os.path.join(os.path.dirname(HERE), "docs")

# Validation ledger: which rows the law has actually touched, with the receipt.
VALIDATED = {
    "1060": ("measured", "reference box - eta measured across the full ladder (FINDINGS)"),
    "rx 5700 xt": ("external",
                   "[issue #1 / E-13](https://github.com/FedericoTs/quantprobe/issues/1): "
                   "predicted 73.1 vs measured 73.18 +/- 0.16 (+0.1%), Vulkan, Windows 11"),
    "5060 ti": ("external", "E-08: Blackwell-generation replication, +2%..+7.6% inside the "
                            "published band"),
    "3090": ("external", "first external replication (Ryzen 8600G box; source of the "
                         "channel-count rule)"),
}


def rows():
    out = []
    for frag, bw, geta, gl in GPU_TABLE:
        vendor = ("AMD" if frag.startswith(("rx", "vega", "radeon"))
                  else "Intel" if frag.startswith("arc") else "NVIDIA")
        st, ev = VALIDATED.get(frag, ("spec", ""))
        out.append(dict(match=frag, vendor=vendor, vram_bw_gbs=bw, geta_hint=geta,
                        gl_hint=gl, status=st, evidence=ev))
    for chip, bw in sorted(MAC_BW.items(), key=lambda x: -x[1]):
        out.append(dict(match=chip, vendor="Apple", vram_bw_gbs=bw, geta_hint=0.26,
                        gl_hint=0.24, status="spec",
                        evidence="unified memory; estimated eta, unvalidated - bench me"))
    return out


def render(rs):
    counts = {s: sum(1 for r in rs if r["status"] == s) for s in ("measured", "external", "spec")}
    L = [
        "# Hardware table - what the tool knows, and how it knows it",
        "",
        "Generated from `quantprobe/detect.py` by `weights/make_hardware_table.py`; a smoke",
        "test fails if this file drifts from the code. Bandwidths are THEORETICAL spec peaks",
        "(the law's eta absorbs realism - same convention everywhere in this project).",
        "",
        "**Status legend:** `measured` = on this project's reference box, full ladder;",
        "`external` = an independent contributor's scored datapoint (cited);",
        "`spec` = spec-sheet number, no one has validated the law on it yet.",
        "",
        f"Current census: **{counts['measured']} measured / {counts['external']} external / "
        f"{counts['spec']} spec-only.** Every `quantprobe bench --contribute` run on a "
        "spec-only card is a chance to move a row up - the most valuable submissions are the "
        "ones that land OUTSIDE the predicted band.",
        "",
        "| card (name match) | vendor | VRAM BW (GB/s, spec) | status | evidence |",
        "|---|---|---|---|---|",
    ]
    order = {"measured": 0, "external": 1, "spec": 2}
    for r in sorted(rs, key=lambda r: (order[r["status"]], r["vendor"], -r["vram_bw_gbs"])):
        L.append(f"| {r['match']} | {r['vendor']} | {r['vram_bw_gbs']} | {r['status']} | "
                 f"{r['evidence']} |")
    L += [
        "",
        "Missing your card? `quantprobe hw` will name it if the driver registry sees it; "
        "pass `--vram-bw` from its spec sheet, run `quantprobe calibrate`, then "
        "`bench --contribute` - that is exactly how the first AMD row above got its receipt.",
        "",
    ]
    return "\n".join(L)


def main():
    rs = rows()
    os.makedirs(DOCS, exist_ok=True)
    md = render(rs)
    # HARDWARE_TABLE.md, not HARDWARE.md - that name is the reference-box spec doc (README
    # links it), and the first version of this script overwrote it. Distinct docs, distinct names.
    with open(os.path.join(DOCS, "HARDWARE_TABLE.md"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(md)
    with open(os.path.join(DOCS, "hardware.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(dict(_schema="rendered from quantprobe/detect.py - do not hand-edit",
                       rows=rs), fh, indent=1)
    print(f"docs/HARDWARE_TABLE.md + docs/hardware.json: {len(rs)} rows")


if __name__ == "__main__":
    main()
