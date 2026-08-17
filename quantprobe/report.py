"""quantprobe report - the one-page forwardable Markdown artifact (docs/DESIGN_REPORT_CMD.md).

The reader of this feature is NEW: an IT manager deciding a hardware buy, a consultant's
client, an ISV writing hardware requirements. They will not run the tool; they will read a
file someone forwarded. Everything here follows from that one fact - the report must carry
its own provenance, its own error bars and its own scope lines, because the person who could
explain them is not in the room.

Two invariants, inherited from the repo's discipline:
  1. No second physics. Every number is produced by the same functions plan already runs:
     this module is a RENDERER over plan.build_rows() plus the same pure advice functions,
     exactly as binding_report() is a renderer over binding_constraint(). If report and plan
     could disagree about the same file on the same box, the feature would be wrong.
  2. Every number is labeled, on the line it qualifies, never only in a footnote:
       [measured]   an instrument, benchmark, or file read on record (source named)
       [derived]    exact arithmetic from measured/estimated inputs
       [est]        a fitted band or preset
       UNVALIDATED  stated, and no measurement covers this exact case

Plain-words rule: register IDs (preregs, C-/L-/U- codes) do not appear in the body - the
reader has no FINDINGS.md. Quoted tool text passes through _plain(); the register pointers
live in the final Sources block, which is the one sanctioned place for them.
"""
from __future__ import annotations
import os
import re
import textwrap
import time

from . import __version__
from . import plan as planmod
from .plan import _pct

COL = 74          # label column: labels right-align here, matching the design mock's layout

# ------------------------------------------------------------------------------------------
# The two mandatory misread-prevention blocks (design section 5). VERBATIM - the smoke suite
# (t_report_template_guards_and_bench_refusal) asserts M1 as a contiguous block and M2 line
# by line (the scope renderer prefixes its lines with "- "/"  ") in every generated report,
# so do not reword either string without updating that test.
M1 = ("Every speed in this report marked PREDICTED was computed from this machine's memory\n"
      "bandwidths - it was not run. Predictions of this kind have a stated error band of\n"
      "+/-25% (printed next to each number), validated at 8.4% median error on the reference\n"
      "machine ladder.")
# Appended to M1 only when nothing measured exists (no bench rows, no anchor ratios): absence
# of measurement is stated, never implied.
M1_TAIL = ("No number in this report was measured on\n"
           "this machine. One command replaces the predictions with measurements:\n"
           "`quantprobe bench --gguf {file}`.")
M2 = ("Every number here is ONE user's speed. Aggregate throughput under concurrent users is a\n"
      "different quantity, and on the one machine where we measured it, concurrency INVERTED\n"
      "the right choice of model: a dense 7B in VRAM served 219 tokens/s aggregate at 32\n"
      "streams while a RAM-offloaded 30B MoE capped near 40 - yet the MoE wins at 1 stream.\n"
      "Do not size a multi-user service from this page; measured multi-user notes, where they\n"
      "exist for this placement, appear above (one Pascal-class box - treat as indicative, not\n"
      "a prediction for your hardware).")

LABELS_KEY = ("Labels: [measured] instrument/benchmark on record - [derived] arithmetic from "
              "measured inputs - [est] fitted band or preset - UNVALIDATED: stated, no "
              "measurement covers it.")

# design section 1: the bench row must be about THIS model before a ratio is quoted -
# "quoting a ratio between two different models is worse than no ratio".
BENCH_PARAM_TOL = 0.02

LEVER_NAME = {"vram_bw": "faster GPU memory bus", "io": "faster storage / NVMe",
              "cpu_compute": "more/faster CPU threads",
              "ram_bw": "faster RAM / XMP / more channels"}


# ------------------------------------------------------------------------------------------
# plain-words scrubber for QUOTED tool text (row warnings, recipe notes, calibration warning).
# It removes register citations, never facts: "(U-23, reported by a user...)" keeps the user
# report and loses the code. The report's own prose is written clean and never needs this.
_RE_PREREG_PAREN = re.compile(r"\s*\((?:pre-?reg(?:istration)?s?)[^)]*\)", re.IGNORECASE)
_RE_CODE_PAREN = re.compile(
    r"\s*\((?:[A-Z]{1,2}-\d+[a-z]?|#\d+)(?:\s*[,/;]\s*(?:[A-Z]{1,2}-\d+[a-z]?|#\d+))*\)")
_RE_CODE_LEAD = re.compile(
    r"\(((?:[A-Z]{1,2}-\d+[a-z]?|#\d+)(?:\s*[,/]\s*(?:[A-Z]{1,2}-\d+[a-z]?|#\d+))*),\s+")
_RE_PREREG_INLINE = re.compile(
    r"pre-?reg(?:istration)?s?\s*#?\d+[a-z]?(?:'s)?(?:\s*(?:and|/|,)\s*#?\d+[a-z]?)*",
    re.IGNORECASE)
_RE_CODE_INLINE = re.compile(r"\b[A-Z]{1,2}-\d+[a-z]?\b(?:'s)?\s*")


def _plain(text):
    if not text:
        return text
    t = _RE_PREREG_PAREN.sub("", text)
    t = _RE_CODE_PAREN.sub("", t)
    t = _RE_CODE_LEAD.sub("(", t)                      # "(U-23, reported by" -> "(reported by"
    t = _RE_PREREG_INLINE.sub("our measurement register", t)
    t = _RE_CODE_INLINE.sub("", t)
    t = re.sub(r"\(\s*\)", "", t)
    t = re.sub(r"  +", " ", t)
    return t.strip()


def _tps(x):
    """tok/s formatting: sub-1 speeds keep two decimals so 0.13 does not print as 0.1."""
    return f"{x:.2f}" if x < 0.95 else f"{x:.1f}"


def _lab(line, label, col=COL):
    if not label:
        return line
    return line + " " * max(2, col - len(line) - len(label)) + label


# Never break inside words or at hyphens: a wrapped `--spec-ngram-simple-size-m` that splits
# at a hyphen is a corrupted command the reader will paste. Long tokens overflow their line
# instead - ugly beats wrong.
_WRAP = dict(break_long_words=False, break_on_hyphens=False)


def _kv(key, text, label=None, keyw=13, col=COL):
    """Aligned key/value block, label right-aligned on the LAST line (the mock's layout)."""
    lines = textwrap.wrap(text, width=col - keyw - 2, **_WRAP) or [""]
    out = [(key if i == 0 else "").ljust(keyw) + ln for i, ln in enumerate(lines)]
    if label:
        out[-1] = _lab(out[-1], label, col)
    return out


def _para(text, width=COL + 6):
    return textwrap.wrap(text, width=width, **_WRAP)


# ------------------------------------------------------------------------------------------
# bench-log ingestion (design 3.6). Same llama-bench markdown-row shape calibrate._bench_anchor
# parses; report NEVER runs llama-bench itself (plan's contract: takes a second, needs only
# Python) - --bench-log ingests a run the user already made.
def _parse_bench_log(path, quiet=False):
    try:
        text = open(path, encoding="utf-8", errors="replace").read()
    except OSError as e:
        raise SystemExit(f"--bench-log {path}: cannot read it ({e})")
    out = dict(rows=[], build=None, size_gib=None, params_b=None, path=path)
    for line in text.splitlines():
        b = re.search(r"\bbuild:\s*([0-9a-fA-F]{6,40})\b", line)
        if b:
            out["build"] = b.group(1)
        if "|" not in line:
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 3:
            continue
        test = next((c for c in cells
                     if re.fullmatch(r"(?:pp|tg)\d+(?:\s*@?\s*d\d+)?", c)), None)
        if not test:
            continue
        # t/s is the last cell; its +/- may arrive as ASCII, as U+00B1, or as U+FFFD when the
        # log's encoding was guessed wrong. All three shapes are accepted (the \u escapes are
        # resolved by the re module itself, keeping this source file plain ASCII).
        m = re.match(r"([0-9]+(?:\.[0-9]+)?)\s*(?:\+/-|\u00b1|\ufffd+)\s*"
                     r"([0-9]+(?:\.[0-9]+)?)", cells[-1])
        if not m:
            continue
        for c in cells:
            pm = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)\s*B", c)
            if pm:
                out["params_b"] = float(pm.group(1))
            sm = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)\s*GiB", c)
            if sm:
                out["size_gib"] = float(sm.group(1))
        out["rows"].append(dict(test=test, tok_s=float(m.group(1)), err=float(m.group(2))))
    if not out["rows"] and not quiet:
        print(f"[quantprobe] --bench-log {path}: no llama-bench result rows found - "
              "the Measured check section will say so instead of inventing one")
    return out


def _bench_context(bench, px):
    """Everything the Verdict and Measured-check sections need, decided ONCE so the two
    sections cannot disagree: decode row, ratio vs the prediction, in/out of band, and the
    param-count guard that refuses a ratio between two different models."""
    if not bench or not bench["rows"]:
        return None
    t = px["inputs"]["t"]
    pb = bench["params_b"]
    mismatch = None
    if pb is None:
        mismatch = ("the log carries no parseable params column, so it cannot be verified "
                    "as THIS model - no predicted-vs-measured ratio is quoted")
    elif t and abs(pb - t) / t > BENCH_PARAM_TOL:
        mismatch = (f"the log reports {pb:g}B params where this file is {t:.1f}B - a "
                    f"different model, so no predicted-vs-measured ratio is quoted (a ratio "
                    f"between two different models is worse than no ratio)")
    decode = next((r for r in bench["rows"] if r["test"].startswith("tg")), None)
    ctx = dict(bench=bench, mismatch=mismatch, decode=decode, ratio=None, inband=None)
    if decode and not mismatch and px["best"][1] > 0:
        ratio = decode["tok_s"] / px["best"][1]
        ctx["ratio"] = ratio
        ctx["inband"] = 0.75 <= ratio <= 1.25         # the +/-25% band plan itself states
    return ctx


def _nothing_measured(px, bctx):
    has_rows = bool(bctx and bctx["bench"]["rows"])
    has_anchor = bool(px["hw"]["gpu_ratio"] or px["hw"]["cpu_ratio"])
    return not (has_rows or has_anchor)


# ------------------------------------------------------------------------------------------
# section builders. Each returns a list of lines.
def _verdict(px, bctx, gguf_base):
    best = px["best"]
    tps = best[1]
    L = []
    if best[0] == "all in VRAM":
        # No point prediction exists for this regime (fits_in_vram_advice): a one-sided,
        # falsifiable floor is what we can state - 13 of 13 measured runs at or above 0.90x.
        L.append(_lab(f"{'PREDICTED decode speed, one user:':<36}{_tps(tps)} tok/s - a FLOOR",
                      "[measured] one-sided"))
        L += _para("Real speed measured >= 0.90x this number in 13 of 13 benchmarks, and in "
                   "12 of 13 it was HIGHER, typically 1.1x-1.8x. No two-sided band is honest "
                   "here: measured efficiency spreads 0.32-0.56 across models on this "
                   "placement, so read the number as a floor, not a point.")
    else:
        L.append(_lab(f"{'PREDICTED decode speed, one user:':<36}{_tps(tps)} tok/s   "
                      f"(band {_tps(tps * 0.75)} - {_tps(tps * 1.25)})", "[derived]"))
    if bctx and bctx["decode"] and not bctx["mismatch"]:
        d = bctx["decode"]
        L.append(_lab(f"{'MEASURED on this machine:':<36}{d['tok_s']:g} +/- {d['err']:g} "
                      f"tok/s  (llama-bench)", "[measured]"))
        if bctx["ratio"]:
            side = ", on the side our misses err (low)" if bctx["ratio"] > 1 else ""
            verdict = (f"inside the +/-25% band{side}." if bctx["inband"] else
                       "OUTSIDE the +/-25% band - trust the measurement, not the prediction; "
                       "a datapoint like this is exactly how the bands improve.")
            L += _para(f"The measurement is {bctx['ratio']:.2f}x the prediction - {verdict}")
    elif bctx and bctx["mismatch"]:
        L += _para(f"A bench log was supplied but {bctx['mismatch']} - see Measured check.")
    L.append("")
    L += _para(_fit_sentences(px))
    L.append("")
    L += M1.splitlines()
    if _nothing_measured(px, bctx):
        tail = M1_TAIL.format(file=gguf_base).splitlines()
        L[-1] = L[-1] + " " + tail[0]
        L += tail[1:]
    return L


def _fit_sentences(px):
    """The 'what does this mean' sentence a decision-maker actually reads. The usability
    phrasing is editorial banding of the PREDICTED number printed beside it - no new number."""
    inp, hw, best = px["inputs"], px["hw"], px["best"]
    name, tps, size = best[0], best[1], px["size"]
    vc, rc = hw["vc"], hw["rc"]
    kv_gb = inp["ctx"] * inp["kvp"] / 1e9 if inp["ctx"] > 0 else 0.0
    if tps >= 15:
        use = "comfortable for interactive chat and coding"
    elif tps >= 5:
        use = "usable for interactive work, with patience on long answers"
    elif tps >= 1:
        use = "usable for short answers and background jobs, not for interactive work"
    else:
        use = "usable for batch and overnight jobs only"
    speed = f"At ~{_tps(tps)} tok/s (PREDICTED) this is {use}."
    if name == "all in VRAM":
        return f"Fit: the {size:.1f} GB file fits this machine's {vc:g} GB VRAM whole. {speed}"
    ram_too = f" or its {rc:g} GB RAM" if size + kv_gb > rc else ""
    if name.startswith("hybrid"):
        return (f"Fit: the {size:.1f} GB file does not fit this machine's {vc:g} GB VRAM "
                f"whole{ram_too}. It runs with attention and the KV cache on the GPU and the "
                f"routed experts in system RAM. {speed}")
    if name.startswith("split experts"):
        return (f"Fit: the {size:.1f} GB file does not fit this machine's {vc:g} GB VRAM "
                f"whole{ram_too}. It runs with attention plus as many expert layers as fit on "
                f"the GPU, the remaining experts in system RAM. {speed}")
    m = re.match(r"split: (\d+)/(\d+) layers", name)
    if m:
        return (f"Fit: the {size:.1f} GB file does not fit this machine's {vc:g} GB VRAM "
                f"whole{ram_too}. It runs SPLIT: {m.group(1)} of {m.group(2)} layers on the "
                f"GPU, the rest in system RAM. {speed}")
    if name.startswith("pure CPU"):
        gpu = "the GPU sits idle" if vc > 0 else "this machine has no usable GPU tier"
        return f"Fit: the {size:.1f} GB file runs from system RAM ({rc:g} GB); {gpu}. {speed}"
    return (f"Fit: the {size:.1f} GB file exceeds this machine's memory tiers ({vc:g} GB "
            f"VRAM, {rc:g} GB RAM) and weights stream from disk every token. {speed}")


def _asked(args, px, gguf_base):
    inp = px["inputs"]
    L = []
    L += _kv("file", f"{gguf_base} - {px['size']:.1f} GB on disk", "[measured]")
    hybrid = inp["kv_layers"] and inp["nlay"] and inp["kv_layers"] < inp["nlay"]
    kind = ("MoE" if inp["moe"] else "dense") + (" hybrid (linear attention)" if hybrid else "")
    L += _kv("model", f"{inp['t']:.1f}B params, {inp['a']:.1f}B active/token, {kind}, "
                      f"{inp['bits']:.2f} effective bits/weight - read from the file itself",
             "[measured]")
    kv_txt = f"{inp['kvp'] / 1024:.0f} KB/pos f16"
    if hybrid:
        kv_txt += f" ({inp['kv_layers']} of {inp['nlay']} layers cache KV)"
    if inp["ctx"] > 0:
        kv_txt += (f"; ctx {inp['ctx']}: +{inp['ctx'] * inp['kvp'] / 1e9:.2f} GB KV read "
                   f"per token is priced in")
    else:
        kv_txt += " (ctx 0 in this report: no KV term is priced)"
    L += _kv("KV cache", kv_txt, "[derived]")
    L += _machine_lines(args, px)
    return L


def _machine_lines(args, px):
    """Machine constants + calibration state + anchors, each with the label its SOURCE earns.
    Preset hint tags travel verbatim; a preset report says out loud that the reader's own box
    was never consulted (the forwarded-reader failure mode, design 3.2)."""
    hw = px["hw"]
    hint = px["inputs"]["machine_hint"]
    consts = (f"VRAM {hw['vc']:g} GB @ {hw['vb']:g} GB/s | RAM {hw['rc']:g} GB @ "
              f"{hw['rb']:g} GB/s | disk {hw['db']:g} GB/s")
    L = []
    preset = getattr(args, "machine", None)
    if preset:
        tag = re.search(r"\[([^\]]+)\]\s*$", hint or "")
        label = f"[{tag.group(1)}]" if tag else "[est]"
        if "unvalidated" in label:
            label = "[est] UNVALIDATED"
        L += _kv("machine", f'"{preset}" preset: {hint or "table entry"}; {consts}', label)
        L += _kv("calibration", "not consulted: a preset machine was named, so these are the "
                                "preset's constants, not the reader's. For YOUR box, run "
                                "without --machine after `quantprobe calibrate`.")
        return L
    if hint is None:
        # explicit --vram/--ram/... flags: the user described a box; nothing here was detected
        L += _kv("machine", f"described by explicit hardware flags; {consts}",
                 "[user-supplied]")
        L += _kv("calibration", "not consulted: explicit flags describe a different box than "
                                "the one generating this report.")
        return L
    L += _kv("machine", f"this machine, auto-detected; {consts}",
             "[measured]" if hw["cal_active"] else "[os]")
    from . import calibrate as calmod
    cal, age = calmod.load()
    if cal and hw["cal_active"]:
        stale = (f"; {age:.0f} days old - re-run `quantprobe calibrate`"
                 if age and age > calmod.STALE_DAYS else "")
        L += _kv("calibration", f"applied, measured {cal.get('date', '?')} on this box, "
                                f"state {cal.get('cal_id', '?')}{stale}", "[measured]")
        for w in calmod.calibration_gap_warning(cal):
            L += _kv("", _plain(w.replace("[quantprobe] ", "")))
    else:
        L += _kv("calibration", "none on file: constants above are OS-reported/auto-detected, "
                                "not measured. `quantprobe calibrate` measures them (~5 min).",
                 "[os]")
    gr, cr = hw["gpu_ratio"], hw["cpu_ratio"]
    if gr or cr:
        parts = [p for p in (f"GPU x{gr:.2f}" if gr else None,
                             f"CPU x{cr:.2f}" if cr else None) if p]
        L += _kv("anchors", "tier ratios from your own calibrate anchor runs: "
                            + ", ".join(parts), "[measured]")
    return L


def _placements(px):
    L = []
    sep_done = False
    for i, row in enumerate(px["rows"]):
        runnable = getattr(row, "runnable", True)
        if not runnable and not sep_done:
            L.append("  ------------- not runnable on stock llama.cpp -------------")
            L += ["  " + ln for ln in _para("rows below need a runtime llama.cpp does not "
                                            "have; each speed is an UPPER BOUND with named "
                                            "unpriced costs - not comparable to the rows "
                                            "above. Shown for capacity context only.",
                                            COL - 2)]
            sep_done = True
        name, tps, warn = row[0], row[1], row[2]
        head = f"  {'RECOMMENDED' if i == 0 else ' ' * 11}  {_tps(tps):>6} tok/s  "
        text = name + (f" - {_plain(warn)}" if warn else "")
        body = textwrap.wrap(text, width=78 - len(head), **_WRAP) or [""]
        lines = [head + body[0]] + [" " * len(head) + b for b in body[1:]]
        lines[-1] = _lab(lines[-1], "[derived]" if runnable else "[derived, upper bound]", 80)
        L += lines
    return L


def _limits(px):
    """Plain-words rendering of binding_constraint + capacity_probe + upgrade_advisor - the
    same data plan prints, with register citations replaced by their facts (design 3.4)."""
    bc, cap, best = px["bc"], px["cap"], px["best"]
    hw = px["hw"]
    if not bc:
        return _para("This placement carries no time decomposition, so nothing is classified "
                     "here. The speed above stands on the law alone.")
    # the staked scope label (variance attribution did not confirm the flag-level mapping;
    # the kill rule ships it at full prominence) - plain words here, the register pointer
    # lives in Sources like every other one
    scope_note = _para("Validation: this classification is derived from the law, not "
                       "confirmed by variance attribution - where we measured which flag "
                       "actually carries decode variance, the top carrier was the placement "
                       "lever, not the flag the mapping expected. The time split itself is "
                       "exact arithmetic and stands; the flag-level reading is the part "
                       "under that label.")
    L = []
    alts = planmod.upgrade_advisor(px["ev"], best[1], hw["rb"])
    if cap:
        gain = max(cap["gain_shave"], cap["gain_lift"])
        head = _para(f"CAPACITY-BOUND ({cap['tier']}): this configuration is "
                     f"{cap['gap_gb']:.1f} GB over the {cap['tier']} boundary, and crossing "
                     f"the boundary is worth {gain:.1f}x.")
        head[-1] = _lab(head[-1], "[derived]")
        L += head
        L += scope_note
        L.append("")
        if cap["gain_shave"] >= planmod.CAP_PROMOTION_MIN:
            L.append(_lab(f"  cross it by   shaving the file (next quant tier down)"
                          f"  -> ~{_tps(cap['shave_tps'])} tok/s", "[derived]"))
            L.append("                (quality cost of that shave: see Quality, below)")
            L.append(_lab(f"           or   fitting {cap['need_gb']:.1f} GB of {cap['tier']}"
                          f"          -> ~{_tps(cap['lift_tps'])} tok/s", "[derived]"))
            if cap["gain_lift"] < 1.05 and bc.get("time_resource"):
                L.append(f"                (little speed gain: "
                         f"{planmod.RESOURCE_LABEL[bc['time_resource']]} then binds instead)")
        else:
            L.append(_lab(f"  cross it by   fitting {cap['need_gb']:.1f} GB of {cap['tier']}"
                          f"  -> ~{_tps(cap['lift_tps'])} tok/s", "[derived]"))
            L += ["                " + ln for ln in
                  _para(f"(shaving the file does not pay here: x{cap['gain_shave']:.1f})",
                        COL - 16)]
        if cap.get("kv_closes"):
            L.append(f"  KV lever      quantize the KV cache (-ctk q8_0 -ctv q8_0): frees")
            L.append(f"                {cap['kv_saved_gb']:.1f} GB of the {cap['kv_gb']:.1f} "
                     f"GB f16 cache and crosses the")
            L.append(_lab(f"                boundary WITHOUT touching the weights"
                          f"  -> ~{_tps(cap['kv_tps'])} tok/s", "[derived]"))
            body = _para("Prefer this at depth over dropping a quant tier. Both halves are "
                         "measured, not assumed: +37% decode at 16k depth vs f16 KV, and "
                         "quality indistinguishable from f16 at the deepest depth f16 even "
                         "runs (PPL ratio 1.00031 +/- 0.0188 against a <=1.02 gate, wikitext, "
                         "reference box) - and f16 KV runs out of VRAM past ~7k on a 6 GB "
                         "card where q8_0 completes. One caveat travels with it: a "
                         "niche-domain report at 100k+ context says judge your own domain.",
                         COL - 16)
            body[-1] = _lab(body[-1], "[measured]", COL - 12)
            L += ["                " + ln for ln in body]
        elif cap.get("kv_gb"):
            body = _para(f"KV lever: quantizing the KV cache (-ctk q8_0 -ctv q8_0) returns "
                         f"{cap['kv_saved_gb']:.1f} GB of the {cap['gap_gb']:.1f} GB gap at a "
                         f"measured-clean quality cost (PPL ratio 1.00031 +/- 0.0188 on "
                         f"wikitext; speed half also measured, +37% decode at 16k depth) - "
                         f"it does not cross the boundary alone here, so pair it with a "
                         f"smaller file shave.", COL - 4)
            body[-1] = _lab(body[-1], "[measured]", COL)
            L += ["  " + ln for ln in body]
        L.append(_lab(f"  until then    {planmod.RESOURCE_LABEL[bc['resource']]} is where the "
                      f"time goes - {_pct(bc['share'])} of every token", "[derived]"))
    else:
        L.append(_lab(f"{bc['klass'].upper()} ({bc['label']}): {_pct(bc['share'])} of every "
                      f"decode token is spent there.", "[derived]"))
        L += scope_note
        if bc.get("margin_x") and bc.get("next_resource"):
            L += ["  " + ln for ln in
                  _para(f"margin: {planmod.RESOURCE_LABEL[bc['resource']]} must get "
                        f"{bc['margin_x']:.2f}x faster before "
                        f"{planmod.RESOURCE_LABEL[bc['next_resource']]} "
                        f"({_pct(bc['next_share'])}) takes over.", COL - 2)]
        else:
            L.append(f"  margin: no second constraint in this model - "
                     f"{planmod.RESOURCE_LABEL[bc['resource']]} is 100% of the modelled "
                     f"token.")
    L.append("")
    if alts:
        L.append(_lab("Upgrades that WOULD move it (the law re-run with one input improved):",
                      "[derived]"))
        for u in alts:
            L.append(f"  {u['name']:<27} -> ~{_tps(u['tps'])} tok/s (x{u['gain']:.2f}, "
                     f"lands on: {u['row']})")
        L.append("")
    helped = {u["resource"] for u in alts}
    wont = []
    for r in ("vram_bw", "io", "cpu_compute", "ram_bw"):
        if r == bc["resource"] or r in helped:
            continue
        sh = bc["shares"].get(r, 0.0)
        # ceiling comes from binding_constraint itself (lever_ceiling), not re-derived here -
        # one Amdahl spelling, exactly like the tok/s numbers all trace to build_rows
        ceil = (bc.get("lever_ceiling") or {}).get(r)
        verdict = ("NO effect (0% of the token)" if sh <= 0 or not ceil
                   else f"capped at {ceil:.2f}x ({_pct(sh)} of the token)")
        wont.append(f"  {LEVER_NAME[r]:<28} {verdict}")
    if "ram_capacity" not in helped:
        extra = ""
        if cap and cap["tier"] == "RAM" and cap["gain_shave"] >= cap["gain_lift"]:
            extra = " (the shave is worth more than the fit)"
        wont.append(f"  {'+16 GB RAM':<28} re-evaluated: does not move this placement{extra}")
    if wont:
        L.append(_lab("Levers that will NOT help here (exact ceilings, not opinions):",
                      "[derived]"))
        L += wont
    srv = _serving_note(best[0])
    if srv:
        L.append("")
        body = _para(srv)
        body[-1] = _lab(body[-1], "[measured, one Pascal-class box]", COL + 8)
        L += body
    depth = _depth_note(px)
    if depth:
        L.append("")
        body = _para(depth)
        body[-1] = _lab(body[-1], "[measured, reference box]", COL + 8)
        L += body
    L.append("")
    L += _para("Scope: DECODE only. Reading long prompts binds differently (compute, not "
               "bandwidth) and is not classified here.")
    return L


def _serving_note(placement):
    """serving_advisory's measured multi-user curves, in plain words. Same two families,
    nothing printed for the untested case - measured numbers only."""
    if not planmod.serving_advisory(placement):
        return None
    if placement == "all in VRAM":
        return ("Multi-user, measured on one Pascal-class box (a 7B Q4 dense model in VRAM): "
                "aggregate decode scaled 23.1 tok/s at 1 stream -> 175.6 at 16 -> 219.4 at "
                "32 (llama-server -np). The jump sits at a width-8-to-9 kernel boundary, so "
                "2-8 concurrent streams are strictly dominated on that card class: serve 1 "
                "stream or >= 9. On newer GPU generations the boundary is unverified.")
    return ("Multi-user, measured on one Pascal-class box: batching does NOT survive this "
            "placement - routed-expert reads from system RAM do not amortise across streams "
            "(aggregate 19.7 -> 40.0 tok/s from 1 to 32 streams, a 2.0x cap, against 9.5x "
            "for a dense model in VRAM on the same box). One user: this row is the right "
            "call. Many users: a dense model that fits VRAM inverts the choice - measured "
            "219 vs 40 aggregate at 32 streams. Prefill is the exception and batches fine "
            "here (45 -> 225 tok/s by 8 streams).")


def _depth_note(px):
    inp = px["inputs"]
    if not planmod.depth_scope_warning(px["best"][0], inp["moe"], inp["ctx"]):
        return None
    return (f"Depth note: this is a dense split at {inp['ctx']} context, where the "
            f"CPU-resident layers run attention over the whole KV cache every token - "
            f"compute on the CPU's threads, not a byte read. The predicted number above "
            f"INCLUDES that cost (1.55 microseconds per position per CPU layer, measured "
            f"across 8-30 CPU layers and 8k-32k depth, validated out of sample; the constant "
            f"is from one 4-thread CPU, so a wider CPU pays less). Measured ways out: raise "
            f"-ngl until attention fits in VRAM; quantize the KV cache (-ctk q8_0 -ctv q8_0: "
            f"+37% decode at 16k depth); or use a MoE model, whose splits keep attention on "
            f"the GPU. Do NOT evict the KV cache to host RAM (-nkvo 1): measured 3.04x WORSE "
            f"at 16k depth for a prefill difference inside the error bar.")


def _command(args, px, gguf_base):
    best, inp, hw = px["best"], px["inputs"], px["hw"]
    # same helper run() calls - if the resident estimate ever drifts, plan and report emit
    # different -ub flags for the same file, which is the disagree-class invariant 1 forbids
    resident = planmod.moe_vram_resident_gb(inp["ne"], inp["bits"], inp["moe"])
    ub = planmod.ubatch_flags(best[0], resident, hw["vc"])
    flags = f"{best[3]} {ub}" if ub else best[3]
    flags, nthreads = planmod.append_threads_flag(flags, best[0])
    L = [f"  llama-server -m {gguf_base} {flags}"]
    if "--threads" in flags:
        L.append("")
        L += ["  " + ln for ln in
              _para(f"--threads {nthreads} = the report-generating machine's logical cores. "
                    f"llama.cpp's own auto-detect may pick physical-only and cost 2x on "
                    f"CPU-bound decode - verify on your build if unsure.", COL - 2)]
    lever = _optional_lever_note(px)
    if lever:
        L.append("")
        body = _para(lever[0])
        body[-1] = _lab(body[-1], lever[1], COL + 8)
        L += body
    return L


def _optional_lever_note(px):
    """One measured speculation/draft lever for this placement family, scoped to what was
    actually measured - or nothing. The numbers are the same cells plan's advice functions
    quote; the reachability bound comes from the row's own decomposition."""
    best, moe = px["best"], px["inputs"]["moe"]
    name = best[0]
    if not moe and "layers->VRAM" in name:
        return ("Optional lever, measured on this placement class (not on this file): a "
                "small same-family draft model (`-md draft.gguf -ngld 0 "
                "--spec-draft-n-max 2`) bought +33% decode on a 14B dense split on the "
                "reference box (5.5 -> 7.4 tok/s, 76% acceptance, novel code), at zero VRAM "
                "cost. Keep the draft length at 2: every measured 3+ landed at or below "
                "baseline. Treat it as an experiment to run, not a promised number.",
                "[measured, other model]")
    if not moe and name == "all in VRAM":
        return ("Optional lever, measured on this placement class (not on this file): on "
                "copy-heavy output (code edits, refactors, quoting) `--spec-type "
                "ngram-simple --spec-ngram-simple-size-m 12 --spec-ngram-simple-size-n 4` "
                "on llama-server measured up to 5.8x single-stream decode on a pre-Ampere "
                "card with byte-identical output; draft lengths 4-7 are strictly dominated "
                "there (never use them), and prose gains nothing at any setting (1.01x). On "
                "newer GPUs the kernel boundary is unverified.", "[measured, other model]")
    if moe and "split experts" in name:
        hx = planmod.spec_headline_x(moe, name)
        reach = planmod.spec_reachable_x(best, hx) if hx else None
        base = ("Optional lever, measured on this placement class: if output REUSES its "
                "context (edits, refactors, RAG quoting), `--spec-type ngram-simple "
                "--spec-ngram-simple-size-m 384 --spec-ngram-simple-size-n 4` on "
                "llama-server measured 4.7x decode at ~3-bit (21.3 -> 98.8 tok/s), no "
                "download. It does nothing on novel generation and cannot fire until the "
                "prompt exceeds 389 tokens.")
        if reach and hx and reach < hx * (1.0 - planmod.SPEC_HEADLINE_TOL):
            base += (f" On THIS row the same drafter is capped at {reach:.2f}x by the row's "
                     f"own time decomposition (CPU attention over the KV cache does not "
                     f"amortize), so the 4.7x is not reachable here.")
        return (base, "[measured, other quant of a same-class model]")
    if moe and "exps=CPU" in (best[3] or ""):
        return ("Speculation note, measured on this placement class: external-draft "
                "speculation does not pay with experts offloaded (+3% ngram, -24% "
                "old-generation multi-token heads - the verify batch unions experts). One "
                "measured exception: models with a trained-in MTP head reached ~93% "
                "acceptance and +11% at `--spec-type draft-mtp --spec-draft-n-max 2`; the "
                "draft length must stay at 2 (4 measured 0.61x).",
                "[measured, other model]")
    return None


def _measured_check(px, bctx, gguf_base, machine_flag):
    hw = px["hw"]
    L = []
    if bctx:
        b = bctx["bench"]
        build = f", build {b['build']}" if b["build"] else ""
        # no blank line before the rows: the [measured] label must sit in the same block as
        # the numbers it covers, not float above a paragraph break
        L.append(_lab(f"llama-bench rows from --bench-log{build}:", "[measured]"))
        for r in b["rows"]:
            line = f"  {r['test']:<8} {r['tok_s']:g} +/- {r['err']:g} tok/s"
            if bctx["decode"] is r and bctx["ratio"]:
                line += (f"    predicted {_tps(px['best'][1])} -> ratio "
                         f"{bctx['ratio']:.2f}, "
                         f"{'IN BAND' if bctx['inband'] else 'OUT OF BAND'}")
            elif r["tok_s"] > 0 and r["err"] > 0.5 * r["tok_s"]:
                line += (f"    error bar is {r['err'] / r['tok_s']:.0%} of the value: "
                         f"quoted for completeness, not usable as a number")
            L.append(line)
        if bctx["mismatch"]:
            L.append("")
            L += _para(f"NO RATIO: {bctx['mismatch']}.")
        if b["size_gib"] and px["size"]:
            gb = b["size_gib"] * 1.073741824
            if abs(gb - px["size"]) / px["size"] < 0.02:
                L.append("")
                L += _para(f"(bench reports the file as {b['size_gib']:.2f} GiB = {gb:.1f} "
                           f"GB - the same file in two units, said out loud so the reader "
                           f"does not see a contradiction.)")
        return L
    if hw["gpu_ratio"] or hw["cpu_ratio"]:
        parts = [p for p in (f"GPU x{hw['gpu_ratio']:.2f}" if hw["gpu_ratio"] else None,
                             f"CPU x{hw['cpu_ratio']:.2f}" if hw["cpu_ratio"] else None) if p]
        body = _para("No llama-bench log was supplied, but this machine's own calibration "
                     "anchors are active: the constants behind every prediction were scaled "
                     "by anchor runs measured on this box (" + ", ".join(parts) + ").")
        body[-1] = _lab(body[-1], "[measured]", COL + 8)
        L += body
        L += _para(f"One command adds a direct check of this file: `quantprobe bench "
                   f"--gguf {gguf_base}{machine_flag}`.")
        return L
    return None                      # nothing measured: the Verdict's M1 tail already says so


def _quality(args, px, rec):
    from . import recipes as recmod
    inp, cap = px["inputs"], px["cap"]
    L = []
    L += _kv("this file", f"{inp['bits']:.2f} bits/weight -> est. quality cost "
                          f"x{px['q']:.2f}. A fitted perplexity band, not a task score.",
             "[est]")
    L.append("")
    if rec:
        p, mm, pr = rec["probe"], rec["model"], rec["provenance"]
        lo, hi = p["fragile_band"]
        base = f", from a {pr['base_quant']} base" if pr.get("base_quant") else ""
        # the raw-log path is a repo pointer, and repo pointers live in Sources (the plain-
        # words rule); _reproduce already prints it there via rec_src
        L += _kv("fragile band", f"measured for THIS architecture ({mm['name']}): layers "
                                 f"{lo}-{hi} break first under low-bit quantization "
                                 f"({p['fragility_ratio']}x the median band). Probed "
                                 f"{pr['measured']} on {pr['eval']}, "
                                 f"{pr['hardware']}{base}; raw log in Sources below.",
                 "[measured]")
        if pr.get("note"):
            L += _kv("", "Scope, from the measurement itself: " + _plain(pr["note"]))
    else:
        n = len(recmod.load_all())
        arch = inp["arch"] or "unknown"
        L += _kv("fragile band", f"no measured recipe for this architecture ({arch}; {n} "
                                 f"model families in the atlas). Nothing is invented in its "
                                 f"place. The probe still works on your model: `quantprobe "
                                 f"probe --gguf <higher-precision source> --eval "
                                 f"wiki.test.raw`.")
    L.append("")
    L += _size_law(inp["t"])
    if cap and cap["gain_shave"] >= planmod.CAP_PROMOTION_MIN and cap.get("fit_scale"):
        L.append("")
        L += _shave_price(px, rec)
    L.append("")
    L += _para("Capability numbers above were measured on other models (Qwen3.5-35B/4B, "
               "full MATH-500/GSM8K/IFEval, temp 0). They scope the method; they do not "
               "score this file. None of the above is a test of your workload. quantprobe "
               "cannot yet run your tasks (planned - see Scope).")
    return L


def _size_law(t):
    """The size-dependence law of docs/QUANT_QUALITY.md, applied to THIS model's size, with
    its measured anchors and its honest gap - the between-sizes case says UNVALIDATED in
    exactly those words (design section 4)."""
    if t >= 30:
        lines = _kv("size law", f"at {t:.0f}B this model is in the size class where a "
                                f"measured recipe plus aggressive 2-bit PAYS: on a 35B, "
                                f"protected 2-bit held MATH-500 at 81.0 vs 57.0 for the "
                                f"naive 2-bit quant.", "[measured, Qwen3.5-35B]")
        if t >= 45:
            lines += _kv("", "Above the 35B measured point the direction is unmeasured - "
                             "stated, not assumed.", "UNVALIDATED")
        return lines
    if t <= 8:
        return _kv("size law", f"at {t:.1f}B this is a SMALL model, and 2-bit is a false "
                               f"economy even with the recipe: 50.2 vs 81.0 BF16 on "
                               f"MATH-500 (naive 2-bit collapses to 2.6). The sensible "
                               f"quant for a small model is Q4, near-lossless there "
                               f"(77.6 vs 81.0).", "[measured, Qwen3.5-4B]")
    return _kv("size law", f"2-bit capability at this size is unmeasured - the measured "
                           f"points are a 4B (fails even with the recipe: 50.2 vs 81.0 on "
                           f"MATH-500) and a 35B (the protected 2-bit holds: 81.0 vs 57.0 "
                           f"naive); this {t:.0f}B sits between them.", "UNVALIDATED")


def _shave_price(px, rec):
    """The one place speed and quality advice meet: the capacity advisor's shave, priced on
    both sides of the trade. Implied bits = bits x fit_scale, fit_scale from capacity_probe
    itself (no re-derived boundary - no second physics)."""
    inp, cap = px["inputs"], px["cap"]
    implied = inp["bits"] * cap["fit_scale"]
    qshave = planmod.qual_of(inp["moe"], implied)
    band = "MoE" if inp["moe"] else "dense"
    L = _kv("the shave", f"crossing the {cap['tier']} boundary needs "
                         f"~{(1 - cap['fit_scale']) * 100:.0f}% off the file: "
                         f"~{implied:.1f} bits/weight. At that tier the fitted quality cost "
                         f"is x{qshave:.2f} ({band} band) - a proxy band, not a task score.",
            "[est]")
    if implied < 3.5:
        key = rec["model"]["key"] if rec else None
        cmd = (f"quantprobe quantize --gguf <higher-precision source> --recipe {key}"
               if key else "quantprobe probe --gguf <higher-precision source> "
                           "--eval wiki.test.raw --apply")
        L += _kv("", f"Below ~3 bits at this size, capability is where the risk lives (see "
                     f"the size law above). If you shave past Q3, build depth-aware with "
                     f"the measured band (`{cmd}`) and treat the result as unproven until "
                     f"scored on your own tasks.", "UNVALIDATED below ~3 bits")
    return L


def _scope(args, px):
    L = []
    for i, ln in enumerate(M2.splitlines()):
        L.append(("- " if i == 0 else "  ") + ln)
    L.append("- Speeds are decode (writing). Prompt reading is a different regime and is "
             "not")
    L.append("  classified here.")
    L.append("- Every speed marked PREDICTED is a prediction, not a guarantee: the stated "
             "band")
    L.append("  is +/-25%, and our misses are published at the same size as our hits.")
    L.append("- Quality lines are proxies or other-model measurements, labeled as such. The")
    L.append("  deciding test for a business decision is your own workload; a "
             "run-your-own-tasks")
    L.append("  mode is planned, not shipped.")
    if getattr(args, "machine", None):
        L.append("- Machine constants are the named preset's, measured or estimated on the "
                 "box")
        L.append("  that preset describes - not on the reader's machine.")
    else:
        L.append("- Machine constants describe the machine that GENERATED this report - not "
                 "the")
        L.append("  reader's machine, if this file was forwarded.")
    return L


_RECON_FLAGS = (("--model", "model"), ("--machine", "machine"), ("--bits", "bits"),
                ("--total", "total"), ("--active", "active"),
                ("--always-active", "always_active"), ("--vram", "vram"),
                ("--vram-bw", "vram_bw"), ("--ram", "ram"), ("--ram-bw", "ram_bw"),
                ("--disk-bw", "disk_bw"), ("--ctx", "ctx"), ("--kv-per-pos", "kv_per_pos"),
                ("--bench-log", "bench_log"), ("--out", "out"))


def _reconstruct_cmd(snap):
    """The exact command that generated this file, from a snapshot taken BEFORE build_rows
    mutated args (autospec fills bits/total/... in place; echoing those back as if the user
    typed them would misdescribe the invocation)."""
    def fmt(v):
        if isinstance(v, list):
            return ",".join(f"{x:g}" for x in v)
        if isinstance(v, float):
            return f"{v:g}"
        s = str(v)
        return f'"{s}"' if " " in s else s
    parts = ["quantprobe report", f"--gguf {fmt(snap['gguf'])}"]
    for flag, attr in _RECON_FLAGS:
        v = snap.get(attr)
        if v is None or (attr == "ctx" and not v):
            continue
        parts.append(f"{flag} {fmt(v)}")
    if snap.get("no_anchors"):
        parts.append("--no-anchors")
    return " ".join(parts)


def _reproduce(snap, px, rec):
    machine_flag = f" --machine {snap['machine']}" if snap.get("machine") else ""
    L = ["  " + _reconstruct_cmd(snap),
         f"  quantprobe bench --gguf {os.path.basename(snap['gguf'])}{machine_flag}",
         "                    (replaces the predictions above with your measurements)", ""]
    rec_src = f"; fragile-band probe {rec['provenance']['raw_log']}" if rec else ""
    L += _para(f"Sources: Law 4 (tok/s = eta x bandwidth / active bytes); placement engine, "
               f"binding-constraint decomposition and capacity probe: quantprobe "
               f"v{__version__} (plan.build_rows - the same functions `quantprobe plan` "
               f"prints from){rec_src}; quality bands: fitted QUAL table; capability "
               f"tables: docs/QUANT_QUALITY.md. Predictions +/-25%; misses published at the "
               f"same size as hits. Register pointers for readers with the repo checkout: "
               f"pre-registrations #19 #25 #88 #91 #95 #98 #100 #101, findings C-17 L-24 "
               f"U-23 U-38/U-39, FINDINGS.md.")
    return L


# ------------------------------------------------------------------------------------------
def render(args, px, bctx, snap, rec):
    gguf_base = os.path.basename(px["inputs"]["gguf"])
    machine_flag = f" --machine {snap['machine']}" if snap.get("machine") else ""
    where = snap.get("machine") or ("this machine" if px["inputs"]["machine_hint"]
                                    else "the described machine")
    doc = []
    doc.append(f"# quantprobe report - {gguf_base} on {where}")
    doc.append("")
    doc.append(f"Generated {time.strftime('%Y-%m-%d')} by quantprobe v{__version__} "
               f"(report v1).")
    doc += _para(LABELS_KEY)
    doc += ["", "## Verdict", ""]
    doc += _verdict(px, bctx, gguf_base)
    doc += ["", "## What was asked", ""]
    doc += _asked(args, px, gguf_base)
    doc += ["", "## Predicted placements (single user, decode)", ""]
    doc += _placements(px)
    doc += ["", "## What limits this machine", ""]
    doc += _limits(px)
    doc += ["", "## The command", ""]
    doc += _command(args, px, gguf_base)
    mc = _measured_check(px, bctx, gguf_base, machine_flag)
    if mc:
        doc += ["", "## Measured check", ""]
        doc += mc
    doc += ["", "## Quality: what this quant costs", ""]
    doc += _quality(args, px, rec)
    doc += ["", "## Scope - read before forwarding", ""]
    doc += _scope(args, px)
    doc += ["", "## Reproduce", ""]
    doc += _reproduce(snap, px, rec)
    doc.append("")
    text = "\n".join(doc)
    # The file must survive forwarding through anything: plain ASCII, stated policy of the
    # design (section 1). Anything non-ASCII that slipped in via a quoted name degrades
    # to '?' rather than to a mojibake surprise in someone's mail client.
    return text.encode("ascii", errors="replace").decode("ascii")


def run(args):
    g = getattr(args, "gguf", None)
    if not g or not os.path.isfile(g):
        raise SystemExit(f"report needs the file it is about: --gguf {g or '<missing>'} "
                         f"not found.\nA preset-only report is a v2 item "
                         f"(docs/DESIGN_REPORT_CMD.md section 6).")
    # Snapshot what the USER passed before build_rows lets autospec fill the gaps in place -
    # the Reproduce section echoes the invocation, not the resolved state.
    snap = {attr: getattr(args, attr, None) for _f, attr in _RECON_FLAGS}
    snap["gguf"] = g
    snap["no_anchors"] = getattr(args, "no_anchors", False)
    px = planmod.build_rows(args)
    if not hasattr(args, "arch"):
        # spec.apply sets .arch only when it actually read the file; without autospec the
        # report's premise ("the file grounds every number") is gone - refuse, don't guess.
        raise SystemExit("report is grounded in the GGUF, and autospec could not read this "
                         "file (see the [quantprobe] line above). No report from guesses.")
    bench = _parse_bench_log(args.bench_log) if getattr(args, "bench_log", None) else None
    bctx = _bench_context(bench, px)
    from . import recipes as recmod
    rec = (recmod.find(arch=px["inputs"]["arch"], n_layer=px["inputs"]["nlay"])
           if px["inputs"]["arch"] and px["inputs"]["nlay"] else None)
    out = getattr(args, "out", None) or (
        f"quantprobe-report-{os.path.splitext(os.path.basename(g))[0]}.md")
    text = render(args, px, bctx, snap, rec)
    with open(out, "w", encoding="ascii", newline="\n") as f:
        f.write(text)
    path = os.path.abspath(out)

    # The 5-line terminal summary. The FILE is the deliverable; this only points at it.
    best = px["best"]
    if best[0] == "all in VRAM":
        v = f"PREDICTED {_tps(best[1])} tok/s decode FLOOR, one user - {best[0]}"
    else:
        v = (f"PREDICTED {_tps(best[1])} tok/s decode, one user "
             f"(band {_tps(best[1] * 0.75)}-{_tps(best[1] * 1.25)}) - {best[0]}")
    if bctx and bctx["ratio"]:
        mline = (f"measured: llama-bench {bctx['decode']['tok_s']:g} tok/s -> ratio "
                 f"{bctx['ratio']:.2f}, {'IN BAND' if bctx['inband'] else 'OUT OF BAND'}")
    elif bctx and bctx["mismatch"]:
        mline = "measured: bench log refused - " + bctx["mismatch"].split(" - ")[0]
    elif px["hw"]["gpu_ratio"] or px["hw"]["cpu_ratio"]:
        mline = "measured: no bench log; this box's calibration anchors back the constants"
    else:
        mline = "measured: nothing - the report says so and names the one-command fix"
    if rec:
        lo, hi = rec["probe"]["fragile_band"]
        qline = (f"quality: {px['inputs']['bits']:.2f} bits/weight, est. cost "
                 f"x{px['q']:.2f} [est]; fragile band {lo}-{hi} [measured]")
    else:
        qline = (f"quality: {px['inputs']['bits']:.2f} bits/weight, est. cost "
                 f"x{px['q']:.2f} [est]; no measured recipe for this arch")
    print(f"[quantprobe] report written: {path}")
    print(f"  verdict  {v}")
    print(f"  {mline}")
    print(f"  {qline}")
    print("  forward the FILE, not a number from it - its labels, bands and scope travel "
          "inside.")
