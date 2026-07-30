"""Experiment #51 - EXTERNAL RETRODICTION: score our shipped decode law against
BigMoeOnEdge's published desktop DRAM ceiling, using THEIR file, not our sibling.

    python weights/exp51_external_retrodiction.py --stake     # compute + freeze the prediction
    python weights/exp51_external_retrodiction.py             # score it against their numbers

WHAT THIS DOES
  1. Identifies the exact GGUF they benchmarked (Qwen3.6-35B-A3B Q4_K_M, "22.3 GB") by
     enumerating a FROZEN candidate list of public repos and selecting on published size.
     Exactly one candidate may match, or the script refuses to run.
  2. Downloads ONLY the GGUF header (HTTP Range, ~11 MB of a 22.3 GB file) and parses the
     tensor table. The parse is proved correct by an accounting identity: header bytes +
     sum(per-tensor bytes) must equal the remote Content-Length to within one alignment pad.
  3. Recomputes active bytes/token with the same arithmetic quantprobe.spec.from_gguf uses,
     INCLUDING finding #76's embedding-gather correction (token_embd is subtracted from the
     always-active set iff a separate output/lm_head tensor exists - it does here).
     The reimplementation is proved faithful by a self-test against the real
     quantprobe.spec.from_gguf on local GGUF files: any disagreement aborts the run.
  4. Emits tok/s = eta_r * ram_bw / active-bytes with SHIPPED constants only. No fitting.
  5. In --score mode, reads the staked numbers back out of the pre-registration, checks the
     fresh computation still reproduces them, then evaluates the kill rule.

WHAT IT REFUSES TO DO
  Every precondition below aborts with a named message rather than producing a number:
    - `requests` or `gguf` not importable
    - quantprobe not importable (the self-test cannot run)
    - no local GGUF available for the mirror self-test
    - the mirror disagrees with quantprobe.spec.from_gguf on any local file
    - the self-test never exercises a MoE file, or never exercises BOTH sides of the #76
      tied/untied branch (an untested branch is an unverified branch)
    - the shipped constants this script quotes are not the constants quantprobe actually ships
    - our predict_tps() does not reproduce plan.evaluate()'s own `pure CPU (GPU idle)` row
    - ANY candidate in the frozen list cannot be sized (the "exactly one matches" rule is only
      an identification if all seven were actually measured)
    - zero or more than one candidate file matches "22.3 GB"
    - the header fetch fails, or the byte-accounting identity does not close
    - the file's own metadata is not the model we think it is
    - the file carries codebook (IQ) tensors, which the shipped eta corrects for and this
      experiment's staked arithmetic does not
    - --score with a missing prereg, an unparseable stake block, or a stake whose OUR-SIDE
      numbers the fresh computation no longer reproduces
    - --score --offline (the target is extracted from the live source; there is no offline target)
    - the live source cannot be parsed into the ten-cell table the target rule needs

WHAT IT DELIBERATELY DOES *NOT* ABORT ON
  Their published numbers moving. The target is re-extracted from the live document under a rule
  fixed in the pre-registration, and a changed target is SCORED, not refused. Aborting on a moved
  target would close the only door through which the kill rule can still fire - see the
  KILL-RULE REACHABILITY block that every --score run prints.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import struct
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "weights", "data")
PREREG = os.path.join(ROOT, "preregistrations", "2026-07-30-external-retrodiction.md")
DEFAULT_GGUF_DIR = r"D:\evo-compress-data\gguf"

# ---------------------------------------------------------------------------------------
# FROZEN INPUTS. Every constant below is either a SHIPPED quantprobe value (with the line it
# lives on) or a literal transcribed from the external source (with the URL). Nothing here is
# fitted to the outcome; --score re-verifies the transcriptions against the live source.
# ---------------------------------------------------------------------------------------

# `eta_r = 0.38 if moe else 0.62` in quantprobe/plan.py - present since commit d590749
# (2026-07-21, "quantprobe v1.0"), i.e. nine days before U-33 was logged and before this
# experiment existed. It is the only MoE RAM-tier efficiency the tool has ever shipped.
#
# ADVERSARIAL REVIEW 2026-07-30: these four literals used to carry hard-coded LINE NUMBERS as
# their provenance, and nothing checked them. Three of the four line numbers were already stale
# (plan.py moved: eta_r is at 904, not 674). Worse, a copied literal that is never compared to
# its source is exactly a silent-substitution channel: if plan.py shipped 0.30, this script would
# have happily predicted with 0.38 and called it "the shipped value". verify_shipped_constants()
# now re-derives all four from quantprobe at runtime and aborts on any disagreement, and the
# whole prediction is additionally checked against plan.evaluate()'s own row.
ETA_R_MOE = 0.38
# quantprobe/plan.py MACHINES: the DDR4-3200 dual-channel entry used by rtx-3060/rtx-3090/gaming.
# U-33 already cited "our DDR4-3200 preset (51 GB/s)" in findings/REGISTER.json before this run.
RAM_BW_GBS = 51.0
MACHINE_FOR_RAM_BW = "rtx-3060"       # the MACHINES key RAM_BW_GBS is taken from
# the activation multiplier on both byte terms of plan.evaluate.
ACT_MULT = 1.15
# attention floor in plan.evaluate. Inactive here (this file is 5.02 effective bits).
ATTN_BIT_FLOOR = 4.5
# Dual-channel DDR4: 2 channels x 64 bit = 16 B per transfer, so GB/s = MT/s * 0.016 exactly.
# 3200 MT/s -> 51.2 GB/s; the shipped preset rounds that to 51.0.
DDR4_DUAL_GBS_PER_MTS = 0.016

SOURCE_README = "https://raw.githubusercontent.com/Helldez/BigMoeOnEdge/main/README.md"
SOURCE_FINDINGS = ("https://raw.githubusercontent.com/Helldez/BigMoeOnEdge/main/docs/"
                   "bench-data/2026-07-24-desktop-qwen36/findings.md")

# Their published desktop cells, transcribed verbatim from SOURCE_FINDINGS at stake time.
# drop=True marks cells run with --drop-cold-experts, which SKIPS routed experts and therefore
# changes the active-byte count this experiment predicts. Those cells are excluded from the
# target by a rule fixed in the pre-registration BEFORE any comparison.
#
# ADVERSARIAL REVIEW 2026-07-30: this list is now the STAKED TRANSCRIPTION ONLY. The target that
# the kill rule binds on is re-extracted from the LIVE document by extract_live_cells() under the
# same fixed rule. Previously the target came from these literals and any change to their
# document ABORTED the run - which, combined with a 0.5% drift gate on a 15% kill threshold,
# made the FAIL verdict mathematically unreachable. A divergence between this list and the live
# document is now loudly reported and SCORED, not refused.
CELLS_STAKED = [
    dict(id="A", cfg="round1 baseline io4.t4",        tps=4.78, compute=0.116, drop=False),
    dict(id="B", cfg="round1 io8",                    tps=5.61, compute=0.116, drop=False),
    dict(id="C", cfg="round1 drop-cold 0.75",         tps=6.82, compute=0.112, drop=True),
    dict(id="D", cfg="round1 t8",                     tps=6.14, compute=0.107, drop=False),
    dict(id="E", cfg="round2 io4.t4 cache5000",       tps=4.72, compute=0.115, drop=False),
    dict(id="F", cfg="round2 io8.t4 cache5000",       tps=4.63, compute=0.117, drop=False),
    dict(id="G", cfg="round2 io4.t8 cache5000",       tps=4.68, compute=0.113, drop=False),
    dict(id="H", cfg="round2 io8.t8 drop0.75",        tps=6.33, compute=0.107, drop=True),
    dict(id="I", cfg="round3 overlap cache5000",      tps=5.14, compute=0.115, drop=False),
    dict(id="J", cfg="round3 overlap+auto+drop0.75",  tps=7.33, compute=0.107, drop=True),
]
HEADLINE_CEILING_STAKED = 9.0  # their prose: "~0.11 s/token floor ... a ~9 tok/s ceiling"
KILL_REL_ERROR = 0.15          # staked in findings/REGISTER.json U-33.predicted_effect
# ^ this threshold is NOT tuned to the outcome: it was written into U-33's `predicted_effect`
#   field ("Kill rule: |error| must be under 15%") before this experiment was written. It is
#   nonetheless LOOSE - the achieved errors are 2.6% and 0.9% - and §3 of the prereg now states
#   the achieved margin next to the band so a reader can judge the test's discriminating power.

# The target-extraction rule, fixed in the pre-registration, applied to the LIVE document.
# A cell row is `| <id> <config> | <tok/s> | <compute s/tok> | ...`. Ids are single letters A-J;
# the markdown header row ("| Cell | tok/s |") cannot match because `[A-J]\b` needs a word
# boundary and "Cell" has none after the C.
CELL_ROW_RE = re.compile(r"^\|\s*([A-J])\b([^|]*)\|\s*([0-9]*\.?[0-9]+)\s*\|\s*([0-9]*\.?[0-9]+)\s*\|")
HEADLINE_RE = re.compile(r"~\s*([0-9]*\.?[0-9]+)\s*tok/s ceiling")
DROP_MARKER = "drop"           # a cell whose config names --drop-cold-experts is reduced-width
EXPECT_CELL_IDS = list("ABCDEFGHIJ")
MIN_FULL_WIDTH_CELLS = 5       # below this the median rule has no content; abort rather than guess

# Candidate public GGUFs that could be the "Qwen3.6-35B-A3B Q4_K_M (22.3 GB)" they ran. The list
# is frozen here; the selection RULE is: keep every candidate whose published size rounds to
# 22.3 GB at 0.1 GB resolution. Exactly one must survive or the run aborts.
CANDIDATES = [
    ("bartowski/Qwen_Qwen3.6-35B-A3B-GGUF",  "Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf"),
    ("unsloth/Qwen3.6-35B-A3B-GGUF",         "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
    ("unsloth/Qwen3.6-35B-A3B-GGUF",         "Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"),
    ("unsloth/Qwen3.6-35B-A3B-GGUF",         "Qwen3.6-35B-A3B-UD-Q4_K_S.gguf"),
    ("unsloth/Qwen3.6-35B-A3B-MTP-GGUF",     "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
    ("ggml-org/Qwen3.6-35B-A3B-GGUF",        "Qwen3.6-35B-A3B-Q4_K_M.gguf"),
    ("lmstudio-community/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-Q4_K_M.gguf"),
]
TARGET_SIZE_GB = 22.3          # their stated file size, to 0.1 GB
EXPECT = dict(arch="qwen35moe", expert_count=256, expert_used_count=8, block_count=41)


class Abort(Exception):
    """A precondition failed. The script prints why and exits non-zero WITHOUT a number."""


# ---------------------------------------------------------------------------------------
# minimal GGUF header parser (works on a truncated prefix; GGUFReader cannot)
# ---------------------------------------------------------------------------------------
_FIXED = {0: ("<B", 1), 1: ("<b", 1), 2: ("<H", 2), 3: ("<h", 2), 4: ("<I", 4), 5: ("<i", 4),
          6: ("<f", 4), 7: ("<?", 1), 10: ("<Q", 8), 11: ("<q", 8), 12: ("<d", 8)}


class _Buf:
    def __init__(self, b):
        self.b, self.i = b, 0

    def take(self, n):
        if self.i + n > len(self.b):
            raise EOFError(f"header truncated: need {self.i + n} bytes, have {len(self.b)}")
        v = self.b[self.i:self.i + n]
        self.i += n
        return v

    def u32(self):
        return struct.unpack("<I", self.take(4))[0]

    def u64(self):
        return struct.unpack("<Q", self.take(8))[0]

    def s(self):
        return self.take(self.u64()).decode("utf-8", "replace")


def _val(b, t):
    if t in _FIXED:
        fmt, n = _FIXED[t]
        return struct.unpack(fmt, b.take(n))[0]
    if t == 8:
        return b.s()
    if t == 9:                                   # array
        et, n = b.u32(), b.u64()
        if et in _FIXED:
            b.take(_FIXED[et][1] * n)
            return f"<array[{n}] type {et}>"
        for _ in range(n):
            _val(b, et)
        return f"<array[{n}]>"
    raise ValueError(f"unknown gguf value type {t}")


def parse_header(data):
    """-> (version, kv dict, [(name, dims, type_id, offset)], header_byte_length)"""
    b = _Buf(data)
    if b.take(4) != b"GGUF":
        raise Abort("fetched bytes are not a GGUF file (bad magic)")
    ver, ntensor, nkv = b.u32(), b.u64(), b.u64()
    kv = {}
    for _ in range(nkv):
        k = b.s()
        kv[k] = _val(b, b.u32())
    tensors = []
    for _ in range(ntensor):
        name, nd = b.s(), b.u32()
        dims = [b.u64() for _ in range(nd)]
        tensors.append((name, dims, b.u32(), b.u64()))
    return ver, kv, tensors, b.i


def field(kv, *suffixes):
    """Same suffix-match rule quantprobe.spec._field uses (arch prefix is not hard-coded)."""
    for k, v in kv.items():
        for s in suffixes:
            if k.endswith(s):
                try:
                    return int(v)
                except (TypeError, ValueError):
                    pass
    return None


def tensor_bytes(dims, type_id, quant_sizes, quant_enum):
    n = 1
    for d in dims:
        n *= int(d)
    blk, size = quant_sizes[quant_enum(type_id)]
    return n, n // blk * size


# ---------------------------------------------------------------------------------------
# the mirror of quantprobe.spec.from_gguf, over a parsed header instead of a local file
# ---------------------------------------------------------------------------------------
def spec_mirror(kv, tensors, file_bytes, quant_sizes, quant_enum):
    total = routed = embd = 0
    all_bytes = exp_bytes = embd_bytes = codebook_bytes = 0
    has_output = False
    k_class_iq = {"IQ4_NL"}                      # C-13 / prereg #70
    for name, dims, tt, _off in tensors:
        n, nb = tensor_bytes(dims, tt, quant_sizes, quant_enum)
        total += n
        all_bytes += nb
        tname = quant_enum(tt).name
        if "exps" in name or "_expert" in name:
            routed += n
            exp_bytes += nb
        if "token_embd" in name:
            embd += n
            embd_bytes += nb
        if name.startswith("output.") or "lm_head" in name:
            has_output = True
        if tname.startswith("IQ") and tname not in k_class_iq:
            codebook_bytes += nb
    # finding #76 (prereg 2026-07-30-embedding-gather): decode GATHERS one embedding row, so a
    # 248k-row token_embd is not an always-active read - UNLESS embeddings are tied, in which
    # case that same tensor IS the output projection and is fully read every token.
    gather_only = embd if has_output else 0
    ne = total - routed - gather_only
    n_exp, n_used = field(kv, ".expert_count"), field(kv, ".expert_used_count")
    if routed and n_exp and n_used:
        active, moe = ne + routed * n_used / n_exp, True
    else:
        active, moe = total - gather_only, False
    n_layer = field(kv, ".block_count") or 32
    kv_lora = field(kv, ".attention.kv_lora_rank")
    if kv_lora:
        kvp = n_layer * (kv_lora + (field(kv, ".rope.dimension_count") or 64)) * 2
    else:
        kv_heads = field(kv, ".attention.head_count_kv") or 8
        k_dim = field(kv, ".attention.key_length")
        v_dim = field(kv, ".attention.value_length")
        if not k_dim:
            emb = field(kv, ".embedding_length") or 4096
            heads = field(kv, ".attention.head_count") or 32
            k_dim = v_dim = emb // heads
        kvp = n_layer * kv_heads * ((k_dim or 128) + (v_dim or k_dim or 128)) * 2
    arch = kv.get("general.architecture")
    return dict(t=total / 1e9, a=active / 1e9, ne=ne / 1e9, moe=moe,
                bits=round(file_bytes * 8 / total, 2), kvp=int(kvp), n_layer=n_layer, arch=arch,
                codebook_share=(codebook_bytes / all_bytes) if all_bytes else 0.0,
                _total=total, _routed=routed, _embd=embd, _has_output=has_output,
                _n_exp=n_exp, _n_used=n_used, _all_bytes=all_bytes,
                _exp_bytes=exp_bytes, _embd_bytes=embd_bytes)


def active_gb_per_token(s):
    """quantprobe/plan.py:640-669, the weight-byte budget of one decoded token."""
    ab = max(s["bits"], ATTN_BIT_FLOOR)
    prot = s["ne"]                               # MoE: `ne` names the protected set exactly
    act_ne = prot * ab / 8 * ACT_MULT
    act_ex = (s["a"] - prot) * s["bits"] / 8 * ACT_MULT
    return act_ne + act_ex, act_ne, act_ex


def predict_tps(act_gb, eta=ETA_R_MOE, bw=RAM_BW_GBS):
    """Law 4, the zero-I/O limit: no KV term (ctx -> 0), no disk term, no context term.
    This is exactly the weight-byte term of plan.evaluate's pure-CPU row - and
    verify_against_plan() proves that claim by calling plan.evaluate instead of asserting it."""
    return eta * bw / act_gb


def bw_for_tps(tps, act_gb, eta=ETA_R_MOE):
    """Invert predict_tps: the RAM bandwidth at which the law would predict exactly `tps`."""
    return tps * act_gb / eta


# ---------------------------------------------------------------------------------------
# provenance: the constants this script quotes must BE the constants quantprobe ships
# ---------------------------------------------------------------------------------------
def verify_shipped_constants(log):
    """A copied literal with a line number in a comment is not provenance - it is a claim.
    Re-derive all four from the installed quantprobe and abort on any disagreement."""
    import inspect
    from quantprobe import plan as planmod
    src = inspect.getsource(planmod.evaluate).splitlines()

    def find(pattern, what):
        rx = re.compile(pattern)
        hits = [(i, ln.strip()) for i, ln in enumerate(src) if rx.search(ln)]
        if len(hits) != 1:
            raise Abort(f"cannot locate {what} in quantprobe.plan.evaluate ({len(hits)} matches "
                        f"for {pattern!r}). The shipped code moved; this script's claim that it "
                        "uses shipped constants can no longer be checked, so it refuses to run.")
        return hits[0]

    checks = []
    _i, ln = find(r"^\s*eta_r\s*=\s*[0-9.]+\s+if\s+moe\s+else\s+[0-9.]+", "the MoE eta_r")
    shipped_eta = float(re.search(r"=\s*([0-9.]+)\s+if", ln).group(1))
    checks.append(("eta_r (MoE)", ETA_R_MOE, shipped_eta, ln))
    _i, ln = find(r"^\s*ab\s*=\s*max\(bits,\s*[0-9.]+\)", "the attention bit floor")
    checks.append(("attn bit floor", ATTN_BIT_FLOOR,
                   float(re.search(r"max\(bits,\s*([0-9.]+)\)", ln).group(1)), ln))
    _i, ln = find(r"^\s*act_ne\s*=\s*prot\s*\*\s*ab\s*/\s*8\s*\*\s*[0-9.]+", "the act_ne multiplier")
    checks.append(("activation mult (ne)", ACT_MULT,
                   float(re.search(r"/\s*8\s*\*\s*([0-9.]+)", ln).group(1)), ln))
    _i, ln = find(r"^\s*act_ex\s*=\s*\(a\s*-\s*prot\)\s*\*\s*bits\s*/\s*8\s*\*\s*[0-9.]+",
                  "the act_ex multiplier")
    checks.append(("activation mult (exp)", ACT_MULT,
                   float(re.search(r"/\s*8\s*\*\s*([0-9.]+)", ln).group(1)), ln))

    mac = planmod.MACHINES.get(MACHINE_FOR_RAM_BW)
    if not mac:
        raise Abort(f"quantprobe.plan.MACHINES has no {MACHINE_FOR_RAM_BW!r} entry; the "
                    "51 GB/s DDR4-3200 preset this experiment cites no longer exists.")
    checks.append((f"ram_bw (MACHINES[{MACHINE_FOR_RAM_BW}])", RAM_BW_GBS, float(mac["rb"]), "MACHINES"))

    bad = []
    for name, ours, shipped, where in checks:
        ok = abs(ours - shipped) < 1e-12
        log(f"  constant {name:28s} script={ours:<7g} shipped={shipped:<7g} {'OK' if ok else 'MISMATCH'}")
        if not ok:
            bad.append(f"{name}: script={ours} shipped={shipped} ({where})")
    if bad:
        raise Abort("this script's 'shipped constants' are NOT what quantprobe ships: "
                    + "; ".join(bad) + ". Refusing to predict with a constant whose provenance "
                    "is false - that is the exact claim the pre-registration rests on.")
    return [dict(name=n, value=o, shipped=s) for n, o, s, _w in checks]


def verify_against_plan(s, act, pred, true_size_gb, log):
    """Prove predict_tps() IS plan.evaluate()'s `pure CPU (GPU idle)` row at ctx=0, by calling it.

    The mirror self-test proves the SPEC half (bytes). Without this, the PLAN half (bytes ->
    tok/s) was an unverified reimplementation - the same defect, one step downstream. rc is set
    large only so the row is emitted at all (it is gated on the model fitting in RAM); nothing
    about the machine enters the ceiling, which is eta * bw / act by construction."""
    from quantprobe import plan as planmod
    size, act_plan, rows = planmod.evaluate(
        s["t"], s["a"], s["ne"], s["moe"], s["bits"],
        0.5, 100.0, 1024.0, RAM_BW_GBS, 3.5, 0.35,
        ctx=0, kvp=s["kvp"], n_layer=s["n_layer"], true_size_gb=true_size_gb,
        codebook_share=s["codebook_share"])
    row = next((r for r in rows if r[0] == "pure CPU (GPU idle)"), None)
    if row is None:
        raise Abort("plan.evaluate did not emit a `pure CPU (GPU idle)` row, so the shipped "
                    "formula this experiment claims to be using could not be exercised. "
                    "Refusing to score an unverified reimplementation of it.")
    if abs(act_plan - act) > 1e-9:
        raise Abort(f"active bytes disagree with plan.evaluate: ours {act:.9f} GB, shipped "
                    f"{act_plan:.9f} GB. The prediction would not be the tool's prediction.")
    if abs(row[1] - pred) > 1e-9:
        raise Abort(f"predicted tok/s disagrees with plan.evaluate's own pure-CPU row: ours "
                    f"{pred:.9f}, shipped {row[1]:.9f}.")
    log(f"  plan.evaluate    act={act_plan:.6f} GB, pure-CPU row={row[1]:.6f} tok/s "
        f"-> reproduces ours EXACTLY (size {size:.2f} GB)")
    return dict(act_gb=round(act_plan, 6), tok_s=round(row[1], 6))


# ---------------------------------------------------------------------------------------
# the target, extracted LIVE under the rule fixed in the pre-registration
# ---------------------------------------------------------------------------------------
def extract_live_cells(doc):
    """Apply the pre-registered extraction rule to their published document as it stands NOW."""
    seen, cells = set(), []
    for line in doc.splitlines():
        m = CELL_ROW_RE.match(line.strip())
        if not m:
            continue
        cid, cfg = m.group(1), m.group(2).strip()
        if cid in seen:
            raise Abort(f"cell id {cid!r} appears twice in {SOURCE_FINDINGS}; the extraction "
                        "rule cannot decide which row is the cell and refuses to guess.")
        seen.add(cid)
        cells.append(dict(id=cid, cfg=cfg, tps=float(m.group(3)), compute=float(m.group(4)),
                          drop=DROP_MARKER in cfg.lower()))
    ids = sorted(c["id"] for c in cells)
    if ids != EXPECT_CELL_IDS:
        raise Abort(f"the live source no longer yields the ten cells A-J the target rule needs "
                    f"(got {ids}). This aborts rather than silently re-deriving the target from "
                    "a table whose shape changed.")
    if sum(1 for c in cells if not c["drop"]) < MIN_FULL_WIDTH_CELLS:
        raise Abort("fewer than %d full-width cells survive the --drop-cold-experts exclusion; "
                    "the median rule has no content." % MIN_FULL_WIDTH_CELLS)
    return sorted(cells, key=lambda c: c["id"])


def extract_live_headline(doc):
    m = HEADLINE_RE.search(doc)
    if not m:
        raise Abort("the '~N tok/s ceiling' headline is no longer in the published source. P-2 "
                    "scores against THEIR stated ceiling; it cannot be scored against our memory "
                    "of it, and it is not re-used from the stake block for exactly that reason.")
    return float(m.group(1))


def target_from(cells):
    """The pre-registered statistic: MEDIAN compute s/token over the full-width cells."""
    full = sorted(c["compute"] for c in cells if not c["drop"])
    med = full[len(full) // 2] if len(full) % 2 else (full[len(full) // 2 - 1]
                                                      + full[len(full) // 2]) / 2
    return 1.0 / med, med, full, max(c["tps"] for c in cells)


# ---------------------------------------------------------------------------------------
# preconditions
# ---------------------------------------------------------------------------------------
def need_imports():
    try:
        import requests                                        # noqa: F401
    except ImportError:
        raise Abort("`requests` is not installed. python -m pip install requests")
    try:
        from gguf import GGML_QUANT_SIZES, GGMLQuantizationType
    except ImportError:
        raise Abort("`gguf` is not installed - per-tensor byte sizes cannot be computed. "
                    "python -m pip install gguf")
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    try:
        from quantprobe import spec                            # noqa: F401
    except ImportError as e:
        raise Abort(f"quantprobe is not importable from {ROOT} ({e}). The mirror self-test "
                    "cannot run, and an unverified reimplementation is exactly the failure "
                    "mode this experiment exists to avoid.")
    return GGML_QUANT_SIZES, GGMLQuantizationType


MOE_HINT = re.compile(r"A3B|A4B|moe|Lite|30B|35B", re.I)


def selftest_mirror(gguf_dir, quant_sizes, quant_enum, log):
    """Prove spec_mirror() reproduces quantprobe.spec.from_gguf EXACTLY on real local files.
    Without this, every number below is an unverified reimplementation."""
    from quantprobe import spec
    if not os.path.isdir(gguf_dir):
        raise Abort(f"self-test needs local GGUFs; {gguf_dir} does not exist. "
                    "Pass --gguf-dir DIR pointing at a folder with at least one .gguf.")
    files = [os.path.join(gguf_dir, f) for f in os.listdir(gguf_dir) if f.endswith(".gguf")]
    if not files:
        raise Abort(f"self-test needs local GGUFs; no .gguf found in {gguf_dir}.")
    files.sort(key=os.path.getsize)
    picked = files[:2]
    for f in files:                              # ensure at least one MoE file is exercised
        if MOE_HINT.search(os.path.basename(f)) and f not in picked:
            picked.append(f)
            break

    def check(path):
        with open(path, "rb") as fh:
            blob = fh.read(min(os.path.getsize(path), 192 << 20))
        _v, kv, tensors, _h = parse_header(blob)
        mine = spec_mirror(kv, tensors, os.path.getsize(path), quant_sizes, quant_enum)
        theirs = spec.from_gguf(path)
        bad = []
        for k in ("t", "a", "ne", "moe", "bits", "kvp", "n_layer", "arch", "codebook_share"):
            mv, tv = mine[k], theirs[k]
            same = (abs(mv - tv) < 1e-9) if isinstance(mv, float) else (mv == tv)
            if not same:
                bad.append(f"{k}: mirror={mv!r} quantprobe={tv!r}")
        log(f"  self-test {os.path.basename(path):52s} "
            f"t={theirs['t']:.4f} a={theirs['a']:.4f} ne={theirs['ne']:.4f} "
            f"moe={mine['moe']!s:5s} untied={mine['_has_output']!s:5s} "
            f"{'OK' if not bad else 'MISMATCH'}")
        if bad:
            raise Abort("the active-byte mirror does NOT reproduce quantprobe.spec.from_gguf "
                        f"on {path}: " + "; ".join(bad))
        return mine, theirs

    # ADVERSARIAL REVIEW 2026-07-30: the old self-test picked the two smallest files plus one
    # name-hinted MoE and declared victory. Nothing checked that the branches the STAKED number
    # actually depends on were ever executed: the MoE branch (`active = ne + routed*k/n`) and
    # BOTH sides of finding #76's tied/untied test. A self-test that never enters the branch it
    # is vouching for is not a self-test. Coverage is now a precondition, and unmet coverage
    # widens the sample instead of passing quietly.
    need = {"moe": lambda m: m["moe"],
            "#76 untied (gather applies)": lambda m: m["_has_output"] and m["_embd"] > 0,
            "#76 tied (gather does not apply)": lambda m: (not m["_has_output"]) and m["_embd"] > 0}
    results, covered, scanned = [], set(), []
    queue = list(picked) + [f for f in files if f not in picked]
    for path in queue:
        if len(covered) == len(need) and path not in picked:
            break
        if len(scanned) >= 8 and path not in picked:
            break
        mine, theirs = check(path)
        scanned.append(path)
        for label, pred in need.items():
            if pred(mine):
                covered.add(label)
        results.append(dict(file=os.path.basename(path), t=theirs["t"], a=theirs["a"],
                            ne=theirs["ne"], bits=theirs["bits"], moe=mine["moe"],
                            untied=bool(mine["_has_output"])))
    missing = [k for k in need if k not in covered]
    if missing:
        raise Abort(f"the mirror self-test never exercised: {missing}. Those branches carry the "
                    "staked number (the #76 gather correction alone is worth 1.29 tok/s here), "
                    "so an unexercised branch is an unverified branch. Point --gguf-dir at a "
                    "folder containing a MoE GGUF, an untied-embedding GGUF and a tied one.")
    log(f"  coverage         {sorted(covered)} over {len(scanned)} local files")
    return results


def http_size(url, sess):
    r = sess.head(url, allow_redirects=True, timeout=60)
    if r.status_code >= 400:
        raise Abort(f"HEAD {url} -> HTTP {r.status_code}")
    n = int(r.headers.get("Content-Length", 0))
    if not n:
        raise Abort(f"HEAD {url} returned no Content-Length; cannot identify the file by size")
    return n


def identify_file(sess, log):
    """Select THE file they benchmarked from the frozen candidate list, on published size."""
    rows, hits = [], []
    for repo, fname in CANDIDATES:
        url = f"https://huggingface.co/{repo}/resolve/main/{fname}"
        try:
            n = http_size(url, sess)
        except Abort as e:
            # ADVERSARIAL REVIEW 2026-07-30: this used to log UNAVAILABLE and carry on. But the
            # identification rule is "exactly one of these seven rounds to 22.3 GB", and that is
            # only an identification if all seven were actually measured. A candidate that
            # silently drops out could be the one that would have made the match ambiguous.
            raise Abort(f"candidate {repo}/{fname} could not be sized ({e}). The selection rule "
                        "requires that EVERY frozen candidate be measured - an unmeasured "
                        "candidate cannot be excluded, and a file identified against a partial "
                        "list is a guess. Re-run with network access to all of Hugging Face.")
        gb = n / 1e9
        match = round(gb, 1) == TARGET_SIZE_GB
        log(f"  candidate {repo:42s} {fname:38s} {gb:7.3f} GB {'  <== MATCH' if match else ''}")
        rows.append(dict(repo=repo, file=fname, bytes=n, gb=round(gb, 4), match=match))
        if match:
            hits.append((repo, fname, n, url))
    if len(hits) != 1:
        raise Abort(f"file identification is ambiguous: {len(hits)} of {len(CANDIDATES)} "
                    f"candidates round to {TARGET_SIZE_GB} GB. The experiment refuses to guess "
                    "which GGUF they ran; add or remove candidates and re-stake.")
    return hits[0], rows


def fetch_header(url, cache, refetch, sess, log):
    """HTTP Range-fetch just enough of the file to parse the tensor table."""
    if os.path.exists(cache) and not refetch:
        log(f"  header cache HIT  {cache} ({os.path.getsize(cache):,} B) - pass --refetch to redo")
        with open(cache, "rb") as f:
            return f.read(), True
    n = 4 << 20
    while n <= (128 << 20):
        try:
            r = sess.get(url, headers={"Range": f"bytes=0-{n - 1}"}, timeout=(30, 180))
        except Exception as e:                                  # noqa: BLE001
            raise Abort(f"header fetch failed ({type(e).__name__}: {e}). This experiment will "
                        "NOT fall back to a sibling GGUF - that is precisely the shortcut that "
                        "left U-33 unscorable. Re-run with network access.")
        if r.status_code not in (200, 206):
            raise Abort(f"header fetch failed: HTTP {r.status_code} for {url}")
        data = r.content
        try:
            hdr_len = parse_header(data)[3]
        except EOFError:
            log(f"  header > {n >> 20} MiB, refetching {n >> 19} MiB")
            n *= 2
            continue
        data = data[:hdr_len]            # keep exactly the header, nothing else
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        with open(cache, "wb") as f:
            f.write(data)
        log(f"  header fetched   {len(data):,} B via Range (file is {url.rsplit('/', 1)[-1]})")
        return data, False
    raise Abort("GGUF header exceeds 128 MiB - refusing to keep downloading a 22 GB file")


def verify_accounting(kv, tensors, hdr_len, remote_bytes, quant_sizes, quant_enum, log):
    """The parse is either exactly right or the identity below does not close."""
    total_tensor_bytes = 0
    align = field(kv, "general.alignment") or 32
    for _name, dims, tt, _off in tensors:
        total_tensor_bytes += tensor_bytes(dims, tt, quant_sizes, quant_enum)[1]
    pad = (-hdr_len) % align
    accounted = hdr_len + pad + total_tensor_bytes
    slack = remote_bytes - accounted
    log(f"  byte accounting  header {hdr_len:,} + pad {pad} + tensors {total_tensor_bytes:,} "
        f"= {accounted:,} vs remote {remote_bytes:,} (slack {slack})")
    if not (0 <= slack < max(align, 64)):
        raise Abort(f"byte accounting does not close (slack {slack} B, alignment {align}). "
                    "The header parse or the quant-size table is wrong; refusing to emit a "
                    "number derived from it.")
    return total_tensor_bytes, align


def verify_identity(kv, log):
    got = dict(arch=kv.get("general.architecture"),
               expert_count=field(kv, ".expert_count"),
               expert_used_count=field(kv, ".expert_used_count"),
               block_count=field(kv, ".block_count"))
    log(f"  file identity    {got}")
    bad = {k: (v, got.get(k)) for k, v in EXPECT.items() if got.get(k) != v}
    if bad:
        raise Abort(f"the fetched file is not the model this experiment was staked on: {bad}")
    return got


def fetch_source(sess, log):
    """Read their published document. This is the SOURCE OF THE TARGET, not a checksum: the
    cells the kill rule binds on are extracted from what is returned here."""
    try:
        doc = sess.get(SOURCE_FINDINGS, timeout=60).text.replace("*", "")
        readme = sess.get(SOURCE_README, timeout=60).text.replace("*", "")
    except Exception as e:                                     # noqa: BLE001
        raise Abort(f"could not read the external source ({type(e).__name__}: {e}). The target "
                    "is extracted live from their document; there is no offline substitute, and "
                    "re-using our own staked transcription as its own confirmation is circular. "
                    "Re-run with network access.")
    if "dual-channel DDR4" not in doc and "dual-channel DDR4" not in readme:
        raise Abort("'dual-channel DDR4' is no longer in the published source. Their memory type "
                    "is the single largest assumption in this retrodiction; if even that is "
                    "gone, the DDR4-3200 preset cannot be justified and no number is emitted.")
    sha = hashlib.sha256(doc.encode()).hexdigest()[:16]
    log(f"  source fetched   {len(doc):,} chars, findings sha256 {sha}")
    return doc, dict(verified=True, findings_sha256=sha, findings_chars=len(doc))


def diff_transcription(live_cells, log):
    """Report, do not refuse. A moved target is an OUTCOME of this experiment, not an error in it."""
    staked = {c["id"]: c for c in CELLS_STAKED}
    diffs = []
    for c in live_cells:
        s = staked.get(c["id"])
        if not s:
            diffs.append(f"{c['id']}: cell is new in the live document")
            continue
        for k in ("tps", "compute", "drop"):
            if s[k] != c[k]:
                diffs.append(f"{c['id']}.{k}: staked {s[k]} -> live {c[k]}")
    live_ids = {c["id"] for c in live_cells}
    diffs += [f"{cid}: staked cell is gone from the live document"
              for cid in staked if cid not in live_ids]
    if diffs:
        log("  TRANSCRIPTION DIVERGENCE - the live source no longer matches what was staked:")
        for d in diffs:
            log(f"    ! {d}")
        log("    The LIVE values are used and SCORED. Refusing here would close the only door "
            "the kill rule can still come through.")
    else:
        log(f"  transcription    all {len(live_cells)} staked cells reproduce verbatim")
    return diffs


# ---------------------------------------------------------------------------------------
# stake block <-> pre-registration
# ---------------------------------------------------------------------------------------
STAKE_KEYS = ("prediction_tok_s", "target_measured_tok_s", "target_headline_tok_s",
              "kill_rel_error", "active_gb_per_token", "max_measured_desktop_tok_s")

# ADVERSARIAL REVIEW 2026-07-30 - THE DEFECT THIS SPLIT FIXES.
#
# Every staked key used to go through one drift gate with a 0.5% tolerance, and any drift
# ABORTED the run. The kill rule fires at 15% relative error. A gate 30x tighter than the kill
# threshold, sitting UPSTREAM of it, makes FAIL unreachable by construction: to fail P-1 the
# prediction would have to move 12%+, which the gate catches at 0.5% and converts into an abort
# ("no number") instead of a miss. Same for P-2 (needs >14% move) and P-3 (needs >18%). The
# verdict was therefore decided the moment the stake block was pasted, and the score run could
# only ever print PASS or ABORT. That is theatre.
#
# The fix is not a looser gate - our own side SHOULD be frozen. It is that the gate must only
# guard the inputs WE control:
#   OUR side  (prediction, active bytes, kill threshold): drift means our code or the remote file
#             moved after staking -> the comparison is void -> abort is correct.
#   THEIR side (the target cells, the headline, their best measured tok/s): drift is not an
#             error, it is the measurement moving. It must flow into the comparison and be
#             allowed to fire the kill rule.
OUR_SIDE_DRIFT_KEYS = ("prediction_tok_s", "active_gb_per_token", "kill_rel_error")
THEIR_SIDE_KEYS = ("target_measured_tok_s", "target_headline_tok_s", "max_measured_desktop_tok_s")


def read_stake(path):
    if not os.path.isfile(path):
        raise Abort(f"pre-registration not found: {path}. --score refuses to run without the "
                    "staked document; the whole point is that the number was fixed first.")
    text = open(path, encoding="utf-8").read()
    m = re.search(r"```stake\s*\n(.*?)```", text, re.S)
    if not m:
        raise Abort(f"no ```stake block in {path}; nothing was staked in a machine-readable form")
    out = {}
    for line in m.group(1).splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        k, _, v = line.partition("=")
        out[k.strip()] = float(v.strip())
    missing = [k for k in STAKE_KEYS if k not in out]
    if missing:
        raise Abort(f"stake block in {path} is missing {missing}")
    return out


def kill_rule_reachability(act, tgt_tps, headline, max_meas, k=KILL_REL_ERROR, eta=ETA_R_MOE):
    """Answer, in the output of every run, the question an adversarial reviewer asks first:
    COULD this run have failed, and what would it have taken?

    Returns the interval of predictions - and the equivalent DRAM bandwidths and DDR4 grades -
    inside which all three predicates hold. If the staked prediction sits far from both edges,
    the run says so in its own verdict block instead of leaving the reader to do the algebra."""
    lo_tps = max(tgt_tps * (1 - k), headline * (1 - k), max_meas)   # P-1 lo, P-2 lo, P-3
    hi_tps = min(tgt_tps * (1 + k), headline * (1 + k))             # P-1 hi, P-2 hi
    return dict(
        pass_band_tok_s=[round(lo_tps, 4), round(hi_tps, 4)],
        binding_low=("P-3 (ceiling exceeded by a measured cell)" if max_meas >= max(
            tgt_tps * (1 - k), headline * (1 - k)) else
            ("P-2 (headline)" if headline * (1 - k) >= tgt_tps * (1 - k) else "P-1 (median cell)")),
        binding_high="P-1 (median cell)" if tgt_tps * (1 + k) <= headline * (1 + k) else "P-2 (headline)",
        pass_band_ram_bw_gbs=[round(bw_for_tps(lo_tps, act, eta), 3),
                              round(bw_for_tps(hi_tps, act, eta), 3)],
        pass_band_ddr4_mts=[int(bw_for_tps(lo_tps, act, eta) / DDR4_DUAL_GBS_PER_MTS) + 1,
                            int(bw_for_tps(hi_tps, act, eta) / DDR4_DUAL_GBS_PER_MTS)])


# ---------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stake", action="store_true",
                    help="compute the prediction ONLY; do not evaluate the comparison")
    ap.add_argument("--prereg", default=PREREG)
    ap.add_argument("--gguf-dir", default=DEFAULT_GGUF_DIR,
                    help="folder of local .gguf files for the mirror self-test")
    ap.add_argument("--refetch", action="store_true", help="ignore the cached GGUF header")
    ap.add_argument("--offline", action="store_true",
                    help="--stake only: skip reading the external source. REFUSED in --score, "
                         "because the target is extracted from the live document and there is no "
                         "offline substitute for it.")
    ap.add_argument("--dram-mts", type=float, default=None,
                    help="THE OPEN FALSIFIER (prereg §5-A). If the upstream author reports the "
                         "laptop's actual DDR4 grade, pass it (e.g. --dram-mts 2666) and the "
                         "retrodiction is re-scored at that bandwidth. The consequence is "
                         "pre-committed in the pre-registration: below the disclosed band the "
                         "kill rule fires and C-06 stays open.")
    a = ap.parse_args()

    os.makedirs(DATA, exist_ok=True)
    logpath = os.path.join(DATA, "exp51_external_retrodiction.log")
    lines = []

    def log(s=""):
        print(s, flush=True)
        lines.append(s)

    t0 = time.time()
    log("=" * 88)
    log("EXPERIMENT #51 - external retrodiction against BigMoeOnEdge's desktop DRAM ceiling")
    log(f"mode: {'STAKE (no comparison)' if a.stake else 'SCORE'}   {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log("=" * 88)

    try:
        if a.offline and not a.stake:
            raise Abort("--offline is refused in --score mode. The target the kill rule binds on "
                        "is extracted from their live document; scoring offline would score our "
                        "own transcription against itself, which cannot fail by construction.")
        quant_sizes, quant_enum = need_imports()
        import requests
        sess = requests.Session()
        sess.headers["User-Agent"] = "quantprobe-exp51/1.0"

        log("\n[0] provenance: the constants this script quotes vs the ones quantprobe ships")
        const_check = verify_shipped_constants(log)

        log("\n[1] mirror self-test: our active-byte arithmetic vs quantprobe.spec.from_gguf")
        st = selftest_mirror(a.gguf_dir, quant_sizes, quant_enum, log)

        log("\n[2] identify THEIR file (frozen candidate list, selected on published size)")
        (repo, fname, remote_bytes, url), cand_rows = identify_file(sess, log)
        log(f"  selected         {repo}/{fname}  {remote_bytes:,} B ({remote_bytes/1e9:.3f} GB)")

        log("\n[3] fetch and validate the GGUF header (Range request, not the 22 GB file)")
        cache = os.path.join(DATA, "exp51_gguf_header.bin")
        blob, cached = fetch_header(url, cache, a.refetch, sess, log)
        _ver, kv, tensors, hdr_len = parse_header(blob)
        log(f"  parsed           {len(tensors)} tensors, {len(kv)} metadata keys")
        tb, align = verify_accounting(kv, tensors, hdr_len, remote_bytes,
                                      quant_sizes, quant_enum, log)
        ident = verify_identity(kv, log)

        log("\n[4] active bytes per token (quantprobe.spec arithmetic + finding #76)")
        s = spec_mirror(kv, tensors, remote_bytes, quant_sizes, quant_enum)
        # `prot = ne` in active_gb_per_token() is only the protected set for a MoE (plan.py uses
        # `prot = ne if moe else min(ne, t*DENSE_PROTECTED_SHARE)`), and the shipped eta is
        # divided by a codebook penalty this experiment's staked arithmetic does not carry
        # (U-17/C-13). Both are true for this file; neither was ever asserted.
        if not s["moe"]:
            raise Abort("the fetched file does not parse as MoE, so `prot = ne` is the wrong "
                        "protected set and this script's arithmetic is not plan.py's.")
        if s["codebook_share"] > 0:
            raise Abort(f"the file carries codebook (IQ) tensors, share {s['codebook_share']:.4f}. "
                        "The shipped eta is divided by (1 + share * IQ_CPU_TG_PENALTY) and the "
                        "staked prediction is not, so the two would silently disagree. Re-stake "
                        "with the penalty applied rather than emitting a number that is neither.")
        act, act_ne, act_ex = active_gb_per_token(s)
        log(f"  total params     {s['t']:.4f} B")
        log(f"  routed (expert)  {s['_routed']/1e9:.4f} B over {s['_n_exp']} experts, top-{s['_n_used']}")
        log(f"  token_embd       {s['_embd']/1e9:.4f} B  untied={s['_has_output']} "
            f"-> #76 gather correction {'APPLIED' if s['_has_output'] else 'not applicable'}")
        log(f"  always-active ne {s['ne']:.4f} B")
        log(f"  active params    {s['a']:.4f} B  (ne + routed x {s['_n_used']}/{s['_n_exp']})")
        log(f"  effective bits   {s['bits']:.2f}  (file bytes x 8 / total params)")
        log(f"  ACTIVE BYTES     {act:.4f} GB/token  (ne {act_ne:.4f} + experts {act_ex:.4f}, "
            f"x{ACT_MULT} activation)")

        log("\n[5] prediction - shipped constants only, zero fitted parameters")
        pred = predict_tps(act)
        log(f"  eta_r (MoE, RAM tier)  {ETA_R_MOE}     quantprobe.plan.evaluate, since v1.0 (d590749)")
        log(f"  RAM bandwidth          {RAM_BW_GBS} GB/s   DDR4-3200 dual-channel preset "
            f"(MACHINES[{MACHINE_FOR_RAM_BW}])")
        log(f"  tok/s = eta x BW / active = {ETA_R_MOE} x {RAM_BW_GBS} / {act:.4f}")
        log(f"  PREDICTED ZERO-I/O CEILING = {pred:.3f} tok/s")
        plan_check = verify_against_plan(s, act, pred, remote_bytes / 1e9, log)

        log("\n[6] sensitivity (disclosure, no kill power)")
        sens = {}
        for label, val in (("DDR4-2400 dual (38.4 GB/s)", predict_tps(act, bw=38.4)),
                           ("DDR4-2666 dual (42.7 GB/s)", predict_tps(act, bw=42.7)),
                           ("DDR4-2933 dual (46.9 GB/s)", predict_tps(act, bw=46.9)),
                           ("DDR4-3200 dual (51.2 GB/s)", predict_tps(act, bw=51.2))):
            sens[label] = round(val, 3)
        # counterfactuals: which shipped correction is load-bearing here?
        s_no76 = dict(s, ne=s["ne"] + s["_embd"] / 1e9, a=s["a"] + s["_embd"] / 1e9)
        act_no76 = active_gb_per_token(s_no76)[0]
        act_raw = act / ACT_MULT
        act_exact = (s["_all_bytes"] - s["_exp_bytes"] - s["_embd_bytes"]
                     + s["_exp_bytes"] * s["_n_used"] / s["_n_exp"]) / 1e9
        # L-19's CPU-attention term is our-box-calibrated and this model is mostly linear
        # attention (full_attention_interval=4), so including it is not defensible - but its
        # magnitude is disclosed rather than left as an unexamined omission.
        cpu_attn_s = s["n_layer"] * 256 * 1.55e-6
        cf = {
            "#76 embedding-gather OFF": round(predict_tps(act_no76), 3),
            "activation x1.15 OFF": round(predict_tps(act_raw), 3),
            "eta 0.30 (the value U-33's prose quoted)": round(predict_tps(act, eta=0.30), 3),
            "exact per-tensor bytes (U-29 convention, no x1.15)": round(predict_tps(act_exact), 3),
            "plus L-19 CPU-attention term at ctx=256": round(1 / (1 / pred + cpu_attn_s), 3),
            "DDR4-2666 AND the L-19 term together": round(
                1 / (1 / predict_tps(act, bw=42.7) + cpu_attn_s), 3),
        }
        for k, v in sens.items():
            log(f"  {k:44s} {v:7.3f} tok/s")
        for k, v in cf.items():
            log(f"  {k:44s} {v:7.3f} tok/s")

        # --- the target. In STAKE mode it comes from the transcription (nothing is scored);
        # in SCORE mode it is re-extracted from the live document under the pre-registered rule.
        if a.stake:
            live_cells, src, tdiff = CELLS_STAKED, dict(verified=False, reason="stake mode"), []
            headline = HEADLINE_CEILING_STAKED
            if not a.offline:
                doc, src = fetch_source(sess, log)
                live_cells = extract_live_cells(doc)
                headline = extract_live_headline(doc)
                tdiff = diff_transcription(live_cells, log)
        else:
            log("\n[6b] read THEIR document and extract the target under the pre-registered rule")
            doc, src = fetch_source(sess, log)
            live_cells = extract_live_cells(doc)
            headline = extract_live_headline(doc)
            tdiff = diff_transcription(live_cells, log)
        tgt_tps, tgt_med, full_cells, max_meas = target_from(live_cells)

        payload = dict(
            experiment=51, kind="external-retrodiction",
            when=time.strftime("%Y-%m-%dT%H:%M:%S"),
            source=dict(readme=SOURCE_README, findings=SOURCE_FINDINGS),
            file=dict(repo=repo, name=fname, url=url, bytes=remote_bytes,
                      gb=round(remote_bytes / 1e9, 4), header_bytes=hdr_len,
                      tensor_bytes=tb, alignment=align, identity=ident,
                      header_cached=cached),
            candidates=cand_rows,
            selftest=st,
            constants_provenance=const_check,
            plan_crosscheck=plan_check,
            spec={k: v for k, v in s.items() if not k.startswith("_")},
            spec_raw={k: v for k, v in s.items() if k.startswith("_")},
            constants=dict(eta_r_moe=ETA_R_MOE, ram_bw_gbs=RAM_BW_GBS, act_mult=ACT_MULT,
                           attn_bit_floor=ATTN_BIT_FLOOR),
            derived=dict(active_gb_per_token=round(act, 6),
                         active_gb_ne=round(act_ne, 6), active_gb_experts=round(act_ex, 6),
                         active_gb_exact_bytes=round(act_exact, 6)),
            prediction_tok_s=round(pred, 4),
            sensitivity=sens, counterfactuals=cf,
            target=dict(rule=("median compute s/token over cells run at the model's own routing "
                              "width (no --drop-cold-experts), extracted from the LIVE document"),
                        full_width_cells=[c["id"] for c in live_cells if not c["drop"]],
                        excluded_cells=[c["id"] for c in live_cells if c["drop"]],
                        compute_s_per_token=full_cells, median_s=tgt_med,
                        measured_tok_s=round(tgt_tps, 4),
                        headline_tok_s=headline,
                        max_measured_desktop_tok_s=max_meas,
                        transcription_divergence=tdiff),
            cells_staked=CELLS_STAKED, cells_live=live_cells,
        )

        if a.stake:
            log("\n[7] STAKE MODE - the comparison is NOT evaluated here.")
            log("    Paste this block into the pre-registration, then run without --stake:\n")
            block = "\n".join([
                "```stake",
                f"prediction_tok_s          = {pred:.4f}",
                f"active_gb_per_token       = {act:.4f}",
                f"target_measured_tok_s     = {tgt_tps:.4f}",
                f"target_headline_tok_s     = {headline:.4f}",
                f"max_measured_desktop_tok_s= {max_meas:.4f}",
                f"kill_rel_error            = {KILL_REL_ERROR:.4f}",
                "```"])
            log(block)
            payload["mode"] = "stake"
            out = os.path.join(DATA, "exp51_stake.json")
            with open(out, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=1)
            log(f"\n  wrote {out}")
            log(f"  elapsed {time.time() - t0:.1f}s")
            return 0

        log("\n[7] re-read the staked pre-registration; OUR side must still reproduce")
        stake = read_stake(a.prereg)
        drift = {}
        for k, fresh in (("prediction_tok_s", pred), ("active_gb_per_token", act),
                         ("kill_rel_error", KILL_REL_ERROR)):
            if abs(stake[k] - fresh) > 5e-3 * max(1.0, abs(fresh)):
                drift[k] = (stake[k], round(fresh, 4))
        if drift:
            raise Abort("the fresh computation no longer reproduces OUR staked numbers "
                        f"{drift}. Our code or the remote file moved after staking; the "
                        "comparison is void until the prereg is re-staked and re-dated.")
        log(f"  our side         staked prediction {stake['prediction_tok_s']:.4f} reproduced "
            f"({pred:.4f}); {len(OUR_SIDE_DRIFT_KEYS)} keys agree -> abort gate satisfied")
        live_now = dict(target_measured_tok_s=tgt_tps, target_headline_tok_s=headline,
                        max_measured_desktop_tok_s=max_meas)
        for k in THEIR_SIDE_KEYS:
            live = live_now[k]
            mark = "unchanged" if abs(stake[k] - live) < 5e-3 * max(1.0, abs(live)) else "MOVED"
            log(f"  their side       {k:28s} staked {stake[k]:7.4f} live {live:7.4f}  {mark} "
                "(scored, never aborted)")

        log("\n[8] THE COMPARISON")
        # The STAKED prediction is what was committed, so it is what gets scored. `pred` is the
        # fresh recomputation and was already required to reproduce it above; using the staked
        # value makes it impossible for a later code edit to move the prediction and the verdict
        # together. --dram-mts replaces the ONE assumption we could not read off their machine.
        scored_pred = stake["prediction_tok_s"]
        bw_used, dram_note = RAM_BW_GBS, "DDR4-3200 preset (ASSUMED - they published only "\
                                         "'dual-channel DDR4')"
        if a.dram_mts:
            bw_used = a.dram_mts * DDR4_DUAL_GBS_PER_MTS
            scored_pred = predict_tps(act, bw=bw_used)
            dram_note = (f"REPORTED DDR4-{a.dram_mts:g} dual-channel = {bw_used:.2f} GB/s "
                         "- the §5-A falsifier has been resolved with real data")
            log(f"  !! --dram-mts {a.dram_mts:g} supplied: the assumed 51.0 GB/s is replaced and the "
                f"prediction is re-scored at {scored_pred:.3f} tok/s")
        reach = kill_rule_reachability(act, tgt_tps, headline, max_meas)
        e_meas = scored_pred / tgt_tps - 1.0
        e_head = scored_pred / headline - 1.0
        log(f"  DRAM assumption       {dram_note}")
        log(f"  full-width cells      {[c['id'] for c in live_cells if not c['drop']]} "
            f"compute s/token {full_cells}")
        log(f"  target  (median)      {tgt_med:.4f} s/token = {tgt_tps:.3f} tok/s")
        log(f"  target  (headline)    {headline:.3f} tok/s")
        log(f"  predicted (scored)    {scored_pred:.3f} tok/s")
        log(f"  P-1 error vs measured {e_meas:+.2%}   (kill: |e| >= {KILL_REL_ERROR:.0%})")
        log(f"  P-2 error vs headline {e_head:+.2%}   (kill: |e| >= {KILL_REL_ERROR:.0%})")
        log(f"  P-3 ceiling not violated: predicted {scored_pred:.3f} > best measured "
            f"{max_meas:.3f}? {scored_pred > max_meas}")
        p1 = abs(e_meas) < KILL_REL_ERROR
        p2 = abs(e_head) < KILL_REL_ERROR
        p3 = scored_pred > max_meas
        verdict = "PASS" if (p1 and p2 and p3) else "FAIL"

        log("\n[9] KILL-RULE REACHABILITY - could this run have failed?")
        lo, hi = reach["pass_band_tok_s"]
        blo, bhi = reach["pass_band_ram_bw_gbs"]
        mlo, mhi = reach["pass_band_ddr4_mts"]
        log(f"  all three predicates hold iff the prediction is in [{lo:.3f}, {hi:.3f}] tok/s")
        log(f"    low edge bound by  {reach['binding_low']}")
        log(f"    high edge bound by {reach['binding_high']}")
        log(f"  equivalently, iff their DRAM delivers [{blo:.2f}, {bhi:.2f}] GB/s, i.e. "
            f"dual-channel DDR4-{mlo} to DDR4-{mhi}")
        margin = min(scored_pred - lo, hi - scored_pred) / scored_pred
        log(f"  scored prediction {scored_pred:.3f} sits {margin:.1%} from the nearer edge")
        # The honest disclosure. With our side frozen by the abort gate above and their document
        # unchanged, no input this run could have produced would have flipped the verdict.
        our_side_frozen = not drift and not a.dram_mts
        reachable = bool(tdiff) or bool(a.dram_mts)
        if not reachable:
            log("  KILL RULE UNREACHABLE THIS RUN. Both operands were public and frozen before")
            log("  staking, and the abort gate holds our side fixed, so this run could only ever")
            log("  print PASS or ABORT. It is a reproduction check, NOT an independent test.")
            log("  The two live routes to a FAIL remain: (i) they revise the published cells,")
            log("  (ii) --dram-mts resolves the DRAM grade below the band printed above.")
        else:
            log("  KILL RULE LIVE this run: " + ("their published cells moved; " if tdiff else "")
                + ("the DRAM grade was supplied and replaced our assumption" if a.dram_mts else ""))

        payload.update(mode="score", stake=stake, source_check=src,
                       scored_prediction_tok_s=round(scored_pred, 4),
                       dram=dict(assumed_gbs=RAM_BW_GBS, used_gbs=round(bw_used, 4),
                                 reported_mts=a.dram_mts, note=dram_note),
                       kill_rule=dict(reachable_this_run=reachable,
                                      our_side_frozen=our_side_frozen,
                                      margin_to_nearer_edge=round(margin, 6), **reach),
                       comparison=dict(error_vs_measured=round(e_meas, 6),
                                       error_vs_headline=round(e_head, 6),
                                       P1_measured_within_15pct=p1,
                                       P2_headline_within_15pct=p2,
                                       P3_ceiling_not_violated=p3,
                                       verdict=verdict))
        log("")
        log("-" * 88)
        log(f"  {verdict}: P-1 {'hit' if p1 else 'MISS'} | P-2 {'hit' if p2 else 'MISS'} | "
            f"P-3 {'hit' if p3 else 'MISS'}")
        if verdict == "PASS":
            log("  C-06's external replication ask is satisfied for the CPU/DRAM tier on a "
                "machine we have never touched, CONDITIONAL on their DRAM delivering "
                f"{blo:.1f}-{bhi:.1f} GB/s (dual-channel DDR4-{mlo}+). That condition is an "
                "assumption, not a measurement, and the register entry must carry it.")
            if not reachable:
                log("  ...and it is a PASS on a run that could not have failed. Weigh it as a "
                    "reproduction of a retrodiction, not as a test.")
        else:
            log("  KILL RULE FIRED. C-06 STAYS OPEN. U-33 must be rewritten to say the match "
                "did not survive contact with their actual file. Publish at equal prominence.")
        log("-" * 88)

        out = os.path.join(DATA, "exp51_external_retrodiction.json")
        with open(out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=1)
        log(f"\n  wrote {out}")
        log(f"  elapsed {time.time() - t0:.1f}s")
        return 0 if verdict == "PASS" else 1

    except Abort as e:
        log("")
        log("!" * 88)
        log(f"ABORTED - precondition failed, NO number produced:\n  {e}")
        log("!" * 88)
        # A previous PASS must not survive on disk as the apparent result of THIS run. An abort
        # that leaves a stale exp51_external_retrodiction.json behind is exactly how a number
        # nobody stands behind gets picked up by a later reader (or by findings.py).
        if not a.stake:
            out = os.path.join(DATA, "exp51_external_retrodiction.json")
            with open(out, "w", encoding="utf-8") as f:
                json.dump(dict(experiment=51, mode="abort", verdict="ABORT",
                               when=time.strftime("%Y-%m-%dT%H:%M:%S"), reason=str(e),
                               note="No number was produced. Any previous scored result at this "
                                    "path has been overwritten deliberately."), f, indent=1)
            log(f"  wrote abort record to {out} (previous result overwritten on purpose)")
        return 2
    finally:
        with open(logpath, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print(f"[log] {logpath}")


if __name__ == "__main__":
    sys.exit(main())
