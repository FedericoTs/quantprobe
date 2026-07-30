"""Experiment #53 - TWO-RESOURCE MODEL FOR THE STREAMING/DISK TIER.

    python weights/exp53_two_resource_disk_tier.py --stake    # compute + print the stake block
    python weights/exp53_two_resource_disk_tier.py            # score it against the prereg

THE CLAIM UNDER TEST (staked in preregistrations/2026-07-30-two-resource-disk-tier.md):

    t_token = max(C, I)   if the row was run with --overlap
    t_token = C + I       otherwise
        C = active_bytes(model, k) / S_c(device)      <- Law 4's compute-byte term
        I = flash_bytes(row)       / S_i(device)      <- the streamed-read term

`S_c` and `S_i` are ONE effective rate per device, shared across every model and every row on
that device.  Nothing is fitted per model and nothing is fitted per row.  `flash_bytes` is the
external authors' own instrumented Flash/token counter; `active_bytes` comes from the GGUF
header via quantprobe's own arithmetic.

WHY THIS FILE EXISTS.  BigMoeOnEdge's k=8 -> k=6 result is 1.25x measured; our Law 4 compute-byte
ratio predicts 1.175x and their flash-byte ratio is 1.364x.  The truth sits strictly between two
single-resource bounds, which is evidence that the disk tier needs a second resource term.  This
script decides whether the two-resource form earns its place BEFORE any of it is wired into
quantprobe.  Their four published README tables are the held-out set.

WHAT IS SCORED
  P-0  byte accounting: the GGUF's own routed-bytes-per-token must reproduce their independent
       no-cache Flash/token counter.  Parameter-free.  Four comparisons, 3% gate.
  Arm A  the model form itself, scored by leave-one-MODEL-out (LOMO) on the 18 phone rows,
       against two one-resource nulls (compute-only, io-only) and two same-cost alternatives
       (additive-always, max-always) plus a partial-overlap variant.
  Arm B  the five clean k -> k' pairs: does the prediction sit strictly between the two
       single-resource bounds and within 10% of measured?
  Arm C  cache hit as a MODELLED quantity: h = (M/N)^beta, one beta for every model and device.
  Desktop  a disclosure arm with no kill power (one usable row cannot support two constants).

WHAT IT REFUSES TO DO - each of these aborts with exit code 2 and produces NO number:
  - `requests` / `gguf` / `quantprobe` not importable
  - weights/exp51_external_retrodiction.py missing (its GGUF parser and its proved-faithful
    mirror of quantprobe.spec.from_gguf are the byte source; re-implementing them here is
    exactly the unverified-reimplementation failure this project exists to avoid)
  - no local GGUF for exp51's mirror self-test, or the mirror disagreeing with quantprobe
  - weights/data/external_bigmoe_tables.md missing, or disagreeing with the table in this file
  - any transcribed row no longer present in the live upstream README (unless --offline, which
    is recorded in the output)
  - a header fetch failing, byte accounting not closing, or a file's own metadata not matching
    the architecture/expert-count/width this experiment was staked on
  - --score with a missing prereg, an unparseable ```stake block, or a stake the fresh
    computation no longer reproduces

IDEMPOTENT.  GGUF headers are cached under weights/data/exp53_headers/ and re-used; the fit is a
deterministic grid + coordinate descent with fixed iteration counts, so two runs on the same
inputs produce byte-identical numbers.  Re-running never mutates anything but its own outputs.

ADVERSARIAL REVIEW 2026-07-30 - what was wrong with this scorer, fixed below.  The staked model
M2 fails its own kill rule on all four gates, so the *staked* arm is not theatre.  But the thing
that actually ships out of this experiment is the pre-declared fallback N3, and N3's gates were
softer than they looked:
  1. P-0 gated M2 and NOT N3.  `verdict = g_m2["PASS"] and p0_ok`, while the fallback's verdict
     was `g_n3["PASS"]` alone - so a P-0 failure would have killed the model that was already
     dead and left the model that gets wired in untouched, in direct contradiction of prereg
     section 8 ("nothing downstream of it may be wired in whatever Arm A says").  P-0 is now a
     conjunct of EVERY gate set.
  2. K-4's `bracketed` conjunct is an ALGEBRAIC TAUTOLOGY for any additive form.  (Ch+Ih)/(Cl+Il)
     is the mediant of Ch/Cl and Ih/Il and therefore always lies between them; 200k random draws
     produce zero violations.  For N3 - additive on every row - `bracketed` cannot fail, so K-4
     collapses to the 10% tolerance alone.  It is now computed, labelled `bracket_forced`, and
     reported as carrying no evidence rather than being counted as if it did.
  3. K-3 cannot fail for the fallback BY CONSTRUCTION.  The fallback was named as the best
     same-cost form, and K-3 asks whether it is the best same-cost form.  Now flagged
     `k3_informative=False` wherever the mode is the argmin of the rival set it is scored against.
  4. The desktop arm printed an implied read rate through an UNGUARDED division that goes
     negative the moment the compute term exceeds the measured token time - the exact shape of
     the profiler-on-an-oversized-model artifact this project was burned by.  It now refuses.
  5. The desktop arm's literals (their overlap A/B, their instrumented compute, their NVMe rate)
     reached the published conclusion through f-strings and were checked against NOTHING.  They
     are named constants now and verified against the transcription and the live README like
     every other number.  One of them was also wrong: their prose says ~0.11 s/token, and the
     write-up rendered that as 115 ms.
  6. The stake-reproduction tolerance was `5e-3 * max(1.0, |value|)` - an ABSOLUTE 0.005 on every
     value below 1.0, i.e. +-12% on n3_lomo_median = 0.0404, while the prereg claimed 0.5%.  Now
     genuinely relative.
  7. --offline silently bought a full publishable verdict for a run that never checked the source.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "weights", "data")
HDR_DIR = os.path.join(DATA, "exp53_headers")
PREREG = os.path.join(ROOT, "preregistrations", "2026-07-30-two-resource-disk-tier.md")
TABLES = os.path.join(DATA, "external_bigmoe_tables.md")
EXP51 = os.path.join(HERE, "exp51_external_retrodiction.py")
DEFAULT_GGUF_DIR = r"D:\evo-compress-data\gguf"

SOURCE_README = "https://raw.githubusercontent.com/Helldez/BigMoeOnEdge/main/README.md"
SOURCE_REPO = "https://github.com/Helldez/BigMoeOnEdge"
SOURCE_READ_DATE = "2026-07-30"

MiB = 1048576.0

# ---------------------------------------------------------------------------------------
# FROZEN INPUTS.  Shipped quantprobe constants carry the line they live on; everything else is
# a literal transcribed from the external source and re-verified against it on every run.
# ---------------------------------------------------------------------------------------

ACT_MULT = 1.15          # quantprobe/plan.py:667-668, activation multiplier on both byte terms
ATTN_BIT_FLOOR = 4.5     # quantprobe/plan.py:640, the always-active bit floor
ETA_R_MOE = 0.38         # quantprobe/plan.py:674, MoE RAM-tier efficiency, shipped since d590749

# The GGUF that each benchmarked model is taken to be.  Chosen BEFORE any fit, by one rule:
# the Q4_K_M (or, for gpt-oss, the native MXFP4 that their README says "streams unchanged")
# whose published size is closest to the size they state.  For Qwen3.6-35B-A3B the file is
# inherited unchanged from experiment #51 so the two experiments cannot disagree about it.
# The rule is NOT what does the work - P-0 is: each file's own routed-byte count has to
# reproduce their independent flash counter, and a file that fails that is disqualified.
FILES = {
    "gpt-oss-120b":   ("ggml-org/gpt-oss-120b-GGUF", "gpt-oss-120b-MXFP4.gguf", "~60 GB"),
    "qwen36-35b-a3b": ("bartowski/Qwen_Qwen3.6-35B-A3B-GGUF",
                       "Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf", "22.3 GB"),
    "qwen3-30b-a3b":  ("ggml-org/Qwen3-30B-A3B-GGUF", "Qwen3-30B-A3B-Q4_K_M.gguf", "18.5 GB"),
    "gemma4-26b-a4b": ("bartowski/google_gemma-4-26B-A4B-it-GGUF",
                       "google_gemma-4-26B-A4B-it-Q4_K_M.gguf", "17.0 GB"),
}
# arch / expert_count / expert_used_count / block_count each file must report, or the run aborts.
EXPECT = {
    "gpt-oss-120b":   dict(arch="gpt-oss",   expert_count=128, expert_used_count=4, block_count=36),
    "qwen36-35b-a3b": dict(arch="qwen35moe", expert_count=256, expert_used_count=8, block_count=41),
    "qwen3-30b-a3b":  dict(arch="qwen3moe",  expert_count=128, expert_used_count=8, block_count=48),
    "gemma4-26b-a4b": dict(arch="gemma4",    expert_count=128, expert_used_count=8, block_count=30),
}

# Their published rows.  Transcribed from README.md on 2026-07-30; see
# weights/data/external_bigmoe_tables.md for provenance, column meanings and the exclusion rules.
# fields: device, model, k, tok/s, Flash/token (MiB), overlap, cache (MiB), cache-hit, label
ROWS = [
    ("P", "gpt-oss-120b",   2, 2.2,  590, True,  2000, 0.32, "k2 cache2000 8lane"),
    ("P", "gpt-oss-120b",   2, 1.8,  909, True,     0, None, "k2 nocache 4lane"),
    ("P", "gpt-oss-120b",   4, 1.3, 1292, True,  2000, 0.27, "k4 cache2000 8lane"),
    ("P", "gpt-oss-120b",   4, 0.7, 1817, True,     0, None, "k4 nocache 4lane"),
    ("P", "qwen36-35b-a3b", 8, 4.3,  206, True,  2000, 0.56, "k8 cache2000"),
    ("P", "qwen36-35b-a3b", 8, 5.0,  144, True,  3000, 0.65, "k8 cache3000"),
    ("P", "qwen36-35b-a3b", 6, 5.4,  137, True,  2000, 0.60, "k6 cache2000"),
    ("P", "qwen36-35b-a3b", 6, 5.8,   91, True,  3000, 0.68, "k6 cache3000"),
    ("P", "qwen3-30b-a3b",  8, 1.7, 1051, False,    0, None, "k8 nocache"),
    ("P", "qwen3-30b-a3b",  8, 2.4,  480, False, 2000, 0.53, "k8 cache2000"),
    ("P", "qwen3-30b-a3b",  8, 4.0,  225, False, 4000, 0.76, "k8 cache4000"),
    ("P", "qwen3-30b-a3b",  8, 5.2,  225, True,  4000, 0.76, "k8 auto4000 overlap"),
    ("P", "qwen3-30b-a3b",  6, 5.0,  165, False, 4000, 0.77, "k6 cache4000"),
    ("P", "gemma4-26b-a4b", 8, 1.6,  904, False,    0, None, "k8 nocache"),
    ("P", "gemma4-26b-a4b", 8, 2.2,  366, False, 2000, 0.58, "k8 cache2000"),
    ("P", "gemma4-26b-a4b", 8, 2.8,  365, True,  2000, 0.58, "k8 cache2000 overlap"),
    ("P", "gemma4-26b-a4b", 8, 4.1,  144, False, 4000, 0.82, "k8 cache4000"),
    ("P", "gemma4-26b-a4b", 6, 5.0,   98, False, 4000, 0.83, "k6 cache4000"),
    ("D", "qwen36-35b-a3b", 8, 4.8,   74, False,    0, 0.84, "desktop k8 cache-auto"),
]

# Rows whose no-cache Flash/token is the model's FULL routed-byte read for that width: the
# parameter-free anchor for P-0.  (model, k, published MiB)
P0_ROWS = [("gpt-oss-120b", 4, 1817), ("gpt-oss-120b", 2, 909),
           ("qwen3-30b-a3b", 8, 1051), ("gemma4-26b-a4b", 8, 904)]
P0_GATE = 0.03                        # 3% - see pre-registration section 3

# The five clean k -> k' pairs: same model, same cache, same overlap setting, only k changes.
# (model, k_hi, k_lo, flash_hi, flash_lo, overlap, measured speedup)
KPAIRS = [
    ("qwen3-30b-a3b",  8, 6,  225, 165, False, 5.0 / 4.0),
    ("gemma4-26b-a4b", 8, 6,  144,  98, False, 5.0 / 4.1),
    ("qwen36-35b-a3b", 8, 6,  206, 137, True,  5.4 / 4.3),
    ("qwen36-35b-a3b", 8, 6,  144,  91, True,  5.8 / 5.0),
    ("gpt-oss-120b",   4, 2, 1292, 590, True,  2.2 / 1.3),
]
KPAIR_TOL = 0.10                      # |pred/meas - 1| must be under this
KPAIR_GATE_COUNT = 4                  # how many of the five must bracket AND land inside KPAIR_TOL

ARMC_ABS_TOL = 0.10                   # absolute cache-hit error the modelled h must stay inside

# Literals used ONLY by the desktop disclosure arm (prereg section 6).  They are NOT in ROWS, so
# until the 2026-07-30 adversarial review NOTHING checked them against the source - they reached
# the published prose through f-strings.  A number that reaches a conclusion is load-bearing
# whether or not it has kill power, so they are named, sourced and verified like everything else.
# (`115 ms` was the number in the write-up; their prose says ~0.11 s/token.  Corrected to 110.)
DESKTOP_OVERLAP_AB = (6.8, 7.3)       # their --drop-cold-experts A/B, without / with --overlap
DESKTOP_OVERLAP_AB_FLASH = (23, 24)   # ...at these flash bytes, i.e. the same work either way
DESKTOP_NVME_GBS = 3.0                # "~3 GB/s" NVMe, their prose
DESKTOP_THEIR_COMPUTE_MS = 110.0      # "~0.11 s/token in every cell", their prose
DESKTOP_RAM_BW_GBS = 51.0             # OUR DDR4-3200 preset (experiment #51).  NOT their number.
# Fragments that must appear verbatim in the transcription AND upstream, exactly like ROWS.
DESKTOP_LITERALS = [
    f"| {DESKTOP_OVERLAP_AB[0]:.1f} | {DESKTOP_OVERLAP_AB_FLASH[0]} MiB |",
    f"| {DESKTOP_OVERLAP_AB[1]:.1f} | {DESKTOP_OVERLAP_AB_FLASH[1]} MiB |",
]
DESKTOP_PROSE = ["0.11 s/token", "3 GB/s"]

# KILL-RULE constants, staked in the pre-registration.
K1_LOMO_MEDIAN_MAX = 0.15             # inherited from prereg #86 / U-33: the project's standing
K1_LOMO_MAX_MAX = 0.35                # external-comparison tolerance, and 2x it for the worst row
K2_MIN_BEAT_FACTOR = 2.0              # the 2nd resource must halve the best one-resource LOMO

MODES = ("M2", "N1", "N2", "N3", "N4", "M2b")
MODE_DOC = {
    "M2":  "STAKED: max(C,I) under --overlap, C+I without",
    "N1":  "null: compute only (Law 4 alone, disk invisible)",
    "N2":  "null: I/O only (compute invisible)",
    "N3":  "alt: C+I always (--overlap earns no byte-level credit)",
    "N4":  "alt: max(C,I) always (perfect overlap everywhere)",
    "M2b": "alt: C+I-phi*min(C,I) under --overlap (partial overlap, 1 extra param)",
}
CONVENTIONS = ("shipped", "exact")     # active-byte conventions; `shipped` is primary


class Abort(Exception):
    """A precondition failed.  The script says why and exits non-zero WITHOUT a number."""


# ---------------------------------------------------------------------------------------
# preconditions
# ---------------------------------------------------------------------------------------
def need_imports():
    try:
        import requests                                                        # noqa: F401
    except ImportError:
        raise Abort("`requests` is not installed.  python -m pip install requests")
    try:
        from gguf import GGML_QUANT_SIZES, GGMLQuantizationType
    except ImportError:
        raise Abort("`gguf` is not installed - per-tensor byte sizes cannot be computed.  "
                    "python -m pip install gguf")
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    try:
        from quantprobe import spec                                            # noqa: F401
    except ImportError as e:
        raise Abort(f"quantprobe is not importable from {ROOT} ({e}).  exp51's mirror self-test "
                    "cannot run, and an unverified byte model is worse than no number.")
    if not os.path.isfile(EXP51):
        raise Abort(f"{EXP51} is missing.  #53 deliberately does NOT re-implement the GGUF header "
                    "parser or the quantprobe.spec mirror; it imports #51's, which ship with a "
                    "self-test against quantprobe itself.")
    s = importlib.util.spec_from_file_location("exp51_external_retrodiction", EXP51)
    mod = importlib.util.module_from_spec(s)
    s.loader.exec_module(mod)
    for fn in ("parse_header", "spec_mirror", "tensor_bytes", "field", "http_size",
               "fetch_header", "selftest_mirror"):
        if not hasattr(mod, fn):
            raise Abort(f"exp51 does not expose {fn}(); its interface changed and #53's byte "
                        "model is no longer provably the same one.  Re-verify before scoring.")
    return GGML_QUANT_SIZES, GGMLQuantizationType, mod


def row_literals(r):
    """The exact fragments that must appear in BOTH the local transcription and the live README."""
    _dev, _m, _k, tps, flash, _ov, _c, hit, _lab = r
    out = [f"| {tps:.1f} | {flash} MiB |"]
    if hit is not None:
        out.append(f"| {tps:.1f} | {flash} MiB | {round(hit * 100)}% |")
    return out


def verify_transcription(log):
    """The data file is not decoration: every row in this script must be in it, verbatim."""
    if not os.path.isfile(TABLES):
        raise Abort(f"{TABLES} is missing.  The external tables are the held-out set; scoring "
                    "against numbers that live only inside the scoring script is circular.")
    text = open(TABLES, encoding="utf-8").read().replace("*", "")
    if SOURCE_README not in text or SOURCE_READ_DATE not in text:
        raise Abort(f"{TABLES} does not carry the source URL and read date this run expects "
                    f"({SOURCE_README}, {SOURCE_READ_DATE}).")
    missing = [f"{r[1]}/{r[8]}" for r in ROWS if any(x not in text for x in row_literals(r))]
    if missing:
        raise Abort(f"rows {missing} are in this script but not in {TABLES}.  The transcription "
                    "and the scorer disagree; fix the data file and re-stake.")
    # The desktop disclosure literals get exactly the same treatment as the fitted rows.
    dmiss = [x for x in DESKTOP_LITERALS + DESKTOP_PROSE if x not in text]
    if dmiss:
        raise Abort(f"the desktop disclosure literals {dmiss} are not in {TABLES}.  Section 6 of "
                    "the pre-registration quotes them; a quoted number that is not in the "
                    "transcription is an unsourced number, kill power or no kill power.")
    log(f"  transcription    all {len(ROWS)} rows + {len(DESKTOP_LITERALS + DESKTOP_PROSE)} "
        f"desktop literals present verbatim in {os.path.basename(TABLES)}")
    return dict(path=TABLES, rows=len(ROWS),
                desktop_literals=DESKTOP_LITERALS + DESKTOP_PROSE)


def verify_source(sess, log, offline):
    """Re-read the upstream README and confirm every transcribed row is still there."""
    if offline:
        log("  source verify    SKIPPED (--offline).  The transcription is UNVERIFIED this run.")
        return dict(verified=False, reason="offline")
    try:
        doc = sess.get(SOURCE_README, timeout=60).text.replace("*", "")
    except Exception as e:                                                     # noqa: BLE001
        raise Abort(f"could not re-read {SOURCE_README} ({type(e).__name__}: {e}).  Scoring a "
                    "transcription nobody re-checked is how a citation error becomes a finding.  "
                    "Re-run with network, or pass --offline to disclose it in the output.")
    missing = [f"{r[1]}/{r[8]}" for r in ROWS if any(x not in doc for x in row_literals(r))]
    missing += [f"desktop-disclosure:{x}" for x in DESKTOP_LITERALS if x not in doc]
    if missing:
        raise Abort(f"rows {missing} no longer appear verbatim upstream.  The source moved or the "
                    "transcription is wrong; re-stake before scoring.")
    for probe in ("Flash/token", "cache hit", "dual-channel DDR4", "UFS 4"):
        if probe not in doc:
            raise Abort(f"the phrase {probe!r} is gone from the upstream README; the dataset this "
                        "experiment was staked on has changed shape.")
    log(f"  source verify    all {len(ROWS)} rows still present verbatim upstream")
    import hashlib
    return dict(verified=True, url=SOURCE_README,
                sha256=hashlib.sha256(doc.encode()).hexdigest()[:16])


# ---------------------------------------------------------------------------------------
# model specs from the GGUF headers
# ---------------------------------------------------------------------------------------
def load_specs(sess, e51, quant_sizes, quant_enum, refetch, log):
    os.makedirs(HDR_DIR, exist_ok=True)
    specs = {}
    for key, (repo, fname, stated) in FILES.items():
        url = f"https://huggingface.co/{repo}/resolve/main/{fname}"
        try:
            remote = e51.http_size(url, sess)
        except Exception as e:                                                 # noqa: BLE001
            raise Abort(f"could not size {repo}/{fname} ({e}).  #53 will not substitute a "
                        "sibling GGUF; that shortcut is what left U-33 unscorable.")
        cache = os.path.join(HDR_DIR, f"{key}.bin")
        blob, cached = e51.fetch_header(url, cache, refetch, sess, lambda s: None)
        _ver, kv, tensors, hdr_len = e51.parse_header(blob)
        tb = sum(e51.tensor_bytes(d, t, quant_sizes, quant_enum)[1] for _n, d, t, _o in tensors)
        align = e51.field(kv, "general.alignment") or 32
        slack = remote - (hdr_len + (-hdr_len) % align + tb)
        if not (0 <= slack < max(align, 1024)):
            raise Abort(f"byte accounting does not close for {key} (slack {slack} B).  The header "
                        "parse or the quant-size table is wrong; refusing to emit a number from it.")
        got = dict(arch=kv.get("general.architecture"),
                   expert_count=e51.field(kv, ".expert_count"),
                   expert_used_count=e51.field(kv, ".expert_used_count"),
                   block_count=e51.field(kv, ".block_count"))
        bad = {k: (v, got.get(k)) for k, v in EXPECT[key].items() if got.get(k) != v}
        if bad:
            raise Abort(f"{key}: the fetched file is not the model this experiment was staked "
                        f"on: {bad}")
        s = e51.spec_mirror(kv, tensors, remote, quant_sizes, quant_enum)
        s.update(_repo=repo, _file=fname, _url=url, _bytes=remote, _stated=stated,
                 _cached=cached, _slack=slack, _header_bytes=hdr_len, _ntensor=len(tensors))
        specs[key] = s
        log(f"  {key:16s} {remote/1e9:7.3f} GB (they state {stated:8s}) {got['arch']:10s} "
            f"E={got['expert_count']:4d} k={got['expert_used_count']} L={got['block_count']:3d} "
            f"bits={s['bits']:.2f} {'cached' if cached else 'fetched'}")
    return specs


def active_gb(s, k, convention):
    """Active weight bytes for ONE decoded token at routing width k.

    `shipped`: quantprobe/plan.py:640-669 exactly - average effective bits with the 4.5-bit
               always-active floor.  This is what the tool ships, so it is the primary.
    `exact`  : the file's real per-tensor bytes (the U-29 convention).  Only differs where a file
               is strongly mixed-precision - which gpt-oss-120b is (its always-active set is
               8.7 effective bits against a 4.34-bit file average).
    Finding #76's embedding-gather correction applies in both: token_embd leaves the always-active
    set iff a separate output/lm_head tensor exists."""
    gather_only = s["_embd"] if s["_has_output"] else 0
    if convention == "shipped":
        bits, ab = s["bits"], max(s["bits"], ATTN_BIT_FLOOR)
        ne_p = s["_total"] - s["_routed"] - gather_only
        ex_p = s["_routed"] * k / s["_n_exp"]
        return (ne_p * ab / 8 + ex_p * bits / 8) * ACT_MULT / 1e9
    embd_b = s["_embd_bytes"] if s["_has_output"] else 0
    ne_b = s["_all_bytes"] - s["_exp_bytes"] - embd_b
    ex_b = s["_exp_bytes"] * k / s["_n_exp"]
    return (ne_b + ex_b) * ACT_MULT / 1e9


def routed_gb_per_token(s, k):
    """Bytes of routed-expert weight a width-k token touches.  P-0 compares this with their
    own instrumented Flash/token on the rows that ran with no cache at all."""
    return s["_exp_bytes"] * k / s["_n_exp"] / 1e9


# ---------------------------------------------------------------------------------------
# the model, the nulls, and a deterministic fit
# ---------------------------------------------------------------------------------------
def t_pred(A, F, ov, sc, si, mode, phi):
    C, I = A / sc, F / si
    if mode == "M2":
        return max(C, I) if ov else C + I
    if mode == "N1":
        return C
    if mode == "N2":
        return I
    if mode == "N3":
        return C + I
    if mode == "N4":
        return max(C, I)
    if mode == "M2b":
        return (C + I - phi * min(C, I)) if ov else (C + I)
    raise Abort(f"unknown model form {mode!r}")


def bracket_forced(mode, ov):
    """Is Arm B's `bracketed` test capable of failing for this (mode, row) at all?

    NO, if the form reduces to `C + I` on both rows of the pair.  t_hi/t_lo is then
    (Ch+Ih)/(Cl+Il), the MEDIANT of Ch/Cl = A_hi/A_lo (the compute-only bound) and
    Ih/Il = F_hi/F_lo (the io-only bound), and a mediant of two ratios always lies between them.
    So for an additive form `bracketed` is an algebraic identity, not a measurement: 200k random
    draws produce zero violations.  It is a real test only where max() can collapse the ratio onto
    a single resource - which is exactly the failure the staked model M2 was hypothesised to
    avoid, and exactly the one it commits.  Counted and reported so that a gate cleared by a
    tautology is never read as a gate cleared by the data."""
    if mode == "N3":
        return True
    if mode in ("M2", "M2b") and not ov:
        return True
    return False


def tol_window(rows, gate):
    """The interval of KPAIR_TOL over which the K-4 verdict would not change.

    A pair that is not bracketed can never pass at any tolerance, so only bracketed pairs count.
    Reported so the reader can see whether the staked 10% sits in the middle of a wide plateau or
    one hair from flipping the answer."""
    e = sorted(r["abs_rel_error"] for r in rows if r["bracketed"])
    if len(e) < gate:
        return dict(reachable=False, n_bracketed=len(e), lo=None, hi=None,
                    note=f"only {len(e)} of {len(rows)} pairs are bracketed at all; a gate of "
                         f"{gate} is UNREACHABLE at every tolerance")
    lo = e[gate - 1]
    hi = e[gate] if len(e) > gate else None
    return dict(reachable=True, n_bracketed=len(e), lo=round(lo, 5),
                hi=(round(hi, 5) if hi is not None else None),
                note=f"pass_count stays {gate} for any tolerance in ({lo:.2%}, "
                     f"{'inf' if hi is None else format(hi, '.2%')}]")


def _loss(pts, sc, si, mode, phi):
    tot = 0.0
    for A, F, ov, meas in pts:
        tot += math.log(1.0 / t_pred(A, F, ov, sc, si, mode, phi) / meas) ** 2
    return tot


def fit(pts, mode, phi_grid=11):
    """Deterministic: a fixed log grid, then a fixed coordinate descent.  No RNG, no scipy."""
    phis = [i / (phi_grid - 1) for i in range(phi_grid)] if mode == "M2b" else [0.0]
    best = None
    for a in range(46):
        sc = 2.0 * (1.08 ** a)                     # 2.0 .. 63 GB/s
        for b in range(46):
            si = 0.15 * (1.09 ** b)                # 0.15 .. 7.2 GB/s
            for ph in phis:
                L = _loss(pts, sc, si, mode, ph)
                if best is None or L < best[0]:
                    best = (L, sc, si, ph)
    L, sc, si, ph = best
    for _ in range(50):
        for scale in (1.02, 1.005, 1.001):
            for cand in (sc * scale, sc / scale):
                L2 = _loss(pts, cand, si, mode, ph)
                if L2 < L:
                    L, sc = L2, cand
            for cand in (si * scale, si / scale):
                L2 = _loss(pts, sc, cand, mode, ph)
                if L2 < L:
                    L, si = L2, cand
        if mode == "M2b":
            for d in (0.02, 0.005):
                for cand in (min(1.0, ph + d), max(0.0, ph - d)):
                    L2 = _loss(pts, sc, si, mode, cand)
                    if L2 < L:
                        L, ph = L2, cand
    return sc, si, ph


def med(xs):
    xs = sorted(xs)
    n = len(xs)
    if not n:
        return float("nan")
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2


def score_mode(rows, pts, mode):
    """In-sample, leave-one-ROW-out and leave-one-MODEL-out relative errors."""
    sc, si, ph = fit(pts, mode)
    ins = [abs(1.0 / t_pred(*pts[i][:3], sc, si, mode, ph) / pts[i][3] - 1.0)
           for i in range(len(pts))]
    loo = []
    for i in range(len(pts)):
        a, b, p = fit(pts[:i] + pts[i + 1:], mode)
        loo.append(abs(1.0 / t_pred(*pts[i][:3], a, b, mode, p) / pts[i][3] - 1.0))
    lomo, by_model = [], {}
    for m in FILES:
        tr = [pts[i] for i, r in enumerate(rows) if r[1] != m]
        te = [(i, pts[i]) for i, r in enumerate(rows) if r[1] == m]
        if not te or not tr:
            continue
        a, b, p = fit(tr, mode)
        e = [abs(1.0 / t_pred(*q[:3], a, b, mode, p) / q[3] - 1.0) for _i, q in te]
        by_model[m] = dict(median=med(e), max=max(e), n=len(e),
                           S_c=round(a, 4), S_i=round(b, 5), phi=round(p, 4))
        lomo += e
    return dict(S_c=sc, S_i=si, phi=ph,
                in_median=med(ins), in_max=max(ins),
                loo_median=med(loo), loo_max=max(loo),
                lomo_median=med(lomo), lomo_max=max(lomo), lomo_by_model=by_model,
                per_row_err=ins)


# ---------------------------------------------------------------------------------------
# stake block <-> pre-registration
# ---------------------------------------------------------------------------------------
STAKE_KEYS = (
    "p0_max_abs_rel_error", "p0_gate",
    "m2_lomo_median", "m2_lomo_max", "m2_loo_median",
    "n1_lomo_median", "n2_lomo_median", "n3_lomo_median", "n4_lomo_median",
    "m2b_lomo_median", "m2b_phi",
    "k1_lomo_median_max", "k1_lomo_max_max", "k2_min_beat_factor",
    "kpair_m2_pass_count", "kpair_n3_pass_count", "kpair_gate_count",
    "armc_beta", "armc_within_count", "armc_null_median", "armc_model_median",
    "verdict_m2_pass", "verdict_n3_pass",
)


def read_stake(path):
    if not os.path.isfile(path):
        raise Abort(f"pre-registration not found: {path}.  --score refuses to run without the "
                    "staked document; the whole point is that the numbers were fixed first.")
    m = re.search(r"```stake\s*\n(.*?)```", open(path, encoding="utf-8").read(), re.S)
    if not m:
        raise Abort(f"no ```stake block in {path}; nothing was staked machine-readably.")
    out = {}
    for line in m.group(1).splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        k, _, v = line.partition("=")
        try:
            out[k.strip()] = float(v.strip())
        except ValueError:
            raise Abort(f"stake line {line!r} in {path} is not `key = number`")
    miss = [k for k in STAKE_KEYS if k not in out]
    if miss:
        raise Abort(f"stake block in {path} is missing {miss}")
    return out


# ---------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stake", action="store_true",
                    help="compute and print the stake block ONLY; evaluate no kill rule")
    ap.add_argument("--prereg", default=PREREG)
    ap.add_argument("--gguf-dir", default=DEFAULT_GGUF_DIR,
                    help="folder of local .gguf files for exp51's mirror self-test")
    ap.add_argument("--refetch", action="store_true", help="ignore the cached GGUF headers")
    ap.add_argument("--offline", action="store_true",
                    help="skip re-verifying the transcription against the live README.  Header "
                         "sizing stays live on purpose.  In SCORE mode this makes the run "
                         "unpublishable: the verdict is stamped -UNVERIFIED-SOURCE and the exit "
                         "code is non-zero even if every gate is cleared.")
    a = ap.parse_args()

    os.makedirs(DATA, exist_ok=True)
    logpath = os.path.join(DATA, "exp53_two_resource_disk_tier.log")
    lines = []

    def log(s=""):
        print(s, flush=True)
        lines.append(s)

    t0 = time.time()
    log("=" * 100)
    log("EXPERIMENT #53 - two-resource model for the streaming/disk tier")
    log(f"held-out set: {SOURCE_REPO} README tables, read {SOURCE_READ_DATE}")
    log(f"mode: {'STAKE (no kill rule evaluated)' if a.stake else 'SCORE'}   "
        f"{time.strftime('%Y-%m-%d %H:%M:%S')}")
    log("=" * 100)

    try:
        quant_sizes, quant_enum, e51 = need_imports()
        import requests
        sess = requests.Session()
        sess.headers["User-Agent"] = "quantprobe-exp53/1.0"

        log("\n[1] preconditions")
        st = e51.selftest_mirror(a.gguf_dir, quant_sizes, quant_enum, log)
        tr = verify_transcription(log)
        src = verify_source(sess, log, a.offline)

        log("\n[2] the four benchmarked models, from their own GGUF headers")
        specs = load_specs(sess, e51, quant_sizes, quant_enum, a.refetch, log)

        log("\n[3] P-0  parameter-free byte accounting: our routed bytes/token vs THEIR "
            "instrumented no-cache Flash/token")
        p0 = []
        for m, k, pub in P0_ROWS:
            ours = routed_gb_per_token(specs[m], k) * 1e9 / MiB
            err = ours / pub - 1.0
            p0.append(dict(model=m, k=k, published_mib=pub, ours_mib=round(ours, 1),
                           rel_error=round(err, 5)))
            log(f"  {m:16s} k={k}  ours {ours:8.1f} MiB   theirs {pub:5d} MiB   {err:+7.2%}")
        p0_max = max(abs(x["rel_error"]) for x in p0)
        p0_ok = p0_max < P0_GATE
        log(f"  P-0: worst |error| {p0_max:.2%}  gate {P0_GATE:.0%}  -> "
            f"{'HOLDS' if p0_ok else 'FAILS'}")
        # qwen35moe has no no-cache row; report the back-derived number as a disclosure only.
        q_back = [(r[4] / (1 - r[7]) * (8 / r[2]), r[8], r[0]) for r in ROWS
                  if r[1] == "qwen36-35b-a3b" and r[7] is not None]
        q_ours = routed_gb_per_token(specs["qwen36-35b-a3b"], 8) * 1e9 / MiB
        q_lo, q_hi = min(x for x, _, _ in q_back), max(x for x, _, _ in q_back)
        log(f"  disclosure: qwen35moe has NO no-cache row.  Back-derived F/(1-hit) scaled to k=8 "
            f"gives {q_lo:.0f}-{q_hi:.0f} MiB against our {q_ours:.0f} MiB "
            f"({q_lo/q_ours - 1:+.0%} to {q_hi/q_ours - 1:+.0%}).  No kill power; see prereg "
            "section 7 defect 4.")
        log(f"  P-0 COVERAGE: {len(P0_ROWS)} comparisons over 3 of the 4 models.  The untested "
            "model is the one whose byte model visibly disagrees, and it carries 4 of the 18 "
            "fitted rows, 2 of the 5 k-pairs and the whole desktop arm.  P-0's 3% gate is "
            "therefore a statement about three models, not four.")
        # ADVERSARIAL REVIEW 2026-07-30: this disclosure existed only as a log line while every
        # passing number went to JSON.  A defect that is harder to machine-read than a success is
        # published at LOWER prominence than a hit, which the protocol forbids.  It is a
        # first-class field now.
        p0_disclosure = dict(
            model="qwen36-35b-a3b", reason="no no-cache row exists, so P-0 cannot test it",
            ours_mib=round(q_ours, 1), back_derived_mib=[round(x, 1) for x, _, _ in q_back],
            back_derived_range_mib=[round(q_lo, 1), round(q_hi, 1)],
            rel_error_range=[round(q_lo / q_ours - 1, 4), round(q_hi / q_ours - 1, 4)],
            devices=sorted({dev for _, _, dev in q_back}),
            would_fail_p0_gate=bool(min(abs(q_lo / q_ours - 1), abs(q_hi / q_ours - 1)) > P0_GATE),
            kill_power=False,
            carries="4 of 18 fitted rows, 2 of 5 k-pairs, and the entire desktop arm")

        log("\n[4] Arm A  fit the model form on the 18 phone rows "
            "(2 device constants, nothing per model, nothing per row)")
        phone = [r for r in ROWS if r[0] == "P"]
        results = {}
        for conv in CONVENTIONS:
            pts = [(active_gb(specs[r[1]], r[2], conv), r[4] * MiB / 1e9, r[5], r[3])
                   for r in phone]
            results[conv] = {}
            log(f"  --- active-byte convention: {conv}"
                f"{'  (PRIMARY - what quantprobe ships)' if conv == 'shipped' else ''}")
            for mode in MODES:
                res = score_mode(phone, pts, mode)
                results[conv][mode] = res
                log(f"    {mode:4s} S_c={res['S_c']:6.2f} GB/s  S_i={res['S_i']*1e9/MiB:7.1f} MiB/s"
                    f"{'  phi=%.3f' % res['phi'] if mode == 'M2b' else '        '}  |  "
                    f"in {res['in_median']:6.2%}  LOO {res['loo_median']:6.2%}  "
                    f"LOMO med {res['lomo_median']:6.2%} max {res['lomo_max']:6.2%}   "
                    f"{MODE_DOC[mode]}")

        prim = results["shipped"]
        log("\n[5] per-row, STAKED model M2 and the surviving alternative, primary convention")
        for mode in ("M2", "N3"):
            r0 = prim[mode]
            log(f"  --- {mode}  S_c={r0['S_c']:.3f} GB/s  S_i={r0['S_i']*1e9/MiB:.1f} MiB/s")
            for r in phone:
                A = active_gb(specs[r[1]], r[2], "shipped")
                F = r[4] * MiB / 1e9
                C, I = A / r0["S_c"], F / r0["S_i"]
                p = 1.0 / t_pred(A, F, r[5], r0["S_c"], r0["S_i"], mode, r0["phi"])
                log(f"    {r[1]:16s} {r[8]:22s} ov={int(r[5])} meas {r[3]:5.2f} pred {p:6.2f} "
                    f"err {p/r[3]-1:+7.1%}   C {C*1e3:6.1f} ms  I {I*1e3:6.1f} ms  "
                    f"BINDS: {'I/O' if I > C else 'COMPUTE'}")

        log("\n[6] Arm B  the five clean k -> k' pairs (does the prediction sit strictly between "
            "the two single-resource bounds, and within 10%?)")
        armb = {}
        for mode in ("M2", "N3", "M2b"):
            r0 = prim[mode]
            rowsb, npass = [], 0
            for (m, kh, kl, fh, fl, ov, meas) in KPAIRS:
                Ah, Al = active_gb(specs[m], kh, "shipped"), active_gb(specs[m], kl, "shipped")
                th = t_pred(Ah, fh * MiB / 1e9, ov, r0["S_c"], r0["S_i"], mode, r0["phi"])
                tl = t_pred(Al, fl * MiB / 1e9, ov, r0["S_c"], r0["S_i"], mode, r0["phi"])
                pred, lo, hi = th / tl, Ah / Al, fh / fl
                brack = min(lo, hi) < pred < max(lo, hi)
                forced = bracket_forced(mode, ov)
                close = abs(pred / meas - 1.0) < KPAIR_TOL
                npass += int(brack and close)
                rowsb.append(dict(model=m, k_hi=kh, k_lo=kl, measured=round(meas, 4),
                                  predicted=round(pred, 4), compute_only_bound=round(lo, 4),
                                  io_only_bound=round(hi, 4), bracketed=brack,
                                  bracket_forced=forced, within_tol=close,
                                  rel_error=round(pred / meas - 1, 5),
                                  abs_rel_error=abs(pred / meas - 1)))
                if mode == "M2" or mode == "N3":
                    log(f"    {mode:4s} {m:16s} k{kh}->k{kl}  measured {meas:.3f}  "
                        f"predicted {pred:.3f}  [compute-only {lo:.3f}, io-only {hi:.3f}]  "
                        f"bracketed {str(brack):5s}"
                        f"{' (FORCED - see below)' if forced else '              '} "
                        f"within-10% {str(close):5s}")
            nforced = sum(1 for r in rowsb if r["bracket_forced"])
            win = tol_window(rowsb, KPAIR_GATE_COUNT)
            armb[mode] = dict(rows=rowsb, pass_count=npass, bracket_forced_count=nforced,
                              bracket_informative_count=len(rowsb) - nforced, tol_window=win)
            log(f"    {mode:4s} -> {npass}/{len(KPAIRS)} pairs bracketed AND inside "
                f"{KPAIR_TOL:.0%} (gate: {KPAIR_GATE_COUNT})")
            if nforced:
                log(f"    {mode:4s}    DISCLOSURE: on {nforced}/{len(rowsb)} of those pairs this "
                    "form is additive, so `bracketed` is the mediant identity and CANNOT fail.  "
                    f"K-4 carries evidence on {len(rowsb) - nforced} pair(s); on the rest it "
                    "reduces to the 10% tolerance alone.")
            log(f"    {mode:4s}    threshold sensitivity: {win['note']}")

        # One beta for every model, PHONE ONLY.  The prereg prose said "and both devices"; the
        # code has always filtered to dev == "P" and must keep doing so - the desktop row
        # publishes `cache auto` with no realised size, and pooling two machines into one
        # constant is the cross-machine-state comparison C-14 forbids.  Prose corrected to match.
        log("\n[7] Arm C  cache hit as a MODELLED quantity: h = (M/N)^beta, one beta for every "
            "model, PHONE ONLY (pooling devices would violate C-14)")
        seen, crows = set(), []
        for r in ROWS:
            dev, m, k, _tps, _f, _ov, cache, hit, lab = r
            if dev != "P" or not cache or hit is None:
                continue
            key = (m, k, cache)
            if key in seen:
                continue
            seen.add(key)
            s = specs[m]
            slot_b = s["_exp_bytes"] / (s["n_layer"] * s["_n_exp"])
            crows.append(dict(model=m, k=k, cache_mib=cache, hit=hit,
                              frac=(cache * MiB / slot_b) / (s["n_layer"] * s["_n_exp"]),
                              slot_mib=slot_b / MiB, label=lab))
        best = None
        for i in range(1, 2001):
            beta = i / 2000.0
            e = med([abs(c["frac"] ** beta - c["hit"]) for c in crows])
            if best is None or e < best[0]:
                best = (e, beta)
        armc_med, beta = best
        null_med = med([abs(c["frac"] - c["hit"]) for c in crows])
        within = 0
        for c in crows:
            c["modelled"] = round(c["frac"] ** beta, 4)
            c["abs_error"] = round(abs(c["modelled"] - c["hit"]), 4)
            c["frac"] = round(c["frac"], 5)
            c["slot_mib"] = round(c["slot_mib"], 3)
            within += int(c["abs_error"] < ARMC_ABS_TOL)
            log(f"    {c['model']:16s} k={c['k']} cache {c['cache_mib']:5d} MiB  "
                f"expert slot {c['slot_mib']:6.2f} MiB  M/N {c['frac']:.4f}  "
                f"modelled h {c['modelled']:.3f}  measured {c['hit']:.2f}  "
                f"|err| {c['abs_error']:.3f}")
        log(f"    beta = {beta:.4f}   median |err| {armc_med:.4f} vs structural null (h=M/N) "
            f"{null_med:.4f}   within {ARMC_ABS_TOL:.2f}: {within}/{len(crows)}")
        armc_ok = (armc_med < null_med) and (within * 2 > len(crows))
        log(f"    Arm C {'HOLDS' if armc_ok else 'FAILS'} "
            "(must beat the structural null AND land inside 0.10 on a majority of rows)")

        log("\n[8] Desktop  DISCLOSURE ONLY - one usable row cannot support two device constants")
        d = [r for r in ROWS if r[0] == "D"][0]
        Ad = active_gb(specs["qwen36-35b-a3b"], 8, "shipped")
        C51 = Ad / (ETA_R_MOE * DESKTOP_RAM_BW_GBS)
        Id = 1.0 / d[3] - C51
        # ADVERSARIAL REVIEW 2026-07-30: `implied_read_gbs = flash / Id` was computed and printed
        # with NO check that Id > 0.  If our compute term ever exceeds the measured token time -
        # which is precisely what happens when the model, the machine or eta is wrong - Id goes
        # negative and this arm prints a negative or near-infinite "measured" read rate with the
        # same confident formatting as a valid one.  That is the shape of the artifact this
        # project nearly published once already.  It refuses now.
        io_share = Id / (1.0 / d[3])
        if Id <= 0 or io_share < 0.05:
            raise Abort(
                f"desktop disclosure arm: the compute term ({C51*1e3:.1f} ms/token) leaves "
                f"{Id*1e3:+.1f} ms of the {1000/d[3]:.1f} ms measured token for I/O "
                f"({io_share:+.1%} of it).  An implied read rate divided out of a residual that "
                "small - or negative - is an artifact, not a measurement, and this arm has no "
                "kill power to justify printing one anyway.  Refusing to emit a number.")
        log(f"    #51's staked compute term      {C51*1e3:7.1f} ms/token "
            f"({1/C51:.2f} tok/s ceiling; their own prose says ~{DESKTOP_THEIR_COMPUTE_MS:.0f} ms "
            "- '~0.11 s/token', verified in the transcription)")
        log(f"    measured                       {1000/d[3]:7.1f} ms/token at {d[3]} tok/s, "
            f"{d[4]} MiB/token")
        log(f"    implied desktop read rate      {d[4]*MiB/1e9/Id:7.3f} GB/s over the "
            f"{Id*1e3:.1f} ms ({io_share:.0%} of the token) the additive model leaves for I/O")
        at_nvme = 1 / (C51 + d[4] * MiB / 1e9 / DESKTOP_NVME_GBS)
        log(f"    their prose calls the NVMe ~{DESKTOP_NVME_GBS:.0f} GB/s.  At that rate the "
            f"additive model predicts {at_nvme:.2f} tok/s against {d[3]} measured "
            f"({at_nvme/d[3]-1:+.0%}).")
        ab = DESKTOP_OVERLAP_AB[1] / DESKTOP_OVERLAP_AB[0]
        log(f"    their --overlap A/B is {DESKTOP_OVERLAP_AB[0]} -> {DESKTOP_OVERLAP_AB[1]} tok/s "
            f"= x{ab:.3f}.  A perfect max() would be x(1+min/max) >= x1.0 and typically x1.2-x1.5 "
            "here.")
        log("    CAVEAT on that A/B: both of its rows are `--drop-cold-experts 0.75` rows, which "
            "section 2 EXCLUDES from every fit because the authors call their output not "
            "reproducible.  It is the only overlap A/B the desktop has; it is quoted as an "
            "indication and must not be read as a measurement.")
        desktop = dict(row=d[8], measured_tok_s=d[3], flash_mib=d[4],
                       compute_s_51=round(C51, 6), implied_io_s=round(Id, 6),
                       io_share_of_token=round(io_share, 4),
                       implied_read_gbs=round(d[4] * MiB / 1e9 / Id, 4),
                       at_nvme_prose_tok_s=round(at_nvme, 4),
                       overlap_ab_measured=round(ab, 4),
                       overlap_ab_from_excluded_rows=True,
                       ram_bw_gbs_is_ours_not_theirs=DESKTOP_RAM_BW_GBS)

        fresh = dict(
            p0_max_abs_rel_error=p0_max, p0_gate=P0_GATE,
            m2_lomo_median=prim["M2"]["lomo_median"], m2_lomo_max=prim["M2"]["lomo_max"],
            m2_loo_median=prim["M2"]["loo_median"],
            n1_lomo_median=prim["N1"]["lomo_median"], n2_lomo_median=prim["N2"]["lomo_median"],
            n3_lomo_median=prim["N3"]["lomo_median"], n4_lomo_median=prim["N4"]["lomo_median"],
            m2b_lomo_median=prim["M2b"]["lomo_median"], m2b_phi=prim["M2b"]["phi"],
            k1_lomo_median_max=K1_LOMO_MEDIAN_MAX, k1_lomo_max_max=K1_LOMO_MAX_MAX,
            k2_min_beat_factor=K2_MIN_BEAT_FACTOR,
            kpair_m2_pass_count=float(armb["M2"]["pass_count"]),
            kpair_n3_pass_count=float(armb["N3"]["pass_count"]),
            kpair_gate_count=float(KPAIR_GATE_COUNT),
            armc_beta=beta, armc_within_count=float(within),
            armc_null_median=null_med, armc_model_median=armc_med,
        )

        def gates(mode, tag, preselected_as_best):
            """P-0 is a conjunct of EVERY gate set.

            It used to gate only M2, while the fallback N3 - the form that actually gets wired in
            when M2 dies, which is the outcome that was staked - was scored on Arm A and Arm B
            alone.  A P-0 failure would then have killed a model that was already dead and left
            the survivor untouched, which is the exact opposite of what prereg section 8 says
            ("nothing downstream of it may be wired in whatever Arm A says").  Arm A takes `I`
            from their flash counter, but `C` is OUR byte model, and P-0 is the only test that
            our byte model describes their engine at all.

            `preselected_as_best` records that K-3 carries no information for this mode: the
            pre-declared fallback was named as the best same-cost form, and K-3 asks whether it is
            the best same-cost form.  It cannot fail for the fallback by construction.  The flag
            does not change the verdict - retuning a gate after seeing the numbers is what this
            protocol forbids - it makes the gate's emptiness part of the published record."""
            r0, best1 = prim[mode], min(prim["N1"]["lomo_median"], prim["N2"]["lomo_median"])
            k1 = (r0["lomo_median"] < K1_LOMO_MEDIAN_MAX) and (r0["lomo_max"] < K1_LOMO_MAX_MAX)
            k2 = r0["lomo_median"] * K2_MIN_BEAT_FACTOR < best1
            rivals = [prim[x]["lomo_median"] for x in ("M2", "N3", "N4") if x != mode]
            k3 = r0["lomo_median"] <= min(rivals) + 1e-12
            k4 = armb[mode]["pass_count"] >= KPAIR_GATE_COUNT if mode in armb else False
            b = armb.get(mode, {})
            return dict(tag=tag, P0_byte_model=p0_ok, K1_absolute=k1, K2_beats_one_resource=k2,
                        K3_best_at_its_cost=k3, K4_kpair=k4,
                        K3_informative=not preselected_as_best,
                        K4_bracket_forced_on=b.get("bracket_forced_count"),
                        K4_bracket_informative_on=b.get("bracket_informative_count"),
                        K4_tol_window=b.get("tol_window"),
                        PASS=bool(p0_ok and k1 and k2 and k3 and k4))

        # M2 was named before the fit; N3 was named as the fallback KNOWING it was the best
        # same-cost form (prereg section 0 states the arithmetic was run before the document).
        g_m2 = gates("M2", "staked model", preselected_as_best=False)
        g_n3 = gates("N3", "declared alternative", preselected_as_best=True)
        fresh["verdict_m2_pass"] = float(g_m2["PASS"])
        fresh["verdict_n3_pass"] = float(g_n3["PASS"])

        payload = dict(
            experiment=53, kind="external-retrodiction + model selection",
            when=time.strftime("%Y-%m-%dT%H:%M:%S"),
            source=dict(repo=SOURCE_REPO, readme=SOURCE_README, read=SOURCE_READ_DATE,
                        transcription=tr, live_check=src),
            constants=dict(act_mult=ACT_MULT, attn_bit_floor=ATTN_BIT_FLOOR, eta_r_moe=ETA_R_MOE,
                           p0_gate=P0_GATE, kpair_tol=KPAIR_TOL,
                           kpair_gate_count=KPAIR_GATE_COUNT, armc_abs_tol=ARMC_ABS_TOL,
                           k1_lomo_median_max=K1_LOMO_MEDIAN_MAX,
                           k1_lomo_max_max=K1_LOMO_MAX_MAX,
                           k2_min_beat_factor=K2_MIN_BEAT_FACTOR,
                           desktop_ram_bw_gbs_OURS=DESKTOP_RAM_BW_GBS,
                           desktop_nvme_gbs_THEIR_PROSE=DESKTOP_NVME_GBS,
                           desktop_their_compute_ms_THEIR_PROSE=DESKTOP_THEIR_COMPUTE_MS),
            selftest=st,
            models={k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
                    for k, v in specs.items()},
            models_raw={k: {kk: vv for kk, vv in v.items() if kk.startswith("_")}
                        for k, v in specs.items()},
            rows=[dict(device=r[0], model=r[1], k=r[2], tok_s=r[3], flash_mib=r[4],
                       overlap=r[5], cache_mib=r[6], cache_hit=r[7], label=r[8]) for r in ROWS],
            P0=dict(rows=p0, worst_abs_rel_error=round(p0_max, 5), gate=P0_GATE, holds=p0_ok,
                    models_tested=sorted({r["model"] for r in p0}),
                    models_untested=sorted(set(FILES) - {r["model"] for r in p0}),
                    untested_disclosure=p0_disclosure),
            armA={c: {m: {k: v for k, v in r.items() if k != "per_row_err"}
                      for m, r in results[c].items()} for c in results},
            armB=armb, armC=dict(beta=round(beta, 5), rows=crows, within=within,
                                 n=len(crows), null_median=round(null_med, 5),
                                 model_median=round(armc_med, 5), holds=armc_ok),
            desktop=desktop, fresh=fresh, gates=dict(M2=g_m2, N3=g_n3),
        )

        if a.stake:
            log("\n[9] STAKE MODE - no kill rule is evaluated here.")
            log("    Paste this block into the pre-registration, then run without --stake:\n")
            block = ["```stake"] + [f"{k:24s} = {fresh[k]:.4f}" for k in STAKE_KEYS] + ["```"]
            log("\n".join(block))
            payload["mode"] = "stake"
            out = os.path.join(DATA, "exp53_stake.json")
            with open(out, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=1)
            log(f"\n  wrote {out}")
            log(f"  elapsed {time.time() - t0:.1f}s")
            return 0

        log("\n[9] re-read the staked pre-registration and check it still reproduces")
        stake = read_stake(a.prereg)
        # ADVERSARIAL REVIEW 2026-07-30: this was `5e-3 * max(1.0, |value|)`, i.e. an ABSOLUTE
        # 0.005 for every staked value below 1.0, while the pre-registration advertised "0.5%".
        # On n3_lomo_median = 0.0404 that permitted +-12.4% of drift before the run would notice -
        # a reproduction check loose enough to absorb the very movement it exists to catch.  It is
        # relative now, with an absolute floor at half the last printed digit so that the 4-decimal
        # stake block cannot fail against its own rounding.
        def stake_tol(v):
            return max(5e-5, 5e-3 * abs(v))

        drift = {k: dict(staked=stake[k], fresh=round(fresh[k], 6),
                         tol=round(stake_tol(fresh[k]), 6)) for k in STAKE_KEYS
                 if abs(stake[k] - fresh[k]) > stake_tol(fresh[k])}
        if drift:
            raise Abort("the fresh computation no longer reproduces the staked numbers "
                        f"{drift}.  Either the code or the source moved after staking; the "
                        "comparison is void until the prereg is re-staked and re-dated.")
        log(f"  all {len(STAKE_KEYS)} staked values reproduced to 0.5% relative")

        log("\n[10] THE KILL RULE")
        best1 = min(prim["N1"]["lomo_median"], prim["N2"]["lomo_median"])
        for g, mode in ((g_m2, "M2"), (g_n3, "N3")):
            name = "M2 (STAKED)" if mode == "M2" else "N3 (declared alternative)"
            r0 = prim[mode]
            log(f"  {name}")
            log(f"    P-0 byte model reproduces their own flash counter (worst {p0_max:.2%} < "
                f"{P0_GATE:.0%})   -> {'hit' if g['P0_byte_model'] else 'MISS'}")
            log(f"    K-1 LOMO median {r0['lomo_median']:.2%} < {K1_LOMO_MEDIAN_MAX:.0%} and max "
                f"{r0['lomo_max']:.2%} < {K1_LOMO_MAX_MAX:.0%}          -> "
                f"{'hit' if g['K1_absolute'] else 'MISS'}")
            log(f"    K-2 LOMO median x{K2_MIN_BEAT_FACTOR:.0f} = "
                f"{r0['lomo_median']*K2_MIN_BEAT_FACTOR:.2%} < best "
                f"one-resource {best1:.2%}      -> "
                f"{'hit' if g['K2_beats_one_resource'] else 'MISS'}")
            log(f"    K-3 no same-cost rival predicts held-out models better              -> "
                f"{'hit' if g['K3_best_at_its_cost'] else 'MISS'}"
                f"{'' if g['K3_informative'] else '   [CARRIES NO EVIDENCE: this form was named '
                                                 'as the best same-cost form, so K-3 asks a '
                                                 'question it was selected to answer]'}")
            log(f"    K-4 k-pair bracketing {armb[mode]['pass_count']}/5 "
                f">= {KPAIR_GATE_COUNT}                                        -> "
                f"{'hit' if g['K4_kpair'] else 'MISS'}")
            if g["K4_bracket_forced_on"]:
                log(f"        of which `bracketed` is an algebraic identity on "
                    f"{g['K4_bracket_forced_on']}/{len(KPAIRS)} pairs (additive form), leaving "
                    f"{g['K4_bracket_informative_on']} informative; {g['K4_tol_window']['note']}")
            log(f"    => {'PASS' if g['PASS'] else 'FAIL'}")
        if not p0_ok:
            log("  P-0 FAILED: the byte model does not reproduce their own counter.  Nothing "
                "downstream of it may be wired in, whatever Arm A says - and that now includes "
                "the pre-declared fallback, which used to escape this gate entirely.")

        verdict = "PASS" if g_m2["PASS"] else "FAIL"
        # An unverified source cannot authorise wiring anything in.  --offline is legitimate for
        # re-deriving numbers offline, but it is not a scoring run: the transcription is the only
        # thing standing between a citation error and a published finding.
        if not src.get("verified"):
            verdict = f"{verdict}-UNVERIFIED-SOURCE"
        publishable = bool(src.get("verified"))
        payload.update(mode="score", stake=stake, verdict=verdict, publishable=publishable)
        log("")
        log("-" * 100)
        if not publishable:
            log("  THIS RUN IS NOT PUBLISHABLE: --offline skipped the live source check, so "
                "nothing below was scored against a source anyone re-read.  Re-run with network.")
        log(f"  {verdict}: the STAKED two-resource form (max under --overlap, sum without) "
            f"{'clears' if g_m2['PASS'] else 'does NOT clear'} its kill rule.")
        if not g_m2["PASS"]:
            log("  KILL RULE FIRED.  M2 does NOT get wired into quantprobe.  Per prereg section 5, "
                "the pre-declared fallback is evaluated next and published either way:")
            log(f"    two-resource ACCOUNTING survives: {'YES' if g_n3['PASS'] else 'NO'} "
                f"(N3 = C+I always, LOMO median {prim['N3']['lomo_median']:.2%} against "
                f"{best1:.2%} for the best one-resource null)")
            log("    the part that dies is the COMBINE RULE: --overlap does not turn sum into "
                f"max.  Fitting a partial-overlap credit gives phi={prim['M2b']['phi']:.3f} and "
                f"still does not beat phi=0 held-out "
                f"({prim['M2b']['lomo_median']:.2%} vs {prim['N3']['lomo_median']:.2%}).")
            if g_n3["PASS"]:
                log("    WHAT THAT FALLBACK PASS IS AND IS NOT (adversarial review 2026-07-30):")
                log("      - N3 was named as the fallback AFTER the arithmetic had been run "
                    "(prereg section 0), so its pass is a SELECTION, not a prediction.  Two of "
                    "its four gates cannot fail for it: K-3 asks whether it is the best "
                    "same-cost form, which is why it was chosen, and K-4's bracketing conjunct "
                    "is the mediant identity for an additive form.")
                log(f"      - What survives that is real: K-1 (LOMO max "
                    f"{prim['N3']['lomo_max']:.2%} < {K1_LOMO_MAX_MAX:.0%}), K-2 "
                    f"({prim['N3']['lomo_median']*K2_MIN_BEAT_FACTOR:.2%} < {best1:.2%}), and "
                    f"K-4's tolerance half, which N3 clears on "
                    f"{armb['N3']['pass_count']}/{len(KPAIRS)} pairs - "
                    f"{armb['N3']['tol_window']['note']}.")
                log("      - Wiring N3 in therefore still requires the separate commit prereg "
                    "section 5 demands, and inherits every caveat in sections 7 and 8.")
        log("-" * 100)

        out = os.path.join(DATA, "exp53_two_resource_disk_tier.json")
        with open(out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=1)
        log(f"\n  wrote {out}")
        log(f"  elapsed {time.time() - t0:.1f}s")
        return 0 if (g_m2["PASS"] and publishable) else 1

    except Abort as e:
        log("")
        log("!" * 100)
        log(f"ABORTED - precondition failed, NO number produced:\n  {e}")
        log("!" * 100)
        return 2
    finally:
        with open(logpath, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print(f"[log] {logpath}")


if __name__ == "__main__":
    sys.exit(main())
