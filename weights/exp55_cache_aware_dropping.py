r"""exp55_cache_aware_dropping.py -- pre-registration #90 (task #55).

Ports BigMoeOnEdge's --drop-cold-experts lever to our GPU/CPU split and measures the thing they
could not: the QUALITY cost of dropping routed experts, with KL divergence and top-token
agreement over thousands of tokens rather than 15 GSM8K questions.

Staked in preregistrations/2026-07-30-cache-aware-dropping.md BEFORE any run.  Read that first;
this file only executes what it stakes.

------------------------------------------------------------------------------------------
THE STRUCTURAL PORT (prereg section 3) -- why a stock binary can bound a feature nobody wrote
------------------------------------------------------------------------------------------
llama.cpp fuses a layer's experts into ONE tensor (blk.N.ffn_{gate,up,down}_exps.weight, last
dim = n_expert), and -ot matches tensor NAMES.  So on our split:

    "fetching this expert costs an I/O read"  ==  "this layer's expert tensor is CPU-resident"

which is STATIC and LAYER-GRANULAR.  Two consequences:

  (A) Our port is REPRODUCIBLE.  Their caveat -- output depends on cache state -- does not
      transfer, because our residency is a load-time placement, not a cache.

  (B) THE BOUNDING LEMMA.  Dropping on a GPU-resident layer costs quality and buys nothing.  So
      at matched speed, placement-aware dropping perturbs a SUBSET of the layers that uniform
      dropping perturbs, and uniform dropping's quality cost UPPER-BOUNDS the port's.

And `--override-kv <arch>.expert_used_count=int:k'` is a genuine member of the drop-rule family:
it drops exactly the experts ranked k'+1..8 BY ROUTER WEIGHT, per token (the rank-threshold
parameterization of their value-threshold F).  Zero code, stock binary, available today.

That is what this script measures.  KR-4 tests the lemma's monotonicity premise rather than
assuming it.

------------------------------------------------------------------------------------------
WHAT THIS IS NOT
------------------------------------------------------------------------------------------
NOT a timing measurement.  Perplexity, KL and top-token agreement are functions of (model,
tokens, routing width), not of clocks or thermals, so C-14 does not bind and no cal_id is
required.  The C-14 ANALOGUE that does bind: base and test logits must come from the same
backend, or the KL measures reduction-order noise instead of the drop.  -ngl is therefore
stamped into the base-logits filename and a mismatch refuses to run.

The speed side (Arm B) is ARITHMETIC OVER GGUF HEADERS, computed before the prereg was written
and labelled a PREDICTION everywhere it appears.  No tok/s is measured here.

Outputs (all under weights/data/):
  exp55_cache_aware_dropping.json    machine-readable results + verdict
  exp55_cache_aware_dropping.log     human-readable transcript
  exp55_runs/<model>__<domain>__k<K>__ngl<N>.json    per-run parsed metrics (idempotency keys)
  exp55_runs/<...>.stdout.txt                        raw captured output of every run

Base logits are large (up to ~4 GB each) and go to --logits-dir, default
D:\evo-compress-data\exp55_logits.  Free space is checked BEFORE the first run.

Run:
  python weights/exp55_cache_aware_dropping.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time

# ======================================================================================
# STAKED CONSTANTS -- these mirror pre-registration #90 and must not be edited after staking.
# ======================================================================================

PREREG = "preregistrations/2026-07-30-cache-aware-dropping.md"
PREREG_ID = 90
TASK_ID = 55

# The routing widths under test.  8 is the models' native width and is the KL base.
K_BASE = 8
K_GRID = (7, 6, 5, 4)

# The drop level the staked speed envelope needs (prereg section 5).  KR-1 is evaluated here.
K_HEADLINE = 6

# --- the bar (prereg section 7).  Set before any number was seen. ---
# Top-1 agreement: at 1% argmax disagreement a 500-token completion changes ~5 tokens.  For code
# that is a broken build.  Anything worse cannot ship as a DEFAULT.
GATE_TOP1_MIN = 0.99
# Mean KL divergence, nats.  Secondary gate: catches distribution shift that survives an
# unchanged argmax.
GATE_MEAN_KLD_MAX = 0.05
# Computed speedup required to justify a runtime change.  It rejects the adjacent option:
# k'=7 fails on both models (1.1282x, 1.0963x), k'=6 passes on the primary (1.2940x).
#
# HONESTY NOTE (added 2026-07-30 by adversarial review, BEFORE any Arm A run).  This gate has
# NO KILL POWER OF ITS OWN.  Arm B is deterministic arithmetic over GGUF headers that was
# computed before the prereg was written (prereg section 0.1), so every number it is compared
# against was already known at staking time, and k'=6/5/4 clear 1.20x by construction.  Under
# KR-4 monotonicity KR-2 therefore reduces to KR-1: the only thing that can make KR-2 fire is
# a QUALITY failure.  Recorded in the JSON as kr2_speed_gate_kill_power = false so that KR-2
# can never be presented as a second, independent hurdle that the lever cleared.
GATE_SPEEDUP_MIN = 1.20

# KR-7 marginality bands.  Inside these the verdict is MARGINAL, never PASS/FAIL.
MARGIN_TOP1 = 0.002
MARGIN_KLD = 0.005

# Terminal verdicts -- a scored outcome, as opposed to VOID/REFUSED/DRY-RUN.  Note there is no
# 'PASS': the all-gates-clear branch is KR-3, and KR-3 refutes the runtime patch.  See
# verdict_for's docstring.
TERMINAL_VERDICTS = ("REFUTED-AS-UNNECESSARY", "REFUTED-LEVER", "MARGINAL", "MARGINAL-UNSOUND",
                     "UNSOUND")
VERDICT_SENTENCE = {
    "REFUTED-AS-UNNECESSARY":
        "KR-3 fired.  Quality and computed speed clear the bar, so the win is already available "
        "from a stock flag and THE RUNTIME PATCH IS REFUTED AS UNNECESSARY.  Ship documentation "
        "of the flag with its measured quality cost.  Write no cache-aware runtime code.",
    "REFUTED-LEVER":
        "KR-2 fired.  No routing width is both affordable in quality and fast enough.  THE LEVER "
        "IS REFUTED for our split.  Build nothing; keep full routing width.  Publish as a miss at "
        "equal prominence with any hit.",
    "MARGINAL":
        "KR-7 fired.  The headline sits inside the marginality band.  This is NOT a result: "
        "re-stake with more chunks.",
    "MARGINAL-UNSOUND":
        "KR-7 and KR-4 both fired.  Marginal quality AND the bounding lemma's monotonicity "
        "premise fails in our own data.",
    "UNSOUND":
        "KR-4 fired.  Mean KLD is not monotone in drop depth, so section 3.3's dominance argument "
        "is unsound and nothing here may be carried into Stage 2.",
    "VOID": "KR-5/KR-6.  Instrument or knobs, not a result.  A void is not a failure.",
    "REFUSED": "A precondition was missing.  Nothing was measured.",
}

# Budget: equal token count per domain so the comparison is not confounded (prereg section 6).
CTX = 512
CHUNKS = 12
# Staked backend placement (prereg section 6).  -ngl 0 is pure CPU, matching #87.  Both are part
# of the headline test: base and test logits must come from the same backend, and thread count
# changes the CPU reduction order, so a run that moves either is a sensitivity run, not the
# headline.
STAKED_NGL = 0
STAKED_THREADS = 0      # 0 = let llama.cpp choose; the point is that base and test agree

# Corpus shards, SHA-256-pinned.  Inherited from prereg #87 / exp52; re-checked at runtime.
# Scoring uses the B shards so #52's ranking surface (A) stays disjoint from ours.
CORPUS_SHA256 = {
    "code_B": "dccbb1bd4af5b47b363d6cce7090bbebd73c9cf112249583f92a98618f5a1593",
    "prose_B": "5fb4087ab6b5c048366614e0525127f3adb70055eb1efeead2e9eb9ac785d304",
}
DOMAINS = ("code_B", "prose_B")

# Model arms.  'primary' carries KR-2's kill power; the replication arm's speed envelope
# straddles the bar (prereg section 5) so it is reported without kill power.
MODELS = {
    "qwen3-30b-a3b-q2k": {
        "path": r"D:\evo-compress-data\gguf\Qwen3-30B-A3B-Q2_K.gguf",
        "arch": "qwen3moe",
        "n_layer": 48,
        "n_expert": 128,
        "n_expert_used": 8,
        "role": "primary",
    },
    "q35-A-shexp": {
        "path": r"D:\evo-compress-data\gguf\q35-A-shexp.gguf",
        "arch": "qwen35moe",
        "n_layer": 40,
        "n_expert": 256,
        "n_expert_used": 8,
        "role": "replication",
    },
}
PRIMARY_MODEL = "qwen3-30b-a3b-q2k"

# Arm B constants.  CALIBRATION INPUTS, not predictions -- see prereg section 5 and #89's
# standing warning that fitted device rates must never be presented as prophecy.
BW_GPU_EFF = 130e9      # GTX 1060 6GB, effective
BW_CPU_EFF = 15.3e9     # DDR4-3200 at the CPU-tier eta
GPU_EXPERT_SHARE = 0.32  # g: share of expert layers the shipped -ot regex keeps resident

DEFAULT_PPL_BIN = r"<repo>\tools\llamacpp-b10098\llama-perplexity.exe"
DEFAULT_LOGITS_DIR = r"D:\evo-compress-data\exp55_logits"

# The build the prereg stakes.  Any other binary is a DIFFERENT INSTRUMENT and the run stops
# being the headline (KR-6).  There is deliberately NO silent fall back to whatever
# llama-perplexity happens to sit on PATH -- see the 2026-07-30 amendment in the prereg.
STAKED_PPL_BIN = DEFAULT_PPL_BIN

# Logging flags every invocation MUST carry.
#
#   -v                   At this build's DEFAULT verbosity the model-info banner is NOT emitted:
#                        a stock base run prints twelve lines and `print_info: n_expert_used`
#                        is not one of them.  KR-5 requires that line as its proof that
#                        --override-kv took effect, so WITHOUT -v every single run of this
#                        script refuses and no kill rule can ever fire.  Verified empirically
#                        on 2026-07-30 (adversarial review) against llamacpp-b10098.
#   --no-log-prefix
#   --no-log-timestamps  The logger injects `<timestamp> I ` mid-line when a message is emitted
#                        through more than one LOG call -- observed splitting the
#                        `validate_override:` line in half.  Turning the prefix off makes the
#                        parsed surface stable instead of relying on that never happening to a
#                        line we parse.
PPL_LOG_FLAGS = ("-v", "--no-log-prefix", "--no-log-timestamps")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, "weights", "data")
RUNS_DIR = os.path.join(DATA, "exp55_runs")
JSON_OUT = os.path.join(DATA, "exp55_cache_aware_dropping.json")
LOG_OUT = os.path.join(DATA, "exp55_cache_aware_dropping.log")

# The corpus shards are emitted by exp52.  We do not re-emit them: one generator, one hash.
CORPUS_PATH = os.path.join(DATA, "exp55_corpus_{domain}.txt")
EXP52_CORPUS_PATH = os.path.join(DATA, "exp52_corpus_{domain}.txt")

# Conservative upper bound on the base-logits file: 4 bytes per vocab entry per token.  The real
# format may be smaller; over-reserving is the safe direction.
LOGITS_BYTES_PER_VOCAB_TOKEN = 4
DISK_HEADROOM_BYTES = 2 * 1024 ** 3


class Refuse(Exception):
    """A precondition is missing.  We stop loudly rather than emit a wrong number."""


# ======================================================================================
# small helpers
# ======================================================================================

def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def log_line(handle, text: str = "") -> None:
    print(text, flush=True)
    if handle is not None:
        handle.write(text + "\n")
        handle.flush()


def gb(n: float) -> str:
    return f"{n / 1e9:.2f} GB"


def _num(text):
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def build_id(binary: str) -> str:
    """Short identity of the INSTRUMENT (exe + its impl/llama DLLs).

    Base logits and test logits must come from the same build or the KL measures build drift
    instead of the drop.  This tag goes into the base-logits filename and into every run
    record, and a cached run from a different build is never reused.
    """
    h = hashlib.sha256()
    folder = os.path.dirname(os.path.abspath(binary))
    parts = [os.path.abspath(binary)]
    for name in ("llama-perplexity-impl.dll", "llama.dll", "ggml-base.dll"):
        candidate = os.path.join(folder, name)
        if os.path.isfile(candidate):
            parts.append(candidate)
    for path in parts:
        stat = os.stat(path)
        h.update(os.path.basename(path).encode())
        h.update(str(stat.st_size).encode())
        h.update(str(int(stat.st_mtime)).encode())
    return h.hexdigest()[:10]


# ======================================================================================
# Arm 0 -- the structural check.  Reads the GGUF header and asserts the facts the port rests on.
# ======================================================================================

def model_facts(path: str, staked: dict) -> dict:
    """Read arch/expert/layout facts and ABORT if they disagree with what the prereg staked.

    The fusion check is the one that matters: if experts were NOT fused per layer, -ot could
    address individual experts, the miss predicate would stop being layer-granular, and the
    entire design in prereg section 3 would be the wrong one for this model.
    """
    from gguf import GGUFReader

    reader = GGUFReader(path)

    def kv(key):
        field = reader.fields.get(key)
        if field is None:
            return None
        try:
            part = field.parts[field.data[0]]
            if field.types and int(field.types[0]) == 8:      # GGUF string
                return bytes(part).decode("utf-8", "replace")
            return part.tolist()[0] if hasattr(part, "tolist") else part
        except Exception:
            return None

    arch = kv("general.architecture")
    if arch != staked["arch"]:
        raise Refuse(
            f"architecture mismatch for {os.path.basename(path)}: header says {arch!r}, "
            f"prereg #90 staked {staked['arch']!r}.  Refusing: the staked model is not this file."
        )

    facts = {
        "arch": arch,
        "n_layer": kv(f"{arch}.block_count"),
        "n_expert": kv(f"{arch}.expert_count"),
        "n_expert_used": kv(f"{arch}.expert_used_count"),
    }
    for key in ("n_layer", "n_expert", "n_expert_used"):
        got, want = facts[key], staked[key]
        if got is None:
            raise Refuse(f"{os.path.basename(path)}: header does not expose {key}.")
        if int(got) != int(want):
            raise Refuse(
                f"{os.path.basename(path)}: {key} = {got}, prereg #90 staked {want}.  Refusing "
                "rather than measure a model the document does not describe."
            )

    # vocab size, for the logits-file budget
    n_vocab = None
    for key in ("tokenizer.ggml.tokens",):
        field = reader.fields.get(key)
        if field is not None:
            try:
                n_vocab = len(field.data)
            except Exception:
                pass
    if not n_vocab:
        raise Refuse(f"{os.path.basename(path)}: cannot determine vocab size; disk budget unknown.")
    facts["n_vocab"] = int(n_vocab)

    # --- the fusion check (prereg section 2 / 3.1) ---
    per_layer_exps, expert_bytes, other_bytes = {}, 0, 0
    for tensor in reader.tensors:
        name, size = tensor.name, int(tensor.n_bytes)
        if "_exps" in name:
            expert_bytes += size
            match = re.match(r"^blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight$", name)
            if not match:
                raise Refuse(
                    f"{os.path.basename(path)}: unexpected expert tensor name {name!r}.  The port "
                    "assumes one fused tensor per layer per projection; refusing."
                )
            per_layer_exps.setdefault(int(match.group(1)), set()).add(match.group(2))
            if int(tensor.shape[-1]) != int(facts["n_expert"]):
                raise Refuse(
                    f"{os.path.basename(path)}: {name} last dim = {int(tensor.shape[-1])}, expected "
                    f"n_expert = {facts['n_expert']}.  Experts are NOT fused as the port assumes; "
                    "prereg section 3.1 does not hold for this file."
                )
        elif "token_embd" in name or name == "output.weight":
            continue
        else:
            other_bytes += size

    n_moe_layers = len(per_layer_exps)
    if n_moe_layers == 0:
        raise Refuse(f"{os.path.basename(path)}: no fused expert tensors found -- not an MoE file.")
    incomplete = [ly for ly, projs in per_layer_exps.items() if projs != {"gate", "up", "down"}]
    if incomplete:
        raise Refuse(
            f"{os.path.basename(path)}: layers {sorted(incomplete)[:5]} do not carry all three "
            "fused expert projections.  Refusing: the byte model would be wrong."
        )

    facts["n_moe_layer"] = n_moe_layers
    facts["expert_bytes"] = expert_bytes
    facts["other_bytes"] = other_bytes
    facts["experts_fused_per_layer"] = True
    facts["has_shared_experts"] = any("shexp" in t.name for t in reader.tensors)
    return facts


# ======================================================================================
# Arm B -- the speed envelope.  COMPUTED, NOT MEASURED (prereg section 5).
# ======================================================================================

def speed_envelope(facts: dict, g: float = GPU_EXPERT_SHARE) -> dict:
    """Two-tier byte model.  Returns {k': speedup vs k=K_BASE}.  A PREDICTION, never a claim."""
    expert_bytes = facts["expert_bytes"]
    other_bytes = facts["other_bytes"]
    n_expert = facts["n_expert"]

    def t_token(k):
        routed = expert_bytes * k / n_expert
        return (routed * g + other_bytes) / BW_GPU_EFF + routed * (1.0 - g) / BW_CPU_EFF

    base = t_token(K_BASE)
    return {k: base / t_token(k) for k in (K_BASE,) + K_GRID}


# ======================================================================================
# the instrument -- stock llama-perplexity
# ======================================================================================

# Format strings read out of llama-perplexity-impl.dll / llama.dll (prereg section 0.3), so the
# parser targets real output rather than a guess.
RE_PPL_FINAL = re.compile(r"Final estimate:\s*PPL\s*=\s*([0-9.eE+-]+)")
RE_MEAN_KLD = re.compile(r"Mean\s+KLD:\s*([0-9.eE+-]+)")
RE_MEDIAN_KLD = re.compile(r"Median\s+KLD:\s*([0-9.eE+-]+)")
RE_MAX_KLD = re.compile(r"Maximum\s+KLD:\s*([0-9.eE+-]+)")
RE_P99_KLD = re.compile(r"99\.0%\s+KLD:\s*([0-9.eE+-]+)")
RE_SAME_TOP = re.compile(r"Same\s+top\s+p:\s*([0-9.eE+-]+)")
RE_PPL_Q = re.compile(r"Mean\s+PPL\(Q\)\s*:\s*([0-9.eE+-]+)")
RE_PPL_BASE = re.compile(r"Mean\s+PPL\(base\)\s*:\s*([0-9.eE+-]+)")
# Anchored on `print_info:` deliberately.  The loader ALSO dumps the raw metadata table, which
# prints the file's ORIGINAL expert_used_count and is explicitly labelled "KV overrides do not
# apply in this output".  Matching that line instead of the post-override hparams would turn
# KR-5 -- the check that proves the override took effect -- into a check that proves nothing.
RE_N_EXPERT_USED = re.compile(r"print_info:\s*n_expert_used\s*=\s*(\d+)")
RE_CHUNKS = re.compile(r"(?:calculating perplexity|computing) over\s+(\d+)\s+chunks")


def parse_ppl_output(text: str, want_kl: bool) -> dict:
    """Parse llama-perplexity output.  REFUSES on a missing required field rather than guess."""
    out = {
        "n_expert_used_seen": None,
        "chunks_seen": None,
        "ppl": _num(RE_PPL_FINAL.search(text).group(1)) if RE_PPL_FINAL.search(text) else None,
        "mean_kld": None, "median_kld": None, "max_kld": None, "p99_kld": None,
        "top1_agreement": None, "ppl_q": None, "ppl_base": None,
    }

    match = RE_N_EXPERT_USED.search(text)
    if match:
        out["n_expert_used_seen"] = int(match.group(1))
    match = RE_CHUNKS.search(text)
    if match:
        out["chunks_seen"] = int(match.group(1))

    if want_kl:
        for key, rx in (("mean_kld", RE_MEAN_KLD), ("median_kld", RE_MEDIAN_KLD),
                        ("max_kld", RE_MAX_KLD), ("p99_kld", RE_P99_KLD),
                        ("ppl_q", RE_PPL_Q), ("ppl_base", RE_PPL_BASE)):
            match = rx.search(text)
            if match:
                out[key] = _num(match.group(1))

        match = RE_SAME_TOP.search(text)
        if match:
            value = _num(match.group(1))
            # llama.cpp prints this as a percentage.  Normalise defensively: a genuine agreement
            # is never between 1.5 and 100 as a fraction, so >1.5 means percent.
            if value is not None:
                out["top1_agreement"] = value / 100.0 if value > 1.5 else value

        missing = [k for k in ("mean_kld", "top1_agreement") if out[k] is None]
        if missing:
            raise Refuse(
                "llama-perplexity produced no " + ", ".join(missing) + ".\n"
                "  The KL-divergence statistics block is absent from the output.  Either the base "
                "logits file was not read, or this build's output format differs from the strings "
                "this parser was written against.  REFUSING rather than emit a partial number."
            )
    else:
        if out["ppl"] is None:
            raise Refuse(
                "llama-perplexity produced no 'Final estimate: PPL = ...' line for the base run.\n"
                "  REFUSING: without a base perplexity the comparison has no anchor."
            )
    return out


def run_key(model_name: str, domain: str, k: int, ngl: int, threads: int) -> str:
    # threads is part of the key because the CPU backend's reduction order depends on it, and a
    # base/test pair that disagrees on it measures thread-count noise instead of the drop.
    return f"{model_name}__{domain}__k{k}__ngl{ngl}__t{threads}"


def run_perplexity(binary, model_name, model_cfg, facts, domain, corpus_path, k, args, handle,
                   bid, created_logits):
    """One llama-perplexity invocation.  Idempotent: a valid cached result is reused."""
    key = run_key(model_name, domain, k, args.ngl, args.threads)
    result_path = os.path.join(RUNS_DIR, key + ".json")
    stdout_path = os.path.join(RUNS_DIR, key + ".stdout.txt")
    is_base = (k == K_BASE)
    # ngl, threads, ctx, chunks AND the build tag are all in the filename.  This is the C-14
    # analogue the prereg stakes: base and test logits must come from the same backend, and a
    # base produced under any other configuration must not be silently reachable.
    base_logits = os.path.join(
        args.logits_dir,
        f"{model_name}__{domain}__base_k{K_BASE}__ngl{args.ngl}__t{args.threads}"
        f"__c{args.ctx}__n{args.chunks}__b{bid}.bin"
    )

    if os.path.isfile(result_path) and not args.force:
        try:
            with open(result_path, "r", encoding="utf-8") as fh:
                cached = json.load(fh)
            fresh = (cached.get("ctx") == args.ctx and cached.get("chunks") == args.chunks
                     and cached.get("build_id") == bid and cached.get("threads") == args.threads
                     and cached.get("ngl") == args.ngl)
            # A cached BASE record is worthless without its logits file, and the script deletes
            # those on a successful run unless --keep-logits.  Reusing the record alone would
            # send every downstream KL run into the "base logits missing" refusal while the log
            # claims the base was satisfied from cache.  Re-run instead.
            if fresh and is_base and not args.dry_run and not (
                    os.path.isfile(base_logits) and os.path.getsize(base_logits) > 0):
                log_line(handle, f"  [redo] {key} cached but its base-logits file is gone -- re-running")
                fresh = False
            if fresh:
                log_line(handle, f"  [skip] {key} cached (idempotent; --force to redo)")
                if is_base:
                    created_logits.add(base_logits)
                return cached
        except Exception:
            pass  # unreadable cache -> just redo it

    if not is_base and not args.dry_run:
        if not os.path.isfile(base_logits) or os.path.getsize(base_logits) == 0:
            raise Refuse(
                f"base logits missing or empty for {model_name}/{domain}: {base_logits}\n"
                f"  KR-5: a KL run without its own base is VOID.  Run k={K_BASE} for this cell first "
                "(the script does this automatically when k is not restricted with --k)."
            )

    cmd = [
        binary,
        "-m", model_cfg["path"],
        "-f", corpus_path,
        "-c", str(args.ctx),
        "--chunks", str(args.chunks),
        "-ngl", str(args.ngl),
        "--seed", "1",
    ]
    cmd += list(PPL_LOG_FLAGS)
    if args.threads:
        cmd += ["-t", str(args.threads)]
    if is_base:
        cmd += ["--save-all-logits", base_logits]
        created_logits.add(base_logits)
    else:
        cmd += ["--override-kv", f"{facts['arch']}.expert_used_count=int:{k}",
                "--kl-divergence", "--kl-divergence-base", base_logits]

    log_line(handle, f"  [run ] {key}")
    log_line(handle, f"         {' '.join(cmd)}")
    if args.dry_run:
        log_line(handle, "         --dry-run: not executed")
        return None

    started = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    elapsed = time.time() - started
    text = (proc.stdout or "") + "\n" + (proc.stderr or "")

    os.makedirs(RUNS_DIR, exist_ok=True)
    with open(stdout_path, "w", encoding="utf-8") as fh:
        fh.write(text)

    if proc.returncode != 0:
        tail = "\n".join(text.splitlines()[-25:])
        raise Refuse(
            f"llama-perplexity failed (exit {proc.returncode}) for {key}\n"
            f"  raw output kept at {stdout_path}\n--- last lines ---\n{tail}"
        )

    parsed = parse_ppl_output(text, want_kl=not is_base)

    # ---- KR-5 instrument checks: refuse rather than silently produce a wrong number ----
    expected_used = K_BASE if is_base else k
    seen = parsed["n_expert_used_seen"]
    if seen is None:
        raise Refuse(
            f"{key}: the loader never printed 'print_info: n_expert_used'.  KR-5 requires proof "
            "that the --override-kv took effect; without it the run is VOID.\n"
            f"  raw output kept at {stdout_path}\n"
            f"  This build suppresses the model-info banner at default verbosity, which is why "
            f"every invocation carries {' '.join(PPL_LOG_FLAGS)}.  If those flags were stripped "
            "or are unsupported by --ppl-bin, the instrument cannot prove the override and no "
            "number from it may be scored."
        )
    if seen != expected_used:
        raise Refuse(
            f"{key}: VOID -- requested n_expert_used = {expected_used} but the model loaded with "
            f"{seen}.  The --override-kv did not take effect for arch {facts['arch']!r}.  "
            "Refusing: this run would have measured the wrong routing width."
        )
    if parsed["chunks_seen"] is None:
        raise Refuse(f"{key}: could not read the processed chunk count.  VOID (KR-5).")
    if parsed["chunks_seen"] < args.chunks:
        raise Refuse(
            f"{key}: VOID -- processed {parsed['chunks_seen']} chunks, staked {args.chunks}.  "
            f"The corpus shard tokenizes to fewer than {args.ctx * args.chunks} tokens at "
            f"-c {args.ctx}.  Lengthen the shard or lower --chunks (which makes the run a "
            "clearly-labelled sensitivity run, never the headline -- KR-6)."
        )

    record = dict(parsed)
    record.update({
        "key": key, "model": model_name, "domain": domain, "k": k,
        "ngl": args.ngl, "threads": args.threads, "ctx": args.ctx, "chunks": args.chunks,
        "build_id": bid, "binary": binary,
        "arch": facts["arch"], "elapsed_s": round(elapsed, 1),
        "base_logits": base_logits if not is_base else None,
        "cmd": cmd,
    })
    with open(result_path, "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2)
    log_line(handle, f"  [done] {key} in {elapsed:.1f}s")
    return record


# ======================================================================================
# scoring -- the staked kill rules
# ======================================================================================

def score_directional(cells, detail, reasons):
    """Score the staked DIRECTIONAL predictions P-3 and P-4 at k = K_HEADLINE.

    These carry no kill power, but a prediction that is never mechanically scored is a
    prediction that can be quietly forgotten when it misses.  Protocol requires misses at equal
    prominence, so they are computed and written out whatever the headline verdict is.
    """
    at_k = {(m, d): by_k[K_HEADLINE] for (m, d), by_k in cells.items()
            if K_HEADLINE in by_k and by_k[K_HEADLINE].get("mean_kld") is not None}

    # --- P-3: code degrades MORE than prose at equal k', within a model. ---
    p3 = []
    for model in {m for m, _ in at_k}:
        code, prose = at_k.get((model, "code_B")), at_k.get((model, "prose_B"))
        if not code or not prose:
            continue
        p3.append({
            "model": model,
            "kld_code": code["mean_kld"], "kld_prose": prose["mean_kld"],
            "top1_code": code["top1_agreement"], "top1_prose": prose["top1_agreement"],
            "hit": bool(code["mean_kld"] > prose["mean_kld"]),
        })
    detail["p3_code_worse_than_prose"] = p3
    if p3:
        hits = sum(1 for r in p3 if r["hit"])
        reasons.append(
            f"P-3 (code degrades more than prose at k={K_HEADLINE}): {hits}/{len(p3)} models in the "
            f"predicted direction -> {'HIT' if hits == len(p3) else 'MISS' if hits == 0 else 'SPLIT'}"
            + "  " + "  ".join(
                f"[{r['model']}: KLD code {r['kld_code']:.5f} vs prose {r['kld_prose']:.5f}]" for r in p3)
        )
    else:
        detail["p3_status"] = "not evaluable"
        reasons.append(f"P-3: not evaluable -- both domains were not scored at k={K_HEADLINE}.")

    # --- P-4: the shared-expert model degrades LESS at equal k'. ---
    # CAVEAT, recorded with the number: mean KLD is compared ACROSS models here, and the two
    # models have different vocabularies, different quantisations and different base
    # distributions.  This is a weak comparison and P-4 is scored as directional evidence only.
    p4 = []
    for domain in DOMAINS:
        prim, repl = at_k.get((PRIMARY_MODEL, domain)), None
        for (m, d), rec in at_k.items():
            if d == domain and m != PRIMARY_MODEL and MODELS.get(m, {}).get("role") == "replication":
                repl = rec
        if not prim or not repl:
            continue
        p4.append({"domain": domain, "kld_primary": prim["mean_kld"], "kld_replication": repl["mean_kld"],
                   "top1_primary": prim["top1_agreement"], "top1_replication": repl["top1_agreement"],
                   "hit": bool(repl["mean_kld"] < prim["mean_kld"])})
    detail["p4_shared_expert_cushion"] = p4
    detail["p4_caveat"] = ("mean KLD compared across two models with different vocab, quantisation "
                           "and base distribution -- directional evidence only, not a matched test.")
    if p4:
        hits = sum(1 for r in p4 if r["hit"])
        reasons.append(
            f"P-4 (shared-expert cushion: replication arm degrades less at k={K_HEADLINE}): "
            f"{hits}/{len(p4)} domains in the predicted direction -> "
            f"{'HIT' if hits == len(p4) else 'MISS' if hits == 0 else 'SPLIT'}  "
            + "  ".join(f"[{r['domain']}: KLD primary {r['kld_primary']:.5f} vs replication "
                        f"{r['kld_replication']:.5f}]" for r in p4)
            + "  (cross-model KLD -- directional only)"
        )
    else:
        detail["p4_status"] = "not evaluable"
        reasons.append("P-4: not evaluable -- the two model arms were not both scored.")


def verdict_for(cells, envelopes, headline, handle):
    """Score KR-1..KR-4 and KR-7.  Returns (status, reasons, detail).

    STATUS VOCABULARY -- deliberately has no member called 'PASS'.  The branch in which every
    gate is cleared is the branch in which KR-3 fires, and KR-3's content is that THE RUNTIME
    PATCH IS REFUTED AS UNNECESSARY.  Labelling that outcome 'PASS' would headline a refutation
    with the word for success, which is exactly the reporting asymmetry protocol forbids.

      REFUTED-AS-UNNECESSARY  quality + speed clear the bar; KR-3 fires; ship documentation of
                              the existing flag with its measured quality cost.  NO CODE.
      REFUTED-LEVER           KR-2 fires; no width is both affordable and fast enough; build
                              nothing and keep full routing width.
      MARGINAL / MARGINAL-UNSOUND   KR-7; re-stake with more chunks.
      UNSOUND                 KR-4; the bounding lemma's premise fails in our own data.
      VOID                    KR-5 / KR-6; instrument or knobs, not a result.
    """
    reasons, detail = [], {}

    if not headline:
        return "VOID", ["KR-6: this run used non-default knobs and may never be the headline."], detail

    # ---- KR-4 first: if the bounding lemma is unsound, nothing downstream may be carried. ----
    non_monotone, thin_cells = [], []
    for (model, domain), by_k in sorted(cells.items()):
        klds = [(k, by_k[k]["mean_kld"]) for k in sorted(by_k, reverse=True)
                if k != K_BASE and by_k[k].get("mean_kld") is not None]
        klds.sort(key=lambda pair: -pair[0])           # k descending: 7, 6, 5, 4
        # A cell with fewer than three measured widths makes ZERO or ONE comparison, and would
        # otherwise hand KR-4 a vacuous pass -- an unfireable kill rule dressed as a check.
        if len(klds) < 3:
            thin_cells.append(f"{model}/{domain} ({len(klds)} widths)")
        for (k_hi, kld_hi), (k_lo, kld_lo) in zip(klds, klds[1:]):
            if kld_lo < kld_hi:
                non_monotone.append(f"{model}/{domain}: KLD(k={k_lo})={kld_lo:.5f} < KLD(k={k_hi})={kld_hi:.5f}")
    detail["kr4_violations"] = non_monotone
    detail["kr4_cells_with_too_few_widths"] = thin_cells
    if thin_cells:
        return "VOID", reasons + [
            "KR-5/KR-4: monotonicity is not testable -- " + ", ".join(thin_cells) + ".  A cell with "
            "fewer than three measured routing widths makes at most one comparison and would give "
            "KR-4 a vacuous pass.  Refusing to score rather than report an untested premise as met."
        ], detail
    lemma_sound = not non_monotone
    reasons.append(
        f"KR-4 bounding-lemma monotonicity: {'PASS' if lemma_sound else 'FAIL'}"
        + ("" if lemma_sound else " -- " + "; ".join(non_monotone[:3]))
    )

    # ---- KR-1: quality at the headline drop level. ----
    # EACH GATE IS APPLIED TO ITS OWN WORST CELL.  Picking one cell by top-1 and then testing
    # BOTH gates on that cell lets a cell that fails only the KLD gate escape KR-1 entirely --
    # the gate would be unable to fire in the very case it exists to catch.
    scorable = {(m, d): by_k[K_HEADLINE] for (m, d), by_k in cells.items()
                if K_HEADLINE in by_k
                and by_k[K_HEADLINE].get("top1_agreement") is not None
                and by_k[K_HEADLINE].get("mean_kld") is not None}
    if not scorable:
        return "VOID", reasons + [f"KR-5: no scorable k={K_HEADLINE} cell was produced."], detail

    (t_model, t_domain), t_rec = min(scorable.items(), key=lambda kv: kv[1]["top1_agreement"])
    (d_model, d_domain), d_rec = max(scorable.items(), key=lambda kv: kv[1]["mean_kld"])
    top1 = t_rec["top1_agreement"]
    kld = d_rec["mean_kld"]
    detail["kr1_worst_top1_cell"] = {"model": t_model, "domain": t_domain, "top1_agreement": top1,
                                     "mean_kld": t_rec["mean_kld"]}
    detail["kr1_worst_kld_cell"] = {"model": d_model, "domain": d_domain, "mean_kld": kld,
                                    "top1_agreement": d_rec["top1_agreement"]}

    kr1_top1 = top1 >= GATE_TOP1_MIN
    kr1_kld = kld <= GATE_MEAN_KLD_MAX
    kr1 = kr1_top1 and kr1_kld
    reasons.append(
        f"KR-1 quality at k={K_HEADLINE}: worst top-1 cell ({t_model}/{t_domain}) = {top1:.5f} vs "
        f"gate >= {GATE_TOP1_MIN} -> {'PASS' if kr1_top1 else 'FAIL'};  worst mean-KLD cell "
        f"({d_model}/{d_domain}) = {kld:.5f} vs gate <= {GATE_MEAN_KLD_MAX} -> "
        f"{'PASS' if kr1_kld else 'FAIL'}"
    )

    # ---- the staked directional predictions, scored whatever the verdict is ----
    score_directional(cells, detail, reasons)

    # ---- KR-7 marginality ----
    if abs(top1 - GATE_TOP1_MIN) < MARGIN_TOP1 or abs(kld - GATE_MEAN_KLD_MAX) < MARGIN_KLD:
        reasons.append(
            f"KR-7: the headline lands within +/-{MARGIN_TOP1} of the top-1 gate or "
            f"+/-{MARGIN_KLD} of the KLD gate -> MARGINAL.  Rerun with more chunks and re-stake; "
            "a marginal number must not be rounded into a verdict in either direction."
        )
        if not lemma_sound:
            # KR-4 says its failure is reported even when other rules pass.  An early return on
            # marginality would swallow it.
            reasons.append(
                "KR-4 ALSO FAILED: monotonicity does not hold in our own data, so section 3.3's "
                "dominance argument is UNSOUND independently of the marginal quality number."
            )
            return "MARGINAL-UNSOUND", reasons, detail
        return "MARGINAL", reasons, detail

    # ---- KR-2: is any quality-passing k' fast enough, on the PRIMARY model? ----
    # The speed half of this rule cannot fail (see GATE_SPEEDUP_MIN).  Recorded, not hidden.
    detail["kr2_speed_gate_kill_power"] = False
    detail["kr2_speed_gate_note"] = (
        "Arm B is deterministic arithmetic over GGUF headers computed before staking, and "
        "k'=6/5/4 clear 1.20x by construction.  Under KR-4 monotonicity KR-2 reduces to KR-1: "
        "only a quality failure can make it fire.  It is not a second independent hurdle.")
    primary_env = envelopes.get(PRIMARY_MODEL, {})
    passing = []
    for k in sorted(set(K_GRID), reverse=True):
        cells_at_k = [by_k[k] for by_k in cells.values()
                      if k in by_k and by_k[k].get("top1_agreement") is not None
                      and by_k[k].get("mean_kld") is not None]
        if not cells_at_k or len(cells_at_k) != len(cells):
            continue          # a width scored in only some cells cannot clear an all-cells gate
        ok = all(c["top1_agreement"] >= GATE_TOP1_MIN and c["mean_kld"] <= GATE_MEAN_KLD_MAX
                 for c in cells_at_k)
        speedup = primary_env.get(k)
        if ok and speedup is not None and speedup >= GATE_SPEEDUP_MIN:
            passing.append((k, speedup))
    detail["kr2_passing_widths"] = [{"k": k, "computed_speedup": round(s, 4)} for k, s in passing]

    kr2 = bool(passing)
    if kr2:
        best_k, best_speed = max(passing, key=lambda pair: pair[0])   # largest k = mildest drop
        reasons.append(
            f"KR-2 worth-building: k={best_k} passes quality in every cell and reaches "
            f"{best_speed:.3f}x COMPUTED (not measured) on the primary model (gate >= "
            f"{GATE_SPEEDUP_MIN}) -> cleared.  The speed half of this rule had no kill power."
        )
    else:
        reasons.append(
            f"KR-2 worth-building: NO routing width both passes quality in every cell and reaches "
            f"{GATE_SPEEDUP_MIN}x computed on the primary model -> FIRES.  The lever is REFUTED for "
            "our split, uniform and placement-aware alike.  Publish as a miss at equal prominence."
        )

    # ---- KR-3: the redundancy rule.  It is a CONCLUSION, not a hurdle: it fires exactly when
    # everything else clears, and what it concludes is that no code may be written. ----
    if kr2:
        best_k, best_speed = max(passing, key=lambda pair: pair[0])
        reasons.append(
            f"KR-3 REDUNDANCY -> THE RUNTIME PATCH IS REFUTED AS UNNECESSARY.  The win is already "
            f"available today via  --override-kv {{arch}}.expert_used_count=int:{best_k}  -- a flag "
            "that costs zero lines of code.  What ships is DOCUMENTATION OF THE FLAG WITH ITS "
            "MEASURED QUALITY COST ATTACHED (never the flag alone -- E-10).  No cache-aware runtime "
            "change may be written on this result."
        )
        detail["kr3_fired"] = True
        detail["kr3_recommended_k"] = best_k
    else:
        detail["kr3_fired"] = False
        reasons.append(
            "KR-3 not reached: nothing passed KR-2, so there is no redundancy to declare.  Note the "
            "bounding lemma means a KR-1/KR-2 failure does NOT prove the placement-aware port fails "
            "-- only that our upper bound is too loose to authorize a patch on this evidence.  A "
            "Stage 2 MEASUREMENT (patch + direct placement-aware KL) may be staked separately; a "
            "Stage 2 SHIP may not."
        )

    if not lemma_sound:
        reasons.append(
            "KR-4 FAILED: monotonicity does not hold in our own data, so section 3.3's dominance "
            "argument is UNSOUND and no result above may be carried into Stage 2 without a direct "
            "placement-aware measurement.  This is reported even though other rules passed."
        )
        return "UNSOUND", reasons, detail

    return ("REFUTED-AS-UNNECESSARY" if (kr1 and kr2) else "REFUTED-LEVER"), reasons, detail


# ======================================================================================
# preflight -- refuse early, refuse loudly
# ======================================================================================

def preflight(args, handle):
    if sys.version_info < (3, 9):
        raise Refuse(f"Python 3.9+ required, running {sys.version.split()[0]}")

    try:
        import gguf  # noqa: F401
    except ImportError:
        raise Refuse(
            "the 'gguf' python package is not importable, so model headers cannot be read.\n"
            "  Install with: python -m pip install gguf"
        )

    # ---- the instrument.  NO PATH FALLBACK. ----
    # The prereg stakes one build.  Silently substituting whatever llama-perplexity happens to
    # sit on PATH would swap the instrument mid-experiment and the run would still be stamped
    # headline -- the exact class of defect that once turned an out-of-VRAM profiler run into a
    # nearly-published streaming artifact.  Refuse instead, and make an explicit override
    # forfeit headline status (KR-6).
    binary = args.ppl_bin
    if not os.path.isfile(binary):
        raise Refuse(
            f"llama-perplexity not found at {binary}\n"
            f"  The prereg stakes this exact build:\n    {STAKED_PPL_BIN}\n"
            "  There is deliberately no fall back to a llama-perplexity found on PATH: a\n"
            "  different build is a different instrument, and base logits written by one build\n"
            "  must never be KL-compared against another.  Restore the staked build, or pass\n"
            "  --ppl-bin <path> explicitly -- which marks the run headline:false (KR-6)."
        )
    if os.path.abspath(binary) != os.path.abspath(STAKED_PPL_BIN):
        log_line(handle, f"  WARNING non-staked binary -> headline:false (KR-6)")
        log_line(handle, f"          staked {STAKED_PPL_BIN}")
        log_line(handle, f"          using  {binary}")

    # ---- corpus: emitted by exp52.  One generator, one hash -- we never re-emit. ----
    corpora = {}
    for domain in DOMAINS:
        path = CORPUS_PATH.format(domain=domain)
        if not os.path.isfile(path):
            source = EXP52_CORPUS_PATH.format(domain=domain)
            if not os.path.isfile(source):
                raise Refuse(
                    f"corpus shard '{domain}' is missing.\n"
                    f"  Expected {path}\n"
                    f"  or the exp52 original at {source}\n"
                    "  Emit it first with:\n"
                    "      python weights\\exp52_expert_usage_skew.py --emit-corpus-only\n"
                    "  #90 deliberately does not re-implement the corpus generator: one generator,\n"
                    "  one hash, so #52 and #55 can never disagree about what was measured."
                )
            shutil.copyfile(source, path)
        digest = sha256_file(path)
        if digest != CORPUS_SHA256[domain]:
            raise Refuse(
                f"corpus shard '{domain}' FAILED its pinned hash (KR-6).\n"
                f"  path     {path}\n  expected {CORPUS_SHA256[domain]}\n  actual   {digest}\n"
                "  The prompt set cannot be swapped after seeing a result.  Refusing."
            )
        corpora[domain] = path
        log_line(handle, f"  corpus  {domain:9s} {os.path.getsize(path):>8,} B  sha256 OK  {path}")

    # ---- models ----
    selected = {}
    for name, cfg in MODELS.items():
        if args.model and name not in args.model:
            continue
        if not os.path.isfile(cfg["path"]):
            raise Refuse(
                f"model GGUF missing: {cfg['path']}\n"
                f"  Required for arm '{name}'.  Restore it, or restrict with --model <name>."
            )
        selected[name] = cfg
    if not selected:
        raise Refuse(f"no models selected.  Known arms: {', '.join(MODELS)}")

    facts = {}
    for name, cfg in selected.items():
        facts[name] = model_facts(cfg["path"], cfg)
        f = facts[name]
        log_line(handle,
                 f"  model   {name:20s} {f['arch']:10s} L={f['n_layer']} E={f['n_expert']} "
                 f"k={f['n_expert_used']} vocab={f['n_vocab']:,} moe_layers={f['n_moe_layer']} "
                 f"experts_fused=YES shared={'YES' if f['has_shared_experts'] else 'no'}")

    # ---- disk budget for the base logits ----
    if not os.path.isdir(args.logits_dir):
        parent = os.path.dirname(os.path.abspath(args.logits_dir)) or "."
        if not os.path.isdir(parent):
            raise Refuse(
                f"--logits-dir parent does not exist: {parent}\n"
                "  Base logits are large; point --logits-dir at a volume with room."
            )
        os.makedirs(args.logits_dir, exist_ok=True)

    tokens = args.ctx * args.chunks
    need = 0
    for name, f in facts.items():
        per_cell = f["n_vocab"] * tokens * LOGITS_BYTES_PER_VOCAB_TOKEN
        need += per_cell * len([d for d in DOMAINS if not args.domain or d in args.domain])
    free = shutil.disk_usage(args.logits_dir).free
    log_line(handle, f"  logits  dir={args.logits_dir}  need<={gb(need)} (upper bound)  free={gb(free)}")
    if free < need + DISK_HEADROOM_BYTES:
        raise Refuse(
            f"not enough free space for the base logits.\n"
            f"  upper-bound need {gb(need)} + {gb(DISK_HEADROOM_BYTES)} headroom, free {gb(free)}\n"
            f"  at {args.logits_dir}\n"
            "  Point --logits-dir at a larger volume, or run one --model / --domain at a time.\n"
            "  Refusing rather than dying mid-run and leaving a truncated base file that would "
            "silently corrupt every KL number computed against it."
        )

    return binary, selected, facts, corpora


# ======================================================================================
# main
# ======================================================================================

def main():
    parser = argparse.ArgumentParser(
        description="exp55 -- cache-aware expert dropping (prereg #90, task #55)")
    parser.add_argument("--model", action="append", choices=sorted(MODELS),
                        help="restrict to one model arm (repeatable)")
    parser.add_argument("--domain", action="append", choices=list(DOMAINS),
                        help="restrict to one domain (repeatable)")
    parser.add_argument("--k", action="append", type=int,
                        help="restrict the routing widths tested (repeatable; k=8 always runs)")
    parser.add_argument("--ngl", type=int, default=0,
                        help="GPU layers.  Default 0 (pure CPU) for determinism, matching #87. "
                             "Base and test runs MUST share this value.")
    parser.add_argument("--threads", type=int, default=0, help="-t for llama-perplexity")
    parser.add_argument("--ctx", type=int, default=CTX, help=f"context (staked {CTX})")
    parser.add_argument("--chunks", type=int, default=CHUNKS, help=f"chunks (staked {CHUNKS})")
    parser.add_argument("--ppl-bin", default=DEFAULT_PPL_BIN)
    parser.add_argument("--logits-dir", default=DEFAULT_LOGITS_DIR)
    parser.add_argument("--keep-logits", action="store_true",
                        help="keep the multi-GB base logits files after scoring")
    parser.add_argument("--force", action="store_true", help="redo cached runs")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the exact commands and the computed speed envelope, run nothing")
    args = parser.parse_args()

    os.makedirs(DATA, exist_ok=True)
    os.makedirs(RUNS_DIR, exist_ok=True)

    # KR-6.  ngl, threads and the binary are in here because each of them changes the backend
    # the logits come from; leaving them out let a run with a swapped backend still be stamped
    # headline:true.  STAKED_NGL is 0 (pure CPU), matching #87.
    headline = (args.ctx == CTX and args.chunks == CHUNKS and not args.k
                and not args.model and not args.domain
                and args.ngl == STAKED_NGL and args.threads == STAKED_THREADS
                and os.path.abspath(args.ppl_bin) == os.path.abspath(STAKED_PPL_BIN))

    # A dry run must never overwrite a scored result: it produces no measurement and its
    # 'DRY-RUN' verdict would silently replace a real one in the file the register reads.
    json_out = JSON_OUT if not args.dry_run else JSON_OUT.replace(".json", ".dryrun.json")
    log_out = LOG_OUT if not args.dry_run else LOG_OUT.replace(".log", ".dryrun.log")

    handle = open(log_out, "w", encoding="utf-8")
    started_at = time.strftime("%Y-%m-%d %H:%M:%S")
    log_line(handle, f"# exp55 cache-aware expert dropping | pre-registration #{PREREG_ID} (task #{TASK_ID})")
    log_line(handle, f"# staked in {PREREG}")
    log_line(handle, f"# gates: KR-1 top-1 >= {GATE_TOP1_MIN} AND mean KLD <= {GATE_MEAN_KLD_MAX} at k={K_HEADLINE};")
    log_line(handle, f"#        KR-2 computed speedup >= {GATE_SPEEDUP_MIN}x on {PRIMARY_MODEL};")
    log_line(handle, f"#        KR-3 a KR-2 pass REFUTES the runtime patch as unnecessary;")
    log_line(handle, f"#        KR-4 mean KLD must be monotone in drop depth or the bound is UNSOUND")
    log_line(handle, f"# started {started_at}")
    log_line(handle, f"# headline run: {headline}"
                     + ("" if headline else "   <-- KR-6: NON-DEFAULT KNOBS, sensitivity only"))
    log_line(handle)

    result = {
        "prereg": PREREG, "prereg_id": PREREG_ID, "task": TASK_ID,
        "started": started_at, "headline": headline,
        "config": {"ctx": args.ctx, "chunks": args.chunks, "ngl": args.ngl,
                   "k_base": K_BASE, "k_grid": list(K_GRID), "k_headline": K_HEADLINE},
        "gates": {"top1_min": GATE_TOP1_MIN, "mean_kld_max": GATE_MEAN_KLD_MAX,
                  "speedup_min": GATE_SPEEDUP_MIN, "primary_model": PRIMARY_MODEL},
        "arm_b_constants": {"bw_gpu_eff": BW_GPU_EFF, "bw_cpu_eff": BW_CPU_EFF,
                            "gpu_expert_share": GPU_EXPERT_SHARE,
                            "NOTE": "CALIBRATION INPUTS, not predictions.  Arm B is arithmetic "
                                    "over GGUF headers -- no tok/s is measured anywhere in #90."},
    }

    try:
        log_line(handle, "## preflight")
        binary, selected, facts, corpora = preflight(args, handle)
        bid = build_id(binary)
        created_logits = set()
        log_line(handle, f"  binary  {binary}  build_id={bid}")
        log_line(handle, f"  flags   every run carries {' '.join(PPL_LOG_FLAGS)} -- without -v this "
                         "build never prints 'print_info: n_expert_used' and KR-5 voids every run")
        result["instrument"] = {"binary": binary, "build_id": bid,
                                "staked_binary": STAKED_PPL_BIN,
                                "log_flags": list(PPL_LOG_FLAGS)}
        log_line(handle)

        # ---- Arm B: the computed speed envelope (a PREDICTION) ----
        log_line(handle, "## Arm B -- computed speed envelope (PREDICTION, NOT A MEASUREMENT)")
        envelopes = {}
        for name, f in facts.items():
            env = speed_envelope(f)
            envelopes[name] = env
            role = MODELS[name]["role"]
            kill = "kill power" if name == PRIMARY_MODEL else "NO kill power (envelope straddles the bar)"
            log_line(handle, f"  {name} ({role}, {kill})")
            log_line(handle, f"    expert bytes {gb(f['expert_bytes'])}  other {gb(f['other_bytes'])}")
            for k in (K_BASE,) + K_GRID:
                log_line(handle, f"      k={k}  computed speedup {env[k]:.3f}x")
        result["arm_b_speed_envelope"] = {n: {str(k): round(v, 4) for k, v in e.items()}
                                          for n, e in envelopes.items()}
        result["arm_0_structural"] = {
            n: {"arch": f["arch"], "n_moe_layer": f["n_moe_layer"], "n_expert": f["n_expert"],
                "experts_fused_per_layer": f["experts_fused_per_layer"],
                "has_shared_experts": f["has_shared_experts"], "n_vocab": f["n_vocab"]}
            for n, f in facts.items()
        }
        log_line(handle)

        if args.dry_run:
            log_line(handle, "## --dry-run: commands only, nothing executed")

        # ---- Arm A: the measurement ----
        log_line(handle, "## Arm A -- quality of dropping (llama-perplexity, KL vs the k=8 base)")
        k_list = sorted({k for k in (args.k or K_GRID) if k != K_BASE}, reverse=True)
        cells = {}
        for name, cfg in selected.items():
            for domain in DOMAINS:
                if args.domain and domain not in args.domain:
                    continue
                log_line(handle, f"  -- {name} / {domain}")
                by_k = {}
                base = run_perplexity(binary, name, cfg, facts[name], domain,
                                      corpora[domain], K_BASE, args, handle, bid, created_logits)
                if base:
                    by_k[K_BASE] = base
                for k in k_list:
                    rec = run_perplexity(binary, name, cfg, facts[name], domain,
                                         corpora[domain], k, args, handle, bid, created_logits)
                    if rec:
                        by_k[k] = rec
                        log_line(handle,
                                 f"         k={k}: top-1 {rec['top1_agreement']:.5f}  "
                                 f"meanKLD {rec['mean_kld']:.5f}  medKLD {rec['median_kld']}  "
                                 f"PPL(Q) {rec['ppl_q']}  PPL(base) {rec['ppl_base']}")
                if by_k:
                    cells[(name, domain)] = by_k
        log_line(handle)

        if args.dry_run:
            log_line(handle, "## dry run complete -- no verdict (nothing was measured)")
            result["overall"] = "DRY-RUN"
            with open(json_out, "w", encoding="utf-8") as fh:
                json.dump(result, fh, indent=2)
            log_line(handle, f"\nwrote {json_out}  (dry runs never overwrite a scored result)")
            return 0

        result["arm_a_cells"] = {f"{m}|{d}": {str(k): v for k, v in by_k.items()}
                                 for (m, d), by_k in cells.items()}

        # ---- scoring ----
        log_line(handle, "## VERDICT against the staked kill rules")
        status, reasons, detail = verdict_for(cells, envelopes, headline, handle)
        for line in reasons:
            log_line(handle, "  " + line)
        result["overall"] = status
        result["verdict_reasons"] = reasons
        result["verdict_detail"] = detail

        result["verdict_sentence"] = VERDICT_SENTENCE.get(status, "")
        log_line(handle)
        log_line(handle, "=" * 78)
        log_line(handle, f"  exp55 / prereg #{PREREG_ID}:  {status}")
        for line in result["verdict_sentence"].split(".  "):
            if line.strip():
                log_line(handle, f"  {line.strip().rstrip('.')}.")
        log_line(handle, "=" * 78)

        if not args.keep_logits:
            # Only the files THIS run created.  Sweeping the whole directory by glob deleted
            # other arms' bases -- e.g. a run restricted with --model wiped the base logits of
            # the model it was not touching, and every later KL run against them refused.
            freed = 0
            for path in sorted(created_logits):
                if os.path.isfile(path):
                    freed += os.path.getsize(path)
                    os.remove(path)
            if freed:
                log_line(handle, f"\nremoved {gb(freed)} of base logits created by this run "
                                 "(--keep-logits to retain)")

    except Refuse as exc:
        log_line(handle, "")
        log_line(handle, "=" * 78)
        log_line(handle, "  REFUSING TO RUN -- a precondition is missing.")
        log_line(handle, "  A wrong number is worse than no number.")
        log_line(handle, "=" * 78)
        log_line(handle, str(exc))
        result["overall"] = "REFUSED"
        result["refuse_reason"] = str(exc)
        with open(json_out, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2)
        handle.close()
        return 2

    with open(json_out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    log_line(handle, f"\nwrote {json_out}")
    log_line(handle, f"wrote {log_out}")
    handle.close()
    return 0 if result["overall"] in TERMINAL_VERDICTS else 2


if __name__ == "__main__":
    sys.exit(main())
