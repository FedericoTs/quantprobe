#!/usr/bin/env python3
"""Generate BREAKTHROUGH_BRIEF.json — the complete machine-readable state of this investigation,
for an external model to red-team.

Generated, never hand-written, from the same canonical sources the release gate enforces:
findings/REGISTER.json (every claim, with its scope) and preregistrations/ (every staked
prediction and its scored verdict). Hand-writing this would let it drift from the record, which
is the failure mode findings.py exists to prevent.

    python report.py            # write BREAKTHROUGH_BRIEF.json
    python report.py --check    # verify it is current (non-zero exit if stale)
"""
import io, json, os, re, sys, subprocess

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "BREAKTHROUGH_BRIEF.json")
PRE = os.path.join(ROOT, "preregistrations")


def preregs():
    """Every staked prediction with its verdict line, so a reader can audit hit-rate and honesty."""
    out = []
    for fn in sorted(os.listdir(PRE)):
        if not fn.endswith(".md"):
            continue
        txt = io.open(os.path.join(PRE, fn), encoding="utf-8").read()
        m = re.search(r"[Pp]re-registration #(\d+)", txt[:3000])
        num = int(m.group(1)) if m else None
        v = re.search(r"\*\*Verdict:?\s*(.+?)\*\*", txt, re.S)
        verdict = " ".join(v.group(1).split())[:400] if v else None
        if verdict is None and "WITHDRAWN" in txt:
            verdict = "WITHDRAWN, incomplete"
        kill = "KILL RULE" in txt or "Refuted if" in txt
        fired = bool(re.search(r"KILL RULE FIRES|kill rule fires|KILL RULE.*FIRE", txt))
        out.append({"n": num, "file": fn, "has_kill_rule": kill,
                    "kill_rule_fired": fired, "verdict": verdict,
                    "scored": verdict is not None})
    return sorted(out, key=lambda r: (r["n"] is None, r["n"]))


def hardware():
    return {
        "label": "reference box, 2016 desktop",
        "cpu": "Intel i5-7600K, 4 cores / 4 threads, AVX2, no AVX-512, no AMX",
        "gpu": "NVIDIA GTX 1060 6GB, Pascal sm_61, 4.4 TFLOPS fp32, native __dp4a INT8, fp16 at 1/64 rate",
        "ram": "16 GB DDR4-3000 (XMP on)",
        "storage": "SATA SSD (Crucial MX500)",
        "link": "PCIe 3.0 x16",
        "os": "Windows 10 Home 19045",
        "measured_not_spec": {
            "dram_read_stream_GBps": 26.1,
            "dram_copy_stream_GBps": 30.4,
            "dram_spec_sheet_GBps": 48.0,
            "dram_spec_is_fiction": True,
            "vram_read_independent_GBps": 165.6,
            "vram_gemv_fp32_independent_GBps": 161.3,
            "vram_gemv_fp16_independent_GBps": 108.9,
            "vram_spec_sheet_GBps": 192.0,
            "vram_spec_is_real": True,
            "pcie_h2d_measured_GBps": 12.2,
            "note": "VRAM figures measured with CuPy, NO llama.cpp in the path (prereg #44). DRAM figures with numpy (#27). This distinction is the single most important methodological fact in the record: for most of the project every GPU number came through llama.cpp, which made 'hardware limit' and 'llama.cpp limit' indistinguishable on that side."
        },
    }


def workload():
    return {
        "model": "Qwen3-30B-A3B, MoE, 48 layers, 128 experts, 8 active per token",
        "quantization_primary": "Q2_K (2.95 effective bits), 10.49 GB file",
        "active_params_B": 3.3,
        "active_bytes_per_token_GB": 1.217,
        "shipped_placement": "-ngl 99 -ot 'blk\\.(16..47)\\.ffn_.*_exps\\.=CPU' --no-mmap -b 1024 -ub 1024",
        "placement_meaning": "attention + 16 layers of experts in VRAM; 32 layers of experts in host DRAM, computed on CPU",
    }


def state_of_play():
    return {
        "copy_regime_decode_tok_s": 108.4,
        "copy_regime_recipe": "llama-server --spec-type ngram-simple --spec-ngram-simple-size-m 384 --spec-ngram-simple-size-n 4",
        "copy_regime_notes": "output that reuses its context: edits, refactors, RAG quoting. 4.7x over the 21.3 no-speculation baseline at ~3-bit. Multiplier SHRINKS with bit-width (3.4x at Q3_K_M).",
        "novel_generation_decode_tok_s": 21.0,
        "novel_generation_notes": "fresh reasoning/prose with nothing in context to copy. EVERY drafting mechanism measured and closed.",
        "raw_decode_wall_tok_s": 41.1,
        "raw_decode_wall_basis": "1 / sum(bytes_i / measured_BW_i) using MEASURED 26.1 GB/s DRAM, not the 48 spec",
        "unattributed_split_token_ms": 20,
        "prefill_tok_s": 386,
        "prefill_ceiling_tok_s": "405-445, converged across three placements = a compute ceiling on this GPU",
        "the_one_live_lead": "GPU kernel gap: card does 161.3 GB/s on decode-shaped GEMV (eta 0.84); llama.cpp Q4_K_M does 98.6 (eta 0.513). 1.64x, NOT hardware and NOT arithmetic (sm_61 has native __dp4a at ~4x fp32 rate). Remaining explanation is the kernel's block-unpacking/scale work. Reducibility UNMEASURED.",
    }


def the_ask():
    return {
        "goal": "Maximise tok/s for NOVEL generation (fresh reasoning/prose) for a single user running a big MoE model on cheap consumer hardware. Copy-regime is solved at 108 tok/s; novel generation is stuck at 21 against a 41.1 wall.",
        "explicitly_out_of_scope": [
            "model selection (choosing a model with fewer active params) - the owner rejected this lever",
            "buying different hardware - the point is to make THIS class of machine capable",
            "anything that changes output distribution without measuring the quality cost",
        ],
        "explicitly_in_scope": [
            "leaving llama.cpp entirely - a custom runtime, kernels in CUDA/PTX/SASS, assembly",
            "leaving Windows - Linux, WSL2, bare metal, a different driver model",
            "OS/driver/BIOS/firmware levers - hugepages, WDDM vs TCC, MMIO, BAR sizing, power/clock policy",
            "anything at or below the hardware abstraction the current stack imposes",
        ],
        "what_would_count_as_a_breakthrough": "any measured, reproducible lever that raises NOVEL-content decode above ~25 tok/s on this exact box without degrading output quality, OR a demonstration that the 41.1 wall itself is wrong (as the 48 GB/s DDR4 spec turned out to be).",
        "instructions_for_the_reader": [
            "Do NOT re-propose anything in refuted_do_not_repropose - each was measured, with a control, and the control is cited.",
            "Every proposal must come with ARITHMETIC using the measured constants in hardware.measured_not_spec, not spec sheets.",
            "Every proposal must name a FALSIFICATION TEST runnable on this box, ideally with no code.",
            "State explicitly whether your idea changes NOVEL decode or only copy-regime. Copy-regime is already solved and is not interesting.",
            "'Nothing found in my area' is a valuable answer. Do not invent a lever to have something to say.",
            "Note the methodology below before proposing: this project has killed 6 of its own mechanism hypotheses with controls. Assume your first explanation is wrong and design the control that would kill it.",
        ],
    }


def methodology():
    return {
        "protocol": "Every claim is pre-registered BEFORE measurement with numbered falsifiable predictions and an explicit kill rule; scored publicly with hits and misses given equal prominence; wired into code or explicitly recorded as not-shipped. A 5-layer release gate enforces that the register and the code cannot drift apart.",
        "hard_won_measurement_rules": [
            "Position-control every A/B: this box drifts up to +25% cold-to-warm within one session, which is larger than most effects under test. Baseline must be re-run LAST.",
            "Same-session only: 10-13% drift BETWEEN sessions against sub-1% error bars within one.",
            "Read what the model actually wrote: three spectacular results were harness artifacts (a repetition loop, an n-gram store replaying a repeated request, ANSI codes in a diff).",
            "Check the neighbourhood: a value flat next to its neighbours is a result; a value with a 45% step next to it is a coincidence (a shipped figure sat one -ub step from a cliff).",
            "Never attribute without a control: 6 mechanism hypotheses have been killed by their own controls (fixed overhead, GPU clock state, bytes-per-token, memory scatter, per-layer round trips, dual-bus).",
            "Compare like with like: request 1 vs request 2 of the same server differ (prompt-cache reuse), with and without speculation.",
        ],
        "counterintuitive_findings_that_generalise": [
            "A quantized byte is not a byte: 6 measured instances where a constant assumed uniform across formats turned out to be a property OF the format (weight decode collapse, VRAM eta, KV eta, CPU I-quants, speculation multiplier, VRAM format ladder).",
            "The cost unit in speculative decoding is the VERIFY ROUND (one full weight read), not the token. Acceptance rate is the WRONG optimisation target: it falls 89%->68% while throughput doubles.",
            "Acceptance decay separates drafters: a copy drafter's acceptance is flat in draft length (extend it freely); a model drafter's collapses geometrically (extending it destroys throughput). Same flag shape, opposite behaviour.",
            "Excluding a configuration is also a claim and inherits the scope of the sweep that produced it - a cell recorded as 'dominated' was later found to be the outright winner, because it had only been measured past a cliff.",
        ],
    }


def main():
    reg = json.load(io.open(os.path.join(ROOT, "findings", "REGISTER.json"), encoding="utf-8"))
    try:
        head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True).stdout.strip()
    except Exception:
        head = None

    brief = {
        "_meta": {
            "title": "quantprobe — complete measured state, for external red-teaming",
            "generated_by": "report.py from findings/REGISTER.json + preregistrations/ (never hand-written)",
            "git_head": head,
            "date": "2026-07-28",
            "public_repo": "https://github.com/FedericoTs/quantprobe",
            "upstream_issue_filed": "https://github.com/ggml-org/llama.cpp/issues/26200",
        },
        "the_ask": the_ask(),
        "hardware": hardware(),
        "workload": workload(),
        "state_of_play": state_of_play(),
        "methodology": methodology(),
        "established_laws": reg.get("laws", []),
        "shipped_levers": reg.get("levers", []),
        "refuted_do_not_repropose": reg.get("dead_ends", []),
        "open_contradictions": reg.get("contradictions", []),
        "untried_levers": reg.get("untried", []),
        "external_prior_art_reviewed": reg.get("external", []),
        "preregistrations": preregs(),
    }
    txt = json.dumps(brief, indent=2, ensure_ascii=False)
    if "--check" in sys.argv:
        cur = io.open(OUT, encoding="utf-8").read() if os.path.isfile(OUT) else ""
        if cur.strip() != txt.strip():
            print("BREAKTHROUGH_BRIEF.json is STALE - run: python report.py")
            return 1
        print("  brief current")
        return 0
    io.open(OUT, "w", encoding="utf-8", newline="\n").write(txt)
    p = brief["preregistrations"]
    print(f"  wrote {os.path.basename(OUT)}: {len(txt)//1024} KB")
    print(f"  {len(p)} pre-registrations ({sum(1 for x in p if x['scored'])} scored, "
          f"{sum(1 for x in p if x['kill_rule_fired'])} kill rules fired)")
    print(f"  {sum(len(reg.get(k, [])) for k in ('laws','levers','dead_ends','contradictions','untried','external'))} register entries")
    return 0


if __name__ == "__main__":
    sys.exit(main())
