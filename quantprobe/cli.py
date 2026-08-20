"""quantprobe CLI - probe / plan / fetch."""
from __future__ import annotations
import argparse

from .plan import numlist, MODELS as _MODELS


def main():
    import sys
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    from . import __version__
    ap = argparse.ArgumentParser(
        prog="quantprobe",
        description="Probe-then-quantize for LLMs: fragility curves, placement plans, recipes. "
                    "Laws + evidence: github.com/FedericoTs/quantprobe")
    # Packagers and bug reports expect `--version`; before this it errored with
    # `unrecognized arguments` and the number was only visible in the no-args banner.
    ap.add_argument("--version", action="version", version=f"quantprobe {__version__}")
    sub = ap.add_subparsers(dest="cmd")   # not required: bare `quantprobe` shows the banner + help

    p = sub.add_parser("probe", help="measure a GGUF's depth-fragility curve, emit the recipe (Law 3)")
    p.add_argument("--gguf", required=True, help="f16/bf16 (or high-precision) source GGUF")
    p.add_argument("--bands", type=int, default=4)
    p.add_argument("--chunks", type=int, default=32)
    p.add_argument("--eval", default="auto",
                   help="raw text eval file (e.g. wikitext-2 test), or 'auto' (the default) to "
                        "fetch WikiText-2 once, 1.3 MB, into --workdir")
    p.add_argument("--ngl", type=int, default=99, help="GPU layers for perplexity (0 for CPU)")
    p.add_argument("--workdir", default=None)
    p.add_argument("--llama-dir", default=None, help="dir containing llama-quantize/llama-perplexity")
    p.add_argument("--apply", action="store_true", help="after probing, BUILD the recommended depth-aware GGUF")
    p.add_argument("--out", default=None, help="output path for --apply (default: <model>-depthaware.gguf)")
    p.add_argument("--imatrix", default=None, help="importance-matrix file for --apply, or 'auto' to generate one first (measured -8.5%% ppl at ~3 bits, free in size and speed)")
    p.add_argument("--imatrix-chunks", type=int, default=100, help="calibration chunks when --imatrix auto (default 100)")
    p.add_argument("--yes", "-y", action="store_true", help="skip the time-commitment confirmation for long runs")
    p.add_argument("--dry-run", action="store_true")

    q = sub.add_parser("plan", help="evaluate every bit/tier placement, predict tok/s (Law 4)")
    q.add_argument("--model", default=None, help="preset: " + " ".join(sorted(_MODELS)))
    q.add_argument("--machine", default=None, help="preset: 2016-xmp 2016 gaming ddr5 colibri")
    q.add_argument("--bits", type=float, default=None, help="effective bits/weight (default: read from --gguf, else 2.5)")
    q.add_argument("--total", type=float, help="total params (B)")
    q.add_argument("--active", type=float, help="active params per token (B)")
    q.add_argument("--always-active", type=float, help="always-active (attention/embed) params (B)")
    q.add_argument("--vram", type=numlist, help="GB; comma-list for multi-GPU: 24,24"); q.add_argument("--vram-bw", type=numlist, help="GB/s; comma-list aggregates x0.85")
    q.add_argument("--ram", type=float); q.add_argument("--ram-bw", type=float)
    q.add_argument("--disk-bw", type=numlist, help="GB/s; comma-list (RAID) aggregates x0.75")
    q.add_argument("--gguf", default=None, help="optional: read total/active/bits/KV exactly from this GGUF")
    q.add_argument("--ctx", type=int, default=0, help="context depth: adds KV reads/token + KV memory to the prediction (Law 4 v2)")
    q.add_argument("--kv-per-pos", type=float, default=None, help="KV bytes per position in KB (default: model preset, or 96)")
    q.add_argument("--no-anchors", action="store_true",
                   help="ignore calibrate's anchor runs; use the plain law constants")

    def hwargs(sp):
        sp.add_argument("--model", default=None); sp.add_argument("--machine", default=None)
        sp.add_argument("--bits", type=float, default=None, help="effective bits/weight (default: read from --gguf, else 2.5)")
        sp.add_argument("--total", type=float); sp.add_argument("--active", type=float)
        sp.add_argument("--always-active", type=float)
        sp.add_argument("--vram", type=numlist); sp.add_argument("--vram-bw", type=numlist)
        sp.add_argument("--ram", type=float); sp.add_argument("--ram-bw", type=float)
        sp.add_argument("--disk-bw", type=numlist)
        sp.add_argument("--ctx", type=int, default=0, help="context depth for the prediction (Law 4 v2)")
        sp.add_argument("--kv-per-pos", type=float, default=None, help="KV bytes per position in KB")
        sp.add_argument("--llama-dir", default=None); sp.add_argument("--dry", action="store_true")
        sp.add_argument("--no-anchors", action="store_true",
                        help="ignore calibrate's anchor runs; use the plain law constants")

    r = sub.add_parser("run", help="plan the best placement, then launch llama.cpp chat with those flags")
    r.add_argument("--gguf", required=True); hwargs(r)
    r.add_argument("--serve", action="store_true", help="launch llama-server instead of interactive chat")
    r.add_argument("--extra", default=None, help="extra flags passed through to llama.cpp")

    b = sub.add_parser("bench", help="measure real tok/s with the planned flags; print predicted vs measured")
    b.add_argument("--gguf", required=True); hwargs(b)
    b.add_argument("--reps", type=int, default=3)
    b.add_argument("--depth", type=int, default=None, help="bench at KV depth N (llama-bench -d): measures the Law 4 v2 context term on YOUR box")
    b.add_argument("--contribute", action="store_true", help="print a pre-filled, opt-in GitHub issue with your predicted-vs-measured point (you review before submitting; nothing auto-sent)")

    rp = sub.add_parser("report", help="one-page forwardable Markdown report: verdict, placement, "
                                       "binding constraint, quality - every number labeled")
    rp.add_argument("--gguf", required=True, help="the file the report is about (autospec grounds every number)")
    hwargs(rp)                       # --model/--machine/--bits/--ctx/... identical to run/bench
    rp.add_argument("--out", default=None,
                    help="output path (default: quantprobe-report-<model>.md in the current directory)")
    rp.add_argument("--bench-log", default=None,
                    help="a llama-bench log to quote as the [measured] column next to the prediction")

    d = sub.add_parser("dashboard", help="launch llama-server with planned flags + a live predicted-vs-measured chat page")
    d.add_argument("--gguf", required=True); hwargs(d)
    d.add_argument("--port", type=int, default=8077); d.add_argument("--server-port", type=int, default=8090)
    d.add_argument("--no-open", action="store_true")

    o = sub.add_parser("audit-ollama", help="price the models already in your ollama store, on this machine")
    hwargs(o)
    o.add_argument("--store", default=None, help="ollama models dir (default: $OLLAMA_MODELS or ~/.ollama/models)")
    o.add_argument("--measure", action="store_true",
                   help="also RUN ollama and report its measured eval rate beside the prediction")

    g = sub.add_parser("target", help="INVERSE planning: give a tok/s target, get the smartest feasible model + the speed-intelligence ladder")
    g.add_argument("--tps", type=float, required=True, help="minimum tok/s you need")
    hwargs(g)
    g.add_argument("--ladder", action="store_true", help="print the full speed-vs-intelligence ladder")

    z = sub.add_parser("quantize", help="COMPRESS: build a depth-aware GGUF from a model (protect the fragile band, rest 2-bit)")
    z.add_argument("--gguf", required=True, help="high-precision source GGUF (f16/bf16/Q8)")
    z.add_argument("--out", default=None, help="output path (default: <model>-depthaware.gguf)")
    z.add_argument("--protect-late", type=int, default=12, help="protect the last N layers at 4-bit (default 12)")
    z.add_argument("--protect", default=None, help="protect an explicit band LO-HI (e.g. 36-47) instead of --protect-late")
    z.add_argument("--imatrix", default=None, help="importance-matrix file (build one with `probe --imatrix auto`); improves quality at zero size cost")
    z.add_argument("--recipe", default=None, help="use a MEASURED fragile band from the community atlas instead of the default guess (see: quantprobe recipes)")
    z.add_argument("--llama-dir", default=None)
    z.add_argument("--dry", action="store_true")

    sub.add_parser("recipes", help="the community fragility atlas: bands already measured, so you can skip the probe")

    o = sub.add_parser("optimize", help="cheapest path to a target speed: search bits x placement x KV x hardware over the law")
    o.add_argument("--tps", type=float, default=None, help="target tok/s (omit for the speed frontier)")
    o.add_argument("--max-quality", type=float, default=None, help="quality-cost ceiling (default 1.12)")
    o.add_argument("--any-runtime", action="store_true", help="include placements needing expert-caching runtimes (ktransformers-class), not just stock llama.cpp")
    o.add_argument("--allow-prune", action="store_true", help="include REAP-class pruned variants (domain-specialized: +39%% out-of-domain ppl measured)")
    hwargs(o)

    au = sub.add_parser("auto", help="ONE command: model in, running setup out - optimizer picks the bits, the closest quant is fetched, run command printed")
    au.add_argument("target", nargs="?", default=None,
                    help="model preset (qwen3-30b, qwen3-coder, glm-air, laguna-s, gemma-12b, mistral-7b, qwen3-235b, glm-4.7, glm-744b, kimi-k2.6, gpt-oss-120b, llama-70b, deepseek-16b) or a HF GGUF repo id; omit it and quantprobe ASKS (interactive)")
    au.add_argument("--dir", default="./models", help="download directory (default ./models)")
    au.add_argument("--run", action="store_true", help="launch chat immediately after the download")
    hwargs(au)
    au.add_argument("--tps", type=float, default=None, help="target tok/s for the optimizer")
    au.add_argument("--max-quality", type=float, default=None)
    au.add_argument("--allow-prune", action="store_true"); au.add_argument("--any-runtime", action="store_true")
    au.add_argument("--custom", action="store_true", help="THE PRODUCT: fetch a high-precision source, probe YOUR model (~30-60 min), build a depth-aware GGUF personalized to it (machine-gated: skipped with an explanation when this box doesn't need sub-3-bit surgery)")
    au.add_argument("--yes", "-y", action="store_true", help="skip time-commitment confirmations")
    au.add_argument("--no-imatrix", action="store_true", help="skip importance-matrix calibration in --custom (faster build, measurably worse quality)")
    au.add_argument("--force-custom", action="store_true", help="build the depth-aware GGUF even when the optimizer says this machine doesn't need it")
    au.add_argument("--serve", action="store_true", help="with --run: llama-server instead of chat")
    au.add_argument("--extra", default=None)

    hwp = sub.add_parser("hw", help="detect THIS machine's memory tiers (no flags needed); every value tagged with its source")
    hwp.add_argument("--measure", default=None, metavar="FILE", help="also MEASURE sequential disk read on a real file (e.g. a GGUF)")

    cal = sub.add_parser("calibrate", help="MEASURE this machine's constants (RAM stream, disk, GPU boost health, decode anchor) - plan uses them automatically")
    cal.add_argument("--model", default=None, metavar="FILE", help="a GGUF on this machine: unlocks disk measurement + a real pure-CPU decode anchor")
    cal.add_argument("--skip-bench", action="store_true", help="measure RAM/disk/GPU only; skip the llama-bench anchor")
    cal.add_argument("--llama-dir", default=None, help="folder containing llama-bench, if not on PATH")

    f = sub.add_parser("fetch", help="robust HF download (Range-resume, retry)")
    f.add_argument("repo", help="HF repo, or a preset: qwen3-30b, glm-air, deepseek-16b, qwen3-0.6b"); f.add_argument("dest"); f.add_argument("files", nargs="*")
    f.add_argument("--force", action="store_true", help="re-download even if a same-named file exists (a stale local file with the same name once fed an incompatible draft model to llama-speculative - three crashed runs before the collision was spotted)")

    a = ap.parse_args()
    if a.cmd is None:
        from . import branding, __version__
        print(branding.banner("v" + __version__))
        ap.print_help()
        return
    if a.cmd == "probe":
        from . import probe
        probe.run(a)
    elif a.cmd == "plan":
        from . import plan
        plan.run(a)
    elif a.cmd == "run":
        from . import runtime
        runtime.run(a)
    elif a.cmd == "bench":
        from . import runtime
        runtime.bench(a)
    elif a.cmd == "report":
        from . import report
        report.run(a)
    elif a.cmd == "dashboard":
        from . import dashboard
        dashboard.dashboard(a)
    elif a.cmd == "target":
        from . import target
        target.run(a)
    elif a.cmd == "auto":
        from . import auto
        auto.run(a)
    elif a.cmd == "optimize":
        from . import optimize
        optimize.run(a)
    elif a.cmd == "hw":
        from . import detect
        detect.run(a)
    elif a.cmd == "calibrate":
        from . import calibrate
        calibrate.run(a)
    elif a.cmd == "recipes":
        from . import recipes
        recipes.run(a)
    elif a.cmd == "quantize":
        from . import probe
        probe.quantize(a)
    elif a.cmd == "audit-ollama":
        from . import ollama as ollamamod
        ollamamod.run(a)
    else:
        from . import fetch
        fetch.run(a)


if __name__ == "__main__":
    main()
