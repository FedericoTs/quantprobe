"""End-to-end spike driver.

  python -m experiments.run_spike --domain time-series --engine ga \
      --population 80 --generations 40 --seed 0 --objective max_ratio

Loads a TRAIN/TEST split, evolves pipelines on TRAIN, selects the champion by
TRAIN fitness, evaluates it (and the standard baselines) on the HELD-OUT TEST
split with byte-exact round-trip, prints a comparison, writes results.json + a
Pareto PNG, appends to EXPERIMENT_LOG.md, and prints a GO/NO-GO verdict.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys

# make the repo root importable when run as a script
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from data import fetch_data  # noqa: E402
from evocompress import backends, report  # noqa: E402
from evocompress.engine import Engine, GAConfig  # noqa: E402
from evocompress.evaluator import score  # noqa: E402
from evocompress.fitness import FitnessConfig  # noqa: E402
from evocompress.genome import Gene, Genome, make_search_space  # noqa: E402
from evocompress.pipeline import Pipeline  # noqa: E402

RESULTS_DIR = os.path.join(_ROOT, "results")
LOG_PATH = os.path.join(_ROOT, "experiments", "EXPERIMENT_LOG.md")


def seed_genomes(domain: str) -> list[Genome]:
    """A few sensible starting points so the champion is never worse than a
    standard baseline and the search has good genetic material to recombine."""
    avail = set(backends.available_backends())
    seeds: list[Genome] = [Genome([], "store", 0)]
    if "zstd" in avail:
        seeds.append(Genome([], "zstd", 19))
    if "brotli" in avail:
        seeds.append(Genome([], "brotli", 11))
    seeds.append(Genome([], "lzma", 9))
    if domain in ("time-series", "ml-weights"):
        be = "zstd" if "zstd" in avail else "lzma"
        lv = 19 if be == "zstd" else 9
        seeds.append(Genome([Gene("transpose", {"stride": 4}), Gene("delta", {"size": 1})], be, lv))
        seeds.append(Genome([Gene("float_split", {"dtype": "f4"}), Gene("delta", {"size": 1})], be, lv))
        seeds.append(Genome([Gene("delta", {"size": 4}), Gene("zigzag", {"size": 4})], be, lv))
    return seeds


def best_general_baseline(rows: list[dict]) -> dict:
    """Pick the strongest *general-purpose* baseline available (zstd-19/brotli-11,
    falling back to lzma/bz2)."""
    prefer = ["zstd-19", "brotli-11", "lzma-9", "bz2-9", "gzip-9"]
    by_name = {r["name"]: r for r in rows}
    for name in prefer:
        if name in by_name:
            return by_name[name]
    return max((r for r in rows if r["name"] != "evolved"), key=lambda r: r["ratio"])


def verdict(champ: dict, general: dict) -> tuple[str, str]:
    cr, gr = champ["ratio"], general["ratio"]
    cd, gd = champ["decode_MBps"], general["decode_MBps"]
    if not champ["roundtrip_ok"]:
        return "INVALID", "champion failed round-trip (should never happen)"
    if cr >= gr * 1.05:
        return "GO", f"evolved ratio {cr:.4f} beats {general['name']} {gr:.4f} by >=5%"
    if cr >= gr * 0.99 and cd >= gd * 2.0:
        return "GO", (
            f"evolved matches {general['name']} ratio ({cr:.4f} vs {gr:.4f}) "
            f"at >=2x decode ({cd:.1f} vs {gd:.1f} MB/s)"
        )
    if cr > gr:
        return "MARGINAL", f"evolved beats {general['name']} ({cr:.4f} > {gr:.4f}) but by <5%"
    return "NO-GO", f"evolved {cr:.4f} did not beat {general['name']} {gr:.4f} on held-out data"


def append_log(entry: dict) -> None:
    line = (
        f"\n### {entry['timestamp']} | domain={entry['domain']} engine={entry['engine']} "
        f"seed={entry['seed']}\n\n"
        f"- config: pop={entry['population']} gen={entry['generations']} "
        f"objective={entry['objective']} max_len={entry['max_len']} "
        f"train={entry['n_train']} test={entry['n_test']} files, evals={entry['evaluated']}\n"
        f"- champion: `{entry['champion']}`\n"
        f"- HELD-OUT ratio: evolved={entry['champ_ratio']:.4f} "
        f"vs {entry['general_name']}={entry['general_ratio']:.4f} "
        f"(best baseline {entry['best_baseline_name']}={entry['best_baseline_ratio']:.4f})\n"
        f"- decode MB/s: evolved={entry['champ_dec']:.1f} "
        f"vs {entry['general_name']}={entry['general_dec']:.1f}\n"
        f"- **verdict: {entry['verdict']}** -- {entry['reason']}\n"
    )
    with open(LOG_PATH, "a", encoding="utf-8") as fh:
        fh.write(line)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="evo-compress feasibility spike")
    ap.add_argument("--domain", default="time-series", choices=fetch_data.DOMAINS)
    ap.add_argument("--engine", default="ga", choices=["ga"])
    ap.add_argument("--population", type=int, default=80)
    ap.add_argument("--generations", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--objective", default="max_ratio",
                    choices=["max_ratio", "ratio_at_speed", "pareto"])
    ap.add_argument("--speed-floor", type=float, default=0.0, help="decode MB/s floor")
    ap.add_argument("--max-len", type=int, default=5, help="max transforms in a pipeline")
    ap.add_argument("--train-frac", type=float, default=0.7)
    ap.add_argument("--max-bytes", type=int, default=0, help="truncate each file (0 = full)")
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--islands", type=int, default=2)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--out-dir", default=RESULTS_DIR)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    print(f"== evo-compress spike :: domain={args.domain} engine={args.engine} seed={args.seed} ==")
    print(f"backends available: {backends.available_backends()}")

    # 1. data + split
    fetch_data.ensure_domain(args.domain, download=args.download,
                             max_bytes=args.max_bytes, seed=args.seed)
    all_paths = fetch_data.list_files(args.domain)
    if len(all_paths) < 2:
        print("need at least 2 files for a train/test split", file=sys.stderr)
        return 2
    train_paths, test_paths = fetch_data.split_train_test(
        all_paths, train_frac=args.train_frac, seed=args.seed)
    train_files = fetch_data.load_files(train_paths, max_bytes=args.max_bytes)
    test_files = fetch_data.load_files(test_paths, max_bytes=args.max_bytes)
    print(f"TRAIN: {len(train_files)} files ({sum(map(len, train_files)):,} B)  "
          f"HELD-OUT TEST: {len(test_files)} files ({sum(map(len, test_files)):,} B)")

    # 2. search space + fitness + engine
    space = make_search_space(args.domain, max_len=args.max_len)
    fcfg = FitnessConfig(objective=args.objective, speed_floor_MBps=args.speed_floor)
    gacfg = GAConfig(population=args.population, generations=args.generations, seed=args.seed,
                     n_islands=max(1, args.islands), patience=args.patience)
    engine = Engine(space, train_files, fcfg, gacfg, seed_genomes=seed_genomes(args.domain))

    print(f"\n-- evolving on TRAIN ({args.population} pop x {args.generations} gen) --")
    logger = (lambda *_: None) if args.quiet else (lambda msg: print("   " + msg))
    result = engine.run(log=logger)
    champ_genome = result.champion
    champ_pipeline = champ_genome.to_pipeline()
    print(f"\nchampion (by TRAIN fitness): {champ_pipeline.describe()}")
    print(f"  TRAIN ratio={result.champion_metrics.ratio:.4f} "
          f"dec={result.champion_metrics.decode_MBps:.1f} MB/s  evals={result.n_evaluated}")

    # 3. HELD-OUT evaluation: champion + baselines
    print("\n-- HELD-OUT TEST: champion vs baselines --")
    os.makedirs(args.out_dir, exist_ok=True)
    meta = {
        "domain": args.domain, "engine": args.engine, "seed": args.seed,
        "population": args.population, "generations": args.generations,
        "objective": args.objective, "n_train": len(train_files), "n_test": len(test_files),
        "champion": champ_pipeline.describe(), "champion_spec": champ_genome.spec(),
        "evaluated": result.n_evaluated,
    }
    cmp = report.compare(champ_pipeline, test_files, args.out_dir, meta, champion_label="evolved")

    # explicit round-trip confirmation on every held-out file
    rt = score(champ_pipeline, test_files)
    print(f"\nround-trip on every held-out file: "
          f"{'CONFIRMED byte-exact' if rt.roundtrip_ok else 'FAILED'} "
          f"({rt.n_files} files)")

    # 4. verdict
    general = best_general_baseline(cmp["rows"])
    champ_row = cmp["champion"]
    best_baseline = cmp["best_baseline"]
    v, reason = verdict(champ_row, general)
    print(f"\n=================== VERDICT: {v} ===================")
    print(f"  {reason}")
    print(f"  results: {cmp['results_path']}")
    if cmp["png_path"]:
        print(f"  pareto : {cmp['png_path']}")

    append_log({
        "timestamp": _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "domain": args.domain, "engine": args.engine, "seed": args.seed,
        "population": args.population, "generations": args.generations,
        "objective": args.objective, "max_len": args.max_len,
        "n_train": len(train_files), "n_test": len(test_files),
        "evaluated": result.n_evaluated, "champion": champ_pipeline.describe(),
        "champ_ratio": champ_row["ratio"], "champ_dec": champ_row["decode_MBps"],
        "general_name": general["name"], "general_ratio": general["ratio"],
        "general_dec": general["decode_MBps"],
        "best_baseline_name": best_baseline["name"], "best_baseline_ratio": best_baseline["ratio"],
        "verdict": v, "reason": reason,
    })

    # machine-readable champion spec alongside results
    with open(os.path.join(args.out_dir, "champion.json"), "w", encoding="utf-8") as fh:
        json.dump({"spec": champ_genome.spec(), "describe": champ_pipeline.describe(),
                   "held_out": champ_row, "verdict": v}, fh, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
