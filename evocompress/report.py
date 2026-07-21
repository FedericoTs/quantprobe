"""Reporting: evaluate the evolved champion against strong baselines on the
HELD-OUT test split, print a table, write a results.json leaderboard, and render a
Pareto scatter (ratio vs decode throughput)."""

from __future__ import annotations

import json
import os
from typing import List, Sequence, Tuple

from . import backends
from .evaluator import Metrics, score
from .pipeline import Pipeline

# (label, backend, level) for the standard baselines.
_BASELINE_SPECS = [
    ("store", "store", 0),
    ("gzip-9", "gzip", 9),
    ("bz2-9", "bz2", 9),
    ("lzma-9", "lzma", 9),
    ("zstd-19", "zstd", 19),
    ("brotli-11", "brotli", 11),
]


def baseline_pipelines() -> List[Tuple[str, Pipeline]]:
    avail = set(backends.available_backends())
    out = []
    for label, backend, level in _BASELINE_SPECS:
        if backend in avail:
            out.append((label, Pipeline([], backend, level)))
    return out


def evaluate_entries(entries: Sequence[Tuple[str, Pipeline]], files: Sequence[bytes]) -> List[dict]:
    rows = []
    for label, pipe in entries:
        m: Metrics = score(pipe, files)
        rows.append(
            {
                "name": label,
                "pipeline": pipe.describe(),
                "spec": pipe.spec(),
                "ratio": round(m.ratio, 4),
                "encode_MBps": round(m.encode_MBps, 2),
                "decode_MBps": round(m.decode_MBps, 2),
                "roundtrip_ok": m.roundtrip_ok,
                "total_in": m.total_in,
                "total_out": m.total_out,
            }
        )
    return rows


def print_table(rows: List[dict], highlight: str | None = None) -> None:
    header = f"{'method':<22} {'ratio':>8} {'enc MB/s':>9} {'dec MB/s':>9} {'out bytes':>12} {'rt':>4}"
    print(header)
    print("-" * len(header))
    for r in sorted(rows, key=lambda x: x["ratio"], reverse=True):
        mark = " *" if highlight and r["name"] == highlight else ""
        rt = "ok" if r["roundtrip_ok"] else "FAIL"
        print(
            f"{r['name']:<22} {r['ratio']:>8.4f} {r['encode_MBps']:>9.2f} "
            f"{r['decode_MBps']:>9.2f} {r['total_out']:>12,} {rt:>4}{mark}"
        )


def write_results_json(path: str, rows: List[dict], meta: dict) -> None:
    payload = {
        "meta": meta,
        "leaderboard": sorted(rows, key=lambda x: x["ratio"], reverse=True),
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def render_pareto(path: str, rows: List[dict], champion: str | None = None) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False

    valid = [r for r in rows if r["roundtrip_ok"] and r["decode_MBps"] > 0]
    if not valid:
        return False

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for r in valid:
        is_champ = champion is not None and r["name"] == champion
        ax.scatter(
            r["decode_MBps"],
            r["ratio"],
            s=140 if is_champ else 70,
            marker="*" if is_champ else "o",
            color="crimson" if is_champ else "steelblue",
            zorder=3 if is_champ else 2,
            edgecolors="black",
            linewidths=0.5,
        )
        ax.annotate(
            r["name"],
            (r["decode_MBps"], r["ratio"]),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=8,
        )
    ax.set_xlabel("decode throughput (MB/s)  -- higher is better")
    ax.set_ylabel("compression ratio  -- higher is better")
    ax.set_title("evo-compress: champion vs baselines (held-out test)")
    ax.grid(True, alpha=0.3)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return True


def compare(
    champion: Pipeline,
    files: Sequence[bytes],
    out_dir: str,
    meta: dict,
    champion_label: str = "evolved",
) -> dict:
    """Full comparison: champion + baselines on held-out files -> table, json, png."""
    entries = [(champion_label, champion)] + baseline_pipelines()
    rows = evaluate_entries(entries, files)
    print_table(rows, highlight=champion_label)

    results_path = os.path.join(out_dir, "results.json")
    png_path = os.path.join(out_dir, "pareto.png")
    write_results_json(results_path, rows, meta)
    png_ok = render_pareto(png_path, rows, champion=champion_label)

    champ = next(r for r in rows if r["name"] == champion_label)
    baselines = [r for r in rows if r["name"] != champion_label]
    best_baseline = max(baselines, key=lambda r: r["ratio"]) if baselines else None
    return {
        "rows": rows,
        "results_path": results_path,
        "png_path": png_path if png_ok else None,
        "champion": champ,
        "best_baseline": best_baseline,
    }
