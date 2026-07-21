"""Evolution tracker: append-only leaderboard.json + EVOLUTION_LOG.md.

Records every codec variant's full config + KPIs with a timestamp, so we never lose
progress and can see the trajectory. The headline KPI is size-weighted save% on the
real-model dataset with byte-exact round-trip; bf16 is reported separately (it is the
priority dtype for modern LLMs).
"""

from __future__ import annotations

import datetime as _dt
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
LEADERBOARD = os.path.join(HERE, "results", "leaderboard.json")
LOG = os.path.join(HERE, "EVOLUTION_LOG.md")


def _now() -> str:
    return _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def record(result: dict, note: str = "") -> None:
    os.makedirs(os.path.dirname(LEADERBOARD), exist_ok=True)
    entries = []
    if os.path.exists(LEADERBOARD):
        with open(LEADERBOARD, encoding="utf-8") as fh:
            entries = json.load(fh).get("entries", [])
    entry = dict(result)
    entry["timestamp"] = _now()
    entry["note"] = note
    entries.append(entry)
    with open(LEADERBOARD, "w", encoding="utf-8") as fh:
        json.dump({"entries": entries}, fh, indent=2)
    _append_log(entry)


def _append_log(entry: dict) -> None:
    ov = entry["overall"]
    bd = entry.get("by_dtype", {})
    bf = bd.get("bf16", {})
    line = (
        f"\n### {entry['timestamp']} | {entry['codec']}\n\n"
        f"- **overall: save {ov['save_pct']}%** (ratio {ov['ratio']}), "
        f"enc {ov['enc_MBps']} / dec {ov['dec_MBps']} MB/s, "
        f"round-trip {'OK' if ov['rt_ok'] else 'FAIL'} ({entry.get('n_tensors','?')} tensors)\n"
        f"- **bf16: save {bf.get('save_pct','-')}%** (ratio {bf.get('ratio','-')}, "
        f"dec {bf.get('dec_MBps','-')} MB/s)\n"
        f"- by dtype: "
        + ", ".join(f"{dt} {m['save_pct']}%" for dt, m in sorted(bd.items()))
        + "\n"
        f"- config: `{json.dumps(entry.get('config', {}))}`\n"
    )
    if entry.get("note"):
        line += f"- note: {entry['note']}\n"
    header = ""
    if not os.path.exists(LOG):
        header = (
            "# Weight-Codec Evolution Log\n\n"
            "Every codec variant's KPIs, newest at the bottom. Headline KPI = "
            "size-weighted **save%** on the real-model dataset with byte-exact "
            "round-trip; **bf16** is the priority dtype. Speeds are MB/s of input.\n\n"
            "Baselines to beat: raw zstd/lzma, and **zipnn-zstd19** (byte-split + zstd,"
            " ~33% on real bf16). Our edge: a context model on the exponent plane +"
            " (creatively) predictive/structural coding to break the mantissa wall.\n\n"
            "---\n"
        )
    with open(LOG, "a", encoding="utf-8") as fh:
        if header:
            fh.write(header)
        fh.write(line)
