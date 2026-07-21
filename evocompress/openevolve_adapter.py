"""OPTIONAL adapter so OpenEvolve (+ an LLM like Gemini/Claude) can drive the same
pipeline search later.  This is a stub: it is NOT required for the GA path and does
not import openevolve.  OpenEvolve evolves a *program* that emits a pipeline spec
JSON; this module's ``evaluate`` scores that spec on a configured corpus and returns
the metrics OpenEvolve maximizes.

Wiring (later):
    1. Point ``EVOCOMPRESS_CORPUS`` at a directory of training files.
    2. Have the evolved program write a pipeline spec JSON (see Pipeline.spec()).
    3. Configure OpenEvolve to call ``evaluate(<spec_path>)`` and maximize
       ``metrics['combined_score']``.

A sample OpenEvolve config is returned by ``sample_config()``.
"""

from __future__ import annotations

import glob
import json
import os
from typing import List

from .evaluator import score
from .pipeline import Pipeline


def _load_corpus(corpus_dir: str, max_bytes_per_file: int = 0) -> List[bytes]:
    files = []
    for path in sorted(glob.glob(os.path.join(corpus_dir, "*"))):
        if os.path.isfile(path):
            with open(path, "rb") as fh:
                data = fh.read()
            if max_bytes_per_file:
                data = data[:max_bytes_per_file]
            files.append(data)
    return files


def evaluate(pipeline_spec_path: str, corpus_dir: str | None = None) -> dict:
    """Score a pipeline spec JSON file.  Returns a metrics dict with a single
    ``combined_score`` for the optimizer plus the raw sub-metrics.

    The corpus directory defaults to the ``EVOCOMPRESS_CORPUS`` env var.
    Invalid (non round-tripping) pipelines get a large negative score.
    """
    corpus_dir = corpus_dir or os.environ.get("EVOCOMPRESS_CORPUS", "")
    if not corpus_dir or not os.path.isdir(corpus_dir):
        raise FileNotFoundError(
            "set EVOCOMPRESS_CORPUS to a directory of training files, or pass corpus_dir"
        )

    with open(pipeline_spec_path, "r", encoding="utf-8") as fh:
        spec = json.load(fh)

    try:
        pipe = Pipeline.from_spec(spec)
        files = _load_corpus(corpus_dir)
        m = score(pipe, files)
    except Exception as exc:  # invalid program / pipeline
        return {"combined_score": -1e9, "error": str(exc), "roundtrip_ok": False}

    if not m.roundtrip_ok:
        return {"combined_score": -1e9, "roundtrip_ok": False, "error": m.error}

    return {
        "combined_score": m.ratio,
        "ratio": m.ratio,
        "encode_MBps": m.encode_MBps,
        "decode_MBps": m.decode_MBps,
        "roundtrip_ok": True,
        "total_in": m.total_in,
        "total_out": m.total_out,
    }


def sample_config() -> dict:
    """A minimal example OpenEvolve config (as a dict; serialize to YAML to use)."""
    return {
        "max_iterations": 200,
        "llm": {"primary_model": "gemini-2.0-flash", "temperature": 0.8},
        "evaluator": {
            "module": "evocompress.openevolve_adapter",
            "function": "evaluate",
            "metric": "combined_score",
            "goal": "maximize",
        },
        "notes": "The evolved program must emit a pipeline spec JSON; see Pipeline.spec().",
    }


if __name__ == "__main__":  # pragma: no cover - manual smoke
    import sys

    if len(sys.argv) > 1:
        print(json.dumps(evaluate(sys.argv[1]), indent=2))
    else:
        print(json.dumps(sample_config(), indent=2))
