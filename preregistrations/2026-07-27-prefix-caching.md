# Pre-registration #29: prefix caching — the lever that changes the asymptote, not the percentage

**Author:** Federico Sciuca · **Date staked:** 2026-07-27, BEFORE the measurement. **Status: STAKED.**

## The claim

Every prefill lever measured so far optimises the COST of doing prefill (L-12: capped by this
GPU's FLOPs at ~430-445 t/s). A reused prefix does not do prefill at all. llama-server's
`cache_prompt` keeps the KV of the previous request's prompt and only processes the suffix, so a
RAG/document-QA workload — the same 2048-token document, many questions — pays the prefill cost
ONCE, and time-to-first-token on every later request collapses to the cost of the new suffix.

## Protocol

llama-server, split placement, one session. A ~2048-token prompt (wikitext) + a short question.
Request 1 with `cache_prompt: true` (cold). Request 2: same document, DIFFERENT question (warm —
the document prefix is cached, the question is not). Request 3: fully identical (best case).
Metric: `prompt_ms` from the server's own timings, r=2 per cell.

## Stakes

- **P-1 (the asymptote moves).** Warm same-document/different-question prompt time is **≥5×
  lower** than cold. Not a percentage — a different regime.
- **P-2 (it is really the prefix).** The fully-identical request is **≥20× lower** than cold.
- **P-3 (decode untouched).** Generation tok/s identical across all three, ±5% — the cache must
  not tax decode.

## Refuted if

P-1 fails — then the server re-processes the document despite the flag, prefix reuse does not
apply to this placement, and the register's U-04 dies for this stack.

## What ships

If it holds: the plan output for long-prompt workloads names `cache_prompt` alongside the
placement, with the measured ratio — because for the workloads the frontier used to serve
(RAG 50:1, document QA 200:1), skipping prefill dwarfs every placement decision we have measured.

---

## Scored (2026-07-27, log: `weights/data/prereg29_prefix_cache.log`)

**Verdict: P-1 HIT (29×), P-2 HIT (110×), P-3 HIT. The asymptote moves.**

| request | prompt time | tokens processed | decode |
|---|---|---|---|
| cold, question A | 5380.8 ms | 1889 | 21.77 |
| **warm, question B** | **183.4 ms** | 8 | 21.74 |
| warm, question C | 203.0 ms | 9 | 21.73 |
| identical repeat | 49.1 ms | 1 | 21.89 |

- **P-1 (≥5×): HIT at 29.3×.** The server processes only the 8-token suffix; the 1889-token
  document is never prefilled again.
- **P-2 (≥20×): HIT at 109.6×.**
- **P-3 (decode ±5%): HIT.** 21.73–21.89 across all four.

For the workloads the old frontier existed to serve — RAG at 50:1, document QA at 200:1 — this
dwarfs every placement decision in the register: the entire prefill side of the total-time
equation collapses to near zero on every request after the first. One caveat, disclosed rather
than discovered later: `cache_prompt` reuses KV across requests within one server process, so it
serves the many-questions-one-document shape, not fresh documents, and a server restart is cold.

**Wired into:** `findings/REGISTER.json:V-10` · `quantprobe/plan.py` long-prompt advice.
