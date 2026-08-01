# Prereg #91 — the quality half of the KV-quant advice (V-08 / E-10 debt)

**Staked:** 2026-07-30, BEFORE the perplexity runs. **Model:** Qwen3-30B-A3B Q2_K,
split placement (the ladder's own emitted config, cal_id lineage a19aeee4).
**Corpus:** weights/data/wikitext2_test.txt.

## Why

We ship `-ctk q8_0 -ctv q8_0` on speed evidence alone: +37.0% vs f16 at d16384
(prereg #35) and 3.04× vs the `-nkvo 1` row we currently emit for deep workloads
(10.59 vs 3.48 tok/s, prereg #25). Zero quality numbers — the gap E-10 conceded
publicly. One perplexity A/B closes it, and prereg #25 pre-wrote both branches.

## Staked gate (D-01's own precedent as the bar)

D-01 killed a 1.335× speed lever for costing 1.206× perplexity, because the bit
ladder bought 1.424× for 1.048×. Therefore:

- **P1:** ppl(q8_0 KV) / ppl(f16 KV) at matched 8k-token chunks is **≤ 1.02**
  → q8_0 KV **replaces** the `-nkvo` deep-workload advice as frontier row 3.
- **KILL:** ratio > 1.02 → q8_0 becomes a *disclosed option only*, advice text
  keeps the "judge quality yourself" line permanently.
- **Pre-committed EITHER WAY (from #25, not renegotiable):** the `-nkvo 1` row is
  withdrawn from deep-workload advice — it is dominated at depth regardless.

## The failing input, constructed BEFORE trusting any result

The lever is depth-only (−5.8% at ~1k). Default `llama-perplexity` chunks at 512
tokens, so the cache never reaches depth and the A/B returns ~1.000× — **a
measurement that cannot vary** (the #85-arms-C/D shape). Guards, all mechanical:

1. Chunks run at **-c 8192** (each chunk is a fresh 8k-deep cache). 512 is refused.
2. The two arms' logs must show **different KV cache types** (grep'd, not assumed).
3. If the two final ppl values agree to 4 decimals, the run is declared
   **UNINFORMATIVE, not a pass** — that is the cannot-vary signature.
4. Same binary, same session, back-to-back; no other CUDA process (checked).

## Disclosure

Wikitext-2 prose cannot see niche-domain degradation (the MoneroApe/E-10 report
concerns 100k+ code contexts we cannot reach on this box). A pass here bounds the
*generic* cost only; the advice line keeps a niche-domain caveat regardless of P1.

---

## Correction (2026-07-31, adversarial final audit — appended, staked text above untouched)

The background paragraph mis-cites the +37.0% speed number to **prereg #35**. It was measured in
**prereg #25** (P-1: q8_0 KV in VRAM vs f16 KV in VRAM at d16384, +37.0%); prereg #35 is the
composition/depth-condition experiment and itself attributes the number to #25 three times. The
mis-citation propagated into the L-24 register entry (corrected there with an inline note) and
into the then-unpublished L-25 draft (fixed before first publication). Nothing about this run's
staked gate, guards, or scored result changes.
