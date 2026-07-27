# External engine: antirez/ds4 "DwarfStar", read 2026-07-27

A purpose-built inference engine for DeepSeek V4 and GLM 5.2 — Metal-first, with CUDA and ROCm
paths. Deliberately narrow: "not a general GGUF runner." Directly relevant to us on three counts.

## 1. Independent convergence on Law 3's structural claim

DwarfStar's quantization scheme, verbatim: *"only the routed MoE experts are quantized"* —
`IQ2_XXS` on up/gate, `Q2_K` on down — while *"the other components (shared experts, projections,
routing) are left untouched to guarantee quality."*

That is our depth-aware recipe's central structural finding, reached independently by someone
building a production engine: **protect the always-active tensors, spend the bits on the routed
experts.** We arrived at it from a causal decomposition (PAPER_MOE §: the KV-latent carries 87% of
the collapse); antirez arrived at it from engineering a shipping product. Convergence from a
different direction is worth more than another confirmation from ours.

Difference worth noting: they keep the protected set at **full precision**; our recipe holds it at
~4.5 bits. Theirs is the more conservative point on the same axis.

## 2. Two scored predictions, both badly wrong, in opposite directions

| case | ours | theirs | error |
|---|---|---|---|
| GLM-5.2 IQ2_XXS, tensor-parallel over 2× M5 Max (Thunderbolt 5) | 34.6 | ~16.8 | **+106%** |
| GLM-5.2 class, SSD streaming, single Mac | 0.7 | ~4.8 | **−85%** |

**The over-prediction is our bug.** `plan.agg_bw(v, 0.85)` applies a flat 0.85 efficiency to any
multi-device aggregate — the same number whether the devices are joined by NVLink, PCIe,
Thunderbolt or WiFi. **Our law has no interconnect term at all.** ds4's own data shows the link is
the dominant variable: on identical hardware and prompt, Thunderbolt 5 gives 582 pp / 25.1 tg,
WiFi 250 / 10.7, internet-over-VPN 114 / 3.6. A 5× spread from the cable alone, which we model as
a constant.

**The under-prediction is a known gap, not a surprise.** Our disk tier models naive LRU streaming;
DwarfStar (like colibri) prefetches. `docs/DEEP-DIVE.md` already states that colibri-style
lookahead is "exactly what closes the gap between my naive-streaming numbers and its engine's."
This quantifies it for the first time: **~7× on the streaming tier.**

## 3. What it says about the fork question we just closed

We concluded (#22 and the ceiling analysis) that a custom runtime buys 1–6% end-to-end, because
the static frontier already sits near both per-axis maxima. That analysis was run **entirely in
the VRAM-resident and host-resident regimes**, which is where our measurements live.

DwarfStar's streaming numbers suggest the conclusion does **not** extend to the disk tier, where a
purpose-built engine with prefetch appears to be worth ~7×, not 6%. Our "don't fork" verdict
should be scoped to the regime it was computed in, rather than stated as a general claim about
custom runtimes.

That is a correction to how I framed it, not to the arithmetic.

## Follow-ups this opens

- **Interconnect term for `agg_bw`** — currently a flat 0.85. ds4 publishes a clean 5× ladder
  (TB5 / WiFi / VPN) that could calibrate it. This is a real gap affecting every multi-device
  prediction we make, and all of those are `[est]`.
- **Streaming-tier prefetch** — our disk model is naive by construction and is ~7× pessimistic
  against an engine that prefetches. Worth stating in the limitations rather than only in a
  deep-dive.
- Their Mac numbers are the first external data against our `mac-*` presets, all of which are
  marked `[est, unvalidated]`. Scoring them properly needs DeepSeek V4 Flash/Pro parameter counts,
  which the README does not state.

Source: https://github.com/antirez/ds4 — numbers quoted from its README, not reproduced by us.
