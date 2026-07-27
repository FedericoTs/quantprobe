# Three external GLM-5.2 datapoints, and a correction to my own analysis

Recorded 2026-07-27. None of these are our measurements; each is scored as reported, with its
confounds stated. Two arrived from Federico (X post, GitHub repo), one from antirez/ds4.

| # | cluster | interconnect | quant | measured decode | confounds |
|---|---|---|---|---|---|
| 1 | 4× "RTX 6000" | not stated | ~3-bit (assumed) | 103 tok/s @262K | card model ambiguous |
| 2 | 4× DGX Spark GB10 | Fabric NFS | INT4/INT8 + NVFP4 KV | 31.5 mean / 41.4 peak | **MTP k=5** |
| 3 | 2× M5 Max (ds4) | Thunderbolt 5 | IQ2_XXS | ~16.8 | custom engine |

## Scored

| cluster | ours | measured | error |
|---|---|---|---|
| 4× DGX Spark, 4.5-bit, η=0.79 | 35.4 | 31.5 | **+12%** |
| 4× RTX 6000 **Ada** (960 GB/s), 3-bit, η=0.50 | 105.1 | 103 | **+2%** |
| 4× RTX PRO 6000 **Blackwell** (1792 GB/s), 3-bit, η=0.62 | 243.3 | 103 | **+136%** |

## The correction

I first reported the RTX-6000 cluster as a **+136% law failure** and built an argument on it: that
`agg_bw`'s flat 0.85 multi-device factor lacks an interconnect term, and that this explained the
error. **Both halves of that were wrong, and the second datapoint is what exposed it.**

- **The interconnect hypothesis is refuted by direction.** If we were missing an interconnect
  penalty, the *network-joined* cluster (Spark over NFS) should be over-predicted worst. It is our
  **best** result at +12%, while the fast-interconnect cluster was the bad one. A missing penalty
  cannot produce that ordering.
- **The +136% was mostly my hardware guess.** "4× 6000s" names two cards nearly two× apart in
  bandwidth: RTX 6000 Ada (960 GB/s, 2022) and RTX PRO 6000 Blackwell (1792 GB/s, 2025). Assuming
  Ada, the law lands at **105.1 against 103 — a 2% error.** I assumed Blackwell and reported a law
  failure that may not exist.

**Cannot be resolved without asking which card.** Recorded as ambiguous rather than scored.

## What survives

The flat 0.85 aggregation factor is still **unvalidated** — ds4's own link ladder (same hardware,
same prompt: Thunderbolt 5 582/25.1, WiFi 250/10.7, VPN 114/3.6, a 5× spread) shows the
interconnect matters enormously in *some* regime. But these two 4-device clusters do not
demonstrate that our constant is wrong, and I should not have said they did.

The honest state: **one multi-device prediction is good to +12%, one is unscoreable until the
hardware is known, and the aggregation factor remains untested.**

## Confounds that must be respected before any of these is used to move a constant

- Datapoint 2 runs **MTP k=5**. Our law does not model speculative decoding at all (v1.10.4:
  "there is no single multiplier to apply"). Its 31.5 is inflated by an unknown factor, so the
  +12% is flattering to us by an unknown amount.
- Datapoint 1's effective bits are not stated; 3-bit is our inference from what fits.
- Datapoint 3 runs a custom engine, not llama.cpp, so it tests the hardware model but not our
  runtime assumptions.

**None of these should move a constant.** They are recorded as external observations with their
uncertainty intact.
