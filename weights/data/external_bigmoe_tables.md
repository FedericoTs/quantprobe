# External dataset: BigMoeOnEdge streaming-MoE benchmark tables, read 2026-07-30

**Source:** `https://github.com/Helldez/BigMoeOnEdge` (Apache-2.0), README.md
**Raw URL used by the script:** `https://raw.githubusercontent.com/Helldez/BigMoeOnEdge/main/README.md`
**Read on:** 2026-07-30. **Not our measurement.** Transcribed verbatim and recorded with its
provenance so that experiment #53 has a held-out set it did not generate.

`weights/exp53_two_resource_disk_tier.py` re-downloads that README on every scoring run and
aborts if any row below no longer appears in it. If you are reading this file after the upstream
README moved, the script is the authority on whether the transcription is still valid.

---

## Device and protocol, in their words

**Phone (device `P`).** "Measured on one device (12 GB RAM, 11.3 GB usable, UFS 4.x storage) over
`adb shell`, 256-token greedy decode. Each number is the best observed for that configuration,
and rows in a table can come from different benchmark sessions." All models Q4_K_M.

**Desktop (device `D`).** "a Windows x86 laptop (8 cores, 16 GB RAM, dual-channel DDR4, NVMe SSD)
with Qwen3.6-35B-A3B at ~1.5x RAM, 256-token generations." Their prose adds "~3 GB/s NVMe" and
"~0.11 s/token in every cell -> a ~9 tok/s ceiling even at zero I/O".

**Column meanings, in their words.** "*Flash/token* is data read from storage per generated token
(lower means the cache is working) and *cache hit* is the share of expert reads served from RAM
instead of flash." *k* is `--n-expert-used`.

**Their own caveat on the Qwen3.6 phone table:** "these Qwen3.6 figures are a single 96-token run
rather than the 256-token best-of protocol: treat them as indicative, and not strictly comparable
to the other models until re-measured under the full protocol."

---

## Table 1 - gpt-oss-120b (Q4_K_M per their prose): "~60 GB on a 12 GB phone"

All rows use `--overlap --dense-weights anon --no-think` (their note under the table), so every
row is `overlap = yes`. Native routing width k = 4.

| Configuration | tok/s | Flash/token | Cache hit |
|---|---:|---:|---:|
| **streamed, k=2, cache 2000 MiB, 8 lanes** | **2.2** | 590 MiB | 32% |
| streamed, k=2, no cache, 4 lanes | 1.8 | 909 MiB | — |
| streamed, default k=4, cache 2000 MiB, 8 lanes | 1.3 | 1292 MiB | 27% |
| streamed, default k=4, no cache, 4 lanes | 0.7 | 1817 MiB | — |
| mmap baseline (no streaming) | 0.09 | — | — |

## Table 2 - Qwen3.6-35B-A3B (Q4_K_M): 22.3 GB

Their prose: "A hybrid attention/SSM MoE (256 experts, top-8, 41 blocks)". All streamed rows use
`--overlap --dense-weights anon`, so every streamed row is `overlap = yes`.

| Configuration | tok/s | Flash/token | Cache hit |
|---|---:|---:|---:|
| mmap baseline (no streaming) | 0.1 (unstable) | — | — |
| streamed, default k=8, cache 2000 MiB, 4 lanes, overlap | 4.3 | 206 MiB | 56% |
| streamed, default k=8, cache 3000 MiB, 4 lanes, overlap | 5.0 | 144 MiB | 65% |
| streamed, k=6, cache 2000 MiB, 4 lanes, overlap | 5.4 | 137 MiB | 60% |
| **streamed, k=6, cache 3000 MiB, 4 lanes, overlap** | **5.8** | 91 MiB | 68% |

## Table 3 - Qwen3-30B-A3B (Q4_K_M): 18.5 GB

Overlap is marked per row here, not table-wide.

| Configuration | tok/s | Flash/token | Cache hit |
|---|---:|---:|---:|
| mmap baseline (no streaming) | 2.0 (unstable) | — | — |
| streamed, default k=8, no cache, 4 lanes | 1.7 | 1051 MiB | — |
| streamed, default k=8, cache 2000 MiB, 4 lanes | 2.4 | 480 MiB | 53% |
| streamed, default k=8, cache 4000 MiB, 4 lanes | 4.0 | 225 MiB | 76% |
| **streamed, default k=8, auto cache (capped 4000 MiB), 4 lanes, overlap** | **5.2** | 225 MiB | 76% |
| streamed, k=6, cache 4000 MiB, 4 lanes | 5.0 | 165 MiB | 77% |

## Table 4 - Gemma-4-26B-A4B (Q4_K_M): 17.0 GB

Overlap is marked per row here, not table-wide. Rows 3 and 4 are the same configuration with and
without `--overlap` at essentially identical flash bytes - the cleanest overlap A/B they publish.

| Configuration | tok/s | Flash/token | Cache hit |
|---|---:|---:|---:|
| mmap baseline (no streaming) | 0.4 | — | — |
| streamed, default k=8, no cache, 4 lanes | 1.6 | 904 MiB | — |
| streamed, default k=8, cache 2000 MiB, 4 lanes | 2.2 | 366 MiB | 58% |
| streamed, default k=8, cache 2000 MiB, 4 lanes, overlap | 2.8 | 365 MiB | 58% |
| streamed, default k=8, cache 4000 MiB, 4 lanes | 4.1 | 144 MiB | 82% |
| **streamed, k=6, cache 4000 MiB, 4 lanes** | **5.0** | 98 MiB | 83% |

## Table 5 - Desktop, Qwen3.6-35B-A3B (Q4_K_M) on the Windows laptop

| Configuration | tok/s | Flash/token | Cache hit |
|---|---:|---:|---:|
| streamed, default k=8, cache auto, 4 lanes | 4.8 | 74 MiB | 84% |
| streamed + `--drop-cold-experts 0.75` | 6.8 | 23 MiB | 92% |
| **streamed + `--overlap` + `--drop-cold-experts 0.75`** | **7.3** | 24 MiB | 92% |

---

## Rows experiment #53 uses, and the ones it refuses

**Scored (phone, 18 rows):** every *streamed* row of tables 1-4.

**Excluded, by a rule fixed in the pre-registration before any fit:**

- **All five `mmap baseline` rows.** They are a different code path (page-cache faulting, not the
  streaming engine), their bytes-per-token are not published, and three of the five are labelled
  "unstable" by the authors. A model of the streaming engine cannot be scored on rows the
  streaming engine did not produce.
- **Desktop rows 2 and 3 (`--drop-cold-experts 0.75`).** Dropping skips routed experts, so the
  active-byte count - the very quantity the compute term predicts - is changed by an amount they
  do not publish. Their `--drop-cold-experts` output is also explicitly "not reproducible".
- **Desktop row 1 is not in the fitted set either.** One row cannot support the two device
  constants the model needs; the desktop is handled as a separate disclosure arm with no kill
  power (pre-registration section 6).

## Cache sizes, as used by the cache-hit arm

`auto cache (capped 4000 MiB)` on Qwen3-30B-A3B is treated as **4000 MiB** because its published
flash/token (225 MiB) and cache hit (76%) are *identical* to the explicit 4000 MiB row. The
desktop `cache auto` row is **excluded** from the cache-hit arm because its realised size is not
published.

## Known defects in this dataset, recorded before it was used

1. **Rows within a table can come from different sessions** (their words). Phone throughput
   "moves a lot with device state (heat, free memory)".
2. **Each number is a best-of, not a mean.** Best-of is biased upward, and unequally so across
   rows with different variance.
3. **The Qwen3.6 phone table is a single 96-token run**, not the 256-token protocol.
4. **I/O lane count varies within table 1** (8 lanes for the cached rows, 4 for the uncached).
   Experiment #53 models one effective read rate per *device* and therefore charges the lane
   difference to model error. This is disclosed, not corrected.
5. **`~60 GB` for gpt-oss-120b is two significant figures** and their prose calls every model
   Q4_K_M, while the only gpt-oss GGUF that streams "unchanged" is MXFP4 at 63.4 GB. #53 resolves
   this by cross-checking the file's own routed-byte count against their published no-cache
   flash/token counter (prediction P-0) rather than by trusting either label.
6. **Cache hit is published to whole percent**, so a flash-byte total back-derived as
   `F / (1 - hit)` carries up to ~3% error at hit = 0.68 and more at higher hit.
