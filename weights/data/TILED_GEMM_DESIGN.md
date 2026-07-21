# Design note: tiled quantized GEMM for the 2-bit trellis (sm_61) — the real speed win

**Why:** the honest head-to-head (cycle 30) showed batched trellis reaches only *parity* with 4-bit
(0.98× at B=16) because the simple 1-warp-per-output-row kernels are **activation-bandwidth-bound** —
they re-read the activation from global for every output row. The trellis's half-the-weight-bytes
advantage never gets to matter. A **tiled GEMM** that reuses the activation from shared memory makes the
kernel **weight-bandwidth-bound**, at which point 2-bit (half the weight traffic of 4-bit) should give
**~1.7–1.9× over 4-bit** for throughput. This is the only lever that turns "denser" into "denser AND faster."

## Kernel structure (Marlin-style, adapted to sm_61, no tensor cores)
Compute `Y[M,N] = X[M,K] @ Wdq[N,K]^T` where `Wdq` is decoded on the fly from the 2-bit trellis.
Tile: `BM × BN` output block per CTA; loop over K in `BK` chunks.
1. **Stage X tile → shared:** cooperatively load `X[BM, BK]` (fp16) into shared, coalesced. Reused by all BN.
2. **Decode W tile → shared (or registers):** each warp decodes its slice of `W[BN, BK]` from the trellis
   master-stream via the sliding-window 3INST path (`trellis_decode4` + `recons3inst`, lookup-free, no LUT
   bank conflicts). Decode is **amortized over BM** (the batch/M dim) — the whole point.
3. **MAC:** standard register-blocked FMA accumulation `acc[tm][tn] += Xs[tm,k]*Ws[tn,k]` over the BK chunk
   (no tensor cores on Pascal → plain fp32 FMA, or dp4a int8 path for 4-MAC/instr, quality +0.006 ppl per
   the prior d0 check).
4. **Epilogue:** scale by per-group `gs` (int8-dequant), de-rotation is already on the activation side
   (`fwht_prep`), add the CSR outlier sidecar, write `Y`.

## sm_61 specifics / tuning knobs
- No tensor cores → register-blocked FMA (e.g. 8×8 per thread) or dp4a. Target BM·BN·… within 255 regs.
- Shared budget 96 KB/SM: X tile (BM×BK fp16) + W decoded tile. Pick BM/BN/BK for ≥2 CTAs/SM occupancy.
- Decode-to-shared once per K-chunk, reused across BM rows → this is what amortizes the decode.
- Start point: BM=64, BN=64, BK=32; sweep. Validate vs `f0_gemv3_batch` (4-bit) and cuBLAS fp16 ceiling.

## Validation / gates
- **Correctness:** tiled output == dense-decode @ X (rel < 1e-3), reuse `decode_trellis` reference.
- **Speed gate:** trellis-tiled per-token < 4-bit-tiled per-token at B≥8 (the 2-bit traffic must win once
  weight-bound). Compare both to cuBLAS fp16 as the ceiling.
- **Effort:** multi-day CUDA build, iterative tuning; the hottest of the remaining options (sustained GPU
  during autotuning). Needs explicit go-ahead given thermal constraints.

## Fallback if Pascal tiling underdelivers
If register/shared limits cap occupancy too low for a win on sm_61 (plausible — Pascal lacks the async-copy
and tensor cores Marlin relies on), the honest conclusion stands: **trellis is a memory play, speed-parity**,
and the speed headline would require newer hardware (Ampere+), which is outside the 1060 mandate.
