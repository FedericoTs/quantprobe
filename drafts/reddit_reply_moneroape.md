# Draft reply to u/MoneroApe (for Federico to review and post)

---

u/MoneroApe — owed you an update, and I'd rather deliver it late with numbers than never.

Your pointer went two directions. The apex-quant half you already saw: it exposed two real gaps
in my recipe (unprotected always-active tensors, no imatrix), both adopted and credited in the
README — that one nudge was worth −9.1% perplexity at 3 bits.

The TurboQuant-on-Pascal half I promised and hadn't run. Straight status: still not run as a
head-to-head. But this week I went deeper than a benchmark would have — I built a standalone CUDA
harness (zero llama.cpp) and measured what ANY quantization format can do at decode on a Pascal
card, at the instruction level. Three results that answer the question you were actually asking:

1. **The decode wall on old cards is unpack instruction cost, not bandwidth.** A matvec with no
   unpacking hits 95% of the card's real streaming ceiling; the same bytes with naive
   nibble-to-float unpacking run at 42%. Same buffer, same bytes, only the instruction changed:
   1.9x. So a format's decode speed here is set by how dp4a-native its unpack is — which prices
   VQ/codebook-class methods (TurboQuant included) before running them:

2. **Codebook/LUT decode paths are structurally dead on this hardware generation.** I did the
   arithmetic on T-MAC-style LUT dequant: the table doesn't fit Pascal's 96 KB/SM shared memory
   (needs ~512 KB), and the bit-plane variant that fits costs more ops than dp4a. Any method whose
   decode is "look the vector up" runs at the naive-unpack floor on Pascal. On Ampere+ with bigger
   smem the story may differ — I can't test that (one GPU).

3. **The one that's immediately actionable for anyone on a pre-Ampere card:** Q4_0 decodes +19%
   faster than Q4_K_M on the same model (bytes explain only 5.7% of that), and Q2_K is strictly
   dominated — measured slower in absolute tok/s than Q4_0 while being 32% smaller AND lower
   quality. The "K-quants are always better" community advice is wrong on old cards, for speed.
   Speed-only claim, one card, may invert on Ampere+ — full scope in the repo.

Everything above is pre-registered (predictions staked before measuring, misses published with
equal prominence — 11 pre-registrations that week, 8 of my own hypotheses killed by their own
kill rules). If you're still interested in TurboQuant proper, its natural home on this card is
the KV cache rather than weights (weight-side VQ dies on point 2), and KV-quant I did measure
(-ctk/-ctv q8_0 numbers are in the register).

The kernel brief with every number and every retraction is in the repo if you want to red-team
it — that's genuinely welcome, the retraction log is the product.

---

*Notes for Federico: post as-is or trim point 2 if too long for the thread. The claims all trace:
1 = prereg #52 + kernelprobe L0/L1/L1b; 2 = brainstorm N8 kill math; 3 = preregs #52/#53;
KV = U-01/prereg #20-era. FUTURE.md E1 can be marked answered (mechanism-level) once posted —
the literal head-to-head remains open if MoneroApe still wants it.*
