# Draft reply to u/MoneroApe (v2 — for Federico to review and post after v1.19 is live)

---

u/MoneroApe — your report was the single most valuable input this project has received, and I
want to give it the answer it deserves: not thanks, but fixes. All five of your suggestions were
real defects or gaps. **v1.19.0 ships every one of them**, plus the two bugs your numbers exposed
underneath. (`pip install -U quantprobe`)

**First, the reconciliation of your 9×, because you deserve to know where it went:**

Your miss was OUR detector, not the physics. Two input bugs compounded:

1. **`detect.py` treated your 4 DIMMs as 4 memory channels.** AM5 is dual-channel regardless of
   stick count, so the tool fed the law 173 GB/s where your platform peaks at ~86. Fixed:
   consumer platforms default to dual-channel (HEDT/server CPUs recognized by name go wider),
   and the output now states the assumption.
2. **Spec vs delivered.** Our own box measures 26.1 GB/s of its 48 "spec" — a lesson we measured
   and then failed to apply to the detector. With both corrected, the law predicts **9.3 tok/s**
   for your rig. analogalok's tuned same-class 4090: 9.26–11.36. Law 4 held; the inputs lied.

The remaining gap to your 3.64 was your configuration, which you diagnosed yourself correctly
(6-of-12 threads + 36.5 GB pinned inside 64 GB). And your prefill "67×" isn't real — 3850 ms /
22 tokens measures startup, not prefill; our published number was pp2048 and now says so.

**Your five suggestions, in v1.19:**

1. **--threads** — emitted in every CPU-resident command now (logical cores, with the caveat
   printed). Your "2 t/s → 9 t/s from this flag alone" matched our C-07 measurement exactly.
2. **Pinned memory** — any `-ot` row that would pin >45% of system RAM now warns and names the
   fallback (drop `-ot`, let auto-placement decide). You found a failure mode our 16 GB box
   could never have hit.
3. **Threads/OpenMP in predictions** — the deeper fix is `quantprobe calibrate` (new command):
   it MEASURES your RAM stream, your disk, your GPU's sustained clocks, and optionally a real
   pure-CPU anchor on your own GGUF. plan then uses your measured numbers, tagged [calibrated].
4. **Batch size for MoE** — the 2048 cap was a 6 GB-card assumption doing the limiting on your
   24 GB card. Raised to 4096 where the buffer math allows (your linked 90→470 prefill datapoint
   is credited in the code comment as external and unvalidated by us — see the ask below).
5. **ngram visibility** — the "novel generation drafts 0 tokens" fact now prints top-line, right
   under the placement list. You independently replicated our D-10 finding (different model,
   hardware, and fork) and then got bitten by our burying it. Both are fixed.

**The ask, since you offered:** two commands on your rig would convert all of this from
"reconciled on paper" to "validated on the second machine this project has ever touched":

```
pip install -U quantprobe
quantprobe calibrate --model your-laguna.gguf
quantprobe plan --gguf your-laguna.gguf
```

Then, if you're willing, the rerun you already planned (`--threads 12 -b 4096 -ub 4096
--no-mmap`, no `-ot`) against plan's new prediction. If calibrate reads your RAM stream at
~45–50 GB/s and the plan lands within ±25% of your measured decode, you're the first external
confirmation of Law 4 on hardware we don't own — and if it misses, that's exactly the datapoint
we publish loudest. Your report is register entry E-06 either way, with every number credited.

(On the TurboQuant promise from before: still owed as a head-to-head. What I can give you now is
the mechanism-level answer from this week's bare-metal work: weight-side VQ/codebook decode is
structurally dead on Pascal-class cards — the LUT doesn't fit shared memory — and on any card the
decode cost of a format is set by how dp4a-native its unpack is. On your Ampere card the smem
math is friendlier; KV-side TurboQuant remains the interesting half, and your fork's
`--cache-type-v turbo3` wiring question is a good probe — if you confirm it's actually active,
I'll stake a prediction for it.)

---

*Notes for Federico: post after v1.19 is confirmed live on PyPI. Claims trace: channel bug =
detect.py fix + t_moneroape_channel_count; reconciliation = E-06; threads = C-07 + his quote;
pinning = D-08 extension; batch cap = SAFE_UBATCH_CAP with the external-datapoint comment;
ngram = D-10. The ±25% promise matches the tool's own printed band.*
