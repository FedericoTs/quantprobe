# Pre-registration #65 (scored inline): the machine ladder, 0.5B-30B, three ways vs the law

**Author:** Federico Sciuca · **Date:** 2026-07-28. The stakes are the anchored predictions
captured in `weights/data/prereg65_ladder.log` BEFORE any measurement ran, by shipped v1.20 code.
Full table, protocol, and footnotes: `MACHINE_LADDER.md` (the deliverable).

## Scored

| arm class | result |
|---|---|
| quantprobe vs naive default | 2.3-3.3x on every model (0.5B-30B) |
| quantprobe vs informed llama.cpp | +26% (16B MoE), +8% (30B MoE), honest parity where all-in-VRAM |
| law vs measured, on recommended arms | 0.6B +2.3%, flagship +7.3% (in band), DS -20% under, 7B Q4_K_M -34% under (U-15 format blindness), 0.5B quasi-in-sample |
| direction of misses | 4 of 5 under-promise; the single over-prediction inside the printed +/-25% band |
| bonus finding | naive big-model CPU arms are bimodal (+/-44-75%) from mmap cold paging - the exact failure the tool's pure-CPU warning describes, now measured |

**Wired into:** `MACHINE_LADDER.md` (published table) · U-15 evidence (the -34% Q4_K_M arm is the
format-blindness datapoint) · E-06 (the same three-way framing his rerun will use).
