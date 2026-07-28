# Pre-registration #62: the resident-expert sweep (U-14) — is ~13% of prompt processing on the table?

**Author:** Federico Sciuca · **Date staked:** 2026-07-28, BEFORE the runs. **STAKED.**

## Where U-14 came from

#61's cold-boot run measured pp2048 at 336.94 on the currently-shipped `-ot` pattern (expert
layers 11–47 → CPU, 37 CPU expert layers) where the original 386.04 was recorded on a pattern
with 32 CPU expert layers. The −12.7% tracks the +15.6% CPU expert-layer count — consistent with
prefill being CPU-expert-bound. If that mechanism is right, resident-expert count is a pp dial
that costs nothing in tg (tg parity across nearby splits is measured three times over).

## Arms — flagship, cold-state box, one session, clocks logged, r=2

`-ot "blk.(K..47).ffn_.*_exps.=CPU"` for K ∈ {11 (shipped), 14, 16, 18}, each measured for
pp2048 AND tg128 at `-b 1024 -ub 1024 -mmp 0`. K=18 puts 7 expert layers + attention in VRAM —
predicted near the VRAM budget at ub 1024; if it OOMs, that fact is the datapoint (the pattern's
VRAM cliff located) and the arm is recorded as OOM, not dropped silently.

## Stakes

- **P-1 (the mechanism).** pp2048 rises monotonically with resident-expert count (falling K→more
  residents… note K is the CPU-start index: HIGHER K = more residents). Predicted shape:
  pp ∝ 1/(CPU expert layers), i.e. K=16 lands at **375–400** (the original 386 was this class).
- **P-2 (the free-lunch check).** tg128 stays within **±5%** across all arms that fit — the dial
  moves pp without a tg price.
- **P-3 (the VRAM edge).** K=18 either fits and continues the pp trend, or OOMs — either way the
  cliff position enters `moe_split_flags`' safety logic.

## KILL RULE

**If P-1 fails** — pp does not track resident-expert count — the #61 explanation for the 336-vs-386
gap was wrong, the difference is elsewhere (flags, build, KV state), and U-14 closes refuted with
the shipped pattern keeping its current form. **If P-1 holds and P-2 holds**, the tool's `-ot`
pattern generation moves from "fixed 25% residency" to "max residents that fit the VRAM budget",
shipped with the sweep as evidence.

**Wired into:** pending; `findings/REGISTER.json:U-14` scores either way.
