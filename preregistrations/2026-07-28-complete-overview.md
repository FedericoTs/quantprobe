# Pre-registration #66: the complete overview — every regime this machine has, 0.5B to 117B

**Author:** Federico Sciuca · **Date staked:** 2026-07-28, all predictions captured by shipped
v1.20.2 code BEFORE any measurement (log: `weights/data/prereg66_overview.log`, box idle
139 MHz / 29 C at capture). Pristine zero-patch binary for every arm, clocks logged per arm.

## The stakes (tool-printed, ±25% printed band)

| arm | prediction | notes |
|---|---|---|
| 7B Q4_K_M AIV @ d4096 | 18.8 | Law 4 v2 KV term |
| 7B @ d16384 | 15.0, placement SWITCHES to 26/28-layer split | KV no longer fits AIV — the switch itself is a stake |
| flagship split @ d4096 | 15.4 (residency drops to 24%) | KV pressure reshapes the split |
| 35B APEX-Mini 13.3GB split | 17.7 | RAM boundary, pins 9/12 GB warning fired |
| Coder-30B Q3_K_M 14.7GB split | 16.1 | pins 11/12 GB — the pinning cliff arm |
| DS IQ2_XS split 58% | 28.0 | IQ warning fired at plan time (96% IQ file) |
| DS IQ2_XS pure CPU | 14.1 | the 2.7x IQ-on-CPU warning's validation arm |
| **Laguna 117B Q2_K_XL, disk stream (cold)** | **1.6** | the NEVER-MEASURED tier; U-06 pre-warns predictions here may be off up to 7x — this arm decides |
| pp2048 columns (5 ladder models) | NOT predicted | the tool makes no per-model pp claims; column is measurement-only, stated as such |

## KILL RULES
- Laguna outside [0.23, 11.2] (the U-06 7x band around 1.6) → the disk tier's model is refuted
  outright and the row ships flagged unusable until re-derived.
- The 16k placement-switch failing to beat AIV-at-16k would refute the switch logic (measured only
  if both arms fit).
- Coder-30B failing to start (pinning) is a VALID outcome — it is the warning's demonstration.

**Wired into:** MACHINE_LADDER.md v2 on scoring; U-06 scores either way.
