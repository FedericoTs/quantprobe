# X post copy — the expert-dial arc

Ready to paste. Three assets in `media/`: `neighbour_effect.png` (the benchmark-contamination
finding — the most repostable standalone), `expert_dial.png` (the ceiling), `cache_trap.png`
(the priming trap). Hosted narrative:
https://claude.ai/code/artifact/983fa798-9be4-4e06-92e0-cf414453e537

Every number below traces to a committed pre-registration (#107–#111) and its raw log.

---

## Option A — the single strongest standalone (lead with this)

Attach: `media/neighbour_effect.png`

> Your llama.cpp benchmark is probably lying to you.
>
> I ran the exact same command five times. It returned 11.3 to 70.7 tok/s.
>
> Nothing changed between runs — except which config I'd benchmarked *just before*. On a model
> bigger than your RAM, the page cache carries the last run's working set into the next one, so
> back-to-back A/B compares cache states as much as configs.
>
> Fix: interleave your arms and repeat. Or measure noise.

**Alt text:** Scatter plot titled "Your benchmark runs contaminate each other." Five readings of
one unchanged command range from 11.3 to 70.7 tok/s. The three whose predecessor used the same
config cluster tightly at ~51 (1.8% spread); the two whose predecessor differed sit far out at
11.3 and 70.7. A 6.3× span from run order alone.

---

## Option B — the full arc (thread, ends on the artifact)

**1/** We spent two days trying to break our own tool's advice about MoE inference speed. Most of
it held. The times it didn't each became a finding — including one that changes how you should
benchmark. The whole scorecard, misses and two voids included:
https://claude.ai/code/artifact/983fa798-9be4-4e06-92e0-cf414453e537

**2/** The claim, staked before any run: turning experts down on a mixture-of-experts model can't
buy more than ~1.24× on decode — the routed experts only own 22% of the active bytes, so 78% of
the work is untouched whatever you do. You can read that ceiling off the file. It held to within
2%.
*(attach `media/expert_dial.png`)*

**3/** Then the same command started returning 11.3–70.7 tok/s — a 6.3× spread from nothing but
run order. On a model bigger than RAM, benchmarks contaminate each other through the page cache.
We shipped the warning into the tool.
*(attach `media/neighbour_effect.png`)*

**4/** The part I'm proudest of isn't a win. We built a hypothesis to explain the rest, then our
own confirming test dissolved it — so we walked the claim back, in public, in the register. The
mechanism is still open, and the page says so in plain type.

**5/** Every prediction staked and timestamped before the measurement. Scored by code written
before the run. Misses published at the same size as the wins. That's the whole method, and it's
the only reason any of this is trustworthy.
`pip install quantprobe` · https://github.com/FedericoTs/quantprobe

**Thread image alt text:**
- expert_dial.png: Two panels. Left, decode speedup vs expert count — measured (teal) tracks the
  dashed "predicted from the file" line to within 2%, reaching 1.45× at one expert of 256. Right,
  perplexity on a log scale climbing from 5.96 to 2277 as experts are removed: the knob is never
  free.
- neighbour_effect.png: see Option A alt text.

---

## Option C — the honest-science angle (for the "who checks the gate" audience)

> Four preregistrations, five days, one tool tested against its own advice. Score: two clean, one
> partial, two void — and the two voids are the best part. One caught a benchmarking bug the whole
> field shares; the other caught *us* overstating a claim, which we then walked back in public.
>
> Measurement you can't check is just a vibe with a number attached.
> https://claude.ai/code/artifact/983fa798-9be4-4e06-92e0-cf414453e537

---

Note: the artifact is private until shared from its own share menu (top-right on the page). The
three PNGs are in `media/` and post directly.
