# Reddit post — benchmark contamination (L-31)

Target: **r/LocalLLaMA**. Image: `media/neighbour_effect.png` (the self-check is on the canvas —
house pattern from the `stuck_boost_reddit` cut).

Every number below is from a committed pre-registration with raw logs. Precision notes for
accuracy while editing: the **11.3–70.7 spread is prefill** (prompt-processing tok/s, 20 arms,
prereg #108); the **13.04 → 14.43 warm-up ramp is decode** (prereg #106). Same page-cache cause,
two different measurements — don't merge them.

---

## Title (pick one)

**A. (recommended)** Consecutive llama.cpp benchmark runs aren't independent when the model is
larger than free RAM

**B.** Benchmarking an MoE: consecutive runs contaminate each other through the page cache

**C.** Run order changed my prefill numbers by 6.3× — measured, with the protocol that fixes it

*Why A: it states the condition and the finding, and nothing else. No "PSA", no "lying to you", no
number stacked on the end as a hook. r/LocalLLaMA reads hype as a tell, and a finding this
specific doesn't need amplifying — the table in the first screen does the work. A also carries the
precondition in the title, which keeps it from being wrong for the many readers whose model fits.*

*B is the humbler "here's what I ran into" register, which also plays well there. C leads with the
6.3× and is the one to avoid unless the others underperform — it's the closest to a hook, and it
front-loads the least trustworthy number (a full span including both outliers).*

---

## Body

I was benchmarking expert-count settings on a 13 GB MoE and the numbers wouldn't sit still. Same
command, same file, same box, five runs: **11.3 to 70.7 tok/s** prompt-processing.

Not thermal throttle. Not background apps. I split all 20 arms by **which configuration ran
immediately before them**, and it fell out clean:

| arm | preceded by same config | preceded by a different one |
|---|---|---|
| k=8 | **1.8%** spread | **72.4%** |
| k=4 | **0.8%** spread | **20.9%** |

Arms whose predecessor matched are tight to under 2%. The outliers are all arms whose predecessor
was a *different* config.

**Why:** the model (13.15 GiB) is bigger than my free RAM (~12.5 GB). It can't all stay in page
cache, so the OS is holding whatever the *last* process touched. A config that read fewer expert
tensors leaves the wrong pages resident for the one that needs more. Your benchmark inherits the
previous benchmark's cache.

This means **back-to-back A/B in that regime compares cache states as much as configurations** —
run A then B and the difference you publish may be entirely run order.

**The fix costs nothing:** interleave your arms and repeat — A B C, then C B A, then A B C — and
compare only readings whose predecessor matched. Three passes put every arm inside 2%.

**Related, same cause, if you benchmark a model that doesn't fit:** six consecutive runs of one
unchanged command climbed `13.04 → 13.14 → 13.89 → 14.33 → 14.43` tok/s decode as the cache
filled. A model that *does* fit held a 2.1% spread with no ramp at all. So a single cold benchmark
in this regime measures the cache, not the model.

And one that surprised me — **don't try to fix it by pre-reading the file.** `cat model.gguf >
/dev/null` before benchmarking measured **11.89 tok/s against a 13.84 mean**. It made things
*worse*. A file bigger than RAM ends with the cache holding its **last** ~12 GB, whereas real runs
leave it holding the pages the model actually re-reads. Position is the wrong key. I'd staked that
one at +1.0 tok/s and got the sign wrong.

**What it cost me:** I'd published 14.86 tok/s for a build of mine. Weeks later the identical
command on the same machine returned **11.0**. Nothing about the file changed — I'd just never
recorded free RAM next to the number, so there was no way to tell the two states apart. I had to
correct my own headline down, publicly.

**If you take one thing:** when you post a tok/s number for a model bigger than your RAM, post
your free RAM with it. Otherwise nobody — including you — can reproduce it.

---

Raw logs, the pre-registrations (predictions written and committed *before* each run, misses and
two voids included), and the chart's generator are all in the repo:
github.com/FedericoTs/quantprobe

The tool I was building when I hit this now warns when you're in that regime, but honestly the
protocol above is the whole finding and it works with plain llama.cpp.

*Box: GTX 1060 6GB / 16GB DDR4-3000 / i5-7600K, llama.cpp b10098. Absolute speeds are that
machine; the contamination effect is about the size relationship, not the hardware.*

---

## Comment-thread prep (likely replies, answered honestly)

- **"Just use `--no-mmap`."** Different lever, and it cuts both ways here — measured 2.9× *slower*
  on one placement near the RAM boundary because non-evictable pages OOM instead of degrading. It
  doesn't remove the predecessor effect; it changes which resource you run out of.
- **"This is just cold cache, everyone knows."** Cold-vs-warm is known; the *directional
  predecessor* effect is the part that isn't — a low-k arm poisons a following high-k arm
  specifically, which is why interleaving (not just warming) is the fix.
- **"Does this apply if my model fits in RAM?"** No — that's the control arm: 2.1% spread, no ramp.
  Nothing to evict, nothing to inherit. Stated as the precondition in the title for that reason.
- **"6.3× seems too large."** It's the full span of individual readings including both outliers
  (11.3 and 70.7); the *median* effect is smaller. Both numbers are in the raw log, which is
  committed.
