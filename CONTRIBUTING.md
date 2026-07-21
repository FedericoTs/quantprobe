# Contributing

The most valuable contribution is a **measurement**. This project runs on falsifiable numbers.

## Contribute an η data point (10 minutes)
Run `quantprobe bench --gguf <model> --model <preset> [--vram N --vram-bw N --ram N --ram-bw N --disk-bw N]`
and open an issue with the "η data point" template: your hardware, the model/bits, predicted vs measured.
Points that land outside the bands are MORE valuable than ones that confirm them.

## Contribute an atlas entry (~1 hour)
Run `quantprobe probe --gguf <model-f16.gguf> --eval wiki.test.raw` on a model family not yet in
[the atlas](weights/GGUF_DEPTH_RECIPE.md) and open an "atlas entry" issue with the band curve.
A model whose fragile end breaks the current pattern is a finding, not a failure.

## Code
PRs welcome for the `quantprobe` package (keep `python tests/smoke.py` green). For claims/laws,
open an issue with data first — the bar for prose is measurements.
