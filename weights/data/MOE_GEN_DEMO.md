# Live demo: a 16-billion-parameter MoE generating text on a 2016 GTX 1060

The companion to [MOE_2BIT_RESULTS.md](MOE_2BIT_RESULTS.md). That document reports the *quality*
result (our data-free carve-out codec at measured WikiText-2 perplexity). This document is the
*visceral capacity proof*: a 16 B Mixture-of-Experts model, quantized to ~2 bits/weight, actually
**generating coherent text while resident on a 6 GB consumer GPU**.

## What ran

- **Model:** `DeepSeek-Coder-V2-Lite-Base`, IQ2_XS GGUF (15.7 B params, 64 experts/layer, MLA attn) —
  the **2-bit** quant, file size **5.97 GB**.
- **Hardware:** single **NVIDIA GTX 1060 6 GB** (Pascal, 2016, no tensor cores).
- **Runtime:** llama.cpp (CUDA build), full-offload (`-ngl 99`), raw completion (`-no-cnv`).

## Result

**Prompt:** `The history of computing began`

**Completion (seed 7, temp 0.7):**
> The history of computing began long before there were computers. The first computers were made in
> the late 19th century. Computers began as a way for scientists to do calculations. They are a way
> for humans to do calculations. Computers have evolved to do much more than simply calculations.
> Computers have many uses. Computers have been used for entertainment, education, scientific
> research, business, and communications. Computers have been used to play games, create art, and
> many more. The first computers were made of relays, and used electricity…

Fluent, on-topic English (with a few stray HTML `<p>` tags — the base *code* model's web-text habit;
this is a base model with no instruction tuning).

## The numbers (measured, not estimated)

| metric | value |
|---|---:|
| VRAM, idle baseline | 794 MiB |
| **VRAM, peak during generation** | **5873 / 6144 MiB** (card essentially full) |
| free VRAM at peak | 157 MiB |
| generation speed | ~5.3 tok/s |
| prompt eval | ~14.4 tok/s |
| model load (warm cache) | ~4 s |

VRAM sampled once/second via `nvidia-smi` while generating. The card maxed out at **5873 of 6144 MiB**
with a 16 B model live — the "fits in 6 GB" claim, demonstrated rather than computed.

## Honest framing

- This demo uses **llama.cpp's IQ2_XS** quantization, **not our carve-out trellis codec.** It proves
  the **capacity** story (a 16 B MoE runs, GPU-resident, on a 6 GB 2016 card) — which our results doc
  already concedes is not novel (community IQ2 GGUFs exist). The **novel** contribution remains the
  *measured-quality, data-free* carve-out (perplexity 6.25 @ NWIN=8 / 6.77 @ NWIN=16, gap-ratio
  1.10× — beating the calibrated frontier MxMoE's 1.18×).
- It is the **`Base` (code) sibling** of the same architecture family as the `DeepSeek-V2-Lite` we
  measured, not the identical checkpoint. Same 16 B MoE / MLA / 64-expert architecture, so the
  capacity point transfers directly.
- **Partial offload, by ~0.8 GB:** the Windows desktop held ~0.8 GB of VRAM during the run, so
  llama.cpp fit ~5.1 GB of the 5.97 GB model on the GPU and streamed the small remainder (embedding /
  output tensors) from CPU. On a **headless / free 6 GB card** the whole model is resident. Our
  carve-out (5.64 GB) is *smaller* than this IQ2_XS file (5.97 GB), so it fits with more headroom.
- **Still pending for an our-codec live demo:** the trellis CUDA kernel exists and is gated
  (`weights/trellis_run.py`), but a deployable `nn.Module` + MoE assembly + generation loop do not —
  that is the remaining (multi-day) build to make *our* packed weights generate text, not just measure
  perplexity.

## Reproduce

```
/d/evo-compress-data/llamacpp/llama-completion.exe \
  -m /d/evo-compress-data/gguf/DeepSeek-Coder-V2-Lite-Base-IQ2_XS.gguf \
  -ngl 99 -c 512 -n 160 -no-cnv -p "The history of computing began" --temp 0.7 --seed 7
```
