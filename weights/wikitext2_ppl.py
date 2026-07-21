"""Campaign-4 (FRONTIER): the EXACT standard WikiText-2 perplexity protocol used by
GPTQ / QuIP# / AQLM / QTIP, so our numbers are directly comparable to their published
Llama-2-7B results (fp16 5.47; QTIP 2-bit 5.86; QuIP# 2-bit 6.19; AQLM 2-bit ~6.2-6.9).

Protocol (identical to the GPTQ repo everyone cites):
  testenc = tokenizer("\\n\\n".join(wikitext2-raw-v1[test]['text'])).input_ids
  seqlen = 2048; nsamples = numel // seqlen
  for each non-overlapping 2048-window: nll += CrossEntropy(logits[:-1], tokens[1:]) * seqlen
  ppl = exp(sum(nll) / (nsamples * seqlen))

Usage:
  python -m weights.wikitext2_ppl <model_dir_or_hf_id>      # fp16 baseline (validate ~5.47)
  (or import wikitext2_ppl(model, tok) for a quantized model)
"""
from __future__ import annotations

import sys
import time

import torch
import torch.nn as nn

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

SEQLEN = 2048


import os
_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "wikitext2_test.txt")


def get_wikitext2_test_ids(tok):
    """Tokenize the FULL wikitext-2-raw-v1 test set, '\\n\\n'-joined (exact GPTQ protocol).
    Prefers the cached plain-text file (avoids the flaky `datasets`/multiprocess teardown)."""
    if os.path.exists(_CACHE):
        text = open(_CACHE, encoding="utf-8").read()
    else:
        from datasets import load_dataset
        text = "\n\n".join(load_dataset("wikitext", "wikitext-2-raw-v1", split="test")["text"])
    return tok(text, return_tensors="pt").input_ids


@torch.no_grad()
def wikitext2_ppl(model, tok, device="cuda", seqlen=SEQLEN, limit=None, verbose=True):
    """Standard GPTQ-protocol WikiText-2 ppl. `limit` caps #windows (for quick checks)."""
    ids = get_wikitext2_test_ids(tok)
    n = ids.numel() // seqlen
    if limit:
        n = min(n, limit)
    loss_fct = nn.CrossEntropyLoss()
    nlls = []
    t0 = time.time()
    for i in range(n):
        batch = ids[:, i * seqlen:(i + 1) * seqlen].to(device)
        logits = model(batch).logits
        shift_logits = logits[:, :-1, :].contiguous().float()
        shift_labels = batch[:, 1:].to(shift_logits.device)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        nlls.append(loss.float() * seqlen)
        if verbose and (i % 10 == 0 or i == n - 1):
            cur = torch.exp(torch.stack(nlls).sum() / ((i + 1) * seqlen)).item()
            print(f"  window {i+1}/{n}  running ppl = {cur:.4f}  ({time.time()-t0:.0f}s)", flush=True)
    ppl = torch.exp(torch.stack(nlls).sum() / (n * seqlen)).item()
    return ppl


def main():
    mid = sys.argv[1] if len(sys.argv) > 1 else "weights/data/llama2_7b_base"
    lim = int(sys.argv[2]) if len(sys.argv) > 2 else None
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(mid, use_fast=False)
    print(f"loading {mid} (fp16) ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(mid, torch_dtype=torch.float16).to(dev).eval()
    ppl = wikitext2_ppl(model, tok, device=dev, limit=lim)
    print(f"\n{mid}  WikiText-2 ppl = {ppl:.4f}  (seqlen {SEQLEN})")
    print("  reference: Llama-2-7B fp16 = 5.47 | QTIP 2b 5.86 | QuIP# 2b 6.19 | AQLM 2b ~6.2-6.9")


if __name__ == "__main__":
    main()
