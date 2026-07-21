"""Debug R3: is the gamma fold functionally exact at the BLOCK level? Compare layer-0 outputs
on random input, original vs folded, in fp32 and fp64. Localizes the +0.0099 ppl discrepancy.
Run:  python -m weights.debug_fold
"""
from __future__ import annotations

import copy
import sys

import torch

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from weights.quant_lab import build_model, load_fp16


def fold_block(block):
    with torch.no_grad():
        g_in = block.input_layernorm.weight.clone()
        for lin in (block.self_attn.q_proj, block.self_attn.k_proj, block.self_attn.v_proj):
            lin.weight.mul_(g_in[None, :])
        block.input_layernorm.weight.fill_(1.0)
        g_post = block.post_attention_layernorm.weight.clone()
        for lin in (block.mlp.gate_proj, block.mlp.up_proj):
            lin.weight.mul_(g_post[None, :])
        block.post_attention_layernorm.weight.fill_(1.0)


def main():
    model = build_model()
    load_fp16(model)
    blk = model.model.layers[0]
    blk2 = copy.deepcopy(blk)
    fold_block(blk2)

    torch.manual_seed(0)
    x = torch.randn(1, 16, model.config.hidden_size) * 0.5
    pos = torch.arange(16).unsqueeze(0)
    # transformers Qwen2 block needs position_embeddings (rotary) in newer versions
    rot = model.model.rotary_emb
    pe = rot(x, pos)
    with torch.no_grad():
        y1 = blk(x, position_embeddings=pe)[0]
        y2 = blk2(x, position_embeddings=pe)[0]
    d = (y1 - y2).abs()
    print(f"block-0 fold: max|dy| = {d.max():.3e}   mean = {d.mean():.3e}   "
          f"rel = {(d.max()/y1.abs().max()):.3e}")
    # also test JUST the attention half vs mlp half by folding selectively
    blk3 = copy.deepcopy(blk)
    with torch.no_grad():
        g_in = blk3.input_layernorm.weight.clone()
        for lin in (blk3.self_attn.q_proj, blk3.self_attn.k_proj, blk3.self_attn.v_proj):
            lin.weight.mul_(g_in[None, :])
        blk3.input_layernorm.weight.fill_(1.0)
    with torch.no_grad():
        y3 = blk3(x, position_embeddings=pe)[0]
    print(f"attn-only fold: max|dy| = {(y1-y3).abs().max():.3e}")
    blk4 = copy.deepcopy(blk)
    with torch.no_grad():
        g_post = blk4.post_attention_layernorm.weight.clone()
        for lin in (blk4.mlp.gate_proj, blk4.mlp.up_proj):
            lin.weight.mul_(g_post[None, :])
        blk4.post_attention_layernorm.weight.fill_(1.0)
    with torch.no_grad():
        y4 = blk4(x, position_embeddings=pe)[0]
    print(f"mlp-only fold:  max|dy| = {(y1-y4).abs().max():.3e}")


if __name__ == "__main__":
    main()
