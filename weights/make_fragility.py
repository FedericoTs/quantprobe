"""The fragility fingerprint - why one quantization recipe cannot fit every model.

Renders the committed probe atlas (quantprobe/recipes/*.json): for each model, the measured
perplexity cost of pushing ONE band of layers to q2_k while the rest stay at base. The point
is the SHAPE - Mistral breaks at the front, every Qwen breaks at the back - which is the whole
argument for probing a model instead of reusing somebody else's quant.

Cite-or-refuse: every band, ratio and reference ppl is read from the atlas at render time.
"""
from __future__ import annotations
import glob, json, os
import brand as B

W, H = 1600, 1200
ROW_H = 150
X0, XW = 300, 900          # strip geometry
SHORT = {"mistral-7b": "Mistral-7B", "qwen2.5-7b": "Qwen2.5-7B",
         "qwen3-30b": "Qwen3-30B-A3B", "qwen3.5-35b": "Qwen3.5-35B-A3B"}


def load():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = []
    for p in sorted(glob.glob(os.path.join(here, "quantprobe", "recipes", "*.json"))):
        d = json.load(open(p, encoding="utf-8"))
        pr = d["probe"]
        out.append(dict(key=d["model"]["key"], n=d["model"]["n_layer"],
                        bands=pr["band_deltas"], frag=pr["fragile_band"],
                        ratio=pr["fragility_ratio"], shape=pr["shape"],
                        ref=pr["reference_ppl"]))
    # early-shaped first so the contrast is the first thing read
    return sorted(out, key=lambda m: (m["shape"] != "early", m["key"]))


def main():
    models = load()
    s = [B.svg_open(W, H),
         B.header(W, "MEASURED · WIKITEXT-2 HELD OUT · ONE BAND TO q2_K AT A TIME",
                  "The fragile layers move",
                  "Perplexity cost of quantizing each band. Mistral breaks at the front; "
                  "every Qwen at the back - so a recipe cannot be reused.")]

    top = 312
    for i, m in enumerate(models):
        y = top + i * ROW_H
        mx = max(b["delta_ppl"] for b in m["bands"])
        s.append(f'<text x="{X0-26}" y="{y+34}" text-anchor="end" fill="{B.INK}" '
                 f'font-size="25" font-weight="bold">{SHORT.get(m["key"], m["key"])}</text>')
        s.append(f'<text x="{X0-26}" y="{y+64}" text-anchor="end" fill="{B.MUT}" '
                 f'font-size="19">{m["n"]} layers · ppl {m["ref"]:.2f}</text>')

        for b in m["bands"]:
            bx = X0 + b["lo"] / m["n"] * XW
            bw = (b["hi"] - b["lo"] + 1) / m["n"] * XW
            frac = b["delta_ppl"] / mx                      # per-model scale, stated below
            fragile = [b["lo"], b["hi"]] == m["frag"]
            col = B.DISK if fragile else B.TEAL_DEEP
            s.append(f'<rect x="{bx+3}" y="{y+8}" width="{bw-6}" height="62" rx="7" '
                     f'fill="{col}" opacity="{0.16 + 0.84*frac:.3f}"/>')
            if fragile:
                s.append(f'<rect x="{bx+3}" y="{y+8}" width="{bw-6}" height="62" rx="7" '
                         f'fill="none" stroke="{B.INK}" stroke-width="2.5"/>')
            lab = f'+{b["delta_ppl"]:.2f}' if b["delta_ppl"] >= 0.01 else f'+{b["delta_ppl"]:.3f}'
            s.append(f'<text x="{bx+bw/2}" y="{y+46}" text-anchor="middle" '
                     f'fill="{B.INK if frac > 0.45 else B.SUB}" font-size="20" '
                     f'font-weight="{"bold" if fragile else "normal"}">{lab}</text>')
            s.append(f'<text x="{bx+bw/2}" y="{y+92}" text-anchor="middle" fill="{B.MUT}" '
                     f'font-size="17">{b["lo"]}-{b["hi"]}</text>')

        tag = "FRAGILE AT THE FRONT" if m["shape"] == "early" else "FRAGILE AT THE BACK"
        s.append(f'<text x="{X0+XW+26}" y="{y+32}" fill="{B.DISK}" font-size="30" '
                 f'font-weight="bold">{m["ratio"]:.1f}x</text>')
        s.append(f'<text x="{X0+XW+26}" y="{y+60}" fill="{B.SUB}" font-size="17">{tag}</text>')
        s.append(f'<text x="{X0+XW+26}" y="{y+84}" fill="{B.MUT}" font-size="16">'
                 f'worst band vs median</text>')

    yb = top + len(models) * ROW_H + 4
    s.append(f'<text x="{X0}" y="{yb+16}" fill="{B.MUT}" font-size="19">shallow layers</text>')
    s.append(f'<text x="{X0+XW}" y="{yb+16}" text-anchor="end" fill="{B.MUT}" '
             f'font-size="19">deep layers</text>')
    s.append(f'<line x1="{X0}" y1="{yb-6}" x2="{X0+XW}" y2="{yb-6}" stroke="{B.GRID}" '
             f'stroke-width="1.5"/>')

    PY = yb + 40
    s.append(B.panel(80, PY, W - 160, 132, stroke=B.TEAL, sw=2))
    s.append(f'<text x="118" y="{PY+38}" fill="{B.TEAL}" font-size="19" '
             f'letter-spacing="3">WHY IT MATTERS</text>')
    s.append(f'<text x="118" y="{PY+76}" fill="{B.INK}" font-size="23">'
             f'Protect the wrong band and you spend bits where they buy nothing. quantprobe '
             f'probes YOUR model, then builds</text>')
    s.append(f'<text x="118" y="{PY+104}" fill="{B.INK}" font-size="23">'
             f'the quantization around what it actually found.</text>')
    s.append(f'<text x="{W-118}" y="{PY+104}" text-anchor="end" fill="{B.MUT}" font-size="16">'
             f'shading per-model; absolute deltas printed on every band</text>')

    s.append(B.footer(W, H, "quantprobe/recipes/*.json · band probe, rest at base · GTX 1060 6GB"))
    s.append("</svg>")
    B.save("fragility_fingerprint.svg", "".join(s))


if __name__ == "__main__":
    main()
