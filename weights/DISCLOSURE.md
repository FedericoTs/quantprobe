# AI-Assistance Disclosure

*(Adapt to the target venue's required format — e.g. NeurIPS/ICML LLM-usage statement,
an Acknowledgments paragraph, or a checklist entry. The content below is the honest
account of how this work was produced.)*

This research made **substantial** use of an AI research/coding assistant (Anthropic's
Claude, via the Claude Code agent) under the direction of the human author(s). The
assistant contributed to:

- **Experiment design and execution** — implementing the data-free trellis quantization
  harness, the memory-bounded streaming WikiText-2 evaluation, the causal-decomposition
  ablations (forced-routing / forced-output), the cross-architecture (Qwen) port, and the
  benchmarking scripts; and proposing, running, and iterating on experiments.
- **Analysis** — interpreting results, computing gap-ratios, and surfacing the central
  mechanistic finding (that the MLA KV-latent projections carry the dominant low-bit error).
- **Writing** — drafting and revising the manuscript, tables, and figures.

The human author(s) set the research goals, directed the investigation, made the scientific
decisions, and take **full responsibility for the correctness of every claim**. Because the
assistant was deeply involved, we took specific steps to guard against AI-introduced error:

1. **Audit** — every number reported in the paper was cross-checked against the raw
   experiment logs.
2. **Reproduction** — the pipeline was re-executed from cached weights, reproducing all
   headline numbers.
3. **Gold-standard re-derivation** — the headline carve-out configuration was re-quantized
   from scratch (no cache), reproducing the reported perplexity to four decimal places,
   confirming the codec is deterministic.
4. **Independent protocol verification** — cross-paper comparison details (e.g., MxMoE's
   evaluation sequence length, and which models each baseline actually evaluates) were
   checked against the cited papers' text and public source code, not assumed.

We disclose this in the interest of transparency and in accordance with venue policy on the
use of large language models in research.
