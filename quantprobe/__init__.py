"""quantprobe — probe-then-quantize for LLMs on commodity hardware.

Three commands, all grounded in measured laws (see LAWS.md in the repo):
  quantprobe probe  — measure a GGUF's depth-fragility curve, emit the depth-aware recipe (Law 3)
  quantprobe plan   — evaluate every bit/tier placement for a model on your machine, predict tok/s (Law 4)
  quantprobe fetch  — robust HuggingFace downloader (Range-resume, retry)

Every constant in `plan` is fitted from measurements published in the repo, including two
pre-registered hardware predictions confirmed within 8%.
"""
__version__ = "1.1.0"
