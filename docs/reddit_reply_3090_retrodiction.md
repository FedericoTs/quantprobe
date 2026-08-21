# Reddit reply — the RTX 3090 / Qwen3.8-27B retrodiction

**Time-sensitive.** Post as a comment in the existing thread
(`r/LocalLLaMA/comments/1vtup5s/`) rather than as a new post — the discussion is already there,
and a comment that adds information reads as participation while a separate post riding the same
number reads as clout-chasing.

Tone check before posting: OP did good work and **published their own caveat**. This reply must
confirm and clarify, never "well actually". If it reads as debunking, it has failed — their
number is real, and our own law says it's plausible.

Attach nothing. Links only if asked.

---

## The comment

Ran this through a decode model I've been staking predictions with, because it's a good test —
different hardware from mine, a different inference stack, and the numbers are public.

For the non-speculative baseline OP publishes (46 tok/s single-stream, realistic prompts), the
model predicts **25.0 tok/s** for a 27B at ~5.78 effective bits on a 3090. Measured is 1.84×
that.

That sounds like a miss and isn't: the prediction is deliberately a **one-sided floor**. Across 8
models and 13 benchmarks, real speed has come in **≥0.90× the printed number every time**, and
typically 1.1–1.8× higher — the all-in-VRAM case is the one this model knows least well, and it
errs low on purpose. So 1.84× is the floor holding on its 14th test, and the first on a stack I
don't use (vLLM) and a hybrid linear-attention model. It's also a new maximum for that range,
which is the more interesting half — one more point above 1.8× and I have to restate the band.

**The part I'd underline for anyone reading the headline:** OP already said it, and it deserves
not to get lost. 381 tok/s is a **25k-token document-reproduction** task. Ordinary chat on the
same stack is **~133**. That's not a caveat, it's the whole mechanism — speculative decoding pays
when the output copies its context and does approximately nothing when the model is inventing
text.

I'd staked that one separately and it's satisfying to see it hold somewhere else: measuring
ngram-simple on a GTX 1060, copy-heavy work got ~2.1× and open-ended prose got **1.01× — no gain
at all**. Same mechanism here, different drafter (DFlash2 + context lookup), different vendor
generation, ~2.9× spread between the two regimes. Independent confirmation across
implementations is worth more than either measurement alone.

So: the number is real, it's physically plausible, and it means "381 when reproducing a document
that's already in the prompt, ~133 when chatting." Both are good on a 3090. Only one is the
number you'll see on your own workload, and which one depends entirely on whether your output
copies its input.

Nice work on the stack, and thanks for publishing the chat figure next to the headline — plenty
of people wouldn't have.

---

## If asked "what model / where's the code"

Keep it one line, no pitch:

> It's `quantprobe` — tok/s = η·BW ÷ bytes-per-token, with every prediction staked before the
> measurement. github.com/FedericoTs/quantprobe. The 3090 row is in docs/ATLAS.md with the
> arithmetic.

## If challenged on the 25.0 figure

Answer with the honest uncertainty rather than defending it:

> Fair — the input I'm least sure of is the streamed byte count. I used 19.5 GB (their stated
> download) over 27B = 5.78 effective bits. If I take the full 19.5 GB against the 3090's 936
> GB/s, the implied efficiency is 0.958, which is above every η I've measured (0.512–0.812), so
> the streamed footprint is almost certainly smaller than the download — embeddings and an int4
> lm_head are gathered per token, not streamed. That correction is in the model, which is why it
> prints 25.0 rather than 48.
