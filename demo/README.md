# Reproduce the 50+ tok/s yourself, from cmd

Two windows. Every number prints in the SERVER window (`print_timing` lines).

## Window 1 - the server (speculation ON)

    cd C:\Users\Federico\Documents\evo-compress
    tools\llamacpp-b10098\llama-server.exe -m D:\evo-compress-data\gguf\Qwen3-30B-A3B-Q2_K.gguf -ngl 99 -ot "blk\.(16|17|18|19|20|21|22|23|24|25|26|27|28|29|30|31|32|33|34|35|36|37|38|39|40|41|42|43|44|45|46|47)\.ffn_.*_exps\.=CPU" --no-mmap -t 4 -b 1024 -ub 1024 --spec-type ngram-simple --spec-ngram-simple-size-m 384 --spec-ngram-simple-size-n 4 --port 8089

## Window 2 - the requests

    curl -s http://127.0.0.1:8089/v1/chat/completions -H "Content-Type: application/json" -d @"C:\Users\Federico\Documents\evo-compress\.claude\worktrees\law5-prefill\demo\edit_task.json"

Watch Window 1. Expected (measured 2026-07-27):

    eval time = ... ( ~9 ms per token,  ~108 tokens per second)
    draft acceptance = ~0.68  <- do NOT tune on this: see preregs #36/#37

Then the honest control - novel generation on the SAME server:

    curl -s http://127.0.0.1:8089/v1/chat/completions -H "Content-Type: application/json" -d @"C:\Users\Federico\Documents\evo-compress\.claude\worktrees\law5-prefill\demo\novel_task.json"

Expected: ~21-22 tok/s, acceptance ~0% - nothing to copy, no gain. This is the scope of the
lever, not a failure of it.

## The baseline A/B (proves the 2.4x)

Ctrl+C the server, relaunch WITHOUT `--spec-type ngram-simple`, re-send edit_task.json:
expected ~21 tok/s for the identical output (byte-identical at temp 0 - verified by SHA-256,
see weights/data/demo_identity_proof.log).

## Rules for a clean number (learned the hard way)

- Close browser windows / anything holding VRAM first: ~250 MB of desktop occupancy is measured
  to flip results by 2x on this 6 GB card (pre-registration #23).
- If a server is already running on 8089, kill it first: `taskkill /F /IM llama-server.exe`
- The speedup needs CONTEXT to copy from. A short prompt gets ~20% acceptance and ~17 tok/s
  (measured live); the full-file edit gets ~95% and 57. Paste real files, not snippets.
- llama-cli ignores --spec-type silently. Only llama-server speculates.
