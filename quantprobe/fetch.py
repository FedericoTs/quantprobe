"""hf_fetch.py -- robust multi-file HF downloader (manual HTTP Range, retry-on-break), bypassing the
hf CLI's Xet-backend stalls. Usage: python -m weights.hf_fetch <repo_id> <dest_dir> <file1> [file2 ...]
Resumes partial .part files; token from HF_TOKEN env or ~/.cache/huggingface/token.
"""
from __future__ import annotations
import os, sys, time
import requests


PRESETS = {
    "qwen3-30b":   ("unsloth/Qwen3-30B-A3B-GGUF", "Qwen3-30B-A3B-Q2_K.gguf"),
    "glm-air":     ("unsloth/GLM-4.5-Air-GGUF", "GLM-4.5-Air-UD-IQ2_XXS.gguf"),
    "deepseek-16b": ("bartowski/DeepSeek-Coder-V2-Lite-Base-GGUF", "DeepSeek-Coder-V2-Lite-Base-IQ2_XS.gguf"),
    "qwen3-0.6b":  ("unsloth/Qwen3-0.6B-GGUF", "Qwen3-0.6B-Q8_0.gguf"),
}


def token():
    t = os.environ.get("HF_TOKEN")
    if t:
        return t.strip()
    p = os.path.expanduser("~/.cache/huggingface/token")
    return open(p).read().strip() if os.path.exists(p) else None


def fetch(repo, dest, fname, tok, tries=100):
    url = f"https://huggingface.co/{repo}/resolve/main/{fname}"
    out = os.path.join(dest, fname)
    part = out + ".part"
    if os.path.exists(out):
        print(f"  {fname}: already complete", flush=True)
        return True
    hdr0 = {"Authorization": f"Bearer {tok}"} if tok else {}
    r = requests.head(url, headers=hdr0, allow_redirects=True, timeout=60)
    total = int(r.headers.get("Content-Length", 0))
    print(f"  {fname}: {total/1e9:.2f} GB", flush=True)
    t = 0
    while t < tries:
        have = os.path.getsize(part) if os.path.exists(part) else 0
        if total and have >= total:
            break
        try:
            h = dict(hdr0)
            if have:
                h["Range"] = f"bytes={have}-"
            r = requests.get(url, headers=h, stream=True, timeout=(30, 120), allow_redirects=True)
            if r.status_code not in (200, 206):
                print(f"    status {r.status_code}, retry", flush=True); time.sleep(5); t += 1; continue
            mode = "ab" if (have and r.status_code == 206) else "wb"
            t0 = last = time.time(); base = have if mode == "ab" else 0
            with open(part, mode) as f:
                for chunk in r.iter_content(1 << 22):
                    if chunk:
                        f.write(chunk)
                    if time.time() - last > 20:
                        sz = os.path.getsize(part)
                        print(f"    {sz/1e9:.2f}/{total/1e9:.2f} GB ({(sz-base)/1e6/max(1e-6,time.time()-t0):.1f} MB/s)", flush=True)
                        last = time.time()
        except (requests.exceptions.ChunkedEncodingError, requests.exceptions.ConnectionError,
                requests.exceptions.ReadTimeout, requests.exceptions.Timeout) as e:
            print(f"    break at {os.path.getsize(part) if os.path.exists(part) else 0:,}, retry {t+1}: {str(e)[:60]}", flush=True)
            time.sleep(3); t += 1
    if total and os.path.exists(part) and os.path.getsize(part) == total:
        os.replace(part, out)
        print(f"  {fname}: DONE", flush=True)
        return True
    print(f"  {fname}: INCOMPLETE", flush=True)
    return False


def run(a):
    import sys as _s
    repo, files = a.repo, a.files
    if repo in PRESETS and not files:
        repo, f = PRESETS[repo]
        files = [f]
        print(f"[quantprobe] preset '{a.repo}' -> {repo}/{f}")
    if not files:
        _s.exit("no files given (or use a preset: " + ", ".join(PRESETS) + ")")
    ok = all(fetch(repo, a.dest, fn, token()) for fn in files)
    _s.exit(0 if ok else 1)


if __name__ == "__main__":
    repo, dest = sys.argv[1], sys.argv[2]
    os.makedirs(dest, exist_ok=True)
    ok = all(fetch(repo, dest, f, token()) for f in sys.argv[3:])
    sys.exit(0 if ok else 1)
