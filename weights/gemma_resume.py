"""Robust manual HTTP Range-resume for google/gemma-4-12B model.safetensors. Bypasses the hf CLI (Xet
stall / etag churn). Loops Range requests, resuming from the current partial size on any connection break,
until the file is complete; then sha256-verifies and places it. Run from the main venv. HFTOK env = token.
"""
import os, sys, time, hashlib, requests

DIR = "D:/evo-compress-data/gemma-4-12b"
DL = DIR + "/.cache/huggingface/download"
EXPECT = "fe054ae05ff7f44318fd8ae90d58992531455c7ed31356704088f0f2d8c8009a"
TOTAL = 23919549408
URL = "https://huggingface.co/google/gemma-4-12B/resolve/main/model.safetensors"
TOKEN = os.environ["HFTOK"]

inc = sorted([os.path.join(DL, f) for f in os.listdir(DL) if f.endswith(".incomplete")],
             key=os.path.getsize, reverse=True)
PARTIAL = inc[0]
print(f"partial: {os.path.basename(PARTIAL)}\nstart: {os.path.getsize(PARTIAL):,} / {TOTAL:,}", flush=True)

tries = 0
while os.path.getsize(PARTIAL) < TOTAL and tries < 100:
    have = os.path.getsize(PARTIAL)
    try:
        h = {"Authorization": f"Bearer {TOKEN}", "Range": f"bytes={have}-"}
        r = requests.get(URL, headers=h, stream=True, timeout=(30, 120), allow_redirects=True)
        if r.status_code not in (206, 200):
            print("bad status", r.status_code, "- retry", flush=True); time.sleep(5); tries += 1; continue
        if r.status_code == 200 and have > 0:
            print("server ignored Range (200) - abort to avoid clobber"); sys.exit(2)
        t0 = last = time.time()
        with open(PARTIAL, "ab") as f:
            for chunk in r.iter_content(1 << 22):
                if chunk:
                    f.write(chunk)
                if time.time() - last > 15:
                    sz = os.path.getsize(PARTIAL)
                    print(f"  {sz/1e9:.2f} GB  ({(sz-have)/1e6/max(1e-6,time.time()-t0):.1f} MB/s)", flush=True)
                    last = time.time()
    except (requests.exceptions.ChunkedEncodingError, requests.exceptions.ConnectionError,
            requests.exceptions.ReadTimeout, requests.exceptions.Timeout) as e:
        print(f"  break at {os.path.getsize(PARTIAL):,} - retry {tries+1}: {str(e)[:70]}", flush=True)
        time.sleep(3); tries += 1

sz = os.path.getsize(PARTIAL)
print(f"download loop done: {sz:,} / {TOTAL:,}  (tries={tries})", flush=True)
if sz != TOTAL:
    print("INCOMPLETE - not verifying"); sys.exit(1)

print("verifying sha256 (a few min)...", flush=True)
hsh = hashlib.sha256()
with open(PARTIAL, "rb") as f:
    for b in iter(lambda: f.read(1 << 24), b""):
        hsh.update(b)
got = hsh.hexdigest()
ok = got == EXPECT
print(f"sha256: {got}\nexpect: {EXPECT}\n-> {'MATCH' if ok else 'MISMATCH'}", flush=True)
if ok:
    import shutil
    dst = DIR + "/model.safetensors"
    shutil.move(PARTIAL, dst)
    print(f"MOVED -> {dst}\nGEMMA DOWNLOAD COMPLETE", flush=True)
else:
    print("SHA MISMATCH -- left partial in place", flush=True); sys.exit(3)
