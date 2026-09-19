"""quantprobe fetch -- robust multi-file HF downloader (manual HTTP Range, retry-on-break),
bypassing the hf CLI's Xet-backend stalls. Grown from weights/hf_fetch.py, whose usage line this
docstring still carried: the entry points are `quantprobe fetch <repo> <dest> [file ...]` and
`python -m quantprobe.fetch <repo_id> <dest_dir> <file1> [file2 ...]`.
Resumes partial .part files; token from HF_TOKEN env or ~/.cache/huggingface/token.
Completeness is a byte count against the remote's Content-Length and nothing else - see `fetch`
for what that does and does not certify.
"""

from __future__ import annotations

import os
import re
import sys
import time

import requests

PRESETS = {
    "qwen3-30b": ("unsloth/Qwen3-30B-A3B-GGUF", "Qwen3-30B-A3B-Q2_K.gguf"),
    "glm-air": ("unsloth/GLM-4.5-Air-GGUF", "GLM-4.5-Air-UD-IQ2_XXS.gguf"),
    "deepseek-16b": (
        "bartowski/DeepSeek-Coder-V2-Lite-Base-GGUF",
        "DeepSeek-Coder-V2-Lite-Base-IQ2_XS.gguf",
    ),
    "qwen3-0.6b": ("unsloth/Qwen3-0.6B-GGUF", "Qwen3-0.6B-Q8_0.gguf"),
}


def token():
    t = os.environ.get("HF_TOKEN")
    if t:
        return t.strip()
    p = os.path.expanduser("~/.cache/huggingface/token")
    return open(p).read().strip() if os.path.exists(p) else None


_CONTENT_RANGE = re.compile(r"^\s*bytes\s+(\d+)-(\d+)/(\d+)\s*$", re.IGNORECASE)


def remote_size(url, hdr, timeout=60):
    """The remote length from a HEAD, or `(None, why)` when the answer cannot be trusted.

    An error response carries a Content-Length too - of its own JSON or HTML body. Reading that
    header without looking at the status is how a 404 hands back a number that may happen to
    match a half-written local file, or to differ from a perfectly good one. Status first, then
    the header, and anything missing, unparseable or non-positive is a refusal rather than a
    silent 0.

    `(None, why)` means "no size", not "wrong size", and the two callers treat it differently:
    a NEW download has nothing but the byte count to certify itself with and refuses to start,
    while an already-present file keeps its long-standing reuse and is reported as present
    rather than verified. Neither path may compare a local length against a number that came
    off an error page.
    """
    try:
        r = requests.head(url, headers=hdr, allow_redirects=True, timeout=timeout)
    except requests.exceptions.RequestException as e:
        return None, f"HEAD request failed ({str(e)[:60]})"
    if r.status_code != 200:
        return None, f"HEAD returned status {r.status_code}"
    raw = r.headers.get("Content-Length")
    if raw is None:
        return None, "the response carried no Content-Length"
    try:
        n = int(str(raw).strip())
    except ValueError:
        return None, f"unparseable Content-Length {raw!r}"
    if n <= 0:
        return None, "Content-Length must be positive for a model download"
    return n, None


def check_content_range(value, want_start, want_total):
    """Bytes a 206 is allowed to deliver, or `(None, why)` if its range is not the one we asked for.

    Completion below is `size == total`, so bytes appended at an offset nobody checked make a
    file of exactly the right length and the wrong contents - which then gets promoted over the
    real one. Everything the server claims is therefore compared against what was requested
    before a byte is written. A server may legitimately answer an open-ended range with less
    than the remainder, so a short-but-consistent range is accepted and the loop asks again from
    the new offset.
    """
    if not value:
        return None, "206 carried no Content-Range"
    m = _CONTENT_RANGE.match(value)
    if not m:
        return None, f"unparseable Content-Range {value!r}"
    try:
        start, end, total = (int(g) for g in m.groups())
    except ValueError:
        return None, "Content-Range numbers are not representable"
    if total != want_total:
        return None, f"remote is now {total:,} B, was {want_total:,} B - the file changed"
    if start != want_start:
        return None, f"range starts at {start:,}, we asked from {want_start:,}"
    if end < start or end >= total:
        return None, f"range {start:,}-{end:,} does not fit a {total:,} B file"
    return end - start + 1, None


def fetch(repo, dest, fname, tok, tries=100, force=False):
    """Download `fname` into `dest`, resuming a `.part`; True when `dest/fname` is usable.

    The only completeness test here is a byte count against the remote's own Content-Length,
    and the two paths ask for it with different stakes:

      NEW DOWNLOAD - the byte count is the whole gate, so an untrusted or non-positive size is
      refused before anything is written or deleted (including before `--force` removes a
      published file to make room).
      ALREADY PRESENT - an unforced skip keeps the file. When the remote answers properly the
      size is compared, and a mismatch is still refused (U-18). When it does not answer, the
      file is reused and reported as PRESENT, not complete: this is a name-and-presence match
      and nothing more. It is NOT a content check - no checksum, ETag or GGUF-header validation
      is performed anywhere in this module, and a matching length is not an authentication
      either.
    """
    url = f"https://huggingface.co/{repo}/resolve/main/{fname}"
    out = os.path.join(dest, fname)
    part = out + ".part"
    hdr0 = {"Authorization": f"Bearer {tok}"} if tok else {}
    if os.path.exists(out) and not force:
        # name-only skip once handed an incompatible file to llama-speculative (a June-era GGUF
        # under the target name): ask the remote first, and call this complete only when a good
        # answer says the two sizes agree. U-18 - the mismatch refusal below - is the other half,
        # and it needs a real length for the same reason completion does.
        have = os.path.getsize(out)
        remote, why = remote_size(url, hdr0)
        if remote is None:
            # No trusted size, so there is nothing to compare and nothing is compared - the
            # error-page length that used to decide this is not consulted in either direction.
            # The file is REUSED, as it always has been when the remote is unobtainable:
            # `auto` hands this path a model the user already downloaded, and refusing it would
            # end the command for anyone offline with a perfectly good file. What changes is
            # only what is claimed. "already complete"/"size matches remote" asserted a check
            # that did not happen; this says presence, names why the size is missing, and
            # leaves the bytes alone (a `.part` beside it is not promoted - nothing is renamed
            # on this path).
            print(
                f"  {fname}: already present, NOT VERIFIED - {why}. {have:,} B are on disk and "
                f"were reused on name and presence alone; this is not a check of the contents. "
                f"Re-run when the remote answers to have the size confirmed, or --force to "
                f"download it again.",
                flush=True,
            )
            return True
        if have != remote:
            print(
                f"  {fname}: EXISTING FILE IS NOT THIS FILE - local {have:,} B vs remote "
                f"{remote:,} B. A same-named file from another source is on disk; it may be an "
                f"incompatible model. Re-run with --force to replace it, or fetch to a "
                f"different dest.",
                flush=True,
            )
            return False
        print(
            f"  {fname}: already complete (size matches remote; --force re-downloads)", flush=True
        )
        return True
    # Asked BEFORE the --force deletion below: that deletion exists to make room for a download,
    # so finding out afterwards that the download cannot start costs the user the file for
    # nothing. Nothing on disk has been touched at this point.
    total, why = remote_size(url, hdr0)
    if not total:
        print(
            f"  {fname}: CANNOT FETCH - {why or 'the remote reports a 0 B file'}. There is no "
            f"completed {fname} here to fall back on, and completion for a new download is a "
            f"byte count, so a download against a size we do not have could never be "
            f"certified; nothing was written or removed.",
            flush=True,
        )
        return False
    if os.path.exists(out) and force:
        os.remove(out)
        if os.path.exists(part):
            os.remove(part)
    print(f"  {fname}: {total / 1e9:.2f} GB", flush=True)
    t = 0
    while t < tries:
        have = os.path.getsize(part) if os.path.exists(part) else 0
        if have >= total:
            break
        try:
            h = dict(hdr0)
            if have:
                h["Range"] = f"bytes={have}-"
            r = requests.get(url, headers=h, stream=True, timeout=(30, 120), allow_redirects=True)
            if r.status_code not in (200, 206):
                print(f"    status {r.status_code}, retry", flush=True)
                time.sleep(5)
                t += 1
                continue
            if r.status_code == 206:
                expect, why = check_content_range(r.headers.get("Content-Range"), have, total)
                if expect is None:
                    # Rejected before the file is opened, so the prefix already on disk is
                    # untouched: it is this response that is wrong, not those bytes.
                    r.close()
                    print(f"    bad range - {why}, retry {t + 1}", flush=True)
                    time.sleep(5)
                    t += 1
                    continue
            else:
                expect = total  # 200 to a Range request is the whole file: restart, not append
            mode = "ab" if (have and r.status_code == 206) else "wb"
            t0 = last = time.time()
            base = have if mode == "ab" else 0
            got = 0
            oversized = False
            with open(part, mode) as f:
                for chunk in r.iter_content(1 << 22):
                    if chunk:
                        got += len(chunk)
                        if got > expect:
                            oversized = True
                            break
                        f.write(chunk)
                    if time.time() - last > 20:
                        sz = os.path.getsize(part)
                        print(
                            f"    {sz / 1e9:.2f}/{total / 1e9:.2f} GB ({(sz - base) / 1e6 / max(1e-6, time.time() - t0):.1f} MB/s)",
                            flush=True,
                        )
                        last = time.time()
            if oversized or got != expect:
                # A body that ENDED CLEANLY at the wrong length is the server contradicting
                # itself - a truncated CDN object, or an error page under a 200 - not a dropped
                # connection. Those arrive as exceptions below and keep their progress, because
                # what they delivered was real bytes at the right offset. This is not, so the
                # file goes back to the length it had before the request.
                r.close()
                os.truncate(part, base)
                print(
                    f"    {got:,} B for a {expect:,} B range - rolled back to {base:,}, "
                    f"retry {t + 1}",
                    flush=True,
                )
                time.sleep(3)
                t += 1
                continue
        except (
            requests.exceptions.ChunkedEncodingError,
            requests.exceptions.ConnectionError,
            requests.exceptions.ReadTimeout,
            requests.exceptions.Timeout,
        ) as e:
            # Interrupted, not contradicted: keep the progress and resume from it next round.
            print(
                f"    break at {os.path.getsize(part) if os.path.exists(part) else 0:,}, retry {t + 1}: {str(e)[:60]}",
                flush=True,
            )
            time.sleep(3)
            t += 1
    size = os.path.getsize(part) if os.path.exists(part) else 0
    if size == total:
        os.replace(part, out)
        print(f"  {fname}: DONE", flush=True)
        return True
    if size > total:
        # `have >= total` ends the loop before any request, so this is where an already-oversized
        # `.part` lands - and "INCOMPLETE" was the one thing it is not. It cannot be a prefix of
        # this file, so no retry can shrink it and it is never promoted; say which file is in the
        # way and what to do with it. Not --force: that only clears a `.part` when a completed
        # file sits beside it, which is exactly the case that does not apply here.
        note = (
            ""
            if os.path.exists(out)
            else f" `--force` does not clear it while no completed {fname} sits next to it."
        )
        print(
            f"  {fname}: PARTIAL FILE IS OVERSIZED - {part} holds {size:,} B, more than the "
            f"{total:,} B the remote reports for {fname}, so it cannot be a prefix of this "
            f"file and nothing was promoted. Move or delete that .part file, then re-run." + note,
            flush=True,
        )
        return False
    print(f"  {fname}: INCOMPLETE", flush=True)
    return False


def resolve_alias(name):
    """Turn a bare model name into (repo, file, why) using every list we have, not just ours.

    `fetch`, `auto` and the recipe atlas each grew their own vocabulary, so a name that works in
    one command used to fail in another - worst of all for a recipe we PUBLISHED A BUILD OF,
    where the file is sitting on HuggingFace and we were the only thing standing between the
    user and it. Returns file=None when we know the repo but not which quant to take."""
    from . import recipes as recmod

    rec = recmod.find(key=name)
    if rec:
        art = recmod.artifact(rec)
        if art and art.get("repo") and art.get("file"):
            return art["repo"], art["file"], "published depth-aware build"
    from .auto import MODEL_REPOS

    if name in MODEL_REPOS:
        return MODEL_REPOS[name][0], None, "`auto` preset"
    return None, None, None


def run(a):
    import sys as _s

    repo, files = a.repo, a.files
    if repo in PRESETS and not files:
        repo, f = PRESETS[repo]
        files = [f]
        print(f"[quantprobe] preset '{a.repo}' -> {repo}/{f}")
    elif not files:
        alias, afile, why = resolve_alias(repo)
        if alias and afile:
            print(f"[quantprobe] '{repo}' -> {alias}/{afile} ({why})")
            repo, files = alias, [afile]
        elif alias:
            # We know WHERE it lives but not which quant is right for this box - and picking
            # that is `auto`'s whole job, so hand off instead of guessing a file.
            _s.exit(
                f"'{repo}' is an {why} -> {alias}, but `fetch` needs a filename and the right\n"
                f"quant depends on your hardware. Either:\n"
                f"  quantprobe auto {repo}            # picks the file your machine can run\n"
                f"  quantprobe fetch {alias} {a.dest} <name.gguf>"
            )
    if not files:
        _s.exit("no files given (or use a preset: " + ", ".join(PRESETS) + ")")
    ok = all(fetch(repo, a.dest, fn, token(), force=getattr(a, "force", False)) for fn in files)
    _s.exit(0 if ok else 1)


if __name__ == "__main__":
    repo, dest = sys.argv[1], sys.argv[2]
    os.makedirs(dest, exist_ok=True)
    ok = all(fetch(repo, dest, f, token()) for f in sys.argv[3:])
    sys.exit(0 if ok else 1)
