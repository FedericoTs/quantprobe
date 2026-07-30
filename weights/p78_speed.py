"""#78: bench the NEW emitted configs against the previous measurements. Same files, same
binary, same session - only the tool's advice changed."""
import json, os, re, subprocess

D = "D:/evo-compress-data/gguf/"
BIN = r"C:\Users\Federico\Documents\evo-compress\tools\llama.cpp-pristine\build\bin\llama-bench.exe"
CUDA = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin"
rows = json.load(open("weights/data/full_ladder_v124.json", encoding="utf-8"))
env = dict(os.environ, PATH=CUDA + os.pathsep + os.environ["PATH"])
out = []
for r in rows:
    e = r.get("emit_final", "")
    if not e or e == r.get("emit") or "-ot" not in e:
        continue
    ot = re.search(r'-ot "([^"]+)"', e).group(1)
    cmd = [BIN, "-m", D + r["file"], "-ngl", "99", "-ot", ot, "-mmp", "0", "-t", "4",
           "-n", "64", "-p", "0", "-r", "3"]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=3600, env=env)
    m = re.search(r"tg64\s*\|\s*([0-9.]+)", p.stdout)
    new = float(m.group(1)) if m else None
    prev = r.get("measured")
    r["measured_new_emit"] = new
    delta = (new - prev) / prev * 100 if (new and prev) else None
    r["delta_pct"] = round(delta, 1) if delta is not None else None
    print(f"{r['name'][:26]:26s} was {prev:6.2f} -> now {new if new else 0:6.2f}  "
          f"{delta:+6.1f}%" if new else f"{r['name'][:26]:26s} FAILED")
    out.append((r["name"], prev, new, delta))
json.dump(rows, open("weights/data/full_ladder_v124.json", "w", encoding="utf-8"), indent=1)
ok = [d for _, _, _, d in out if d is not None]
print(f"\narms re-benched: {len(ok)} | improved: {sum(1 for d in ok if d > 0)} | "
      f"regressed: {sum(1 for d in ok if d < -1)} | mean {sum(ok)/len(ok):+.1f}%")
