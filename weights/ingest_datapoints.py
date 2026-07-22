"""ingest_datapoints.py -- pull community eta-datapoint issues from GitHub, parse them into
community_etas.json, which scaling_law.py overlays on the law chart. Closes the contribute loop:
a user runs `quantprobe bench --contribute` -> reviews -> submits an issue -> this ingests it ->
the chart and the eta fit grow. Maintainer-run (read-only, public API, no token needed for public issues).

    python -m weights.ingest_datapoints            # fetch + write weights/data/community_etas.json
"""
from __future__ import annotations
import json, os, re, urllib.request

REPO = "FedericoTs/quantprobe"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "community_etas.json")


def fetch_issues():
    url = f"https://api.github.com/repos/{REPO}/issues?labels=eta-datapoint&state=all&per_page=100"
    req = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json",
                                               "User-Agent": "quantprobe-ingest"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def parse(body):
    d = {}
    for key in ("hardware", "model", "placement", "predicted", "measured"):
        m = re.search(rf"^{key}:\s*(.+)$", body or "", re.M | re.I)
        if m:
            d[key] = m.group(1).strip()
    pred = re.search(r"([0-9.]+)", d.get("predicted", ""))
    meas = re.search(r"([0-9.]+)", d.get("measured", ""))
    if not (pred and meas):
        return None
    return {"hardware": d.get("hardware", "?"), "model": d.get("model", "?"),
            "placement": d.get("placement", "?"),
            "predicted": float(pred.group(1)), "measured": float(meas.group(1))}


def main():
    try:
        issues = fetch_issues()
    except Exception as e:
        print(f"  fetch failed ({e}); leaving existing {os.path.basename(OUT)} untouched")
        return
    pts = []
    for it in issues:
        if it.get("pull_request"):
            continue
        p = parse(it.get("body"))
        if p:
            p["issue"] = it.get("number")
            pts.append(p)
    json.dump(pts, open(OUT, "w"), indent=2)
    print(f"  ingested {len(pts)} community data points -> {OUT}")
    for p in pts[:10]:
        d = (p["measured"] / p["predicted"] - 1) * 100 if p["predicted"] else 0
        print(f"    #{p.get('issue','?')}: {p['model']} on {p['hardware'][:32]} "
              f"pred {p['predicted']} meas {p['measured']} ({d:+.0f}%)")


if __name__ == "__main__":
    main()
