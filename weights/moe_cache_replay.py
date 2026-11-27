"""Replay llama.cpp's MoE expert cache over a recorded routing trace (LLAMA_MOE_CACHE_TRACE).

    python weights/moe_cache_replay.py TRACE --mib 3000 [--share 0.95]
    python weights/moe_cache_replay.py TRACE --sweep 500,1000,2000,3000 --shares 0,0.95
    python weights/moe_cache_replay.py --self-check

A port of the cache policy in llama.cpp's src/llama-moe-cache.cpp (moe_cache_lru: one LRU list per
frequency class, a CRF with a 128-token half-life, victim = the max age/2^class) and of the MCT1
reader in scripts/moe-cache-tune.py, both at release b12815 (MIT License, Copyright (c) 2023-2026
The ggml authors).

Why an exact port: it makes a falsifiable check. Replaying a trace at the size and CPU share it was
recorded with must reproduce the counters llama.cpp logs when the context closes,
    llama_moe_cache: ubatch <= 8: hits = H, misses = M (C on the CPU), hit rate = R%, uploaded = U MiB
and once it does, the same replay at OTHER sizes is what the cache would have done there: the
hit-rate curve a planner needs, from one recorded run. llama.cpp computes the CRF in float32 with
the platform's exp2f/log2f; this port rounds every step to float32 but uses correctly rounded
exp2/log2, so a class boundary can resolve differently in the last ulp. Expect agreement to the
counter in practice, and treat a small residual as that, not as a policy difference.
"""

import argparse
import math
import os
import struct
import sys
import tempfile

MAX_BATCH = 32  # llama-moe-cache.cpp: larger ubatches bypass the cache and are not traced
N_CLASS = 8
CRF_HALF_LIFE = 128.0


def f32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]


def pad(x, n):
    return (x + n - 1) // n * n


# ---------------------------------------------------------------------------- MCT1
def read_trace(path):
    with open(path, "rb") as f:
        data = f.read()
    if data[:4] != b"MCT1":
        raise SystemExit(f"{path}: not an MCT1 MoE cache trace")
    pos = 4

    def take(fmt):
        nonlocal pos
        v = struct.unpack_from("<" + fmt, data, pos)
        pos += struct.calcsize("<" + fmt)
        return v

    n_layer, n_expert_used, n_group = take("3I")
    groups = []
    for _ in range(n_group):
        device, n_expert, n_layers = take("3I")
        layers = list(take(f"{n_layers}I"))
        split, split_sum = take("2d")
        host_bytes, dev_host_bytes, alignment = take("3Q")
        (n_tensors,) = take("I")
        nb2 = list(take(f"{n_tensors}Q"))
        groups.append({"device": device, "n_expert": n_expert, "layers": layers, "split": split,
                       "split_sum": split_sum, "host_bytes": host_bytes, "dev_host_bytes": dev_host_bytes,
                       "alignment": alignment, "nb2": nb2, "expert_bytes": sum(nb2)})
    records = []
    while pos < len(data):
        il, n_used, n_ids = struct.unpack_from("<3i", data, pos)
        pos += 12
        records.append((il, n_used, struct.unpack_from(f"<{n_ids}i", data, pos)))
        pos += 4 * n_ids
    return {"n_layer": n_layer, "n_expert_used": n_expert_used, "groups": groups, "records": records}


def write_trace(path, n_layer, n_expert_used, groups, records):
    """MCT1 writer, for synthetic traces in the self-check."""
    with open(path, "wb") as f:
        f.write(b"MCT1" + struct.pack("<3I", n_layer, n_expert_used, len(groups)))
        for g in groups:
            f.write(struct.pack("<3I", g["device"], g["n_expert"], len(g["layers"])))
            f.write(struct.pack(f"<{len(g['layers'])}I", *g["layers"]))
            f.write(struct.pack("<2d", g["split"], g["split_sum"]))
            f.write(struct.pack("<3Q", g["host_bytes"], g["dev_host_bytes"], g["alignment"]))
            f.write(struct.pack("<I", len(g["nb2"])) + struct.pack(f"<{len(g['nb2'])}Q", *g["nb2"]))
        for il, n_used, ids in records:
            f.write(struct.pack("<3i", il, n_used, len(ids)) + struct.pack(f"<{len(ids)}i", *ids))


# ---------------------------------------------------------------------------- sizing
def n_slots_of(g, size_mib, n_zero, n_expert_used):
    """Slots of a group's cache: as many as fit its share of the budget (llama-moe-cache.cpp)."""
    budget = int(float(size_mib * 1024 * 1024) * g["split"] / g["split_sum"]
                 * float(g["host_bytes"]) / float(g["dev_host_bytes"]))

    def alloc(n):
        return sum(pad(nb * (n + n_zero + 1), g["alignment"]) for nb in g["nb2"])

    lo, hi = 0, len(g["layers"]) * g["n_expert"]
    if alloc(0) > budget:
        return 0
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if alloc(mid) <= budget:
            lo = mid
        else:
            hi = mid - 1
    return lo if lo >= n_expert_used else 0


# ---------------------------------------------------------------------------- policy
class ClassLRU:
    """moe_cache_lru: one LRU list per frequency class over the experts of a group of layers."""

    def __init__(self, n_layer, n_expert, n_slots):
        self.n_expert, self.n_slots = n_expert, n_slots
        self.slot_map = {}  # il -> [slot per expert], n_slots ("empty") if not cached
        self.key_of = [-1] * n_slots
        self.prev = [s - 1 for s in range(n_slots)]
        self.next = [s + 1 if s + 1 < n_slots else -1 for s in range(n_slots)]
        self.head = [0] + [-1] * (N_CLASS - 1)
        self.tail = [n_slots - 1] + [-1] * (N_CLASS - 1)
        self.list_of = [0] * n_slots
        self.n_filled = 0
        self.crf = {}  # key -> float32 CRF, 0 before the first use
        self.last_use = {}  # key -> layer clock at the last use
        self.clock = [0] * n_layer
        self.seen = [0] * n_expert
        self.seen_gen = 0

    def slots(self, il):
        if il not in self.slot_map:
            self.slot_map[il] = [self.n_slots] * self.n_expert
        return self.slot_map[il]

    def move(self, s, c):
        frm = self.list_of[s]
        if frm == c and s == self.tail[frm]:
            return
        prev, nxt, head, tail = self.prev, self.next, self.head, self.tail
        if prev[s] >= 0:
            nxt[prev[s]] = nxt[s]
        else:
            head[frm] = nxt[s]
        if nxt[s] >= 0:
            prev[nxt[s]] = prev[s]
        else:
            tail[frm] = prev[s]
        prev[s], nxt[s] = tail[c], -1
        if tail[c] >= 0:
            nxt[tail[c]] = s
        else:
            head[c] = s
        tail[c] = s
        self.list_of[s] = c

    def class_of(self, key):
        return min(N_CLASS - 1, int(f32(math.log2(self.crf[key]))))

    def decayed_crf(self, il, key):
        crf = self.crf.get(key, 0.0)
        if crf == 0.0:
            return 0.0
        x = f32(f32(-(self.clock[il] - self.last_use[key])) / f32(CRF_HALF_LIFE))
        return f32(crf * f32(2.0 ** x))

    def use(self, il, key):
        self.crf[key] = f32(self.decayed_crf(il, key) + 1.0)
        self.last_use[key] = self.clock[il]

    def is_selected(self, il, key):
        return key >= 0 and key // self.n_expert == il and self.seen[key % self.n_expert] == self.seen_gen

    def victim(self, il):
        res, res_score = -1, -1.0
        for c in range(N_CLASS):
            s = self.head[c]
            while s >= 0 and self.is_selected(il, self.key_of[s]):
                s = self.next[s]
            if s < 0:
                continue
            key = self.key_of[s]
            if key < 0:
                return s  # an unused slot
            score = math.ldexp(float(self.clock[key // self.n_expert] - self.last_use[key]), -c)
            if score > res_score:
                res, res_score = s, score
        if res < 0:
            raise RuntimeError("no MoE cache victim")
        return res

    def plan(self, il, ids, n_used, n_zero, n_cpu):
        """-> (hits, filled, cpu) for one lookup of layer il; ids holds n_tokens*n_used experts."""
        self.seen_gen += 1
        seen, gen = self.seen, self.seen_gen
        uniq = []
        for e in ids:
            if seen[e] != gen:
                seen[e] = gen
                uniq.append(e)
        if len(uniq) > self.n_slots:
            raise RuntimeError(f"layer {il}: {len(uniq)} experts for {self.n_slots} slots")
        slots, n_slots = self.slots(il), self.n_slots
        n_tokens = len(ids) // n_used
        hits, misses = 0, []
        for e in uniq:
            key = il * self.n_expert + e
            self.use(il, key)
            if slots[e] < n_slots:
                self.move(slots[e], self.class_of(key))
                hits += 1
            else:
                misses.append(e)

        # the misses with the lowest CRF go to the CPU; each takes a zero slot, at most once per token
        n_c = min(n_cpu(len(misses)), len(misses))
        n_cpu_done = 0
        if n_c > 0:
            misses.sort(key=lambda e: (self.crf[il * self.n_expert + e], e))
            token_zero = [0] * n_tokens
            n_keep, i = n_c, 0
            while i < n_keep:
                e = misses[i]
                toks = [t for t in range(n_tokens) if e in ids[t * n_used:(t + 1) * n_used]]
                used = 0
                for t in toks:
                    used |= token_zero[t]
                z = 0
                while z < 64 and (used >> z) & 1:
                    z += 1
                if z >= n_zero:
                    n_keep -= 1
                    misses[i], misses[n_keep] = misses[n_keep], misses[i]
                    continue
                for t in toks:
                    token_zero[t] |= 1 << z
                slots[e] = n_slots + z
                n_cpu_done += 1
                i += 1
            del misses[:n_keep]

        misses.sort()  # filled in expert order, as llama.cpp does
        for e in misses:
            s = self.victim(il)
            key = self.key_of[s]
            if key >= 0:
                self.slots(key // self.n_expert)[key % self.n_expert] = n_slots
            else:
                self.n_filled += 1
            key = il * self.n_expert + e
            self.key_of[s] = key
            slots[e] = s
            self.move(s, self.class_of(key))
        self.clock[il] += n_tokens
        return hits, len(misses), n_cpu_done


# ---------------------------------------------------------------------------- replay
def replay(trace, size_mib, share):
    """Counters as llama.cpp logs them ("ubatch <= 8" and "ubatch > 8"), plus lookups the cache
    could not serve at this size (layers left uncached, or a batch wider than the slots)."""
    n_layer, n_expert_used = trace["n_layer"], trace["n_expert_used"]
    share = f32(share)
    share_q = int(math.floor(f32(share * 65536.0) + 0.5)) if share > 0 else 0  # std::lround
    n_zero = min(n_expert_used, 64) if share > 0.0 else 0
    caches, group_of, n_slots = [], {}, []
    for ig, g in enumerate(trace["groups"]):
        n = n_slots_of(g, size_mib, n_zero, n_expert_used)
        n_slots.append(n)
        caches.append(ClassLRU(n_layer, g["n_expert"], n) if n > 0 else None)
        group_of.update((il, ig) for il in g["layers"])
    acc = [0] * n_layer
    stats = {k: {"hits": 0, "misses": 0, "cpu": 0, "bytes": 0} for k in ("small", "large")}
    stats["uncached"] = {"lookups": 0, "expert_uses": 0, "bytes": 0}
    first = min(il for il, _, _ in trace["records"]) if trace["records"] else 0
    tokens = 0
    for il, n_used, ids in trace["records"]:
        n_tokens = len(ids) // n_used
        if il == first:
            tokens += n_tokens
        g, c = trace["groups"][group_of[il]], caches[group_of[il]]
        if c is None or n_tokens > MAX_BATCH or min(n_tokens * n_used, g["n_expert"]) > c.n_slots:
            u = stats["uncached"]
            u["lookups"] += 1
            u["expert_uses"] += len(set(ids))
            u["bytes"] += len(set(ids)) * g["expert_bytes"]
            continue
        q = share_q if c.n_filled == c.n_slots else 0  # the share applies once the cache is full

        def n_cpu(n_miss, il=il, q=q):
            v = n_miss * q + acc[il]
            acc[il] = v & 0xFFFF
            return v >> 16

        hits, filled, cpu = c.plan(il, ids, n_used, n_zero, n_cpu)
        st = stats["small" if len(ids) <= 8 * n_expert_used else "large"]
        st["hits"] += hits
        st["misses"] += filled + cpu
        st["cpu"] += cpu
        st["bytes"] += filled * g["expert_bytes"]
    return {"size_mib": size_mib, "share": share, "n_slots": n_slots, "tokens": tokens, **stats}


def log_line(name, st):
    """The line llama.cpp prints for these counters, for a direct comparison."""
    n = st["hits"] + st["misses"]
    if n == 0:
        return f"llama_moe_cache: {name}: (no lookups)"
    return (f"llama_moe_cache: {name}: hits = {st['hits']}, misses = {st['misses']} ({st['cpu']} on the CPU), "
            f"hit rate = {100.0 * st['hits'] / n:.2f}%, uploaded = {st['bytes'] / 1024.0 / 1024.0:.2f} MiB")


# ---------------------------------------------------------------------------- self-check
def _group(layers, n_expert=16, nb2=(1000, 1000, 1000), host_bytes=None):
    hb = host_bytes or len(layers) * n_expert * sum(nb2)
    return {"device": 0, "n_expert": n_expert, "layers": list(layers), "split": 1.0, "split_sum": 1.0,
            "host_bytes": hb, "dev_host_bytes": hb, "alignment": 1, "nb2": list(nb2)}


def self_check():
    tmp = tempfile.mkdtemp(prefix="mct1_")
    path = os.path.join(tmp, "t.mct1")
    g = _group([0, 1])
    # round trip
    recs = [(0, 2, (3, 5)), (1, 2, (3, 7)), (0, 2, (3, 5)), (1, 2, (8, 7))]
    write_trace(path, 2, 2, [g], recs)
    t = read_trace(path)
    assert t["records"] == recs and t["groups"][0]["expert_bytes"] == 3000, "MCT1 round trip"
    # sizing: 3000 B per slot, alignment 1, no zero slots: (n+1)*3000 <= budget
    assert n_slots_of(g, 32000 / 2**20, 0, 2) == 9, n_slots_of(g, 32000 / 2**20, 0, 2)
    assert n_slots_of(g, 4000 / 2**20, 0, 2) == 0  # below n_expert_used -> not cached

    # 1. a cache that holds everything misses each (layer, expert) once, then always hits
    recs = [(il, 2, (e, (e + 1) % 16)) for _ in range(5) for il in (0, 1) for e in range(0, 16, 2)]
    write_trace(path, 2, 2, [g], recs)
    r = replay(read_trace(path), 1.0, 0.0)
    distinct = len({(il, e) for il, _, ids in recs for e in ids})
    assert r["small"]["misses"] == distinct and r["small"]["hits"] == 2 * len(recs) - distinct, r["small"]
    assert r["small"]["bytes"] == distinct * 3000

    # 2. a hot pair reused every token survives a stream of one-off experts in a 4-slot cache
    stream = [(0, 2, (0, 1))] + [(0, 2, (0, 1)) if i % 2 else (0, 2, (2 + i % 14, 2 + (i + 7) % 14)) for i in range(400)]
    write_trace(path, 1, 2, [_group([0])], stream)
    r4 = replay(read_trace(path), 5 * 3000 / 2**20, 0.0)  # a bank holds n_slots + n_zero + 1 slices
    assert r4["n_slots"] == [4], r4["n_slots"]
    hot_hits = sum(1 for _, _, ids in stream if ids == (0, 1)) * 2 - 2  # all but the first use
    assert r4["small"]["hits"] >= hot_hits, (r4["small"], hot_hits)

    # 3. a CPU share takes effect only once the cache is full, at the rate the carry implies
    rnd = [(0, 2, ((7 * i) % 16, (11 * i + 3) % 16)) for i in range(2000)]
    rnd = [(il, n, ids if ids[0] != ids[1] else (ids[0], (ids[0] + 1) % 16)) for il, n, ids in rnd]
    write_trace(path, 1, 2, [_group([0])], rnd)
    tr = read_trace(path)
    r0, r5 = replay(tr, 8 * 3000 / 2**20 + 3 * 3000 / 2**20, 0.0), replay(tr, 8 * 3000 / 2**20 + 3 * 3000 / 2**20, 0.5)
    assert r0["small"]["cpu"] == 0 and r5["small"]["cpu"] > 0
    frac = r5["small"]["cpu"] / r5["small"]["misses"]
    assert 0.35 < frac < 0.55, frac  # 0.5 of the misses after the fill; the fill itself uploads

    # 4. a batch wider than the slots bypasses the cache, as get_slot_map does
    wide = [(0, 2, tuple(range(16)))]
    write_trace(path, 1, 2, [_group([0])], wide)
    rw = replay(read_trace(path), 5 * 3000 / 2**20, 0.0)
    assert rw["uncached"]["lookups"] == 1 and rw["small"]["hits"] + rw["small"]["misses"] == 0
    os.remove(path)
    os.rmdir(tmp)
    print("self-check: MCT1 round trip, sizing, full-cache, hot-set retention, CPU share, wide-batch bypass: PASS")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace", nargs="?")
    ap.add_argument("--mib", type=float)
    ap.add_argument("--share", type=float, default=0.0)
    ap.add_argument("--sweep", help="comma-separated cache sizes in MiB")
    ap.add_argument("--shares", default="0", help="comma-separated CPU shares for --sweep")
    ap.add_argument("--self-check", action="store_true")
    a = ap.parse_args()
    if a.self_check:
        return self_check()
    if not a.trace or (a.mib is None and not a.sweep):
        ap.error("pass TRACE with --mib or --sweep, or --self-check")
    t = read_trace(a.trace)
    print(f"{a.trace}: {t['n_layer']} layers, {t['n_expert_used']} experts per token, "
          f"{len(t['groups'])} group(s), {len(t['records'])} lookups")
    sizes = [a.mib] if a.mib is not None else [float(x) for x in a.sweep.split(",")]
    shares = [a.share] if a.mib is not None else [float(x) for x in a.shares.split(",")]
    for mib in sizes:
        for sh in shares:
            r = replay(t, mib, sh)
            print(f"\n--moe-cache-mib {mib:g} --moe-cache-cpu-share {sh:g}: slots {r['n_slots']}, {r['tokens']} tokens")
            print("  " + log_line("ubatch <= 8", r["small"]))
            print("  " + log_line("ubatch  > 8", r["large"]))
            if r["uncached"]["lookups"]:
                u = r["uncached"]
                print(f"  not served by the cache: {u['lookups']} lookups, {u['bytes'] / 2**20:.2f} MiB of experts")
    return 0


if __name__ == "__main__":
    sys.exit(main())
