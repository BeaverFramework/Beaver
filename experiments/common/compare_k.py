"""Compare expand_batch_size (k) runs: speed and quality drift.

Transition counts are NOT comparable across k -- the stop check runs after a
whole round, so a k=16 run can overshoot epsilon by up to 15 expansions. The
commensurable cost measures are forward passes (model calls) and wall clock.
"""

import argparse
import glob
import json
import os

import numpy as np

_dec = json.JSONDecoder()


def load(path):
    s = open(path).read()
    i, out = 0, []
    while i < len(s):
        while i < len(s) and s[i] in " \n\t\r":
            i += 1
        if i >= len(s):
            break
        o, i = _dec.raw_decode(s, i)
        out.append(o)
    return out


def summarise(run_dir):
    logs = sorted(glob.glob(os.path.join(run_dir, "logs_*")))
    if not logs:
        return None
    d = logs[-1]
    rounds = fwd = 0
    vmass, times, batch = [], [], []
    for f in glob.glob(os.path.join(d, "*.jsonl")):
        recs = load(f)
        done = [r for r in recs if r.get("event") == "instance_complete"]
        if not done:
            continue
        vmass.append(done[0]["final_violation_prob"])
        times.append(done[0]["instance_run_time"])
        for t in recs:
            if t.get("event") != "transition":
                continue
            rounds += 1
            b = t.get("batch_size", 1)
            fwd += b
            batch.append(b)
    if not vmass:
        return None
    v = np.array(vmass)
    return {
        "n": len(v),
        "rounds": rounds,
        "fwd_passes": fwd,
        "mean_batch": float(np.mean(batch)) if batch else 1.0,
        "wall_h": float(np.sum(times) / 3600),
        "s_per_inst": float(np.median(times)),
        "vmass": float(v.mean()),
        "rdr09": float(((1 - v) <= 0.9).mean()),
        "rdr08": float(((1 - v) <= 0.8).mean()),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", nargs="?", default="logging/k_sweep")
    args = ap.parse_args()
    rows = []
    for d in sorted(glob.glob(os.path.join(args.root, "k*"))):
        r = summarise(d)
        if r:
            r["k"] = os.path.basename(d)
            rows.append(r)
    if not rows:
        print("no completed k runs")
        return
    print(f"{'run':>5} {'n':>4} {'rounds':>8} {'fwd':>8} {'batch':>6} "
          f"{'s/inst':>8} {'wall_h':>7} {'vmass':>8} {'RDR@.9':>8} {'RDR@.8':>8}")
    for r in rows:
        print(f"{r['k']:>5} {r['n']:4d} {r['rounds']:8d} {r['fwd_passes']:8d} "
              f"{r['mean_batch']:6.1f} {r['s_per_inst']:8.2f} {r['wall_h']:7.3f} "
              f"{r['vmass']:8.4f} {r['rdr09']:8.1%} {r['rdr08']:8.1%}")
    base = next((r for r in rows if r["k"] == "k1"), None)
    if base:
        print(f"\nvs k1:")
        for r in rows:
            if r is base:
                continue
            print(f"  {r['k']:>4}  wall {base['wall_h']/max(r['wall_h'],1e-9):5.2f}x faster   "
                  f"fwd passes {r['fwd_passes']/max(base['fwd_passes'],1):5.2f}x   "
                  f"vmass {r['vmass']-base['vmass']:+.4f}   RDR@.9 {r['rdr09']-base['rdr09']:+.1%}")


if __name__ == "__main__":
    main()
