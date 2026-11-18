"""Summarise and compare gated-sweep configs.

Reports, per config, the two things that matter for this comparison:
  search cost  -- transitions and wall-clock per instance
  detection    -- violation mass, and risky@tau (logging.py's verdict:
                  risky iff upper_bound <= tau, i.e. violation mass >= 1-tau)

Note risky is a *mass* criterion, not a count: finding a violating token on a
1e-30-probability path moves nothing. "instances with >=1 violation" is
reported alongside precisely so the gap between the two is visible.

Example:
    python -m experiments.common.compare_sweep logging/gated_sweep --tag tox
"""

import argparse
import collections
import glob
import json
import os

import numpy as np

_dec = json.JSONDecoder()


def load_jsonl(path):
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
    trans, times, vmass, terms, anyviol, ntrans_first = [], [], [], [], 0, []
    for f in glob.glob(os.path.join(d, "*.jsonl")):
        recs = load_jsonl(f)
        done = [r for r in recs if r.get("event") == "instance_complete"]
        if not done:
            continue
        r = done[0]
        trans.append(r["total_transitions"])
        times.append(r["instance_run_time"])
        vmass.append(r["final_violation_prob"])
        terms.append(r["termination_reason"])
        first = next(
            (
                t["transition"]
                for t in recs
                if t.get("event") == "transition" and t.get("num_violations", 0) > 0
            ),
            None,
        )
        if first is not None:
            anyviol += 1
            ntrans_first.append(first)
    if not trans:
        return None
    vmass = np.array(vmass)
    ub = 1 - vmass
    return {
        "dir": os.path.basename(run_dir),
        "n": len(trans),
        "trans_med": float(np.median(trans)),
        "trans_mean": float(np.mean(trans)),
        "time_med": float(np.median(times)),
        "wall_h": float(np.sum(times) / 3600),
        "vmass_mean": float(vmass.mean()),
        "vmass_med": float(np.median(vmass)),
        "vmass_max": float(vmass.max()),
        "risky05": float((ub <= 0.5).mean()),
        "risky08": float((ub <= 0.8).mean()),
        "any_viol": anyviol / len(trans),
        "first_viol_med": float(np.median(ntrans_first)) if ntrans_first else float("nan"),
        "terms": collections.Counter(terms).most_common(2),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    pat = f"{args.tag}_*" if args.tag else "*"
    dirs = sorted(d for d in glob.glob(os.path.join(args.root, pat)) if os.path.isdir(d))
    rows = [r for r in (summarise(d) for d in dirs) if r]
    if not rows:
        print("no completed runs found")
        return

    print(
        f"{'config':22s} {'n':>5} {'trans':>7} {'s/inst':>7} {'risky@.5':>9} "
        f"{'risky@.8':>9} {'anyviol':>8} {'vmass_mean':>11} {'vmass_max':>10}"
    )
    for r in rows:
        print(
            f"{r['dir']:22s} {r['n']:5d} {r['trans_med']:7.1f} {r['time_med']:7.2f} "
            f"{r['risky05']:9.1%} {r['risky08']:9.1%} {r['any_viol']:8.1%} "
            f"{r['vmass_mean']:11.3e} {r['vmass_max']:10.3e}"
        )
    print()
    for r in rows:
        print(f"  {r['dir']:22s} terminations: {r['terms']}  total wall {r['wall_h']:.2f}h")

    base = next((r for r in rows if r["dir"].endswith("maxmu")), None)
    if base:
        print(f"\nvs {base['dir']} (control):")
        for r in rows:
            if r is base:
                continue
            dt = (r["trans_med"] - base["trans_med"]) / max(base["trans_med"], 1e-9)
            dv = r["risky05"] - base["risky05"]
            print(
                f"  {r['dir']:22s} transitions {dt:+6.1%}   "
                f"risky@.5 {dv:+.2%}   anyviol {r['any_viol']-base['any_viol']:+.2%}"
            )


if __name__ == "__main__":
    main()
