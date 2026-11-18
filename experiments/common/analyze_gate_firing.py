"""When did the gate actually change which node got picked?

`RepresentationSelector._pick_gated_batch` already logs everything needed
(gate_theta, n_pool, n_gated, and every pool candidate tagged `in_gate`), so
this reads a completed run rather than instrumenting a new one.

The distinction that matters. `picked = (gated + rest)[:k]` over a pool held in
descending-logprob order, so with k=1:

    gated empty            -> picks pool[0]          == max-mu.  Gate inert.
    pool[0] cleared gate   -> picks pool[0]          == max-mu.  Gate fired but
                                                                 changed nothing.
    pool[0] rejected,
      something else kept  -> picks gated[0] != pool[0].         Gate CHANGED
                                                                 the pick.

Only the third case can make a gated run differ from its max-mu baseline. A run
whose arms are bit-identical has zero of them, and this script says which of the
first two is responsible -- a gate that never fires is a different bug from a
gate that fires and always agrees.

Writes one JSONL record per changed pick (what was rejected, what was taken
instead, and the z-scores of both) plus a summary JSON.

Example:
    python -m experiments.common.analyze_gate_firing \\
        --run-dir logging/runs/qwen3_4b_secure_code/gated_shard0/logs_20260826113532 \\
        --out-jsonl gate_changes.jsonl --out-summary gate_summary.json
"""

import argparse
import glob
import json
import os
from collections import Counter


def iter_objects(path):
    """Yield top-level JSON objects from a file of concatenated pretty-printed
    JSON (the run logs are indented, so they are not JSON-lines)."""
    dec = json.JSONDecoder()
    text = open(path).read()
    i, n = 0, len(text)
    while i < n:
        while i < n and text[i] in " \t\r\n":
            i += 1
        if i >= n:
            return
        try:
            obj, j = dec.raw_decode(text, i)
        except ValueError:
            return
        yield obj
        i = j


def pct(a, b):
    return 0.0 if not b else 100.0 * a / b


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--run-dir", required=True, help="a logs_<timestamp> directory")
    ap.add_argument("--out-jsonl", required=True, help="one record per changed pick")
    ap.add_argument("--out-summary", required=True)
    ap.add_argument("--max-examples", type=int, default=200,
                    help="cap on rejected-candidate detail kept per record")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.run_dir, "*.jsonl")))
    files = [f for f in files if os.path.basename(f)[0].isdigit()]
    if not files:
        raise SystemExit(f"no instance jsonl files in {args.run_dir}")

    n_tx = 0            # transitions carrying gate telemetry
    n_active = 0        # gate had at least one survivor
    n_changed = 0       # survivor set actually moved the pick off pool[0]
    n_pool_trivial = 0  # pool had <2 candidates -- nothing to choose between
    theta = None
    mu = sd = None
    pool_sizes, gated_sizes, incompletes = Counter(), Counter(), Counter()
    all_z = []
    changed = []

    n_sibling_tie = 0   # every pool candidate shares one z -> gate cannot rank

    for f in files:
        inst = os.path.splitext(os.path.basename(f))[0]
        for outer in iter_objects(f):
            if not isinstance(outer, dict):
                continue
            # The selector's telemetry is merged into the transition record's
            # "pick_info", not the top level.
            rec = outer.get("pick_info") if isinstance(outer.get("pick_info"), dict) else outer
            if "gate_theta" not in rec:
                continue
            n_tx += 1
            theta = rec.get("gate_theta", theta)
            mu = rec.get("gate_mu", mu)
            sd = rec.get("gate_sd", sd)
            n_pool = rec.get("n_pool", 0)
            n_gated = rec.get("n_gated", 0)
            pool_sizes[n_pool] += 1
            gated_sizes[n_gated] += 1
            incompletes[rec.get("n_incomplete", 0)] += 1
            if n_gated:
                n_active += 1
            if n_pool < 2:
                n_pool_trivial += 1

            cands = rec.get("top_candidates") or []
            for c in cands:
                if c.get("violation_z") is not None:
                    all_z.append(float(c["violation_z"]))
            # _z() scores an element's PARENT, and an expansion's children all
            # share one parent -- so siblings carry an identical z and the gate
            # can only accept or reject them as a block. When the whole pool
            # ties, the gate cannot reorder anything by construction.
            zs_here = {round(float(c.get("violation_z", 0.0)), 12) for c in cands}
            if cands and len(zs_here) == 1:
                n_sibling_tie += 1

            # The pick moved only if the most probable candidate was rejected
            # while something else survived.
            if n_gated and cands and not cands[0].get("in_gate", False):
                n_changed += 1
                taken = next((c for c in cands if c.get("in_gate")), None)
                changed.append({
                    "instance": inst,
                    "n_pool": n_pool,
                    "n_gated": n_gated,
                    "n_incomplete": rec.get("n_incomplete"),
                    "rejected_top": {
                        k: cands[0].get(k)
                        for k in ("element_id", "logprob", "violation_estimate",
                                  "violation_z", "decoded")
                    },
                    "taken_instead": None if taken is None else {
                        k: taken.get(k)
                        for k in ("element_id", "logprob", "violation_estimate",
                                  "violation_z", "decoded")
                    },
                })

    with open(args.out_jsonl, "w") as fh:
        for c in changed[: args.max_examples]:
            fh.write(json.dumps(c) + "\n")

    all_z.sort()

    def q(p):
        return None if not all_z else float(all_z[min(len(all_z) - 1, int(p * len(all_z)))])

    summary = {
        "run_dir": args.run_dir,
        "n_instances": len(files),
        "gate_theta": theta,
        "gate_mu": mu,
        "gate_sd": sd,
        "transitions_with_gate": n_tx,
        "gate_active": n_active,
        "gate_active_pct": pct(n_active, n_tx),
        "pick_changed": n_changed,
        "pick_changed_pct": pct(n_changed, n_tx),
        "all_candidates_tied_on_z": n_sibling_tie,
        "all_candidates_tied_on_z_pct": pct(n_sibling_tie, n_tx),
        "pool_size_lt2": n_pool_trivial,
        "pool_size_lt2_pct": pct(n_pool_trivial, n_tx),
        "pool_size_hist": dict(sorted(pool_sizes.items())[:12]),
        "gated_size_hist": dict(sorted(gated_sizes.items())[:12]),
        "n_incomplete_hist": dict(sorted(incompletes.items())[:12]),
        "candidate_z": {
            "n": len(all_z),
            "min": q(0.0), "p10": q(.10), "p50": q(.50), "p90": q(.90),
            "max": None if not all_z else float(all_z[-1]),
            "frac_ge_theta": None if (not all_z or theta is None)
                             else sum(1 for z in all_z if z >= theta) / len(all_z),
        },
    }
    json.dump(summary, open(args.out_summary, "w"), indent=1)

    print(f"instances={summary['n_instances']}  transitions with gate telemetry={n_tx}")
    print(f"theta={theta}  mu={mu}  sd={sd}")
    print(f"gate active (>=1 survivor) : {n_active} ({summary['gate_active_pct']:.1f}%)")
    print(f"PICK ACTUALLY CHANGED      : {n_changed} ({summary['pick_changed_pct']:.1f}%)")
    print(f"pool had <2 candidates     : {n_pool_trivial} "
          f"({summary['pool_size_lt2_pct']:.1f}%)   <- nothing to choose between")
    print(f"whole pool tied on z       : {n_sibling_tie} "
          f"({summary['all_candidates_tied_on_z_pct']:.1f}%)   <- gate cannot reorder")
    print(f"pool size hist   : {summary['pool_size_hist']}")
    print(f"gated size hist  : {summary['gated_size_hist']}")
    print(f"n_incomplete hist: {summary['n_incomplete_hist']}")
    cz = summary["candidate_z"]
    if cz["n"]:
        print(f"candidate z (runtime): n={cz['n']} min={cz['min']:.3f} p10={cz['p10']:.3f} "
              f"p50={cz['p50']:.3f} p90={cz['p90']:.3f} max={cz['max']:.3f}")
        print(f"  fraction clearing theta={theta}: {cz['frac_ge_theta']:.3f}")
    print(f"\nwrote {args.out_jsonl} ({min(len(changed), args.max_examples)} records) "
          f"and {args.out_summary}")


if __name__ == "__main__":
    main()
