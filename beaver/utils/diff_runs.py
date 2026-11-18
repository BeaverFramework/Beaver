"""Diagnostic: compare two BEAVER runs and explain why their results differ (or don't).

Typical use: compare a baseline frontier run against a representation-verifier run
that is supposed to do better thanks to a trained probe. Produces a report that
tells you whether:
  * the runs were run with apples-to-apples params,
  * the probe actually changed which branch was expanded at each step,
  * the probe outputs are informative (non-constant / calibrated),
  * UB/LB diverge per-instance, and
  * early termination would have changed outcomes.

Usage:
  python -m beaver.utils.diff_runs <baseline_dir> <repr_dir> [--out <file.json>]
"""

import argparse
import json
from pathlib import Path

import numpy as np

from beaver.logging import get_log_data


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def load_run(run_dir: Path):
    """Return (run_args_dict, {instance_id: [transition_entry, ...]})."""
    args_file = run_dir / "run_args.json"
    run_args = json.loads(args_file.read_text()) if args_file.exists() else {}
    log_data = get_log_data(run_dir)
    # drop the header entry (no "transition" key), index by int
    filtered = {}
    for inst, entries in log_data.items():
        trans = [e for e in entries if "transition" in e]
        filtered[int(inst)] = trans
    return run_args, filtered


def param_diff(a: dict, b: dict):
    """Dict of differing run-arg keys."""
    diffs = {}
    keys = set(a) | set(b)
    for k in sorted(keys):
        if a.get(k) != b.get(k):
            diffs[k] = (a.get(k), b.get(k))
    return diffs


def final_bounds(entries):
    if not entries:
        return None
    e = entries[-1]
    return {
        "ub": e.get("upper_bound", 1.0),
        "lb": e.get("lower_bound", 0.0),
        "cvp": 1.0 - e.get("upper_bound", 1.0),
        "trans": e.get("transition", 0),
        "viol_prob": e.get("violation prob sum", 0.0),
        "complete_size": e.get("complete_size", 0),
        "incomplete_size": e.get("incomplete_size", 0),
    }


def tokens_of(entry):
    return tuple(entry.get("expanded element", []))


def expanded_sequence(entries):
    """List of token-tuples expanded at each transition (the 'trace')."""
    return [tokens_of(e) for e in entries]


def first_divergence(seq_a, seq_b):
    """First index where the two expansion traces diverge; None if one is prefix."""
    for i, (x, y) in enumerate(zip(seq_a, seq_b)):
        if x != y:
            return i
    return None


def transition_at_threshold(entries, threshold):
    """First transition where upper_bound dips below threshold (None if never)."""
    for e in entries:
        if e.get("upper_bound", 1.0) < threshold:
            return e["transition"]
    return None


def probe_stats(entries):
    """Stats on violation_estimate field from repr-verifier logs."""
    ves = [e["violation_estimate"] for e in entries if "violation_estimate" in e]
    if not ves:
        return None
    arr = np.asarray(ves, dtype=float)
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "frac_gt_0.5": float((arr > 0.5).mean()),
        "frac_gt_0.1": float((arr > 0.1).mean()),
        "frac_eq_0": float((arr == 0.0).mean()),
    }


def probe_vs_prob_picks(entries):
    """For each transition, compare the element actually picked vs the argmax-prob pick
    among the top incomplete seqs recorded in `probe_data`.

    Returns:
        n_records  — transitions with probe_data
        n_changed  — transitions where probe ranking != logprob ranking at top
        mean_rank_of_prob_top — avg rank (by violation score) of the argmax-prob child
    """
    n_changed = 0
    n_records = 0
    ranks = []
    for e in entries:
        pd = e.get("probe_data") or []
        if len(pd) < 2:
            continue
        n_records += 1
        # probe_data is already sorted by the violation-aware score (descending).
        # Rank items by logprob as well; see if the argmax-prob item is at rank 0.
        by_prob = sorted(pd, key=lambda s: s["logprob"], reverse=True)
        prob_top = by_prob[0]
        try:
            rank_of_prob_top = next(
                i for i, s in enumerate(pd) if s["element_id"] == prob_top["element_id"]
            )
        except StopIteration:
            rank_of_prob_top = -1
        ranks.append(rank_of_prob_top)
        if rank_of_prob_top != 0:
            n_changed += 1
    return {
        "transitions_with_probe_data": n_records,
        "transitions_where_probe_reorders": n_changed,
        "frac_reordered": (n_changed / n_records) if n_records else 0.0,
        "mean_rank_of_prob_argmax_in_probe_order": (
            float(np.mean(ranks)) if ranks else None
        ),
    }


def calibration_check(entries):
    """Does the probe predict future violation rate at this node?
    Rough proxy: correlate entry's violation_estimate with the *next few* entries'
    total_violation_prob / num_violations within the same instance trace.
    """
    ves = []
    future_viol = []
    for i, e in enumerate(entries):
        if "violation_estimate" not in e:
            continue
        ve = e["violation_estimate"]
        # look at violation mass in the next 5 transitions as a proxy
        window = entries[i + 1 : i + 6]
        fv = sum(w.get("total_violation_prob", 0.0) for w in window)
        ves.append(ve)
        future_viol.append(fv)
    if len(ves) < 10:
        return None
    ves = np.asarray(ves)
    fv = np.asarray(future_viol)
    if ves.std() == 0 or fv.std() == 0:
        return {"pearson_r": 0.0, "n": int(ves.size), "note": "zero-variance signal"}
    r = float(np.corrcoef(ves, fv)[0, 1])
    return {"pearson_r": r, "n": int(ves.size)}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def build_report(base_dir: Path, repr_dir: Path, threshold: float = 0.9):
    base_args, base_data = load_run(base_dir)
    repr_args, repr_data = load_run(repr_dir)

    shared = sorted(set(base_data) & set(repr_data))
    only_base = sorted(set(base_data) - set(repr_data))
    only_repr = sorted(set(repr_data) - set(base_data))

    # ── run-arg diff ─────────────────────────────────────────────────────
    pdiff = param_diff(base_args, repr_args)

    # ── per-instance comparison ──────────────────────────────────────────
    per_inst = []
    ub_diffs, lb_diffs, cvp_diffs, trans_diffs = [], [], [], []
    sat_flip = 0  # instances where constraint verdict differs
    divergence_points = []
    early_term_base, early_term_repr = [], []

    for idx in shared:
        b, r = base_data[idx], repr_data[idx]
        fb, fr = final_bounds(b), final_bounds(r)
        if fb is None or fr is None:
            continue

        ub_diff = fr["ub"] - fb["ub"]
        lb_diff = fr["lb"] - fb["lb"]
        cvp_diff = fr["cvp"] - fb["cvp"]
        trans_diff = fr["trans"] - fb["trans"]
        ub_diffs.append(ub_diff)
        lb_diffs.append(lb_diff)
        cvp_diffs.append(cvp_diff)
        trans_diffs.append(trans_diff)

        b_sat = fb["ub"] >= threshold
        r_sat = fr["ub"] >= threshold
        if b_sat != r_sat:
            sat_flip += 1

        div = first_divergence(expanded_sequence(b), expanded_sequence(r))
        divergence_points.append(div if div is not None else -1)

        etb = transition_at_threshold(b, threshold)
        etr = transition_at_threshold(r, threshold)
        if etb is not None:
            early_term_base.append(etb)
        if etr is not None:
            early_term_repr.append(etr)

        per_inst.append(
            {
                "idx": idx,
                "base": fb,
                "repr": fr,
                "ub_diff": ub_diff,
                "lb_diff": lb_diff,
                "cvp_diff": cvp_diff,
                "trans_diff": trans_diff,
                "first_divergence": div,
                "base_sat": b_sat,
                "repr_sat": r_sat,
            }
        )

    # ── probe-specific stats (repr only) ─────────────────────────────────
    ps_all, reorder_all, calib_all = [], [], []
    for idx in shared:
        r = repr_data[idx]
        ps = probe_stats(r)
        if ps:
            ps_all.append(ps)
        ro = probe_vs_prob_picks(r)
        if ro["transitions_with_probe_data"] > 0:
            reorder_all.append(ro)
        cal = calibration_check(r)
        if cal:
            calib_all.append(cal)

    def _agg(items, keys):
        if not items:
            return {}
        out = {}
        for k in keys:
            vals = [i[k] for i in items if i.get(k) is not None]
            if vals:
                out[k + "_mean"] = float(np.mean(vals))
        return out

    # ── summary numbers ─────────────────────────────────────────────────
    def _stats(arr):
        if not arr:
            return None
        a = np.asarray(arr)
        return {
            "mean": float(a.mean()),
            "median": float(np.median(a)),
            "std": float(a.std()),
            "min": float(a.min()),
            "max": float(a.max()),
        }

    report = {
        "threshold": threshold,
        "param_diff": pdiff,
        "coverage": {
            "shared": len(shared),
            "only_base": only_base,
            "only_repr": only_repr,
        },
        "bounds_delta": {
            "ub_diff (repr - base)": _stats(ub_diffs),
            "lb_diff (repr - base)": _stats(lb_diffs),
            "cvp_diff (repr - base)": _stats(cvp_diffs),
            "trans_diff (repr - base)": _stats(trans_diffs),
        },
        "constraint_verdict_flips": sat_flip,
        "divergence_transition": _stats(
            [d for d in divergence_points if d is not None and d >= 0]
        ),
        "n_identical_traces": sum(1 for d in divergence_points if d == -1),
        "early_term_transition@threshold": {
            "baseline": _stats(early_term_base),
            "repr": _stats(early_term_repr),
            "n_base_dipped": len(early_term_base),
            "n_repr_dipped": len(early_term_repr),
        },
        "probe_output_dist": _agg(
            ps_all,
            ["mean", "std", "min", "max", "frac_gt_0.5", "frac_gt_0.1", "frac_eq_0"],
        ),
        "probe_reordering": _agg(
            reorder_all,
            [
                "frac_reordered",
                "mean_rank_of_prob_argmax_in_probe_order",
            ],
        ),
        "probe_calibration_proxy": _agg(calib_all, ["pearson_r"]),
        "n_instances_per_inst": len(per_inst),
    }
    return report, per_inst


def print_report(report, per_inst, top_k_per_inst=10):
    print("=" * 72)
    print("RUN DIFF REPORT")
    print("=" * 72)

    print("\n[run-arg diff]")
    if not report["param_diff"]:
        print("  (identical run args)")
    else:
        for k, (a, b) in report["param_diff"].items():
            print(f"  {k:35s}  baseline={a}   repr={b}")

    print("\n[coverage]")
    c = report["coverage"]
    print(f"  shared instances: {c['shared']}")
    if c["only_base"]:
        print(f"  only-in-baseline: {c['only_base']}")
    if c["only_repr"]:
        print(f"  only-in-repr:     {c['only_repr']}")

    print("\n[final bounds delta (repr - base)]")
    for k, v in report["bounds_delta"].items():
        if v is None:
            continue
        print(
            f"  {k:30s}  mean={v['mean']:+.4f}  median={v['median']:+.4f}  "
            f"std={v['std']:.4f}  min={v['min']:+.4f}  max={v['max']:+.4f}"
        )

    print(f"\n[constraint-verdict flips @ threshold={report['threshold']}]")
    print(f"  instances flipped: {report['constraint_verdict_flips']}")

    print("\n[expansion-trace divergence]")
    d = report["divergence_transition"]
    print(f"  identical traces: {report['n_identical_traces']}/{report['n_instances_per_inst']}")
    if d:
        print(
            f"  first divergence — mean={d['mean']:.1f} median={d['median']:.1f}  "
            f"min={d['min']}  max={d['max']}"
        )

    print("\n[early-termination transition @ threshold]")
    et = report["early_term_transition@threshold"]
    if et["baseline"]:
        v = et["baseline"]
        print(
            f"  baseline UB<thr: n={et['n_base_dipped']}  "
            f"mean_trans={v['mean']:.1f}  median={v['median']:.1f}"
        )
    else:
        print(f"  baseline UB<thr: never (n={et['n_base_dipped']})")
    if et["repr"]:
        v = et["repr"]
        print(
            f"  repr     UB<thr: n={et['n_repr_dipped']}  "
            f"mean_trans={v['mean']:.1f}  median={v['median']:.1f}"
        )
    else:
        print(f"  repr     UB<thr: never (n={et['n_repr_dipped']})")

    print("\n[probe output distribution (repr run)]")
    for k, v in report["probe_output_dist"].items():
        print(f"  {k:25s} {v:.4f}")

    print("\n[probe reordering power]")
    for k, v in report["probe_reordering"].items():
        print(f"  {k:45s} {v:.4f}")

    print("\n[probe calibration proxy — Pearson r of ve vs future violation mass]")
    for k, v in report["probe_calibration_proxy"].items():
        print(f"  {k:20s} {v:+.4f}")

    # worst offenders: largest |lb_diff| — repr giving up most on LB
    print("\n[per-instance outliers: largest LB regression (repr < base)]")
    worst = sorted(per_inst, key=lambda x: x["lb_diff"])[:top_k_per_inst]
    for x in worst:
        print(
            f"  idx={x['idx']:3d}  ub {x['base']['ub']:.3f}->{x['repr']['ub']:.3f}  "
            f"lb {x['base']['lb']:.3f}->{x['repr']['lb']:.3f}  "
            f"trans {x['base']['trans']}->{x['repr']['trans']}  "
            f"diverge@{x['first_divergence']}"
        )

    print("\n[per-instance outliers: largest UB regression (repr > base, worse)]")
    worst = sorted(per_inst, key=lambda x: -x["ub_diff"])[:top_k_per_inst]
    for x in worst:
        print(
            f"  idx={x['idx']:3d}  ub {x['base']['ub']:.3f}->{x['repr']['ub']:.3f}  "
            f"lb {x['base']['lb']:.3f}->{x['repr']['lb']:.3f}  "
            f"trans {x['base']['trans']}->{x['repr']['trans']}  "
            f"diverge@{x['first_divergence']}"
        )
    print()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("baseline_dir", type=Path)
    p.add_argument("repr_dir", type=Path)
    p.add_argument("--threshold", type=float, default=0.9)
    p.add_argument("--out", type=Path, default=None, help="Write JSON report here.")
    p.add_argument("--top_k", type=int, default=10)
    args = p.parse_args()

    report, per_inst = build_report(args.baseline_dir, args.repr_dir, args.threshold)
    print_report(report, per_inst, top_k_per_inst=args.top_k)

    if args.out:
        args.out.write_text(
            json.dumps({"report": report, "per_instance": per_inst}, indent=2)
        )
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
