#!/usr/bin/env python3
"""Compute & compare forward-pass cost of verification runs.

Cost model (forward passes = FP):
  warmup FP      = warm_start_forward_passes  (cached frontier only; 0 otherwise)
  verification FP = number of transition entries (1 model_generate call each)
  total FP       = warmup + verification

An instance is "risky" (constraint unsatisfied) when its final upper bound
UB < threshold, where UB = incomplete + complete + pruned prob mass. This
mirrors beaver/logging.py's satisfied/unsatisfied logic and works for both
the cached-frontier logs (with warm_start / instance_complete events) and the
plain frontier logs (transition entries only).

Headline metric: total compute spent per risky instance found = sum(total_fp)
over all instances / number of risky instances.

When >= 2 logs are given, also cross-checks methods against each other:
  - risky/safe/open set comparison per pair (common/only-A/only-B, Jaccard),
    matched by dataset idx (not group -- ungrouped/frontier runs log every
    instance under the placeholder group "all", so group isn't a safe join
    key across methods)
  - status x status contingency table over instances both methods verified
  - "hard conflicts": instances proven risky by one method and safe by the
    other -- a soundness bug, since both are proofs against the same
    threshold and can't both be right (unlike a risky/open split, which just
    means one method searched less before giving up)
  - agreement of final UB/LB bounds on commonly-verified instances
  - termination-reason breakdown for risky verdicts, to catch a method
    reaching "risky" through a systematically weaker mechanism
  outputs: printed to stdout, plus status_set_comparison.csv and
  hard_conflicts.csv under --out.

Usage:
    python cost_analysis.py \
        cached=logging/logs_20260617222227 \
        frontier=logging/dist_verf/full_dist_secure_code_qwen3-4b \
        --threshold 0.9 --out cost_out
"""
import argparse
import os
from pathlib import Path
from collections import defaultdict

import numpy as np

from beaver.logging import read_log_file


def instance_records(log_dir, threshold):
    """Yield one cost record per instance .jsonl in ``log_dir``.

    Record: dict(idx, group, warmup_fp, verif_fp, total_fp, risky,
                 warm_skip, determined)
    """
    log_dir = Path(log_dir)
    for fp in sorted(log_dir.glob("*.jsonl"), key=lambda p: int(p.stem) if p.stem.isdigit() else 1 << 30):
        if not fp.stem.isdigit():
            continue
        entries = read_log_file(fp)

        header = next((e for e in entries if e.get("event") == "instance_start"), {})
        group = header.get("group", "all")
        idx = header.get("idx", int(fp.stem))
        start_time = header.get("start_time")

        transitions = [e for e in entries if "transition" in e]
        verif_fp = len(transitions)

        warm = next((e for e in entries if e.get("event") == "warm_start"), None)
        ic = next((e for e in entries if e.get("event") == "instance_complete"), None)

        # Real wall-clock completion time (start_time + how long the instance
        # took), comparable across groups and multiprocessing workers since
        # they share one wall clock. None for older logs that predate the
        # instance_start event.
        instance_run_time = ic.get("instance_run_time") if ic else None
        completion_time = (
            start_time + instance_run_time
            if start_time is not None and instance_run_time is not None
            else None
        )

        warmup_fp = 0
        if warm is not None:
            warmup_fp = warm.get("warm_start_forward_passes", 0)
        elif ic is not None:
            warmup_fp = ic.get("warm_start_forward_passes", 0)

        warm_skip = bool(ic.get("warm_skip", False)) if ic else False

        # Semantic-check cost (gross). Cached runs log the grand total (warm-start
        # + main loop) in instance_complete; cold runs have no such total → sum
        # the per-transition counts. This is the number of sequences sent to the
        # constraint before caching/dedup reduce it.
        if ic is not None and "total_semantic_checks" in ic:
            semantic_attempted = ic.get("total_semantic_checks", 0)
        else:
            semantic_attempted = sum(e.get("num_semantic_checks", 0) for e in transitions)

        # Warm-cache state at the start of this instance. cache_total (antichain
        # size carried from earlier instances of the same group) grows
        # monotonically with processing order, so it lets us recover the order
        # instances ran in within a group. Frontier (cold) logs have no warm
        # event → all zero.
        warm_cache_total = warm.get("cache_total", 0) if warm else 0
        warm_viol_prob = warm.get("warm_violation_prob", 0.0) if warm else 0.0
        new_viol = ic.get("new_violations_found", 0) if ic else 0
        term_reason = ic.get("termination_reason", "") if ic else ""
        # Explicit within-group processing order, if logged (newer cached runs).
        # Falls back to None → order recovered from warm_cache_total downstream.
        group_position = None
        if warm is not None and warm.get("group_position") is not None:
            group_position = warm.get("group_position")
        elif ic is not None and ic.get("group_position") is not None:
            group_position = ic.get("group_position")

        # Bounds on P(constraint satisfied / safe):
        #   UB = incomplete + complete + pruned mass (= 1 - proven-violation)
        #   LB = complete mass (proven-safe)
        # Classification: safe if LB > threshold, risky if UB <= threshold,
        # else open (undecided — needs more search to close the gap).
        determined = True
        if ic is not None:
            inc = ic.get("final_incomplete_prob")
            comp = ic.get("final_complete_prob", 0.0)
            viol = ic.get("final_violation_prob", 0.0)
            pru = ic.get("final_pruned_prob", 0.0)
            ub = (1.0 - viol) if inc is None else (inc + comp + pru)
            lb = comp
        elif transitions:
            last = transitions[-1]
            ub = (
                last.get("incomplete prob sum", 0.0)
                + last.get("complete prob sum", 0.0)
                + last.get("pruned prob sum", 0.0)
            )
            lb = last.get("complete prob sum", 0.0)
        else:
            ub, lb = 1.0, 0.0
            determined = False  # no verdict, no transitions

        if not determined:
            status = "undetermined"
        elif ub <= threshold:
            status = "risky"
        elif lb > threshold:
            status = "safe"
        else:
            status = "open"
        risky = status == "risky"
        safe = status == "safe"
        is_open = status == "open"

        yield {
            "idx": idx,
            "group": group,
            "start_time": start_time,
            "completion_time": completion_time,
            "warmup_fp": warmup_fp,
            "verif_fp": verif_fp,
            "total_fp": warmup_fp + verif_fp,
            "ub": ub,
            "lb": lb,
            "status": status,
            "risky": risky,
            "safe": safe,
            "open": is_open,
            "warm_skip": warm_skip,
            "determined": determined,
            "warm_cache_total": warm_cache_total,
            "warm_viol_prob": warm_viol_prob,
            "new_viol": new_viol,
            "term_reason": term_reason,
            "group_position": group_position,
            "semantic_attempted": semantic_attempted,
        }


# ── aggregation ──────────────────────────────────────────────────────────────
def aggregate(records):
    """Aggregate a list of instance records into summary stats."""
    n = len(records)
    n_risky = sum(r["risky"] for r in records)
    n_safe = sum(r["safe"] for r in records)
    n_open = sum(r["open"] for r in records)
    tot_warm = sum(r["warmup_fp"] for r in records)
    tot_verif = sum(r["verif_fp"] for r in records)
    tot = tot_warm + tot_verif
    per_risky = (tot / n_risky) if n_risky else float("nan")
    tot_sem_att = sum(r["semantic_attempted"] for r in records)
    return {
        "n_instances": n,
        "n_risky": n_risky,
        "n_safe": n_safe,
        "n_open": n_open,
        "n_warm_skip": sum(r["warm_skip"] for r in records),
        "total_warmup_fp": tot_warm,
        "total_verif_fp": tot_verif,
        "total_fp": tot,
        "fp_per_risky": per_risky,
        "warmup_fp_per_risky": (tot_warm / n_risky) if n_risky else float("nan"),
        "verif_fp_per_risky": (tot_verif / n_risky) if n_risky else float("nan"),
        "warmup_fp_per_instance": (tot_warm / n) if n else float("nan"),
        "verif_fp_per_instance": (tot_verif / n) if n else float("nan"),
        "mean_fp_per_instance": (tot / n) if n else float("nan"),
        "median_fp_per_instance": float(np.median([r["total_fp"] for r in records])) if n else float("nan"),
        # semantic-check cost (gross: sent to constraint before caching/dedup)
        "total_semantic": tot_sem_att,
        "semantic_per_instance": (tot_sem_att / n) if n else float("nan"),
    }


# ── table ────────────────────────────────────────────────────────────────────
TABLE_COLS = [
    ("method", "{:<22}"),
    ("group", "{:<14}"),
    ("n", "{:>5}"),
    ("safe", "{:>5}"),
    ("risky", "{:>6}"),
    ("open", "{:>5}"),
    ("warm_fp", "{:>9}"),
    ("verif_fp", "{:>9}"),
    ("total_fp", "{:>9}"),
    ("fp/risky", "{:>10}"),
    ("warm/risky", "{:>11}"),
    ("verif/risky", "{:>12}"),
    ("fp/inst", "{:>9}"),
    ("sem/inst", "{:>9}"),
    ("sem_tot", "{:>9}"),
]


def _fmt(v):
    if isinstance(v, float):
        return "nan" if v != v else f"{v:.1f}"
    return str(v)


def print_table(rows, cols=TABLE_COLS):
    header = "  ".join(f.replace(">", "<").format(name) for name, f in cols)
    print(header)
    print("-" * len(header))
    for row in rows:
        print("  ".join(f.format(_fmt(row[name])) for name, f in cols))


def build_rows(method, records, by_group):
    """One overall row + (optionally) one row per group."""
    rows = []
    agg = aggregate(records)
    rows.append({"method": method, "group": "ALL", **{
        "n": agg["n_instances"], "safe": agg["n_safe"], "risky": agg["n_risky"],
        "open": agg["n_open"],
        "warm_fp": agg["total_warmup_fp"], "verif_fp": agg["total_verif_fp"],
        "total_fp": agg["total_fp"], "fp/risky": agg["fp_per_risky"],
        "warm/risky": agg["warmup_fp_per_risky"], "verif/risky": agg["verif_fp_per_risky"],
        "fp/inst": agg["mean_fp_per_instance"],
        "sem/inst": agg["semantic_per_instance"], "sem_tot": agg["total_semantic"],
    }})
    if by_group:
        groups = defaultdict(list)
        for r in records:
            groups[r["group"]].append(r)
        for g in sorted(groups):
            if g == "all":
                continue
            ga = aggregate(groups[g])
            rows.append({"method": method, "group": g, **{
                "n": ga["n_instances"], "safe": ga["n_safe"], "risky": ga["n_risky"],
                "open": ga["n_open"],
                "warm_fp": ga["total_warmup_fp"], "verif_fp": ga["total_verif_fp"],
                "total_fp": ga["total_fp"], "fp/risky": ga["fp_per_risky"],
                "warm/risky": ga["warmup_fp_per_risky"], "verif/risky": ga["verif_fp_per_risky"],
                "fp/inst": ga["mean_fp_per_instance"],
                "sem/inst": ga["semantic_per_instance"], "sem_tot": ga["total_semantic"],
            }})
    return rows


# ── plotting ─────────────────────────────────────────────────────────────────
# status -> (marker, size, alpha). Shape encodes safe/risky/open across plots.
STATUS_STYLE = {
    "safe": ("o", 22, 0.55),
    "risky": ("X", 70, 1.0),
    "open": ("s", 7, 0.18),
}


def plot(method_aggs, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = list(method_aggs)
    warm = [method_aggs[m]["warmup_fp_per_risky"] for m in methods]
    verif = [method_aggs[m]["verif_fp_per_risky"] for m in methods]
    warm = [0 if v != v else v for v in warm]
    verif = [0 if v != v else v for v in verif]

    x = np.arange(len(methods))
    fig, ax = plt.subplots(figsize=(max(6, 1.8 * len(methods)), 6))
    ax.bar(x, warm, label="warmup", color="#4C72B0")
    ax.bar(x, verif, bottom=warm, label="verification", color="#C44E52")
    for i, m in enumerate(methods):
        tot = warm[i] + verif[i]
        ax.text(i, tot, f"{tot:.0f}", ha="center", va="bottom", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15, ha="right")
    ax.set_ylabel("Forward passes per risky instance found")
    ax.set_title("Compute cost per risky instance (warmup vs verification)")
    ax.legend()
    fig.tight_layout()
    out = Path(out_dir) / "fp_per_risky.png"
    fig.savefig(out, dpi=150)
    print(f"\nPlot saved to: {out}")


def plot_per_instance(method_aggs, out_dir):
    """Same stacked bar as ``plot()`` but normalized by *all* instances
    (safe + risky + open), not just risky ones — the fairer efficiency
    metric since it doesn't ignore compute spent on instances that didn't
    turn out risky."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = list(method_aggs)
    warm = [method_aggs[m]["warmup_fp_per_instance"] for m in methods]
    verif = [method_aggs[m]["verif_fp_per_instance"] for m in methods]
    warm = [0 if v != v else v for v in warm]
    verif = [0 if v != v else v for v in verif]

    x = np.arange(len(methods))
    fig, ax = plt.subplots(figsize=(max(6, 1.8 * len(methods)), 6))
    ax.bar(x, warm, label="warmup", color="#4C72B0")
    ax.bar(x, verif, bottom=warm, label="verification", color="#C44E52")
    for i, m in enumerate(methods):
        tot = warm[i] + verif[i]
        ax.text(i, tot, f"{tot:.1f}", ha="center", va="bottom", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15, ha="right")
    ax.set_ylabel("Forward passes per instance (safe + risky + open)")
    ax.set_title("Compute cost per instance verified (warmup vs verification)")
    ax.legend()
    fig.tight_layout()
    out = Path(out_dir) / "fp_per_instance.png"
    fig.savefig(out, dpi=150)
    print(f"Plot saved to: {out}")


def global_completion_order(records):
    """Sort one method's records into the order instances actually finished
    verification in, using each instance's real wall-clock completion time
    (``start_time`` from its ``instance_start`` event + the logged
    ``instance_run_time``) — comparable across groups and across
    multiprocessing workers since they all share one wall clock. Falls back
    to idx order for older logs that predate the ``instance_start`` event.
    """
    def key(r):
        if r["completion_time"] is not None:
            return (0, r["completion_time"])
        return (1, r["idx"])
    return sorted(records, key=key)


# ── per-group processing order ───────────────────────────────────────────────
def canonical_order(per_method_records):
    """Recover the within-group processing order, shared across methods.

    Preference order:
      1. explicit ``group_position`` logged by the verifier (newer cached runs);
      2. warm-cache size (``warm_cache_total``), which grows with position since
         each instance warm-starts from the cache the previous ones grew;
      3. idx order (cold runs with no warm signal at all).

    Returns: {group: {idx: position}}.
    """
    from collections import defaultdict

    sizes = defaultdict(dict)   # group -> idx -> max warm_cache_total seen
    explicit = defaultdict(dict)  # group -> idx -> logged group_position
    members = defaultdict(set)  # group -> {idx}
    for recs in per_method_records.values():
        for r in recs:
            members[r["group"]].add(r["idx"])
            cur = sizes[r["group"]].get(r["idx"], 0)
            sizes[r["group"]][r["idx"]] = max(cur, r["warm_cache_total"])
            if r.get("group_position") is not None:
                explicit[r["group"]][r["idx"]] = r["group_position"]

    order = {}
    for g, idxs in members.items():
        sz = sizes[g]
        exp = explicit[g]
        if all(i in exp for i in idxs):
            ordered = sorted(idxs, key=lambda i: (exp[i], i))
        elif any(sz.get(i, 0) > 0 for i in idxs):
            ordered = sorted(idxs, key=lambda i: (sz.get(i, 0), i))
        else:
            ordered = sorted(idxs)  # cold: best-effort idx order
        order[g] = {i: p for p, i in enumerate(ordered)}
    return order


TRAJ_COLS = [
    ("pos", "{:>3}"), ("idx", "{:>5}"), ("warm_cache", "{:>10}"),
    ("warm_viol", "{:>9}"), ("warm_fp", "{:>8}"), ("verif_fp", "{:>8}"),
    ("total_fp", "{:>8}"), ("new_viol", "{:>8}"), ("skip", "{:>5}"),
    ("status", "{:>6}"), ("reason", "{:<28}"),
]


def print_group_trajectories(method, records, order):
    """Print, per group, the instances in processing order and how each fared
    starting from the cache the previous ones left behind."""
    from collections import defaultdict

    by_g = defaultdict(list)
    for r in records:
        by_g[r["group"]].append(r)

    print(f"\n{'=' * 80}\nProcessing-order trajectories — method: {method}\n{'=' * 80}")
    for g in sorted(by_g, key=lambda x: str(x)):
        pos_map = order.get(g, {})
        rows = sorted(by_g[g], key=lambda r: pos_map.get(r["idx"], r["idx"]))
        print(f"\n── group {g}  (n={len(rows)}) " + "─" * 40)
        header = "  ".join(f.replace(">", "<").format(c) for c, f in TRAJ_COLS)
        print(header)
        run_total = 0
        for r in rows:
            run_total += r["total_fp"]
            vals = {
                "pos": pos_map.get(r["idx"], "?"),
                "idx": r["idx"],
                "warm_cache": r["warm_cache_total"],
                "warm_viol": f"{r['warm_viol_prob']:.3f}",
                "warm_fp": r["warmup_fp"],
                "verif_fp": r["verif_fp"],
                "total_fp": r["total_fp"],
                "new_viol": r["new_viol"],
                "skip": "Y" if r["warm_skip"] else "",
                "status": r["status"],
                "reason": (r["term_reason"] or "")[:28],
            }
            print("  ".join(f.format(vals[c]) for c, f in TRAJ_COLS))


def plot_group_order(per_method_records, order, out_dir):
    """Per-group subplots, instances in processing order:
      top of each cell  = per-instance verification FP vs position
      (shows the cache warming up — later instances get cheaper)
    plus a second figure with cumulative total FP vs instances verified.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from collections import defaultdict

    groups = sorted(order, key=lambda x: str(x))
    methods = list(per_method_records)
    colors = ["#C44E52", "#4C72B0", "#55A868", "#8172B2"]

    # index records by (method, group)
    rec_by = defaultdict(lambda: defaultdict(list))
    for m, recs in per_method_records.items():
        for r in recs:
            rec_by[m][r["group"]].append(r)

    def ordered(m, g):
        pm = order.get(g, {})
        return sorted(rec_by[m][g], key=lambda r: pm.get(r["idx"], r["idx"]))

    ncol = 2
    nrow = (len(groups) + ncol - 1) // ncol

    for kind, ykey, axlab, fname, title in [
        ("perinst", "verif_fp", "Verification FP (per instance)",
         "fp_vs_order_per_group.png",
         "Per-instance verification cost vs processing order"),
        ("cumul", "total_fp", "Cumulative total FP",
         "cumcost_vs_verified_per_group.png",
         "Instances verified vs cumulative cost (processing order)"),
    ]:
        fig, axes = plt.subplots(nrow, ncol, figsize=(7 * ncol, 3.2 * nrow), squeeze=False)
        for gi, g in enumerate(groups):
            ax = axes[gi // ncol][gi % ncol]
            for mi, m in enumerate(methods):
                rows = ordered(m, g)
                if not rows:
                    continue
                pos = list(range(1, len(rows) + 1))
                vals = [r[ykey] for r in rows]
                c = colors[mi % len(colors)]
                if kind == "cumul":
                    # forward passes on x, instances verified on y
                    cum = list(np.cumsum(vals))
                    ax.plot(cum, pos, lw=1.5, color=c, label=m)
                    for st, (mk, sz, al) in STATUS_STYLE.items():
                        sy = [k for k, r in zip(pos, rows) if r["status"] == st]
                        sx = [cum[k - 1] for k in sy]
                        if sx:
                            ax.scatter(sx, sy, color=c, marker=mk,
                                       s=sz, alpha=al, zorder=5)
                else:
                    ax.plot(pos, vals, marker="o", ms=3, lw=1.5, color=c, label=m)
                    # mark warm-skipped instances (cache alone settled it)
                    sx = [x for x, r in zip(pos, rows) if r["warm_skip"]]
                    sy = [r[ykey] for r in rows if r["warm_skip"]]
                    if sx:
                        ax.scatter(sx, sy, color=c, marker="x", s=40, zorder=5)
            ax.set_title(f"group {g}", fontsize=10)
            if kind == "cumul":
                ax.set_xlabel("cumulative forward passes", fontsize=8)
                ax.set_ylabel("instances verified", fontsize=8)
            else:
                ax.set_xlabel("processing position", fontsize=8)
                ax.set_ylabel(axlab, fontsize=8)
            ax.grid(True, alpha=0.25)
            if gi == 0:
                ax.legend(fontsize=8)
        for j in range(len(groups), nrow * ncol):
            axes[j // ncol][j % ncol].axis("off")
        fig.suptitle(title, fontsize=13, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.98))
        out = Path(out_dir) / fname
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"Plot saved to: {out}")


# ── cross-run set / soundness comparison ─────────────────────────────────────
def compare_status_sets(per_method_records):
    """Pairwise comparison of safe/risky/open classifications across methods,
    keyed by ``idx`` alone so the *same* instance can be matched across two
    runs over the same dataset (e.g. cachedfrontier vs frontier).

    ``idx`` is the original dataset index and is globally unique within a run
    even when instances are partitioned into groups -- ``group`` is not a
    safe join key across methods because grouped (cachedfrontier) runs assign
    real group ids while ungrouped (frontier) runs log every instance under
    the placeholder group "all". Group is carried along per-record for
    display only.

    This checks agreement, not just aggregate counts: two runs can report the
    same n_risky totals while disagreeing on *which* instances are risky, and
    two runs can each definitively prove opposite verdicts on one instance
    (proven-risky by one, proven-safe by the other) -- a hard contradiction
    since both are claims of a mathematical bound crossing the same
    threshold, and at most one of them can be correct. That's a soundness bug
    somewhere, unlike a risky/open or safe/open split, which just means one
    run searched more before giving up.

    Returns {(method_a, method_b): {...}} for every unordered pair of methods
    present in ``per_method_records`` (insertion order preserved).
    """
    methods = list(per_method_records)
    rec_by = {
        m: {r["idx"]: r for r in per_method_records[m]}
        for m in methods
    }

    pairwise = {}
    for i in range(len(methods)):
        for j in range(i + 1, len(methods)):
            a, b = methods[i], methods[j]
            ra, rb = rec_by[a], rec_by[b]
            keys_a, keys_b = set(ra), set(rb)
            common = keys_a & keys_b

            contingency = defaultdict(int)
            hard_conflicts = []  # risky vs safe on the same instance -- unsound
            ub_diffs = []
            lb_diffs = []
            for k in sorted(common):
                sa, sb = ra[k]["status"], rb[k]["status"]
                contingency[(sa, sb)] += 1
                if {sa, sb} == {"safe", "risky"}:
                    hard_conflicts.append(k)
                ub_diffs.append(abs(ra[k]["ub"] - rb[k]["ub"]))
                lb_diffs.append(abs(ra[k]["lb"] - rb[k]["lb"]))

            per_status = {}
            for status in ("risky", "safe", "open"):
                sa_keys = {k for k in keys_a if ra[k]["status"] == status}
                sb_keys = {k for k in keys_b if rb[k]["status"] == status}
                inter = sa_keys & sb_keys
                union = sa_keys | sb_keys
                per_status[status] = {
                    "n_a": len(sa_keys),
                    "n_b": len(sb_keys),
                    "n_common": len(inter),
                    "a_only": sorted(sa_keys - sb_keys),
                    "b_only": sorted(sb_keys - sa_keys),
                    "jaccard": (len(inter) / len(union)) if union else float("nan"),
                }

            pairwise[(a, b)] = {
                "n_a": len(keys_a),
                "n_b": len(keys_b),
                "n_common_instances": len(common),
                "only_a": sorted(keys_a - keys_b),
                "only_b": sorted(keys_b - keys_a),
                "contingency": dict(contingency),
                "hard_conflicts": hard_conflicts,
                "per_status": per_status,
                "mean_abs_ub_diff": float(np.mean(ub_diffs)) if ub_diffs else float("nan"),
                "max_abs_ub_diff": float(np.max(ub_diffs)) if ub_diffs else float("nan"),
                "mean_abs_lb_diff": float(np.mean(lb_diffs)) if lb_diffs else float("nan"),
                "max_abs_lb_diff": float(np.max(lb_diffs)) if lb_diffs else float("nan"),
            }
    return pairwise


def print_status_comparison(pairwise):
    statuses = ("safe", "risky", "open")
    for (a, b), res in pairwise.items():
        print(f"\n{'=' * 80}\nSet comparison: {a}  vs  {b}\n{'=' * 80}")
        print(f"instances: {a}={res['n_a']}  {b}={res['n_b']}  "
              f"common={res['n_common_instances']}")
        if res["only_a"]:
            extra = " ..." if len(res["only_a"]) > 10 else ""
            print(f"  only in {a} ({len(res['only_a'])}): {res['only_a'][:10]}{extra}")
        if res["only_b"]:
            extra = " ..." if len(res["only_b"]) > 10 else ""
            print(f"  only in {b} ({len(res['only_b'])}): {res['only_b'][:10]}{extra}")

        print(f"\n  status x status contingency (rows={a}, cols={b}), "
              f"over {res['n_common_instances']} common instances:")
        header = "          " + "  ".join(f"{s:>8}" for s in statuses)
        print(header)
        for sa in statuses:
            row = "  ".join(
                f"{res['contingency'].get((sa, sb), 0):>8}" for sb in statuses
            )
            print(f"  {sa:<8}{row}")

        print("\n  per-status set comparison:")
        for status, d in res["per_status"].items():
            print(
                f"    {status:<6} {a}={d['n_a']:<5} {b}={d['n_b']:<5} "
                f"common={d['n_common']:<5} jaccard={d['jaccard']:.3f}  "
                f"({a}-only={len(d['a_only'])}, {b}-only={len(d['b_only'])})"
            )

        n_conf = len(res["hard_conflicts"])
        if n_conf:
            print(
                f"\n  !! SOUNDNESS WARNING: {n_conf} instance(s) proven 'risky' by "
                f"one method and 'safe' by the other -- a formal contradiction "
                f"(both claim a proof against the same threshold). "
                f"First few (group, idx): {res['hard_conflicts'][:10]}"
            )
        else:
            print(
                f"\n  no risky/safe contradictions -- consistent on all "
                f"{res['n_common_instances']} common instances"
            )

        print(
            f"\n  final-bound agreement on common instances: "
            f"mean|Δub|={res['mean_abs_ub_diff']:.4f} max|Δub|={res['max_abs_ub_diff']:.4f}  "
            f"mean|Δlb|={res['mean_abs_lb_diff']:.4f} max|Δlb|={res['max_abs_lb_diff']:.4f}"
        )


def write_status_comparison_csvs(pairwise, per_method_records, out_dir):
    """Write set_comparison.csv (per-pair per-status summary) and
    hard_conflicts.csv (every risky-vs-safe contradiction, with both methods'
    bounds/termination reasons, for manual inspection)."""
    import csv

    rec_by = {
        m: {r["idx"]: r for r in per_method_records[m]}
        for m in per_method_records
    }

    set_path = Path(out_dir) / "status_set_comparison.csv"
    with open(set_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "method_a", "method_b", "status", "n_a", "n_b", "n_common",
            "jaccard", "n_a_only", "n_b_only",
        ])
        for (a, b), res in pairwise.items():
            for status, d in res["per_status"].items():
                w.writerow([
                    a, b, status, d["n_a"], d["n_b"], d["n_common"],
                    f"{d['jaccard']:.4f}", len(d["a_only"]), len(d["b_only"]),
                ])
    print(f"\nSet-comparison CSV saved to: {set_path}")

    conf_path = Path(out_dir) / "hard_conflicts.csv"
    with open(conf_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "method_a", "method_b", "group", "idx",
            "status_a", "status_b", "ub_a", "lb_a", "ub_b", "lb_b",
            "term_reason_a", "term_reason_b",
        ])
        for (a, b), res in pairwise.items():
            ra, rb = rec_by[a], rec_by[b]
            for idx in res["hard_conflicts"]:
                w.writerow([
                    a, b, ra[idx]["group"], idx,
                    ra[idx]["status"], rb[idx]["status"],
                    f"{ra[idx]['ub']:.4f}", f"{ra[idx]['lb']:.4f}",
                    f"{rb[idx]['ub']:.4f}", f"{rb[idx]['lb']:.4f}",
                    ra[idx]["term_reason"], rb[idx]["term_reason"],
                ])
    print(f"Hard-conflict CSV saved to: {conf_path}")


def print_term_reason_distribution(per_method_records):
    """Termination-reason breakdown per method, for risky-classified
    instances only. A method reaching 'risky' by a systematically different
    mechanism than another (e.g. mostly max_iterations/early-stop rather than
    a bound tightening past the threshold) is a sign its risky verdicts rest
    on weaker evidence, even when the raw risky counts/sets match."""
    print(f"\n{'=' * 80}\nTermination-reason breakdown (risky instances only)\n{'=' * 80}")
    for m, recs in per_method_records.items():
        risky = [r for r in recs if r["status"] == "risky"]
        counts = defaultdict(int)
        for r in risky:
            counts[r["term_reason"] or "(none)"] += 1
        print(f"\n  {m}  (n_risky={len(risky)})")
        for reason, c in sorted(counts.items(), key=lambda kv: -kv[1]):
            pct = f"{100 * c / len(risky):.1f}%" if risky else "n/a"
            print(f"    {reason:<40} {c:>6}  ({pct})")


def plot_cost_histogram(per_method_records, out_dir):
    """Distribution of per-instance total cost (forward passes), one
    overlaid histogram per method — shows the cache shifting the cost
    distribution toward cheaper instances, distinct from the aggregate
    warmup/verification bar chart in ``plot()``."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = list(per_method_records)
    colors = ["#C44E52", "#4C72B0", "#55A868", "#8172B2"]

    all_costs = [r["total_fp"] for recs in per_method_records.values() for r in recs]
    if not all_costs:
        return
    bins = np.linspace(0, max(all_costs) + 1, 31)

    fig, ax = plt.subplots(figsize=(9, 6))
    for mi, m in enumerate(methods):
        costs = [r["total_fp"] for r in per_method_records[m]]
        if not costs:
            continue
        c = colors[mi % len(colors)]
        ax.hist(costs, bins=bins, alpha=0.55, color=c, label=f"{m} (n={len(costs)})")
        ax.axvline(float(np.mean(costs)), color=c, linestyle="--", lw=1.5)

    ax.set_xlabel("Total forward passes per instance", fontsize=12)
    ax.set_ylabel("# instances", fontsize=12)
    ax.set_title("Cost distribution per instance — dashed line = mean",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = Path(out_dir) / "cost_histogram.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Plot saved to: {out}")


def plot_cost_vs_verified_order(per_method_records, out_dir):
    """Single plot: x = cumulative cost (total_fp) accumulated in each
    method's real completion order (not sorted cheapest-first, not faceted
    by group); y = two cumulative-count lines per method — instances
    verified (conclusive safe/risky verdict) and instances found risky.

    A method whose per-instance cost shrinks over the run (the cache
    warming up) visibly bends the line upward as it goes; a method with no
    such effect stays close to a straight, locally jagged trend.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = list(per_method_records)
    colors = ["#C44E52", "#4C72B0", "#55A868", "#8172B2"]

    fig, ax = plt.subplots(figsize=(11, 7))
    for mi, m in enumerate(methods):
        recs = global_completion_order(per_method_records[m])
        if not recs:
            continue
        cum_cost = list(np.cumsum([r["total_fp"] for r in recs]))
        verified_count = list(np.cumsum(
            [1 if r["status"] in ("safe", "risky") else 0 for r in recs]
        ))
        risky_count = list(np.cumsum(
            [1 if r["status"] == "risky" else 0 for r in recs]
        ))
        c = colors[mi % len(colors)]
        ax.plot(cum_cost, verified_count, color=c, lw=1.8, linestyle="-",
                label=f"{m} — verified")
        ax.plot(cum_cost, risky_count, color=c, lw=1.8, linestyle="--",
                label=f"{m} — risky")

    ax.set_xlabel("Cumulative cost (forward passes, true completion order)",
                  fontsize=12)
    ax.set_ylabel("Cumulative instances", fontsize=12)
    ax.set_title("Instances verified / risky-found vs cumulative cost "
                 "(true processing order)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = Path(out_dir) / "cost_vs_verified_order.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Plot saved to: {out}")


def plot_verified_vs_walltime(per_method_records, out_dir):
    """Single combined plot: x = elapsed wall-clock time since each method's
    own run started (t=0 at its first instance's start_time — so methods
    launched at different times/machines overlay fairly); y = cumulative
    instances verified (safe+risky) and cumulative instances found risky,
    one pair of lines per method.

    Requires ``completion_time`` (needs both ``start_time`` and
    ``instance_run_time`` logged), which predates older logs. Methods with
    no records carrying usable timing data are skipped with a warning.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = list(per_method_records)
    colors = ["#C44E52", "#4C72B0", "#55A868", "#8172B2"]

    fig, ax = plt.subplots(figsize=(11, 7))
    plotted_any = False
    for mi, m in enumerate(methods):
        recs = [r for r in per_method_records[m] if r["completion_time"] is not None]
        if not recs:
            print(f"[plot_verified_vs_walltime] skipping '{m}': no instances with "
                  f"usable timing data (missing start_time/instance_run_time)")
            continue
        recs = sorted(recs, key=lambda r: r["completion_time"])
        t0 = min(r["start_time"] for r in recs)
        elapsed = [r["completion_time"] - t0 for r in recs]
        verified_count = list(np.cumsum(
            [1 if r["status"] in ("safe", "risky") else 0 for r in recs]
        ))
        risky_count = list(np.cumsum(
            [1 if r["status"] == "risky" else 0 for r in recs]
        ))
        c = colors[mi % len(colors)]
        ax.plot(elapsed, verified_count, color=c, lw=1.8, linestyle="-",
                label=f"{m} — verified")
        ax.plot(elapsed, risky_count, color=c, lw=1.8, linestyle="--",
                label=f"{m} — risky")
        plotted_any = True

    if not plotted_any:
        plt.close(fig)
        print("[plot_verified_vs_walltime] no method had usable timing data — skipping plot")
        return

    ax.set_xlabel("Elapsed wall-clock time (s, since each method's own run start)",
                  fontsize=12)
    ax.set_ylabel("Cumulative instances", fontsize=12)
    ax.set_title("Instances verified / risky-found vs wall-clock time",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = Path(out_dir) / "verified_vs_walltime.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Plot saved to: {out}")


def plot_overall(per_method_records, out_dir):
    """Single combined plot across all groups: instances verified (y) vs
    cumulative forward passes (x), one line per method. Instances pooled and
    sorted cheapest-first, so a higher/lefter curve = more instances verified
    per unit compute. Risky instances overlaid as markers along each curve."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    methods = list(per_method_records)
    colors = ["#C44E52", "#4C72B0", "#55A868", "#8172B2"]
    # marker shape encodes status; method colour encodes the method.
    status_style = STATUS_STYLE

    fig, ax = plt.subplots(figsize=(11, 7))
    for mi, m in enumerate(methods):
        recs = sorted(per_method_records[m], key=lambda r: r["total_fp"])
        count = list(range(1, len(recs) + 1))
        cum = list(np.cumsum([r["total_fp"] for r in recs]))
        c = colors[mi % len(colors)]
        agg = aggregate(recs)
        ax.plot(cum, count, lw=2.0, color=c,
                label=f"{m}  ({agg['n_safe']} safe, {agg['n_risky']} risky, "
                      f"{agg['n_open']} open, {cum[-1]:.0f} FP)")
        for st, (mk, sz, al) in status_style.items():
            ys = [k for k, r in zip(count, recs) if r["status"] == st]
            xs = [cum[k - 1] for k in ys]
            if xs:
                ax.scatter(xs, ys, color=c, marker=mk, s=sz, alpha=al, zorder=5)

    ax.set_xlabel("Cumulative forward passes", fontsize=12)
    ax.set_ylabel("Instances verified (cheapest-first)", fontsize=12)
    ax.set_title("Instances verified vs cost — all groups pooled",
                 fontsize=13, fontweight="bold")
    method_leg = ax.legend(fontsize=10, loc="lower right")
    ax.add_artist(method_leg)
    status_handles = [
        Line2D([], [], marker=mk, color="gray", linestyle="", markersize=8, label=st)
        for st, (mk, _sz, _al) in status_style.items()
    ]
    ax.legend(handles=status_handles, fontsize=9, loc="upper left", title="status")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = Path(out_dir) / "cost_vs_verified_all.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Plot saved to: {out}")


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="+", help="label=path entries (or bare paths)")
    ap.add_argument("--threshold", type=float, default=0.9)
    ap.add_argument("--out", default="cost_out", help="output dir for plot + csv")
    ap.add_argument("--no-groups", action="store_true", help="skip per-group rows")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    specs = []
    for item in args.logs:
        if "=" in item:
            label, path = item.split("=", 1)
        else:
            label, path = Path(item).name, item
        specs.append((label, path))

    all_rows = []
    method_aggs = {}
    per_method_records = {}
    for label, path in specs:
        recs = list(instance_records(path, args.threshold))
        per_method_records[label] = recs
        method_aggs[label] = aggregate(recs)
        all_rows.extend(build_rows(label, recs, by_group=not args.no_groups))

    print(
        f"\nThreshold = {args.threshold}  "
        f"(safe = LB > thr,  risky = UB <= thr,  open = otherwise)\n"
    )
    print_table(all_rows)

    # CSV
    import csv
    csv_path = Path(args.out) / "cost_table.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        cols = [c[0] for c in TABLE_COLS]
        w.writerow(cols)
        for row in all_rows:
            w.writerow([_fmt(row[c]) for c in cols])
    print(f"\nTable saved to: {csv_path}")

    plot(method_aggs, args.out)
    plot_per_instance(method_aggs, args.out)

    # ── per-group processing-order analysis ──────────────────────────────────
    order = canonical_order(per_method_records)

    for label in per_method_records:
        print_group_trajectories(label, per_method_records[label], order)

    # Per-instance trajectory CSV (all methods, in recovered processing order)
    traj_path = Path(args.out) / "instance_trajectory.csv"
    with open(traj_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "method", "group", "pos", "idx", "warm_cache_total", "warm_viol_prob",
            "warm_fp", "verif_fp", "total_fp", "new_viol", "warm_skip",
            "lb", "ub", "status", "term_reason",
        ])
        for label, recs in per_method_records.items():
            for r in sorted(recs, key=lambda r: (str(r["group"]),
                                                 order.get(r["group"], {}).get(r["idx"], r["idx"]))):
                w.writerow([
                    label, r["group"], order.get(r["group"], {}).get(r["idx"], ""),
                    r["idx"], r["warm_cache_total"], f"{r['warm_viol_prob']:.4f}",
                    r["warmup_fp"], r["verif_fp"], r["total_fp"], r["new_viol"],
                    int(r["warm_skip"]),
                    f"{r['lb']:.4f}", f"{r['ub']:.4f}",
                    r["status"], r["term_reason"],
                ])
    print(f"Trajectory CSV saved to: {traj_path}")

    plot_group_order(per_method_records, order, args.out)
    plot_overall(per_method_records, args.out)
    plot_cost_histogram(per_method_records, args.out)
    plot_cost_vs_verified_order(per_method_records, args.out)
    plot_verified_vs_walltime(per_method_records, args.out)

    # ── cross-run set / soundness comparison (needs >= 2 methods) ───────────
    print_term_reason_distribution(per_method_records)
    if len(per_method_records) >= 2:
        pairwise = compare_status_sets(per_method_records)
        print_status_comparison(pairwise)
        write_status_comparison_csvs(pairwise, per_method_records, args.out)


if __name__ == "__main__":
    main()
