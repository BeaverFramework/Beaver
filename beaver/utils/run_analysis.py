"""Run analysis for BEAVER frontier-type run logs (frontier, cachedfrontier).

Reads the current per-instance logs (<idx>.jsonl: instance_start, optional
warm_start, transition..., instance_complete) and reports, per run:

  verdicts      safe / risky / open at tau, termination reason cross-tab
  bounds        final bounds on P(safe) as logged: LB = certified-safe
                (complete) mass, UB = 1 - certified-violation mass. Bin counts
                and histograms of LB, UB and the width UB - LB, and the shape
                (exact 0 / exact 1 / tight interior / wide)
  responses     number of violating and satisfying (complete, safe) leaves found
                per instance and their mass: mean / std / median / p90 / max
  depth         token depth at which violations and satisfactions were found
                (all of them, and the first one per instance) and the forward
                passes spent before the first one
  cost          forward passes, wall time, trie size per instance
  horizon       cachedfrontier only: by position in group, the accumulated
                trie the instance inherits (cache_total / violations /
                completions / incomplete), the mass it already decides at warm
                start, warm-start and search forward passes, and evictions

Several runs (--batch, or several paths) also get cross-run tables: one row per
run, with the bound bin counts side by side.

    python beaver/utils/run_analysis.py logging/runs/dist300_qwen3_4b_enron/vllm_frontier
    python beaver/utils/run_analysis.py logging/runs/agentdojo_d12x48_n300_seed0_hf --batch --full
    python beaver/utils/run_analysis.py 'logging/runs/dist300_*' --batch --csv out.csv

A path may be a logs_* folder, an arm folder holding logs_* (the newest is
used), or any folder above them (with --batch, every arm below it). Glob
patterns are expanded. Sampling-verifier runs are skipped (no frontier).
Imports nothing from beaver, so it runs without the model stack installed.
"""

import argparse
import csv
import glob
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

BOUND_BINS = [0.0, 1e-3, 1e-2, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 0.999, 1.0 + 1e-9]
BOUND_LABELS = ["[0,.001)", "[.001,.01)", "[.01,.1)", "[.1,.25)", "[.25,.5)", "[.5,.75)",
                "[.75,.9)", "[.9,.99)", "[.99,.999)", "[.999,1]"]
DEPTH_BINS = [1, 2, 6, 11, 21, 41, 81, 161, 10**9]
DEPTH_LABELS = ["1", "2-5", "6-10", "11-20", "21-40", "41-80", "81-160", ">160"]

_TERMINATION_LABELS = {
    "max_incomplete_below_epsilon": "epsilon_tolerance",
    "incomplete_below_epsilon": "epsilon_tolerance",  # legacy logs
    "early_termination_threshold": "epsilon_tolerance",  # legacy logs
    "max_iterations": "max_steps",
    "frontier_empty": "frontier_empty",
}
_NP_SCALAR = re.compile(r"np\.\w+\(([^()]*)\)")


# ── reading ─────────────────────────────────────────────────────────────────


def read_events(path: Path):
    """All JSON objects in a log file (one per line or pretty-printed)."""
    dec, s, i, out = json.JSONDecoder(), path.read_text(), 0, []
    while i < len(s):
        while i < len(s) and s[i].isspace():
            i += 1
        if i >= len(s):
            break
        try:
            obj, i = dec.raw_decode(s, i)
        except json.JSONDecodeError:
            break  # truncated tail of a killed worker
        out.append(obj)
    return out


def _num(v, default=0.0):
    if v is None:
        return default
    if isinstance(v, str):  # log_json stringifies numpy scalars as np.float64(x)
        v = _NP_SCALAR.sub(r"\1", v)
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def find_log_dirs(path: Path):
    """Run log folders under path: the newest logs_* per arm folder."""
    # a logs folder itself (iterative runs keep instances in base_<g>/ below it)
    if path.name.startswith("logs_") or any(f.stem.isdigit() for f in path.glob("*.jsonl")):
        return [path]
    out = []
    arms = {p.parent for p in path.rglob("logs_*") if p.is_dir()}
    for arm in sorted(arms):
        logs = sorted(p for p in arm.glob("logs_*") if p.is_dir())
        if logs:
            out.append(logs[-1])
    return out


def run_label(log_dir: Path):
    arm = log_dir.parent
    return f"{arm.parent.name}/{arm.name}"


# ── per instance ────────────────────────────────────────────────────────────


def parse_instance(path: Path):
    ev = read_events(path)
    ic = next((e for e in ev if e.get("event") == "instance_complete"), None)
    if ic is None:
        return None
    ws = next((e for e in ev if e.get("event") == "warm_start"), None)
    trans = [e for e in ev if e.get("event") == "transition" or ("transition" in e and "event" not in e)]

    inc = _num(ic.get("final_incomplete_prob"))
    comp = _num(ic.get("final_complete_prob"))
    viol = _num(ic.get("final_violation_prob"))
    pru = _num(ic.get("final_pruned_prob"))
    r = {
        "instance": path.stem,
        # clipped: the summed masses overshoot 1 by float error (1.0000000x)
        "lb": min(max(comp, 0.0), 1.0),
        "ub": min(max(inc + comp + pru, 0.0), 1.0),
        "viol_mass": viol,
        "sat_mass": comp,
        "unresolved_mass": inc + pru,
        "reason": _TERMINATION_LABELS.get(ic.get("termination_reason"), ic.get("termination_reason") or "unknown"),
        "wall_s": _num(ic.get("instance_run_time")),
        "group": ic.get("group"),
        "group_position": ic.get("group_position", ws.get("group_position") if ws else None),
    }
    r["width"] = r["ub"] - r["lb"]

    fp = 0
    viol_depths, viol_probs, sat_depths, sat_probs = [], [], [], []
    first_viol = first_sat = None  # (depth, fp before it)
    prev_complete = _num(ws.get("cache_completions")) if ws else 0.0
    prev_complete_mass = _num(ws.get("warm_completion_prob")) if ws else 0.0
    exp_depths = []
    for t in trans:
        if str(t.get("skipped", "")):  # sampling-verifier zero-mass records
            continue
        els = t.get("expanded elements") or ([t["expanded element"]] if "expanded element" in t else [])
        depths = [len(el.get("tokens", el) if isinstance(el, dict) else el) + 1 for el in els] or [0]
        exp_depths.extend(d - 1 for d in depths)
        # violations: per expanded element (child depth = prefix length + 1)
        texts = t.get("violation_texts") or []
        k = 0
        for el, d in zip(els, depths):
            nv = int(_num(el.get("num_violations"))) if isinstance(el, dict) else 0
            for _ in range(nv):
                viol_depths.append(d)
                viol_probs.append(_num(texts[k].get("seq_prob")) if k < len(texts) and isinstance(texts[k], dict) else np.nan)
                k += 1
            if nv and first_viol is None:
                first_viol = (d, fp)
        if not any(isinstance(el, dict) and "num_violations" in el for el in els):
            nv = int(_num(t.get("num_violations")))  # old logs: per transition only
            viol_depths += [depths[0]] * nv
            viol_probs += [np.nan] * nv
            if nv and first_viol is None:
                first_viol = (depths[0], fp)
        # satisfactions: growth of the complete set / mass at this transition
        cs, cm = _num(t.get("complete_size"), prev_complete), _num(t.get("complete prob sum"), prev_complete_mass)
        new = int(round(cs - prev_complete))
        if new > 0:
            d = int(round(np.mean(depths)))
            sat_depths += [d] * new
            sat_probs += [(cm - prev_complete_mass) / new] * new
            if first_sat is None:
                first_sat = (d, fp)
        prev_complete, prev_complete_mass = max(prev_complete, cs), max(prev_complete_mass, cm)
        fp += int(_num(t.get("batch_size"), 1)) or 1

    warm_fp = int(_num(ws.get("warm_start_forward_passes"))) if ws else 0
    r.update(
        search_fp=fp,
        warm_fp=warm_fp,
        fp=fp + warm_fp,
        n_viol_new=len(viol_depths),
        n_sat_new=len(sat_depths),
        n_viol=len(viol_depths) + (int(_num(ws.get("cache_violations"))) if ws else 0),
        n_sat=int(round(prev_complete)),
        viol_depths=viol_depths,
        viol_probs=viol_probs,
        sat_depths=sat_depths,
        sat_probs=sat_probs,
        first_viol_depth=first_viol[0] if first_viol else None,
        first_viol_fp=first_viol[1] if first_viol else None,
        first_sat_depth=first_sat[0] if first_sat else None,
        first_sat_fp=first_sat[1] if first_sat else None,
        max_depth=max(exp_depths) if exp_depths else 0,
        mean_depth=float(np.mean(exp_depths)) if exp_depths else 0.0,
        trie_size=int(_num(ic.get("total_elements"), 1 + sum(_num(t.get("num_new_elements")) for t in trans))),
    )
    if ws:
        r.update(
            cache_total=int(_num(ws.get("cache_total"))),
            cache_violations=int(_num(ws.get("cache_violations"))),
            cache_completions=int(_num(ws.get("cache_completions"))),
            cache_incomplete=int(_num(ws.get("cache_incomplete"))),
            warm_viol_mass=_num(ws.get("warm_violation_prob")),
            warm_sat_mass=_num(ws.get("warm_completion_prob")),
            warm_incomplete_mass=_num(ws.get("warm_incomplete_prob")),
            warm_skip=bool(ic.get("warm_skip", False)),
            cache_evicted=int(_num(ic.get("cache_evicted"))),
        )
    return r


def load_run(log_dir: Path):
    args_f = log_dir / "run_args.json"
    run_args = json.loads(args_f.read_text()) if args_f.exists() else {}
    files = [f for f in log_dir.rglob("*.jsonl") if f.stem.isdigit()]
    rows, missing = [], 0
    for f in sorted(files, key=lambda p: (str(p.parent), int(p.stem))):
        r = parse_instance(f)
        if r is None:
            missing += 1
        else:
            rows.append(r)
    return run_args, rows, missing


# ── formatting helpers ──────────────────────────────────────────────────────


def stats(vals):
    a = np.array([v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))], dtype=float)
    if a.size == 0:
        return "n=0"
    return (f"n={a.size} mean={a.mean():.4g} std={a.std():.4g} median={np.median(a):.4g} "
            f"p90={np.percentile(a, 90):.4g} max={a.max():.4g}")


def mean_std(vals, fmt="{:.1f}"):
    a = np.array([v for v in vals if v is not None], dtype=float)
    if a.size == 0:
        return "-"
    return f"{fmt.format(a.mean())} ± {fmt.format(a.std())}"


def bin_counts(vals, bins):
    return np.histogram(np.asarray(vals, dtype=float), bins=bins)[0]


def print_bin_table(columns, labels, bins, bar_col=None, width=40):
    """columns: {name: values}. One row per bin; a bar for bar_col."""
    counts = {k: bin_counts(v, bins) for k, v in columns.items()}
    names = list(columns)
    print("| bin | " + " | ".join(names) + (" | " + bar_col + " histogram" if bar_col else "") + " |")
    print("|---" * (len(names) + 1 + bool(bar_col)) + "|")
    peak = max(max(c.max() for c in counts.values()), 1)
    for i, lab in enumerate(labels):
        row = f"| {lab} | " + " | ".join(str(int(counts[k][i])) for k in names)
        if bar_col:
            row += " | `" + "█" * int(round(counts[bar_col][i] / peak * width)) + "`"
        print(row + " |")


def shape(rows):
    lb, ub = np.array([r["lb"] for r in rows]), np.array([r["ub"] for r in rows])
    w = ub - lb
    safe1 = lb >= 0.99           # P(safe) ~ 1
    viol1 = ub <= 0.01           # P(safe) ~ 0 (always violates)
    tight = (w <= 0.01) & ~safe1 & ~viol1
    return {
        "exact_safe": int(safe1.sum()), "exact_violating": int(viol1.sum()), "tight_interior": int(tight.sum()),
        "w_.01-.1": int(((w > 0.01) & (w <= 0.1)).sum()), "w_.1-.5": int(((w > 0.1) & (w <= 0.5)).sum()),
        "w_>.5": int((w > 0.5).sum()), "median_width": float(np.median(w)), "mean_width": float(w.mean()),
    }


def verdicts(rows, tau):
    s = sum(r["lb"] > tau for r in rows)
    k = sum(r["ub"] <= tau for r in rows)
    return s, k, len(rows) - s - k


# ── one run ─────────────────────────────────────────────────────────────────


def report(label, run_args, rows, missing, tau):
    a = run_args
    print(f"\n## {label}")
    print(f"model={a.get('model')}  verifier={a.get('verifier')}  backend={a.get('backend')}  "
          f"eps={a.get('epsilon')}  max_iter={a.get('max_iterations')}  gen_length={a.get('gen_length')}  "
          f"T={a.get('temperature')}  top_p={a.get('top_p')}  top_k={a.get('top_k')}  "
          f"prefix_closed={a.get('prefix_closed')}  strategy={a.get('frontier_scoring_strategy')}")
    print(f"instances with a final verdict: {len(rows)}  (no instance_complete: {missing})")
    if not rows:
        return

    s, k, o = verdicts(rows, tau)
    print(f"\n### Verdicts (tau={tau}: safe if LB > tau, risky if UB <= tau)")
    print(f"safe {s} / risky {k} / open {o}")
    reasons = sorted({r["reason"] for r in rows})
    print("\n| termination | n | safe | risky | open | mean width |\n|---|---|---|---|---|---|")
    for re_ in reasons:
        sub = [r for r in rows if r["reason"] == re_]
        ss, kk, oo = verdicts(sub, tau)
        print(f"| {re_} | {len(sub)} | {ss} | {kk} | {oo} | {np.mean([r['width'] for r in sub]):.3f} |")

    print("\n### Final bounds on P(safe)  (violation mass = 1 - UB, unresolved = UB - LB)")
    sh = shape(rows)
    print(f"exact safe (LB>=.99) {sh['exact_safe']}, exact violating (UB<=.01) {sh['exact_violating']}, "
          f"tight interior (width<=.01) {sh['tight_interior']}, width .01-.1 {sh['w_.01-.1']}, "
          f"width .1-.5 {sh['w_.1-.5']}, width >.5 {sh['w_>.5']}; "
          f"median width {sh['median_width']:.3f}, mean {sh['mean_width']:.3f}")
    print(f"LB: {stats([r['lb'] for r in rows])}")
    print(f"UB: {stats([r['ub'] for r in rows])}\n")
    print_bin_table({"LB": [r["lb"] for r in rows], "UB": [r["ub"] for r in rows],
                     "width": [r["width"] for r in rows]}, BOUND_LABELS, BOUND_BINS, bar_col="UB")

    print("\n### Responses found per instance (leaves; cached runs include inherited ones)")
    print(f"violating leaves:  {stats([r['n_viol'] for r in rows])}  "
          f"(instances with >=1: {sum(r['n_viol'] > 0 for r in rows)})")
    print(f"satisfying leaves: {stats([r['n_sat'] for r in rows])}  "
          f"(instances with >=1: {sum(r['n_sat'] > 0 for r in rows)})")
    print(f"violation mass:    {stats([r['viol_mass'] for r in rows])}")
    print(f"satisfying mass:   {stats([r['sat_mass'] for r in rows])}")
    print(f"unresolved mass:   {stats([r['unresolved_mass'] for r in rows])}")

    vd = [d for r in rows for d in r["viol_depths"]]
    sd = [d for r in rows for d in r["sat_depths"]]
    print("\n### Depth (tokens) at which leaves were found by the search")
    print(f"all violations:     {stats(vd)}")
    print(f"all satisfactions:  {stats(sd)}")
    print(f"first violation:    {stats([r['first_viol_depth'] for r in rows])}  "
          f"fp before it: {stats([r['first_viol_fp'] for r in rows])}")
    print(f"first satisfaction: {stats([r['first_sat_depth'] for r in rows])}  "
          f"fp before it: {stats([r['first_sat_fp'] for r in rows])}")
    print(f"deepest expansion:  {stats([r['max_depth'] for r in rows])}\n")
    print_bin_table({"violations": vd, "satisfactions": sd,
                     "first viol": [r["first_viol_depth"] for r in rows if r["first_viol_depth"]],
                     "first sat": [r["first_sat_depth"] for r in rows if r["first_sat_depth"]]},
                    DEPTH_LABELS, DEPTH_BINS)
    vp = [p for r in rows for p in r["viol_probs"] if not np.isnan(p)]
    if vp:
        print(f"\nviolating-leaf probability: {stats(vp)}")

    print("\n### Cost per instance")
    print(f"forward passes: {stats([r['fp'] for r in rows])}  total {sum(r['fp'] for r in rows)}")
    print(f"wall s:         {stats([r['wall_s'] for r in rows])}")
    print(f"trie size:      {stats([r['trie_size'] for r in rows])}")

    if any("cache_total" in r for r in rows):
        horizon_report(rows)


def horizon_report(rows):
    """Accumulated horizon of a cachedfrontier run, by position in group."""
    groups = Counter(r["group"] for r in rows)
    sizes = Counter(groups.values())
    print("\n### Accumulated horizon (cachedfrontier)")
    print(f"{len(groups)} groups; size histogram " + ", ".join(f"{k}:{v}" for k, v in sorted(sizes.items())))
    print(f"warm_skip {sum(r.get('warm_skip', False) for r in rows)}, "
          f"instances with evictions {sum(r.get('cache_evicted', 0) > 0 for r in rows)}")
    by = defaultdict(list)
    for r in rows:
        if r.get("group_position") is not None:
            p = int(r["group_position"])
            by[p if p < 5 else (5 if p < 10 else 10)].append(r)
    print("\n| position | n | inherited trie nodes | inherited violations | inherited satisfactions | "
          "inherited incomplete | mass decided at warm start (viol + sat) | share of final viol mass already inherited | "
          "warm-start fp | search fp | total fp | new violations | new satisfactions |")
    print("|---" * 13 + "|")
    names = {5: "5-9", 10: "10+"}
    for p in sorted(by):
        sub = by[p]
        share = [r["warm_viol_mass"] / r["viol_mass"] for r in sub if r["viol_mass"] > 1e-12]
        print(f"| {names.get(p, p)} | {len(sub)} | {mean_std([r['cache_total'] for r in sub])} | "
              f"{mean_std([r['cache_violations'] for r in sub])} | {mean_std([r['cache_completions'] for r in sub])} | "
              f"{mean_std([r['cache_incomplete'] for r in sub])} | "
              f"{mean_std([r['warm_viol_mass'] + r['warm_sat_mass'] for r in sub], '{:.3f}')} | "
              f"{mean_std(share, '{:.2f}') if share else '-'} | {mean_std([r['warm_fp'] for r in sub])} | "
              f"{mean_std([r['search_fp'] for r in sub])} | {mean_std([r['fp'] for r in sub])} | "
              f"{mean_std([r['n_viol_new'] for r in sub])} | {mean_std([r['n_sat_new'] for r in sub])} |")


# ── several runs ────────────────────────────────────────────────────────────


def cross_run_tables(runs, tau):
    print("\n## Cross-run summary")
    print(f"tau={tau}; bounds on P(safe); ± is std over instances; depth in tokens; "
          "first viol/sat = first leaf found by the search\n")
    print("| run | model | n | safe/risky/open | exact safe | exact viol | tight interior | width .01-.1 | "
          "width .1-.5 | width >.5 | median width | #violating | #satisfying | depth 1st viol | "
          "depth 1st sat | fp |")
    print("|---" * 16 + "|")
    for label, a, rows in runs:
        sh, (s, k, o) = shape(rows), verdicts(rows, tau)
        print(f"| {label} | {str(a.get('model', '')).split('/')[-1]} | {len(rows)} | {s}/{k}/{o} | "
              f"{sh['exact_safe']} | {sh['exact_violating']} | {sh['tight_interior']} | {sh['w_.01-.1']} | "
              f"{sh['w_.1-.5']} | {sh['w_>.5']} | {sh['median_width']:.3f} | "
              f"{mean_std([r['n_viol'] for r in rows])} | {mean_std([r['n_sat'] for r in rows])} | "
              f"{mean_std([r['first_viol_depth'] for r in rows])} | {mean_std([r['first_sat_depth'] for r in rows])} | "
              f"{mean_std([r['fp'] for r in rows], '{:.0f}')} |")
    for name, key in (("LB", "lb"), ("UB", "ub")):
        print(f"\n### {name} bin counts\n")
        print("| run | " + " | ".join(BOUND_LABELS) + " |")
        print("|---" * (len(BOUND_LABELS) + 1) + "|")
        for label, _, rows in runs:
            c = bin_counts([r[key] for r in rows], BOUND_BINS)
            print(f"| {label} | " + " | ".join(str(int(x)) for x in c) + " |")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help="run folders (logs_*, arm, or above with --batch); globs ok")
    ap.add_argument("--batch", action="store_true", help="every arm below each path, plus cross-run tables")
    ap.add_argument("--full", action="store_true", help="with --batch: also the full per-run report")
    ap.add_argument("--threshold", type=float, default=None,
                    help="verdict tau (default: the run's early_termination_threshold, else 0.9)")
    ap.add_argument("--csv", default=None, help="write per-instance rows (all runs) to this CSV")
    args = ap.parse_args()

    paths = [Path(p) for pat in args.paths for p in (sorted(glob.glob(pat)) or [pat])]
    log_dirs = []
    for p in paths:
        assert p.is_dir(), f"not a directory: {p}"
        found = find_log_dirs(p)
        log_dirs += found if args.batch else found[-1:]
    runs = []
    for d in log_dirs:
        run_args, rows, missing = load_run(d)
        if run_args.get("verifier") == "sampling":
            print(f"(skipping sampling-verifier run {run_label(d)})")
            continue
        tau = args.threshold if args.threshold is not None else (run_args.get("early_termination_threshold") or 0.9)
        if not args.batch or args.full:
            report(run_label(d), run_args, rows, missing, tau)
        if rows:
            runs.append((run_label(d), run_args, rows))
    if len(runs) > 1 or args.batch:
        tau = args.threshold if args.threshold is not None else 0.9
        cross_run_tables(runs, tau)
    if args.csv:
        keys = [k for k in runs[0][2][0] if not isinstance(runs[0][2][0][k], list)]
        keys += [k for r in runs for row in r[2] for k in row if k not in keys and not isinstance(row[k], list)]
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["run"] + keys, extrasaction="ignore")
            w.writeheader()
            for label, _, rows in runs:
                for row in rows:
                    w.writerow({"run": label, **row})
        print(f"\nper-instance rows -> {args.csv}")


if __name__ == "__main__":
    main()
