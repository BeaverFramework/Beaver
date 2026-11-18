#!/usr/bin/env python3
"""Side-by-side comparison of BEAVER runs: full summaries + timings.

Prints, for each pair (or set) of runs:
  1. the complete summary.json fields, aligned side by side
  2. the forward-pass / cost breakdown
  3. the per-call timing breakdown (ordinary FP vs token-trie refill prefill)

Usage:
    python -m beaver.utils.compare_runs \\
        frontier=logging/.../logs_A cached=logging/.../logs_B
    python -m beaver.utils.compare_runs --pairs-under logging/vllm-local-cached-timings/runs
"""
import argparse
import json
from pathlib import Path

# summary.json fields worth showing, in display order
SUMMARY_FIELDS = [
    ("num_instances", "instances", "{:.0f}"),
    ("num_instances_determined", "determined", "{:.0f}"),
    ("avg_transitions_to_completion", "avg transitions", "{:.2f}"),
    ("avg_transitions_to_threshold", "avg transitions->thr", "{:.2f}"),
    ("avg_semantic_checks_per_instance", "semantic checks/inst", "{:.2f}"),
    ("total_semantic_checks", "semantic checks total", "{:.0f}"),
    ("avg_ub", "avg UB", "{:.6f}"),
    ("avg_lb", "avg LB", "{:.6f}"),
    ("avg_ub_minus_lb", "avg UB-LB", "{:.6f}"),
    ("avg_CVP", "avg CVP (1-UB)", "{:.6f}"),
    ("worst_CVP", "worst-case CVP", "{:.6f}"),
    ("max_transitions", "max transitions", "{:.0f}"),
    ("min_transitions", "min transitions", "{:.0f}"),
    ("num_constraint_satisfied", "satisfied (UB>=thr)", "{:.0f}"),
    ("num_constraint_unsatisfied", "unsatisfied (UB<thr)", "{:.0f}"),
    ("num_safe", "safe  (LB>thr)", "{:.0f}"),
    ("num_risky", "risky (UB<=thr)", "{:.0f}"),
    ("num_open", "open", "{:.0f}"),
    ("n_instances_warm_cache_had_violations", "warm cache had viols", "{:.0f}"),
    ("n_instances_warm_skip", "early-skipped by warm", "{:.0f}"),
    ("avg_warm_violation_prob_all", "avg warm viol prob", "{:.6f}"),
    ("total_new_violations_found", "new violations found", "{:.0f}"),
]

TIMING_FIELDS = [
    ("total_time", "total_time / inst"),
    ("warm_start_time", "warm_start_time / inst"),
    ("model_generate", "model_generate / inst"),
    ("get_distribution", "get_distribution / inst"),
    ("semantic_check", "semantic_check / inst"),
    ("decode_and_complete_check", "decode+complete / inst"),
    ("frontier_add", "frontier_add / inst"),
]


def load_summary(d):
    p = Path(d) / "summary.json"
    return json.load(open(p)) if p.exists() else {}


def load_profile(d):
    p = Path(d) / "profiling_summary.json"
    if not p.exists():
        return {}
    return json.load(open(p)).get("file_avg_profiles", {})


def load_args(d):
    p = Path(d) / "run_args.json"
    return json.load(open(p)) if p.exists() else {}


def percall(prof):
    """(fp_ms_per_request, prefill_ms_per_request, prefill_ms_per_row, cached_tokens)."""
    out = {}
    for label in ("model_generate_next_token_logprobs", "model_score_sequences_per_token"):
        net = nreq = nrow = cached = None
        for k, v in prof.items():
            if not k.endswith(f"{label}.net_s"):
                continue
            base = k[: -len("net_s")]
            net = v
            nreq = prof.get(base + "n_requests")
            nrow = prof.get(base + "n_rows")
            cached = prof.get(base + "num_cached_tokens")
            break
        out[label] = {
            "per_request_ms": 1000 * net / nreq if net and nreq else None,
            "per_row_ms": 1000 * net / nrow if net and nrow else None,
            "n_requests": nreq, "n_rows": nrow, "cached_tokens": cached,
        }
    return out


def show(names, dirs):
    sums = [load_summary(d) for d in dirs]
    profs = [load_profile(d) for d in dirs]
    argss = [load_args(d) for d in dirs]

    w = 26
    colw = 20
    line = "=" * (w + colw * len(names) + 4)

    print(line)
    print("RUN CONFIGURATION")
    print(line)
    print(f"{'':<{w}}" + "".join(f"{n:>{colw}}" for n in names))
    for key, lab in [("model", "model"), ("verifier", "verifier"),
                     ("backend", "backend"), ("gen_length", "gen_length"),
                     ("max_workers", "max_workers"), ("num_logprobs", "num_logprobs")]:
        vals = [str(a.get(key, "-")) for a in argss]
        vals = [v.split("/")[-1][:colw - 1] for v in vals]
        print(f"{lab:<{w}}" + "".join(f"{v:>{colw}}" for v in vals))
    print()

    print(line)
    print("COMPLETE SUMMARY (summary.json)")
    print(line)
    print(f"{'':<{w}}" + "".join(f"{n:>{colw}}" for n in names))
    for key, lab, fmt in SUMMARY_FIELDS:
        if not any(key in s for s in sums):
            continue
        cells = []
        for s in sums:
            v = s.get(key)
            cells.append(fmt.format(v) if isinstance(v, (int, float)) else "-")
        print(f"{lab:<{w}}" + "".join(f"{c:>{colw}}" for c in cells))
    print()

    print(line)
    print("AVG TIMING PROFILE (seconds per instance)")
    print(line)
    print(f"{'':<{w}}" + "".join(f"{n:>{colw}}" for n in names))
    for key, lab in TIMING_FIELDS:
        if not any(key in p for p in profs):
            continue
        cells = []
        for p in profs:
            v = p.get(key)
            cells.append(f"{v:.4f}" if isinstance(v, (int, float)) else "-")
        print(f"{lab:<{w}}" + "".join(f"{c:>{colw}}" for c in cells))
    print()

    print(line)
    print("PER-CALL TIMING: ordinary FP vs token-trie refill (prefill)")
    print(line)
    print(f"{'':<{w}}" + "".join(f"{n:>{colw}}" for n in names))
    pcs = [percall(p) for p in profs]
    rows = [
        ("FP ms/call", "model_generate_next_token_logprobs", "per_request_ms", "{:.1f}"),
        ("FP #calls", "model_generate_next_token_logprobs", "n_requests", "{:.0f}"),
        ("prefill ms/call", "model_score_sequences_per_token", "per_request_ms", "{:.1f}"),
        ("prefill ms/row", "model_score_sequences_per_token", "per_row_ms", "{:.1f}"),
        ("prefill #rows", "model_score_sequences_per_token", "n_rows", "{:.0f}"),
        ("prefix-cached tokens", "model_score_sequences_per_token", "cached_tokens", "{:.0f}"),
    ]
    for lab, label, field, fmt in rows:
        cells = []
        for pc in pcs:
            v = pc.get(label, {}).get(field)
            cells.append(fmt.format(v) if isinstance(v, (int, float)) else "-")
        print(f"{lab:<{w}}" + "".join(f"{c:>{colw}}" for c in cells))
    # ratio line
    cells = []
    for pc in pcs:
        fp = pc["model_generate_next_token_logprobs"]["per_request_ms"]
        pr = pc["model_score_sequences_per_token"]["per_row_ms"]
        cells.append(f"{pr / fp:.2f}x" if fp and pr else "-")
    print(f"{'prefill/row ÷ FP/call':<{w}}" + "".join(f"{c:>{colw}}" for c in cells))
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*", help="name=logdir")
    ap.add_argument("--pairs-under", default=None,
                    help="dir holding <exp>_frontier/ and <exp>_cached/ subdirs")
    args = ap.parse_args()

    if args.pairs_under:
        root = Path(args.pairs_under)
        stems = sorted({p.name.replace("_frontier", "").replace("_cached", "")
                        for p in root.iterdir() if p.is_dir()})
        for stem in stems:
            pair = []
            for suffix, nm in (("_frontier", "frontier"), ("_cached", "cachedfrontier")):
                d = root / f"{stem}{suffix}"
                logs = sorted(d.glob("logs_*")) if d.exists() else []
                if logs:
                    pair.append((nm, logs[-1]))
            if len(pair) == 2:
                print("\n" + "#" * 78)
                print(f"### {stem}")
                print("#" * 78)
                show([p[0] for p in pair], [p[1] for p in pair])
        return

    names, dirs = [], []
    for spec in args.runs:
        n, _, d = spec.partition("=")
        names.append(n or Path(d).name)
        dirs.append(d)
    show(names, dirs)


if __name__ == "__main__":
    main()
