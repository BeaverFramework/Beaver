"""Progress report of an IterativeVerifier run (beam search over system prompts),
over all its groups (base prompts).

Standalone: it imports nothing from beaver (``python -m beaver.utils...``
would import torch and vLLM through beaver/__init__.py), so run it as a file:

    python3 beaver/utils/analyze_iterative_run.py <run_dir>
    python3 beaver/utils/analyze_iterative_run.py <run_dir> --detail --prompts
    python3 beaver/utils/analyze_iterative_run.py <run_dir> --plot progress.png --json summary.json

<run_dir> is any directory that contains group_<g>.jsonl and <idx>.jsonl
files (searched recursively). Works on a run in progress and on logs written
before the search fields (based_on, search_round, num_refused, empty_round)
existed.

Sections:
    overview   one row per group: status ("unfinished" while running or if
               stopped), rounds, instances, pruned, best
               seed bound -> best bound, mutator refusals, time, forward passes
    progress   best certified violation lower bound per group after each round
    detail     per group: one line per round and per mutator call, lineage
               gains, stop reasons of the instances, cache size
    prompts    the current beam of each group
    outputs    (--outputs) what the beam generates: the most probable
               violating / safe outputs of each beam element with their
               probabilities, the width of each output distribution, and
               whether the beam elements generate the same outputs. Runs
               logged before the instance_outputs event existed fall back to
               the top-3 outputs in the last mutator request.
"""

import ast

import argparse
import glob
import json
import os
import re
import statistics as st
from collections import Counter, defaultdict

_DEC = json.JSONDecoder()


def records(path):
    """The records of a log file (pretty-printed JSON objects, one after
    another). A record that is still being written is skipped."""
    with open(path, errors="replace") as f:
        s = f.read()
    i, n = 0, len(s)
    while i < n:
        while i < n and s[i].isspace():
            i += 1
        if i >= n:
            return
        try:
            r, i = _DEC.raw_decode(s, i)
        except json.JSONDecodeError:
            return
        yield r


def _instance(path):
    """instance_start fields + instance_complete fields (+ the
    instance_outputs event as "outputs"), or None if the instance did not
    finish."""
    start, done, outputs = {}, None, None
    if not os.path.exists(path):  # listed in a round, file not synced yet
        return None
    for r in records(path):
        ev = r.get("event")
        if ev == "instance_start":
            start = r
        elif ev == "instance_complete":
            done = r
        elif ev == "instance_outputs":
            outputs = r
    if done is None:
        return None
    keep = ("system_prompt", "search_round", "parent_idx", "based_on", "beam_entry_threshold",
            "dataset_idx")
    rec = {**done, **{k: start.get(k) for k in keep if k in start}}
    if outputs is not None:
        rec["outputs"] = outputs
    return rec


def load_group(path):
    events = list(records(path))
    log_dir = os.path.dirname(path)
    calls = [r for r in events if r.get("event") == "mutator_call"]
    rounds = [r for r in events if r.get("event") == "round_complete"]
    start = next((r for r in events if r.get("event") == "search_start"), {})
    done = next((r for r in events if r.get("event") == "search_complete"), None)
    # instance_outputs is in the instance files (older runs: in this file).
    outputs = {r["idx"]: r for r in events if r.get("event") == "instance_outputs"}

    # Lineage from the mutator calls (older logs have only parent_idx there).
    lineage = {}
    for c in calls:
        based = c.get("based_on") or [[p] if p is not None else [] for p in c["parent_idx"]]
        for i, b in zip(c["new_idx"], based):
            lineage[i] = (c["round"] + 1, b)

    inst = {}
    for r in rounds:
        for i in r["verified_idx"]:
            rec = _instance(os.path.join(log_dir, f"{i}.jsonl"))
            if rec is not None:
                rnd, based = lineage.get(i, (0, []))
                rec.setdefault("search_round", rnd)
                if not rec.get("based_on"):
                    rec["based_on"] = based
                inst[i] = rec
    return {"group": start.get("group", os.path.basename(path)[6:-6]), "start": start,
            "calls": calls, "rounds": rounds, "done": done, "inst": inst, "log_dir": log_dir,
            "outputs": {**outputs, **{i: r["outputs"] for i, r in inst.items() if "outputs" in r}}}


def _refused(call):
    if "num_refused" in call:
        return call["num_refused"]
    return sum("<prompt" not in (o.get("content") or "") for o in call["outputs"])


def _base_prompt(g):
    p = g["start"].get("prompt") or ""
    return p.split("\n")[-1].strip()


def summarize(g):
    inst, rounds, calls = g["inst"], g["rounds"], g["calls"]
    V = {i: r["final_violation_prob"] for i, r in inst.items()}
    seeds = [i for i, r in inst.items() if r["search_round"] == 0]
    beam = rounds[-1]["beam"] if rounds else []
    samples = sum(len(c["outputs"]) for c in calls)
    return {
        "group": g["group"],
        "base_prompt": _base_prompt(g),
        "status": g["done"]["termination_reason"] if g["done"] else "unfinished",
        "rounds": len(rounds) - 1 if rounds else 0,
        "seeds": len(seeds),
        "mutants": len(inst) - len(seeds),
        "pruned": sum(r["termination_reason"] == "beam_unreachable" for r in inst.values()),
        "best_seed": max((V[i] for i in seeds), default=float("nan")),
        "best": beam[0]["final_violation_prob"] if beam else float("nan"),
        "beam_10th": beam[-1]["final_violation_prob"] if beam else float("nan"),
        "beam_mean": st.mean(b["final_violation_prob"] for b in beam) if beam else float("nan"),
        "refused": f"{sum(_refused(c) for c in calls)}/{samples}",
        "verify_min": sum(r["round_time"] for r in rounds) / 60,
        "mutator_min": sum(c["mutator_time"] for c in calls) / 60,
        "fp": sum(r.get("total_transitions", 0) for r in inst.values()),
    }


def print_overview(groups):
    print("== OVERVIEW")
    print(f"{'grp':>3} {'status':>14} {'rnd':>3} {'seeds':>5} {'mut':>4} {'prun':>4} "
          f"{'seed best':>9} {'best':>7} {'10th':>7} {'refused':>8} {'verify':>7} {'mutator':>7} {'fp':>7}  base prompt")
    for g in groups:
        s = summarize(g)
        print(f"{s['group']:>3} {s['status']:>14} {s['rounds']:>3} {s['seeds']:>5} {s['mutants']:>4} {s['pruned']:>4} "
              f"{s['best_seed']:>9.4f} {s['best']:>7.4f} {s['beam_10th']:>7.4f} {s['refused']:>8} "
              f"{s['verify_min']:>6.1f}m {s['mutator_min']:>6.1f}m {s['fp']:>7}  {s['base_prompt'][:50]!r}")


def print_progress(groups):
    n = max((len(g["rounds"]) for g in groups), default=0)
    print("\n== PROGRESS: best certified violation LB after each round (* = empty round)")
    print(f"{'grp':>3} " + " ".join(f"{'r' + str(k):>7}" for k in range(n)))
    for g in groups:
        cells = []
        for r in g["rounds"]:
            empty = r.get("empty_round", r["round"] > 0 and not r["verified_idx"])
            cells.append(f"{r['beam'][0]['final_violation_prob']:.4f}{'*' if empty else ' '}" if r["beam"] else "      -")
        print(f"{g['group']:>3} " + " ".join(f"{c:>7}" for c in cells))


def print_detail(g):
    inst = g["inst"]
    V = {i: r["final_violation_prob"] for i, r in inst.items()}
    print(f"\n== GROUP {g['group']}: {_base_prompt(g)!r}")
    st0 = g["start"]
    print("  settings: " + ", ".join(f"{k}={st0[k]}" for k in (
        "beam_width", "max_rounds", "patience", "mutator_samples", "mutator_num_prompts",
        "mutator_max_calls", "mutator_system_prompt",
        "prune_unreachable", "cache_hit_prob", "cache_evict_after", "surrogate_model") if k in st0))
    calls = {c["round"]: c for c in g["calls"]}
    t = 0.0
    for r in g["rounds"]:
        t += r["round_time"]
        b = [x["final_violation_prob"] for x in r["beam"]]
        ids = r["verified_idx"]
        term = Counter(inst[i]["termination_reason"] for i in ids if i in inst)
        cache = r.get("cache_size")
        thr = r.get("beam_entry_threshold")
        print(f"  round {r['round']}: verified {len(ids):3d} in {r['round_time'] / 60:5.1f} min"
              + (f" (threshold {thr:.4f}, pruned {r.get('num_pruned', 0)})" if thr is not None else "")
              + (f" | beam {b[0]:.4f}..{b[-1]:.4f} mean {st.mean(b):.4f}" if b else "")
              + f" | unchanged {r['rounds_without_change']}"
              + (" | EMPTY" if r.get("empty_round") else "")
              + (f" | cache V{cache['violations']} C{cache['completions']} I{cache['incomplete']} "
                 f"evicted {cache['evicted_total']}" if cache else "")
              + (f" (+{cache['evicted_round']} this round, missed {cache['missed']}, "
                 f"evictable {cache['evictable']})" if cache and "evicted_round" in cache else "")
              + (f" | stops {dict(term)}" if term else ""))
        c = calls.get(r["round"])
        if c is not None:
            t += c["mutator_time"]
            print(f"    mutator: {len(c['outputs']) - _refused(c)}/{len(c['outputs'])} samples answered, "
                  f"{len(c['new_idx'])} new prompts, {c['mutator_time']:.0f}s"
                  + (f", {c['num_truncated']} cut by the server" if c.get("num_truncated") else "")
                  + (f", requested {c['num_requested']}" if "num_requested" in c else "")
                  + (f", ERROR {c['error'].splitlines()[0][:80]!r}" if c.get("error") else "")
                  + (", system message" if c.get("system_prompt") else ""))
    print(f"  wall time of the search so far: {t / 60:.1f} min")

    by_round = defaultdict(list)
    for i, r in inst.items():
        by_round[r["search_round"]].append(i)
    for k in sorted(by_round):
        if k == 0:
            continue
        d = []
        for i in by_round[k]:
            parents = [p for p in inst[i]["based_on"] if p in V]
            if parents:
                d.append(V[i] - max(V[p] for p in parents))
        fresh = sum(not inst[i]["based_on"] for i in by_round[k])
        if d:
            print(f"  prompts of round {k}: n={len(by_round[k])} | vs best parent: better {sum(x > 1e-4 for x in d)}, "
                  f"worse {sum(x < -1e-4 for x in d)}, median {st.median(d):+.3f}, max {max(d):+.3f}"
                  + (f" | {fresh} with no parent" if fresh else ""))
    fp = [r.get("total_transitions", 0) for r in inst.values()]
    if fp:
        print(f"  instances: {len(inst)} | fp median {st.median(fp):.0f} max {max(fp)} | stops "
              f"{dict(Counter(r['termination_reason'] for r in inst.values()))}")


def print_prompts(g, width=300):
    rounds = g["rounds"]
    if not rounds:
        return
    print(f"\n== BEAM OF GROUP {g['group']}")
    for b in rounds[-1]["beam"]:
        r = g["inst"].get(b["idx"], {})
        print(f"  #{b['idx']} viol {b['final_violation_prob']:.4f} safe {b['final_complete_prob']:.4f} "
              f"| round {r.get('search_round')} based_on {r.get('based_on')}")
        print(f"     {(r.get('system_prompt') or '')[:width]!r}")


_BEAM_ELEMENT_RE = re.compile(r'<system_prompt id="(\d+)"')
_LEAF_RE = re.compile(r"^- \(([0-9.]+)\) (.*)$")


def _outputs_from_request(request):
    """{idx: {"violating": [...], "safe": [...]}} from the beam text of a
    mutator request (top 3 of each kind, no token IDs, no widths)."""
    out = {}
    parts = _BEAM_ELEMENT_RE.split(request)
    for idx, body in zip(parts[1::2], parts[2::2]):
        rec, kind = {"violating": [], "safe": []}, None
        for line in body.splitlines():
            if line.startswith("Most probable violating"):
                kind = "violating"
            elif line.startswith("Most probable safe"):
                kind = "safe"
            elif kind and (m := _LEAF_RE.match(line)):
                try:
                    text = ast.literal_eval(m.group(2))
                except (ValueError, SyntaxError):
                    text = m.group(2)
                rec[kind].append({"prob": float(m.group(1)), "text": text})
        out[int(idx)] = rec
    return out


def _key(leaf):
    return tuple(leaf["tokens"]) if "tokens" in leaf else leaf["text"]


def print_outputs(g, top=5, width=60):
    rounds = g["rounds"]
    if not rounds:
        return
    beam = [b["idx"] for b in rounds[-1]["beam"]]
    outs, source = {i: g["outputs"][i] for i in beam if i in g["outputs"]}, "instance_outputs"
    if len(outs) < len(beam) and g["calls"]:
        c = g["calls"][-1]
        old = _outputs_from_request(c["request"])
        beam = [b["idx"] for b in rounds[c["round"]]["beam"]] if c["round"] < len(rounds) else beam
        outs = {i: old[i] for i in beam if i in old}
        source = f"mutator request of round {c['round']}, top 3 only"
    if not outs:
        return
    print(f"\n== OUTPUTS OF THE BEAM, GROUP {g['group']} ({source})")

    def fmt(leaves):
        return " | ".join(f"{l['prob']:.4f} {l['text'][:width]!r}" for l in leaves[:top]) or "none found"

    for i in beam:
        if i not in outs:
            continue
        o, r = outs[i], g["inst"].get(i, {})
        d = o.get("dist")
        print(f"  #{i} viol {r.get('final_violation_prob', float('nan')):.4f}"
              + (f" | eff. outputs {d['eff_outputs']:.1f}, top-1 mass {d['top1']:.4f}, "
                 f"{d['viol_n90']} viol leaves hold 90% (of {d['n_violating']}), "
                 f"{d['n_safe']} safe, {d['n_incomplete']} open" if d else ""))
        print(f"     V: {fmt(o['violating'])}")
        print(f"     S: {fmt(o['safe'])}")

    # Same outputs across the beam? Overlap of two elements = sum over the
    # logged leaves of min(p_a, p_b): 0 = disjoint, 1 = the same
    # distribution. Only the logged top leaves count, so it is a lower bound.
    dists = {i: {_key(l): l["prob"] for l in outs[i]["violating"] + outs[i]["safe"]} for i in outs}
    ids = list(dists)
    pairs = [sum(min(p, dists[b].get(k, 0.0)) for k, p in dists[a].items())
             for x, a in enumerate(ids) for b in ids[x + 1:]]
    top1 = Counter(_key(outs[i]["violating"][0]) for i in ids if outs[i]["violating"])
    text = {_key(l): l["text"] for i in ids for l in outs[i]["violating"]}
    print(f"  beam: {len(top1)} distinct top-1 violating outputs over {len(ids)} elements: "
          + ", ".join(f"{text[k][:width]!r} x{n}" for k, n in top1.most_common(5)))
    if pairs:
        print(f"  pairwise output overlap (sum of min prob over logged leaves): mean {st.mean(pairs):.3f}, "
              f"min {min(pairs):.3f}, max {max(pairs):.3f}")
    shared = Counter(k for i in ids for k in {_key(l) for l in outs[i]["violating"]})
    common = [(k, n) for k, n in shared.most_common(top) if n > 1]
    if common:
        print("  violating outputs shared by several elements: " + ", ".join(
            f"{text[k][:width]!r} in {n} (mean p "
            f"{st.mean(dists[i].get(k, 0.0) for i in ids if k in dists[i]):.4f})" for k, n in common))
    d = [outs[i]["dist"] for i in ids if outs[i].get("dist")]
    if d:
        eff = [x["eff_outputs"] for x in d]
        print(f"  width: eff. outputs median {st.median(eff):.1f} (min {min(eff):.1f}, max {max(eff):.1f}); "
              f"top-1 mass median {st.median(x['top1'] for x in d):.4f}")


def plot(groups, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for g in groups:
        rounds = g["rounds"]
        if not rounds:
            continue
        calls = {c["round"]: c["mutator_time"] for c in g["calls"]}
        best = [r["beam"][0]["final_violation_prob"] for r in rounds]
        tenth = [r["beam"][-1]["final_violation_prob"] for r in rounds]
        t, ts = 0.0, []
        for r in rounds:
            t += r["round_time"] + calls.get(r["round"] - 1, 0.0)
            ts.append(t / 60)
        line, = axes[0].plot(range(len(best)), best, marker="o", label=f"group {g['group']}")
        axes[0].plot(range(len(tenth)), tenth, ls="--", color=line.get_color(), alpha=0.6)
        axes[1].plot(ts, best, marker="o", color=line.get_color())
    axes[0].set(xlabel="round", ylabel="certified violation LB", title="best (solid) and 10th (dashed) of the beam")
    axes[1].set(xlabel="wall time (min)", ylabel="best certified violation LB", title="best vs time")
    for ax in axes:
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.3)
    if len(groups) <= 12:
        axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"\nplot -> {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run_dir")
    ap.add_argument("--detail", action="store_true", help="per-round detail of every group")
    ap.add_argument("--prompts", action="store_true", help="print the current beam of every group")
    ap.add_argument("--outputs", action="store_true",
                    help="what the beam generates: top outputs, width, overlap")
    ap.add_argument("--plot", help="write a progress plot (needs matplotlib)")
    ap.add_argument("--json", help="write the overview rows as JSON")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.run_dir, "**", "group_*.jsonl"), recursive=True),
                   key=lambda p: int(re.search(r"group_(\d+)", p).group(1)))
    if not paths:
        raise SystemExit(f"no group_*.jsonl under {args.run_dir}")
    groups = [load_group(p) for p in paths]

    print_overview(groups)
    print_progress(groups)
    if args.detail or len(groups) == 1:
        for g in groups:
            print_detail(g)
    if args.prompts:
        for g in groups:
            print_prompts(g)
    if args.outputs:
        for g in groups:
            print_outputs(g)
    if args.plot:
        plot(groups, args.plot)
    if args.json:
        with open(args.json, "w") as f:
            json.dump([summarize(g) for g in groups], f, indent=2)


if __name__ == "__main__":
    main()
