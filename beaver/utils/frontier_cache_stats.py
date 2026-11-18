import os
import json
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from beaver.logging import get_log_data, get_profile_data


def get_cache_data(cache_dir):
    """Load the FINAL persisted cache snapshot per group (<group>.json)."""
    assert cache_dir.is_dir(), f"The provided path '{cache_dir}' is not a valid directory."
    group_cache_data = {}
    for f in cache_dir.glob("*.json"):
        if f.is_file() and f.name.split(".")[0].isdigit():
            group = int(f.name.split(".")[0])
            with open(f) as fh:
                group_cache_data[group] = json.load(fh)
    return group_cache_data


def _cache_sizes(group_cache):
    """(n_violations, n_satisfactions, n_trie_nodes) from a persisted cache dict.

    Handles all cache formats: violations-only, violations+satisfactions, and
    trie+violations.
    """
    n_viol = len(group_cache.get("violations", []))
    n_sat = len(group_cache.get("satisfactions", []))

    def count_nodes(node):
        if not node:
            return 0
        return 1 + sum(count_nodes(c) for c in node.get("c", {}).values())

    trie = group_cache.get("trie")
    n_nodes = count_nodes(trie) if trie else 0
    return n_viol, n_sat, n_nodes


def _entry(entries, event):
    return next((e for e in entries if e.get("event") == event), None)


def build_group_timelines(log_data):
    """Reconstruct per-instance cache evolution, ordered within each group.

    Instances in a group are warm-started in ascending idx order (== processing
    order), so sorting by idx replays how the cache grew over time.

    Returns: dict group -> list of per-instance dicts ordered by idx, each with
      idx, warm_violations, warm_nodes, warm_violation_prob,
      new_violations_found, new_trie_nodes, new_antichain_size,
      final_violation_prob, total_transitions
    """
    groups = defaultdict(list)
    for entries in log_data.values():
        if not entries:
            continue
        head = entries[0]
        group = head.get("group")
        idx = head.get("idx")
        if group is None:
            continue
        ws = _entry(entries, "warm_start") or {}
        ic = _entry(entries, "instance_complete") or {}
        groups[group].append({
            "idx": idx,
            "warm_violations": ws.get("cache_violations", 0),
            # No per-instance persisted-cache trie-node count is ever
            # logged (only the final on-disk cache snapshot has one) — this
            # is a known, documented gap, not a bug. get_cache_stats only
            # reads it in its rare no-persisted-cache fallback branch.
            "warm_nodes": 0,
            "warm_violation_prob": ws.get("warm_violation_prob", 0.0),
            # cache buildup: component sizes seen at warm-start (i.e. BEFORE this
            # instance adds to the cache) -> how big the cache was already.
            "cache_violations": ws.get("cache_violations", 0),
            "cache_completions": ws.get("cache_completions", 0),
            "cache_incomplete": ws.get("cache_incomplete", 0),
            "cache_total": ws.get(
                "cache_total",
                ws.get("cache_violations", 0)
                + ws.get("cache_completions", 0)
                + ws.get("cache_incomplete", 0),
            ),
            # timing: how long warm-start (re-scoring cached seqs) took, and the
            # full instance run time. warm_start_time grows with cache_total.
            "warm_start_time": ws.get("warm_start_time",
                                      ic.get("warm_start_time", 0.0)),
            "instance_run_time": ic.get("instance_run_time", 0.0),
            "new_violations_found": ic.get("new_violations_found", 0),
            # Real field on every instance_complete: this instance's own
            # local frontier trie size — the closest available proxy for
            # "new trie nodes" (not identical to nodes actually merged into
            # the persistent cache, but real data instead of a dead key
            # that never existed in any logged event).
            "new_trie_nodes": ic.get("total_elements", 0),
            "new_antichain_size": (
                ic.get("new_violations_found", 0)
                + ic.get("new_completions_found", 0)
            ),
            "final_violation_prob": ic.get("final_violation_prob", 0.0),
            "total_transitions": ic.get("total_transitions", 0),
        })
    for g in groups:
        groups[g].sort(key=lambda r: (r["idx"] if r["idx"] is not None else 0))
    return dict(groups)


# Sections build_frontier_from_cache times internally (tokentrie_cache.py's
# _tick calls) -- everything under warm_start_time besides the sections
# themselves is unaccounted overhead (dict/tuple construction between ticks).
_CACHE_SECTION_KEYS = [
    "build_antichain",
    "get_distribution",
    "insert_parents",
    "flag_terminals",
    "classify_leaves",
    "calc_prob_sum",
]


def build_cache_section_timelines(log_dir):
    """Per-group, per-position breakdown of build_frontier_from_cache's internal
    sections, ordered by group_position (the true within-group processing order).

    Only populated for CachedFrontierVerifier runs -- the cold-start
    FrontierVerifier never calls build_frontier_from_cache, so its
    *.profile.json files have no "warm_start" event and this returns {}.

    Returns dict group -> list of per-instance dicts (idx, group_position,
    warm_start_time, and every key in _CACHE_SECTION_KEYS found).
    """
    profile_data = get_profile_data(log_dir)
    groups = defaultdict(list)
    for entries in profile_data.values():
        for e in entries:
            if e.get("event") != "warm_start":
                continue
            group = e.get("group")
            if group is None:
                continue
            groups[group].append(e)
    for g in groups:
        groups[g].sort(
            key=lambda r: (r.get("group_position") if r.get("group_position") is not None else 0)
        )
    return dict(groups)


def get_cache_stats(log_dir, plot=True):
    cache_dir = log_dir / "frontier_cache"
    log_data = get_log_data(log_dir)
    cache = get_cache_data(cache_dir) if cache_dir.is_dir() else {}

    timelines = build_group_timelines(log_data)
    all_groups = sorted(timelines.keys())

    # ── Final cache state per group (from persisted snapshots) ──────────────
    final_viol = {}
    final_sat = {}
    final_nodes = {}
    for g in all_groups:
        if g in cache:
            nv, ns, nn = _cache_sizes(cache[g])
        else:
            # Fall back to reconstruction: last instance's cumulative state.
            # (satisfactions are not logged per-instance -> unknown, report 0).
            tl = timelines[g]
            nv = tl[-1]["warm_violations"] + tl[-1]["new_violations_found"] if tl else 0
            ns = 0
            nn = tl[-1]["warm_nodes"] + tl[-1]["new_trie_nodes"] if tl else 0
        final_viol[g] = nv
        final_sat[g] = ns
        final_nodes[g] = nn

    groups_with_violations = [g for g in all_groups if final_viol[g] > 0]
    groups_with_satisfactions = [g for g in all_groups if final_sat[g] > 0]
    has_satisfactions = any(final_sat.values())

    # ── Per-position aggregates: does warm start improve with more instances? ─
    # position k = the (k+1)-th instance processed in its group.
    max_pos = max((len(timelines[g]) for g in all_groups), default=0)
    by_pos_warm_vp = defaultdict(list)       # warm_violation_prob at position k
    by_pos_warm_viol = defaultdict(list)     # cached violations seen at position k
    by_pos_new_viol = defaultdict(list)      # NEW violations found at position k
    by_pos_transitions = defaultdict(list)   # work done at position k
    # cache component buildup + the time it costs, by position within group.
    by_pos_cache_viol = defaultdict(list)    # cached violations at warm-start
    by_pos_cache_comp = defaultdict(list)    # cached completions at warm-start
    by_pos_cache_inc = defaultdict(list)     # cached incomplete at warm-start
    by_pos_cache_total = defaultdict(list)   # total cache size at warm-start
    by_pos_warm_time = defaultdict(list)     # warm-start (re-score) time
    by_pos_run_time = defaultdict(list)      # full instance run time
    for g in all_groups:
        for k, r in enumerate(timelines[g]):
            by_pos_warm_vp[k].append(r["warm_violation_prob"])
            by_pos_warm_viol[k].append(r["warm_violations"])
            by_pos_new_viol[k].append(r["new_violations_found"])
            by_pos_transitions[k].append(r["total_transitions"])
            by_pos_cache_viol[k].append(r["cache_violations"])
            by_pos_cache_comp[k].append(r["cache_completions"])
            by_pos_cache_inc[k].append(r["cache_incomplete"])
            by_pos_cache_total[k].append(r["cache_total"])
            by_pos_warm_time[k].append(r["warm_start_time"])
            by_pos_run_time[k].append(r["instance_run_time"])

    def _mean(d, k):
        return float(np.mean(d[k])) if d[k] else 0.0

    def _median(d, k):
        return float(np.median(d[k])) if d[k] else 0.0

    positions = list(range(max_pos))
    n_groups_by_pos = [len(by_pos_warm_vp[k]) for k in positions]  # sample size
    avg_warm_vp_by_pos = [_mean(by_pos_warm_vp, k) for k in positions]
    median_warm_vp_by_pos = [_median(by_pos_warm_vp, k) for k in positions]
    avg_warm_viol_by_pos = [_mean(by_pos_warm_viol, k) for k in positions]
    avg_new_viol_by_pos = [_mean(by_pos_new_viol, k) for k in positions]
    avg_transitions_by_pos = [_mean(by_pos_transitions, k) for k in positions]
    # cache buildup + time-cost curves by position
    avg_cache_viol_by_pos = [_mean(by_pos_cache_viol, k) for k in positions]
    avg_cache_comp_by_pos = [_mean(by_pos_cache_comp, k) for k in positions]
    avg_cache_inc_by_pos = [_mean(by_pos_cache_inc, k) for k in positions]
    avg_cache_total_by_pos = [_mean(by_pos_cache_total, k) for k in positions]
    avg_warm_time_by_pos = [_mean(by_pos_warm_time, k) for k in positions]
    median_warm_time_by_pos = [_median(by_pos_warm_time, k) for k in positions]
    avg_run_time_by_pos = [_mean(by_pos_run_time, k) for k in positions]

    # ── build_frontier_from_cache section breakdown, by within-group position ─
    # From *.profile.json warm_start entries (empty for cold-start FrontierVerifier
    # runs, since those never call build_frontier_from_cache).
    section_timelines = build_cache_section_timelines(log_dir)
    by_pos_section = {key: defaultdict(list) for key in _CACHE_SECTION_KEYS}
    for g, rows in section_timelines.items():
        for k, r in enumerate(rows):
            for key in _CACHE_SECTION_KEYS:
                if key in r:
                    by_pos_section[key][k].append(r[key])
    section_max_pos = max((len(rows) for rows in section_timelines.values()), default=0)
    section_positions = list(range(section_max_pos))
    avg_section_by_pos = {
        key: [_mean(by_pos_section[key], k) for k in section_positions]
        for key in _CACHE_SECTION_KEYS
    }
    n_groups_by_section_pos = [
        len({g for g, rows in section_timelines.items() if len(rows) > k})
        for k in section_positions
    ]

    summary = {
        "num_groups": len(all_groups),
        "num_groups_with_violations": len(groups_with_violations),
        "pct_groups_with_violations": (
            100.0 * len(groups_with_violations) / len(all_groups) if all_groups else 0.0
        ),
        "avg_final_violations_per_group": (
            float(np.mean([final_viol[g] for g in all_groups])) if all_groups else 0.0
        ),
        "avg_final_trie_nodes_per_group": (
            float(np.mean([final_nodes[g] for g in all_groups])) if all_groups else 0.0
        ),
        "max_final_violations_in_a_group": (
            max((final_viol[g] for g in all_groups), default=0)
        ),
        "max_group_size": max_pos,
        "avg_group_size": (
            float(np.mean([len(timelines[g]) for g in all_groups])) if all_groups else 0.0
        ),
        # satisfactions (only present in caches that track them)
        "has_satisfactions": has_satisfactions,
        "num_groups_with_satisfactions": len(groups_with_satisfactions),
        "avg_final_satisfactions_per_group": (
            float(np.mean([final_sat[g] for g in all_groups])) if all_groups else 0.0
        ),
        "max_final_satisfactions_in_a_group": (
            max((final_sat[g] for g in all_groups), default=0)
        ),
        # warm-start improvement as more instances accumulate in a group.
        # n_groups_by_position is the sample size behind each point: high
        # positions have few groups, so their averages are noisy/untrustworthy.
        "n_groups_by_position": n_groups_by_pos,
        "avg_warm_violation_prob_by_position": avg_warm_vp_by_pos,
        "median_warm_violation_prob_by_position": median_warm_vp_by_pos,
        "avg_warm_cached_violations_by_position": avg_warm_viol_by_pos,
        "avg_new_violations_found_by_position": avg_new_viol_by_pos,
        "avg_transitions_by_position": avg_transitions_by_pos,
        # cache component buildup over instances in a group, and the warm-start
        # (cached-sequence re-scoring) time it costs. The warm-time curve rising
        # with cache_total is the warm-up slowdown made explicit.
        "avg_cache_violations_by_position": avg_cache_viol_by_pos,
        "avg_cache_completions_by_position": avg_cache_comp_by_pos,
        "avg_cache_incomplete_by_position": avg_cache_inc_by_pos,
        "avg_cache_total_by_position": avg_cache_total_by_pos,
        "avg_warm_start_time_by_position": avg_warm_time_by_pos,
        "median_warm_start_time_by_position": median_warm_time_by_pos,
        "avg_instance_run_time_by_position": avg_run_time_by_pos,
        # build_frontier_from_cache's internal section breakdown by position
        # (empty lists for cold-start FrontierVerifier runs / older logs
        # predating this instrumentation).
        "n_groups_by_section_position": n_groups_by_section_pos,
        "avg_cache_section_time_by_position": avg_section_by_pos,
    }

    out = log_dir / "frontier_cache_stats.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'=' * 50}")
    print(f"Frontier cache stats  ({log_dir.name})")
    print(f"{'=' * 50}")
    print(f"groups                     : {summary['num_groups']}")
    print(f"groups with violations     : {summary['num_groups_with_violations']} "
          f"({summary['pct_groups_with_violations']:.1f}%)")
    print(f"avg violations / group     : {summary['avg_final_violations_per_group']:.2f}")
    print(f"avg trie nodes / group     : {summary['avg_final_trie_nodes_per_group']:.2f}")
    print(f"max violations in a group  : {summary['max_final_violations_in_a_group']}")
    if has_satisfactions:
        print(f"groups with satisfactions  : {summary['num_groups_with_satisfactions']}")
        print(f"avg satisfactions / group  : {summary['avg_final_satisfactions_per_group']:.2f}")
    print(f"avg / max group size       : {summary['avg_group_size']:.1f} / {summary['max_group_size']}")
    # warm-start time growth: first vs last populated position (the slowdown).
    pop = [k for k in positions if by_pos_warm_time[k]]
    if pop:
        first_t = _mean(by_pos_warm_time, pop[0])
        last_t = _mean(by_pos_warm_time, pop[-1])
        max_t = max((max(by_pos_warm_time[k]) for k in pop), default=0.0)
        print(f"avg warm-start time        : {first_t:.1f}s @pos1 -> "
              f"{last_t:.1f}s @pos{pop[-1] + 1}  (max {max_t:.1f}s)")
        first_c = _mean(by_pos_cache_total, pop[0])
        last_c = _mean(by_pos_cache_total, pop[-1])
        print(f"avg cache_total            : {first_c:.0f} @pos1 -> "
              f"{last_c:.0f} @pos{pop[-1] + 1}")
    if section_positions:
        print(f"\nbuild_frontier_from_cache section breakdown (avg seconds, pos1 -> pos{section_positions[-1] + 1}):")
        for key in _CACHE_SECTION_KEYS:
            vals = avg_section_by_pos[key]
            nz = [k for k in section_positions if by_pos_section[key][k]]
            if not nz:
                continue
            print(f"  {key:<18s}: {vals[nz[0]]:.4f}s -> {vals[nz[-1]]:.4f}s")
    print(f"\nwrote {out}")

    if plot:
        _make_plots(log_dir, timelines, final_viol, final_sat, final_nodes,
                    has_satisfactions, positions, n_groups_by_pos,
                    avg_warm_vp_by_pos, median_warm_vp_by_pos,
                    avg_warm_viol_by_pos, avg_new_viol_by_pos)
        _make_buildup_plots(
            log_dir, timelines, positions, n_groups_by_pos,
            avg_cache_viol_by_pos, avg_cache_comp_by_pos, avg_cache_inc_by_pos,
            avg_cache_total_by_pos, avg_warm_time_by_pos, median_warm_time_by_pos,
            avg_run_time_by_pos)
        if section_positions:
            _make_cache_section_plots(
                log_dir, section_positions, n_groups_by_section_pos, avg_section_by_pos
            )

    return summary


def _cumulative(tl, key):
    """Cumulative cache size over instance positions for one group."""
    out, run = [], 0
    for r in tl:
        run += r[key]
        out.append(run)
    return out


def _make_plots(log_dir, timelines, final_viol, final_sat, final_nodes,
                has_satisfactions, positions, n_groups_by_pos,
                avg_warm_vp, median_warm_vp, avg_warm_viol, avg_new_viol):
    fig, axes = plt.subplots(3, 2, figsize=(15, 16))

    # (1) Cache size growth over time, per group (cumulative violations) ──────
    ax = axes[0, 0]
    big = sorted(timelines.keys(), key=lambda g: -len(timelines[g]))[:30]
    for g in big:
        ax.plot(range(1, len(timelines[g]) + 1),
                _cumulative(timelines[g], "new_violations_found"),
                alpha=0.5, linewidth=1)
    ax.set_xlabel("instance # within group")
    ax.set_ylabel("cumulative cached violations")
    ax.set_title("Cache size growth per group (top 30 by size)")

    # (2) Histogram of group cache size (violations per group) ────────────────
    ax = axes[0, 1]
    vvals = list(final_viol.values())
    vmax = max(vvals) if vvals else 0
    bins = np.arange(0, vmax + 2) - 0.5 if vmax <= 40 else 40
    ax.hist(vvals, bins=bins, color="#c44", edgecolor="white")
    ax.set_xlabel("cache size (# violations in group)")
    ax.set_ylabel("# groups")
    ax.set_title("Histogram of group cache size (violations)")
    ax.axvline(np.mean(vvals), color="k", linestyle="--", linewidth=1,
               label=f"mean {np.mean(vvals):.2f}")
    ax.legend()

    # (3) Histogram of group trie size (nodes) OR group sizes if no trie ──────
    ax = axes[1, 0]
    nvals = list(final_nodes.values())
    if any(nvals):
        ax.hist(nvals, bins=40, color="#48a", edgecolor="white")
        ax.set_xlabel("cache size (# trie nodes in group)")
        ax.set_title("Histogram of group cache size (trie nodes)")
    else:
        gsizes = [len(timelines[g]) for g in timelines]
        ax.hist(gsizes, bins=np.arange(0, max(gsizes) + 2) - 0.5 if gsizes else 10,
                color="#48a", edgecolor="white")
        ax.set_xlabel("# instances in group")
        ax.set_title("Histogram of group sizes (no trie nodes in cache)")
    ax.set_ylabel("# groups")

    # (4) Warm-start strength as instances accumulate (median + sample size) ──
    # avg looks noisy at high positions because few groups reach them and early
    # termination (warm_vp >= threshold) skips/removes high-vp instances, biasing
    # the average. Show median + n per point so low-sample tail is obvious.
    ax = axes[1, 1]
    x = [p + 1 for p in positions]
    ax.plot(x, avg_warm_vp, marker="o", markersize=3, alpha=0.5,
            color="#a44", label="mean warm viol prob")
    ax.plot(x, median_warm_vp, marker="s", markersize=3,
            color="#822", label="median warm viol prob")
    ax.set_xlabel("instance # within group")
    ax.set_ylabel("warm-start violation prob")
    ax.set_title("Warm start vs instance position in group")
    ax2 = ax.twinx()
    ax2.bar(x, n_groups_by_pos, color="#ccc", alpha=0.4, width=0.9, zorder=0)
    ax2.set_ylabel("# groups at this position (sample size)", color="#999")
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)
    ax.legend(loc="upper right")

    # (5) New violations found per position (should decay as cache warms) ─────
    ax = axes[2, 0]
    ax.plot(x, avg_new_viol, marker="o", markersize=3, color="#a44")
    ax.set_xlabel("instance # within group")
    ax.set_ylabel("avg NEW violations found")
    ax.set_title("New work per instance (decays as cache warms)")

    # (6) Histogram of group cache size (satisfactions) ──────────────────────
    ax = axes[2, 1]
    if has_satisfactions:
        svals = list(final_sat.values())
        smax = max(svals)
        sbins = np.arange(0, smax + 2) - 0.5 if smax <= 40 else 40
        ax.hist(svals, bins=sbins, color="#4a4", edgecolor="white")
        ax.axvline(np.mean(svals), color="k", linestyle="--", linewidth=1,
                   label=f"mean {np.mean(svals):.2f}")
        ax.set_xlabel("cache size (# satisfactions in group)")
        ax.set_ylabel("# groups")
        ax.set_title("Histogram of group cache size (satisfactions)")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "no satisfactions tracked\nin this cache",
                ha="center", va="center", transform=ax.transAxes, color="#999")
        ax.set_title("Satisfactions (n/a)")
        ax.axis("off")

    fig.tight_layout()
    out = log_dir / "frontier_cache_stats.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"wrote {out}")


def _make_buildup_plots(log_dir, timelines, positions, n_groups_by_pos,
                        avg_cache_viol, avg_cache_comp, avg_cache_inc,
                        avg_cache_total, avg_warm_time, median_warm_time,
                        avg_run_time):
    """Cache-component buildup over a group and the warm-start time it costs.

    Answers: how does each cache component grow across instances in a group, and
    how much does warm-start (cached-sequence re-scoring) time grow with it.
    """
    x = [p + 1 for p in positions]
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))

    # (a) Component buildup vs position ──────────────────────────────────────
    ax = axes[0, 0]
    ax.plot(x, avg_cache_viol, marker="o", markersize=3, color="#c44",
            label="violations")
    ax.plot(x, avg_cache_comp, marker="o", markersize=3, color="#4a4",
            label="completions")
    ax.plot(x, avg_cache_inc, marker="o", markersize=3, color="#48a",
            label="incomplete")
    ax.plot(x, avg_cache_total, marker="s", markersize=3, color="k",
            label="total")
    ax.set_xlabel("instance # within group")
    ax.set_ylabel("avg cache size at warm-start")
    ax.set_title("Cache component buildup over instances in a group")
    ax.legend()

    # (b) Warm-start time vs position (+ sample size) ────────────────────────
    ax = axes[0, 1]
    ax.plot(x, avg_warm_time, marker="o", markersize=3, alpha=0.6, color="#a44",
            label="mean warm-start time")
    ax.plot(x, median_warm_time, marker="s", markersize=3, color="#822",
            label="median warm-start time")
    ax.set_xlabel("instance # within group")
    ax.set_ylabel("warm-start time (s)")
    ax.set_title("Warm-start time grows as cache builds up")
    ax2 = ax.twinx()
    ax2.bar(x, n_groups_by_pos, color="#ccc", alpha=0.4, width=0.9, zorder=0)
    ax2.set_ylabel("# groups at this position (sample size)", color="#999")
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)
    ax.legend(loc="upper left")

    # (c) Time increase due to cache size: warm time vs cache_total ──────────
    ax = axes[1, 0]
    xs, ys = [], []
    for tl in timelines.values():
        for r in tl:
            xs.append(r["cache_total"])
            ys.append(r["warm_start_time"])
    ax.scatter(xs, ys, s=8, alpha=0.3, color="#a44")
    ax.set_xlabel("cache size (cache_total at warm-start)")
    ax.set_ylabel("warm-start time (s)")
    ax.set_title("Time increase due to cache size (per instance)")

    # (d) Full instance run time vs position ─────────────────────────────────
    ax = axes[1, 1]
    ax.plot(x, avg_run_time, marker="o", markersize=3, color="#558")
    ax.set_xlabel("instance # within group")
    ax.set_ylabel("avg instance run time (s)")
    ax.set_title("Total per-instance run time over a group")

    fig.tight_layout()
    out = log_dir / "frontier_cache_buildup.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"wrote {out}")


def _make_cache_section_plots(log_dir, positions, n_groups_by_pos, avg_section_by_pos):
    """build_frontier_from_cache's internal section times vs within-group
    position -- which section actually drives the cached-frontier warm-start
    slowdown as a group's cache grows (get_distribution re-scoring cached
    sequences against the model? insert_parents rebuilding the trie? etc).
    """
    x = [p + 1 for p in positions]
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    colors = {
        "build_antichain": "#888",
        "get_distribution": "#a44",
        "insert_parents": "#48a",
        "flag_terminals": "#4a4",
        "classify_leaves": "#a84",
        "calc_prob_sum": "#84a",
    }

    ax = axes[0]
    for key in _CACHE_SECTION_KEYS:
        vals = avg_section_by_pos[key]
        if not any(vals):
            continue
        ax.plot(x, vals, marker="o", markersize=3, label=key, color=colors.get(key))
    ax.set_xlabel("instance # within group")
    ax.set_ylabel("avg section time (s)")
    ax.set_title("build_frontier_from_cache section breakdown vs position")
    ax.legend(fontsize=8)

    ax = axes[1]
    stack = [avg_section_by_pos[key] for key in _CACHE_SECTION_KEYS if any(avg_section_by_pos[key])]
    labels = [key for key in _CACHE_SECTION_KEYS if any(avg_section_by_pos[key])]
    if stack:
        ax.stackplot(x, *stack, labels=labels,
                     colors=[colors.get(k) for k in labels], alpha=0.85)
    ax.set_xlabel("instance # within group")
    ax.set_ylabel("avg section time (s), stacked")
    ax.set_title("Where warm-start time goes, by section")
    ax2 = ax.twinx()
    ax2.bar(x, n_groups_by_pos, color="#ccc", alpha=0.3, width=0.9, zorder=0)
    ax2.set_ylabel("# groups at this position (sample size)", color="#999")
    ax.legend(fontsize=8, loc="upper left")

    fig.tight_layout()
    out = log_dir / "frontier_cache_section_breakdown.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute frontier cache stats for BEAVER run logs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m beaver.utils.frontier_cache_stats logging/logs_20260316185322\n"
        ),
    )
    parser.add_argument("path", type=str, help="Run log folder directory")
    parser.add_argument("--no-plot", action="store_true", help="Skip generating plots")
    args = parser.parse_args()
    log_dir = Path(args.path)
    assert log_dir.is_dir(), f"The provided path '{log_dir}' is not a valid directory."

    get_cache_stats(log_dir, plot=not args.no_plot)
