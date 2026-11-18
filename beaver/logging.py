import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from scipy.stats import beta



# Non-timing fields that show up in *.profile.json entries for joins/filtering
# (see cached_frontier_verifier.py, frontier_step.py) -- excluded from the
# numeric aggregation in summarize_profile_data / get_profile_stats.
_PROFILE_METADATA_KEYS = {"event", "idx", "group", "group_position", "transition"}


def read_log_file(file_path: str):
    """Read a log file and parse JSON entries."""
    with open(file_path, "r") as f:
        content = f.read()
        if not content.strip():
            # Empty file: instance was created but no entry was ever
            # written (e.g. worker killed/crashed before its first log).
            return []
        # Split by newline-separated JSON objects
        parts = content.replace("}\n{", "}###SPLIT###{").split("###SPLIT###")
        entries = []
        for i, part in enumerate(parts):
            try:
                entries.append(json.loads(part))
            except json.JSONDecodeError:
                if i != len(parts) - 1:
                    raise
    return entries


def get_profile_data(run_logs_folder: Path):
    """Read all profile data and return raw entries per file."""
    profile_json_files = [
        f for f in os.listdir(run_logs_folder) if f.endswith(".profile.json")
    ]
    if len(profile_json_files) == 0:
        return {}

    all_profile_data = {}
    for file in profile_json_files:
        file_path = os.path.join(run_logs_folder, file)
        entries = read_log_file(file_path)
        all_profile_data[file] = entries

    return all_profile_data


def summarize_profile_data(run_logs_folder: Path):
    profile_json_files = [
        f for f in os.listdir(run_logs_folder) if f.endswith(".profile.json")
    ]
    if len(profile_json_files) == 0:
        return
    else:
        all_profiles = {}
        for file in profile_json_files:
            file_path = os.path.join(run_logs_folder, file)
            entries = read_log_file(file_path)
            all_profiles[file] = {}
            for entry in entries:
                for key, value in entry.items():
                    # Metadata fields (event, idx, group, group_position,
                    # transition) aren't timings -- summing/averaging them is
                    # meaningless, and non-numeric ones (e.g. "event") crash
                    # sum() outright.
                    if key in _PROFILE_METADATA_KEYS:
                        continue
                    if not isinstance(value, (int, float)) or isinstance(value, bool):
                        continue
                    if key not in all_profiles[file]:
                        all_profiles[file][key] = []
                    all_profiles[file][key].append(value)
        file_avg_profiles = {}
        file_max_profiles = {}
        avg_profiles = {}
        for file, fdata in all_profiles.items():
            for key, values in fdata.items():
                if key not in avg_profiles:
                    avg_profiles[key] = []
                avg_profiles[key].extend(values)
                if key not in file_avg_profiles:
                    file_avg_profiles[key] = []
                    file_max_profiles[key] = []
                file_total = sum(values)
                file_avg_profiles[key].append(file_total)
                file_max_profiles[key].append(file_total)

        for key, values in avg_profiles.items():
            avg_profiles[key] = np.mean(values)
        for key in file_avg_profiles:
            file_max_profiles[key] = float(np.max(file_avg_profiles[key]))
            file_avg_profiles[key] = float(np.mean(file_avg_profiles[key]))

        summary = {
            "avg_profiles": avg_profiles,
            "file_avg_profiles": file_avg_profiles,
            "file_max_profiles": file_max_profiles,
        }

        summary_file = run_logs_folder / "profiling_summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=4)

        # print("Average Timing Profiles over transitions (in seconds):")
        # print(json.dumps(avg_profiles, indent=4))
        print("Average Timing Profiles over tasks (in seconds):")
        print(json.dumps(file_avg_profiles, indent=4))
        # print("Max Timing Profiles over tasks (in seconds):")
        # print(json.dumps(file_max_profiles, indent=4))

        return


def get_run_args(log_folder: Path):
    with open(log_folder / "run_args.json") as f:
        run_args = json.load(f)
    return run_args

def get_log_data(log_folder: Path):
    all_data = {}
    for file_path in log_folder.glob("*.jsonl"):
        if file_path.is_file() and file_path.name.split(".")[0].isdigit():
            # is log file
            entries = read_log_file(log_folder / file_path.name)
            instance = file_path.name.split(".")[0]
            all_data[instance] = entries

    return all_data


def summarize_log_data(
    all_data, run_logs_folder: Path, use_median: bool = False, threshold: float = 0.9, print_results = True
):
    """Summarize a run's logs.

    Every instance's outcome is read from its single ``instance_complete``
    event — the authoritative final verdict emitted by every verifier
    strategy (cold-start ``FrontierVerifier`` and warm-started
    ``CachedFrontierVerifier`` alike), including the ``warm_skip`` early-exit
    path. Instances with no such event (crashed mid-run, or logs predating
    the unified event) are counted in ``num_instances_no_data`` and excluded
    from every aggregate below.
    """
    num_instances = len(all_data)

    complete_events = {}
    for instance, entries in all_data.items():
        ic = next((e for e in entries if e.get("event") == "instance_complete"), None)
        if ic is not None:
            complete_events[instance] = ic

    num_with_data = len(complete_events)
    num_no_data = num_instances - num_with_data

    transitions = [ic.get("total_transitions", 0) for ic in complete_events.values()]

    # A "transition" is one ROUND of the search loop. With
    # expand_batch_size=k a round expands k nodes and therefore costs k
    # forward passes, so rounds are not comparable across k -- forward passes
    # are. Each transition record carries batch_size (absent == 1 for every
    # pre-batching run), so sum it rather than counting records.
    forward_passes = []
    for instance in complete_events:
        forward_passes.append(
            sum(
                e.get("batch_size", 1)
                for e in all_data[instance]
                if "transition" in e
            )
        )

    # Last round at which the upper bound was still >= threshold (how far the
    # search ran before the bound dropped below it), or 1 if it never did.
    # Reported in BOTH rounds and forward passes for the same reason.
    transitions_to_threshold = []
    fps_to_threshold = []
    for instance, ic in complete_events.items():
        entries = [e for e in all_data[instance] if "transition" in e]
        crossed = [e for e in entries if e.get("upper_bound", 0.0) >= threshold]
        transitions_to_threshold.append(
            crossed[-1]["transition"] + 1 if crossed else 1
        )
        if crossed:
            last = crossed[-1]["transition"]
            fps_to_threshold.append(
                sum(e.get("batch_size", 1) for e in entries if e["transition"] <= last)
            )
        else:
            fps_to_threshold.append(1)

    # ── Final bounds per instance, read straight from instance_complete ──────
    #     UB = incomplete + complete + pruned        LB = complete
    final_ub = []
    final_lb = []
    for ic in complete_events.values():
        inc = ic.get("final_incomplete_prob")
        comp = ic.get("final_complete_prob", 0.0)
        viol = ic.get("final_violation_prob", 0.0)
        pru = ic.get("final_pruned_prob", 0.0)
        # inc is None when the warm-start gate cut the incomplete set; then
        # UB = 1 - violation (total mass minus the proven-violating mass).
        ub = (1.0 - viol) if inc is None else (inc + comp + pru)
        final_ub.append(ub)
        final_lb.append(comp)

    final_ub_minus_lb = [ub - lb for ub, lb in zip(final_ub, final_lb)]
    num_determined = len(final_ub)  # == num_with_data; kept for summary.json compat

    # Calculate constraint satisfaction metrics over all determined instances
    num_satisfied = sum(1 for ub in final_ub if ub >= threshold)
    num_unsatisfied = sum(1 for ub in final_ub if ub < threshold)
    pct_satisfied = (
        (num_satisfied / num_determined * 100) if num_determined > 0 else 0.0
    )
    pct_unsatisfied = (
        (num_unsatisfied / num_determined * 100) if num_determined > 0 else 0.0
    )

    # ── Three-way verdict over determined instances ───────────────────────────
    # Bounds on P(constraint satisfied / safe): LB = proven-safe (complete) mass,
    # UB = 1 - proven-violation mass. An instance is:
    #   safe   if LB  > threshold  (already proven safe past the bar)
    #   risky  if UB <= threshold  (cannot reach the bar even in the best case)
    #   open   otherwise           (gap still straddles the threshold)
    num_safe = sum(1 for lb in final_lb if lb > threshold)
    num_risky = sum(1 for ub in final_ub if ub <= threshold)
    num_open = num_determined - num_safe - num_risky
    pct_safe = (num_safe / num_determined * 100) if num_determined > 0 else 0.0
    pct_risky = (num_risky / num_determined * 100) if num_determined > 0 else 0.0
    pct_open = (num_open / num_determined * 100) if num_determined > 0 else 0.0



    # Choose aggregation function
    agg_func = np.median if use_median else np.mean
    metric_label = "Median" if use_median else "Avg"


    # ── Warm-start stats (cache_extra fields present only on cached-verifier
    # instance_complete/warm_start events; empty/zero for the cold verifier) ──
    warm_by_instance = {}
    for instance, entries in all_data.items():
        ws = next((e for e in entries if e.get("event") == "warm_start"), None)
        if ws is not None:
            warm_by_instance[instance] = ws

    warm_start_events = list(warm_by_instance.values())
    has_warm_start = len(warm_start_events) > 0

    # Instances where cache already held violations when the instance began
    warm_active = [ws for ws in warm_start_events if ws.get("cache_violations", 0) > 0]
    n_warm_active = len(warm_active)

    # Early-skip triggered by warm-start violation prob >= threshold
    n_warm_skip = sum(1 for ic in complete_events.values() if ic.get("warm_skip", False))
    # Instances that actually ran the search loop (not early-skipped by warm start)
    n_actually_ran = num_with_data - n_warm_skip

    # Warm violation prob across all instances (zero for those with empty cache)
    all_warm_vp = [ws.get("warm_violation_prob", 0.0) for ws in warm_start_events]
    avg_warm_vp_all = float(np.mean(all_warm_vp)) if all_warm_vp else 0.0
    # Among instances where the cache actually had violations
    active_warm_vp = [ws.get("warm_violation_prob", 0.0) for ws in warm_active]
    avg_warm_vp_active = float(np.mean(active_warm_vp)) if active_warm_vp else 0.0

    # New violations found during search, excluding sequences already in the
    # warm-start cache (new_violations_found only counts seqs found this instance)
    total_new_violations = sum(
        ic.get("new_violations_found", 0)
        for ic in complete_events.values()
        if not ic.get("warm_skip", False)
    )
    instances_with_new_violations = sum(
        1
        for ic in complete_events.values()
        if not ic.get("warm_skip", False) and ic.get("new_violations_found", 0) > 0
    )

    # Per-transition violation-prob growth (new prob only, net of warm start)
    viol_deltas = []
    for instance, ic in complete_events.items():
        ws = warm_by_instance.get(instance)
        if ws is None or ic.get("warm_skip", False):
            continue
        n_trans = ic.get("total_transitions", 0)
        if n_trans == 0:
            continue
        warm_vp = ws.get("warm_violation_prob", 0.0)
        final_vp = ic.get("final_violation_prob", 0.0)
        # Net new violation prob accumulated during this instance's search
        new_vp = max(final_vp - warm_vp, 0.0)
        viol_deltas.append(new_vp / n_trans)

    avg_viol_delta_per_transition = float(np.mean(viol_deltas)) if viol_deltas else 0.0

    # ── Semantic-check cost (gross) ───────────────────────────────────────────
    # Per instance: prefer the instance_complete grand total (cached runs log the
    # warm-start + main-loop sum); otherwise sum the per-transition counts (cold
    # frontier logs no such grand total). This is the number of sequences sent to
    # the semantic constraint before caching/dedup reduce it — the "everything
    # checked before optimizations" cost.
    total_semantic_attempted = 0
    for inst_entries in all_data.values():
        ic = next(
            (e for e in inst_entries if e.get("event") == "instance_complete"), None
        )
        if ic is not None and "total_semantic_checks" in ic:
            total_semantic_attempted += ic.get("total_semantic_checks", 0)
        else:
            trans = [e for e in inst_entries if "transition" in e]
            total_semantic_attempted += sum(
                e.get("num_semantic_checks", 0) for e in trans
            )

    avg_semantic_attempted = (
        total_semantic_attempted / num_instances if num_instances else 0.0
    )

    # Create summary dictionary
    summary = {
        "num_instances": num_instances,
        "num_instances_with_data": num_with_data,
        "num_instances_no_data": num_no_data,
        "num_instances_determined": num_determined,
        "avg_transitions_to_completion": float(agg_func(transitions)),
        "avg_transitions_to_threshold": float(agg_func(transitions_to_threshold)),
        # Forward passes = sum of batch_size over rounds. THE cost measure to
        # compare across expand_batch_size; rounds are not commensurable.
        "avg_forward_passes_to_completion": float(agg_func(forward_passes)),
        "avg_forward_passes_to_threshold": float(agg_func(fps_to_threshold)),
        "total_forward_passes": int(sum(forward_passes)),
        "avg_batch_size": (
            float(sum(forward_passes) / sum(transitions)) if sum(transitions) else 1.0
        ),
        "avg_ub": float(agg_func(final_ub)),
        "avg_CVP": float(agg_func(1 - np.array(final_ub))) if final_ub else 0.0,
        "worst_CVP": float(max(1 - np.array(final_lb))) if final_lb else 0.0,
        "avg_lb": float(agg_func(final_lb)),
        "avg_ub_minus_lb": float(agg_func(final_ub_minus_lb)),
        "max_transitions": int(max(transitions)) if transitions else 0,
        "min_transitions": int(min(transitions)) if transitions else 0,
        "constraint_threshold": threshold,
        "num_constraint_satisfied": num_satisfied,
        "num_constraint_unsatisfied": num_unsatisfied,
        "pct_constraint_satisfied": float(pct_satisfied),
        "pct_constraint_unsatisfied": float(pct_unsatisfied),
        # three-way verdict (safe / risky / open) over determined instances
        "num_safe": num_safe,
        "num_risky": num_risky,
        "num_open": num_open,
        "pct_safe": float(pct_safe),
        "pct_risky": float(pct_risky),
        "pct_open": float(pct_open),
        # warm-start fields (present regardless; zeros when no warm start)
        "warm_start_enabled": has_warm_start,
        "n_instances_warm_cache_had_violations": n_warm_active,
        "n_instances_warm_skip": n_warm_skip,
        "n_instances_actually_ran": n_actually_ran,
        "avg_warm_violation_prob_all": avg_warm_vp_all,
        "avg_warm_violation_prob_active": avg_warm_vp_active,
        "total_new_violations_found": total_new_violations,
        "instances_with_new_violations": instances_with_new_violations,
        "avg_new_violation_prob_per_transition": avg_viol_delta_per_transition,
        # semantic-check cost (gross: sequences sent to the constraint before
        # caching/dedup optimizations)
        "total_semantic_checks": total_semantic_attempted,
        "avg_semantic_checks_per_instance": float(avg_semantic_attempted),
    }

    # Print summary
    print(f"\n{'=' * 50}")
    print(
        f"Summary for {num_instances} instances ({num_no_data} had no instance_complete verdict)"
    )
    print(f"{'=' * 50}")
    print(
        f"{metric_label} transitions to completion: {summary['avg_transitions_to_completion']:.2f}"
    )
    print(
        f"{metric_label} transitions to reach threshold {threshold}: {summary['avg_transitions_to_threshold']:.2f}"
    )
    print(
        f"{metric_label} FORWARD PASSES to completion: "
        f"{summary['avg_forward_passes_to_completion']:.2f}"
    )
    print(
        f"{metric_label} FORWARD PASSES to reach threshold {threshold}: "
        f"{summary['avg_forward_passes_to_threshold']:.2f}"
    )
    print(f"Avg batch size (fp per round): {summary['avg_batch_size']:.2f}")
    print(
        f"Semantic checks (gross, pre-cache):  "
        f"{summary['avg_semantic_checks_per_instance']:.2f}/instance, "
        f"{summary['total_semantic_checks']} total"
    )
    print(f"{metric_label} UB :                     {summary['avg_ub']:.6f}")
    print(f"{metric_label} LB :                     {summary['avg_lb']:.6f}")
    print(f"{metric_label} UB-LB:                   {summary['avg_ub_minus_lb']:.6f}")
    print(f"{metric_label} CVP (1-UB):              {summary['avg_CVP']:.6f}")
    print(f"Worst-case CVP:              {summary['worst_CVP']:.6f}")
    print(f"Max transitions:             {summary['max_transitions']}")
    print(f"Min transitions:             {summary['min_transitions']}")
    print(
        f"\nConstraint Satisfaction (threshold={threshold}, over {num_determined} determined instances):"
    )
    print(
        f"  Satisfied (UB >= {threshold}):     {num_satisfied} ({pct_satisfied:.1f}%)"
    )
    print(
        f"  Unsatisfied (UB < {threshold}):    {num_unsatisfied} ({pct_unsatisfied:.1f}%)"
    )
    print(
        f"\nThree-way verdict (over {num_determined} determined instances):"
    )
    print(f"  Safe  (LB >  {threshold}):          {num_safe} ({pct_safe:.1f}%)")
    print(f"  Risky (UB <= {threshold}):          {num_risky} ({pct_risky:.1f}%)")
    print(f"  Open  (otherwise):             {num_open} ({pct_open:.1f}%)")
    if num_no_data > 0:
        print(f"  No verdict (instance_complete missing): {num_no_data}")

    if has_warm_start:
        print(f"\n{'─' * 50}")
        print(f"Warm-Start Cache Statistics")
        print(f"{'─' * 50}")
        print(
            f"  Instances where cache held violations at start: "
            f"{n_warm_active} / {num_instances}"
        )
        print(
            f"  Instances actually ran search:                  "
            f"{n_actually_ran} / {len(complete_events)}"
        )
        print(
            f"  Instances early-skipped (warm viol prob >= threshold): "
            f"{n_warm_skip}"
        )
        print(
            f"  {metric_label} warm violation prob (all instances):    "
            f"{avg_warm_vp_all:.6f}"
        )
        if n_warm_active > 0:
            print(
                f"  {metric_label} warm violation prob (cache-active only): "
                f"{avg_warm_vp_active:.6f}"
            )
        print(
            f"  New violations found during search (excl. cached): "
            f"{total_new_violations} across {instances_with_new_violations} instance(s)"
        )
        print(
            f"  {metric_label} new violation prob gained per transition: "
            f"{avg_viol_delta_per_transition:.2e}"
        )

    print(f"{'=' * 50}\n")

    # Frontier cache stats (only when a cache was persisted for this run)
    cache_dir = run_logs_folder / "frontier_cache"
    if cache_dir.is_dir():
        try:
            from beaver.utils.frontier_cache_stats import get_cache_stats

            cache_summary = get_cache_stats(run_logs_folder, plot=True)
            summary["frontier_cache"] = cache_summary
        except Exception as e:
            print(f"  (frontier cache stats skipped: {e})")

    # Save summary to JSON
    summary_path = run_logs_folder / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)
    print(f"Summary saved to: {summary_path}\n")

    return summary


def create_time_plots(all_data, all_profile_data, run_logs_folder: Path):

    import seaborn as sns
    # Set seaborn style for better-looking plots
    sns.set_theme(style="whitegrid", palette="muted")
    sns.set_context("notebook", font_scale=1.1)

    """
    Create plots for UB and LB vs time using profiling data.
    Each plot shows individual instance traces (low alpha) and average across all instances.
    """
    # Create plots directory
    plots_dir = run_logs_folder / "plots"
    plots_dir.mkdir(exist_ok=True)

    # Collect data for all instances
    instances_data = {}
    max_time = 0.0

    for instance_id, entries in all_data.items():
        # Filter to only transition entries
        transition_entries = [e for e in entries if "transition" in e]

        if not transition_entries:
            continue

        # Get corresponding profile data
        profile_file = f"{instance_id}.profile.json"
        if profile_file not in all_profile_data:
            continue

        # CachedFrontierVerifier writes one extra warm_start profiling entry
        # per instance ahead of the per-transition ones; only the "transition"
        # entries line up 1:1 with transition_entries, so filter to those
        # (older logs without an "event" tag have only transition-shaped
        # entries anyway, so this is a no-op for them).
        profile_entries = [
            e for e in all_profile_data[profile_file]
            if e.get("event", "transition") == "transition"
        ]

        # Ensure we have the same number of transition and profile entries
        if len(transition_entries) != len(profile_entries):
            print(f"Warning: Mismatch in entry count for instance {instance_id}")
            continue

        # Extract metrics and time for each transition
        ub_values = []
        lb_values = []
        time_values = []
        cumulative_time = 0.0

        for transition_entry, profile_entry in zip(transition_entries, profile_entries):
            incomplete = transition_entry.get("incomplete prob sum", 0.0)
            complete = transition_entry.get("complete prob sum", 0.0)

            ub = incomplete + complete
            lb = complete

            # Get time from profile entry
            total_time = profile_entry.get("total_time", 0.0)
            cumulative_time += total_time

            ub_values.append(ub)
            lb_values.append(lb)
            time_values.append(cumulative_time)

        max_time = max(max_time, cumulative_time)

        instances_data[instance_id] = {
            "time": time_values,
            "UB": ub_values,
            "LB": lb_values,
        }

    if not instances_data:
        print("No profiling data available for time-based plots")
        return

    # Create combined UB and LB vs time plot
    fig, ax = plt.subplots(figsize=(12, 7))

    # Define colors from seaborn palette
    lb_color = sns.color_palette("muted")[0]  # Blue
    ub_color = sns.color_palette("muted")[2]  # Green

    # Plot individual instances for LB with low alpha
    for instance_id, data in instances_data.items():
        ax.plot(
            data["time"],
            data["LB"],
            marker="",
            alpha=0.15,
            linewidth=1.0,
            color=lb_color,
        )

    # Plot individual instances for UB with low alpha
    for instance_id, data in instances_data.items():
        ax.plot(
            data["time"],
            data["UB"],
            marker="",
            alpha=0.15,
            linewidth=1.0,
            color=ub_color,
        )

    # Calculate and plot average across instances at regular time intervals
    # Sample at regular intervals to get average
    num_samples = 100
    time_samples = np.linspace(0, max_time, num_samples)
    avg_ub_samples = []
    avg_lb_samples = []

    for t in time_samples:
        ub_at_t = []
        lb_at_t = []

        for instance_id, data in instances_data.items():
            # Find the value at time t (use last value before or at t)
            idx = 0
            for i, time_val in enumerate(data["time"]):
                if time_val <= t:
                    idx = i
                else:
                    break

            ub_at_t.append(data["UB"][idx])
            lb_at_t.append(data["LB"][idx])

        avg_ub_samples.append(np.mean(ub_at_t))
        avg_lb_samples.append(np.mean(lb_at_t))

    # Define colors from seaborn palette
    lb_color = sns.color_palette("muted")[0]  # Blue
    ub_color = sns.color_palette("muted")[2]  # Green

    # Plot average LB
    ax.plot(
        time_samples,
        avg_lb_samples,
        marker="",
        alpha=1.0,
        linewidth=3.0,
        color=lb_color,
        label="Average LB",
    )

    # Plot average UB
    ax.plot(
        time_samples,
        avg_ub_samples,
        marker="",
        alpha=1.0,
        linewidth=3.0,
        color=ub_color,
        label="Average UB",
    )

    ax.set_xlabel("Time (seconds)", fontsize=13)
    ax.set_ylabel("Probability", fontsize=13)
    ax.set_title(
        "Upper Bound and Lower Bound vs Time", fontsize=15, fontweight="bold", pad=20
    )
    ax.legend(fontsize=11, frameon=True, shadow=True)
    ax.grid(True, alpha=0.25, linestyle="--")

    plt.tight_layout()

    # Save plot
    output_path = plots_dir / "UB_LB_vs_time_plot.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to: {output_path}")
    plt.close()


def create_plots(all_data, run_logs_folder: Path):


    import seaborn as sns
    # Set seaborn style for better-looking plots
    sns.set_theme(style="whitegrid", palette="muted")
    sns.set_context("notebook", font_scale=1.1)
    """
    Create plots for UB, LB, and UB-LB vs transitions.
    Each plot shows individual instance traces (low alpha) and average across all instances.
    Each metric is saved as a separate image in a 'plots' subfolder.
    """
    # Create plots directory
    plots_dir = run_logs_folder / "plots"
    plots_dir.mkdir(exist_ok=True)

    # Collect data for all instances
    instances_data = {}
    max_transitions = 0

    for instance_id, entries in all_data.items():
        # Filter to only transition entries
        transition_entries = [e for e in entries if "transition" in e]

        if not transition_entries:
            continue

        num_transitions = transition_entries[-1][
            "transition"
        ]  # Get total transitions for this instance
        max_transitions = max(max_transitions, num_transitions)

        # Extract metrics for each transition
        ub_values = []
        lb_values = []
        ub_minus_lb_values = []
        pruned_values = []

        for entry in transition_entries:
            incomplete = entry.get("incomplete prob sum", 0.0)
            complete = entry.get("complete prob sum", 0.0)
            pruned = entry.get("pruned prob sum", 0.0)

            ub = incomplete + complete
            lb = complete
            ub_minus_lb = incomplete

            ub_values.append(ub)
            lb_values.append(lb)
            ub_minus_lb_values.append(ub_minus_lb)
            pruned_values.append(pruned)

        instances_data[instance_id] = {
            "transitions": list(range(1, num_transitions + 1)),
            "UB": ub_values,
            "LB": lb_values,
            "UB-LB": ub_minus_lb_values,
            "Pruned": pruned_values,
        }

    # Calculate average at each transition point
    avg_ub = []
    avg_lb = []
    avg_ub_minus_lb = []
    avg_pruned = []

    for t in range(1, max_transitions + 1):
        ub_at_t = []
        lb_at_t = []
        ub_minus_lb_at_t = []
        pruned_at_t = []

        for instance_id, data in instances_data.items():
            if t <= len(data["transitions"]):
                # Use value at transition t
                ub_at_t.append(data["UB"][t - 1])
                lb_at_t.append(data["LB"][t - 1])
                ub_minus_lb_at_t.append(data["UB-LB"][t - 1])
                pruned_at_t.append(data["Pruned"][t - 1])
            else:
                # Use last available value if instance ended before this transition
                ub_at_t.append(data["UB"][-1])
                lb_at_t.append(data["LB"][-1])
                ub_minus_lb_at_t.append(data["UB-LB"][-1])
                pruned_at_t.append(data["Pruned"][-1])

        avg_ub.append(np.mean(ub_at_t))
        avg_lb.append(np.mean(lb_at_t))
        avg_ub_minus_lb.append(np.mean(ub_minus_lb_at_t))
        avg_pruned.append(np.mean(pruned_at_t))

    # Create separate plots for each metric
    metrics = ["UB", "LB", "UB-LB", "Pruned"]
    avg_data = {
        "UB": avg_ub,
        "LB": avg_lb,
        "UB-LB": avg_ub_minus_lb,
        "Pruned": avg_pruned,
    }

    for metric in metrics:
        fig, ax = plt.subplots(figsize=(12, 7))

        # Get color from seaborn palette
        metric_color = sns.color_palette("muted")[3]  # Purple/red for individual
        avg_color = sns.color_palette("deep")[3]  # Deeper red for average

        # Plot individual instances with low alpha
        for instance_id, data in instances_data.items():
            ax.plot(
                data["transitions"],
                data[metric],
                marker="",
                alpha=0.15,
                linewidth=1.0,
                color=metric_color,
            )

        # Plot average with high alpha
        ax.plot(
            range(1, max_transitions + 1),
            avg_data[metric],
            marker="",
            alpha=1.0,
            linewidth=3.0,
            color=avg_color,
            label="Average",
        )

        ax.set_xlabel("Transitions", fontsize=13)
        ax.set_ylabel(metric, fontsize=13)
        ax.set_title(f"{metric} vs Transitions", fontsize=15, fontweight="bold", pad=20)
        ax.legend(fontsize=11, frameon=True, shadow=True)
        ax.grid(True, alpha=0.25, linestyle="--")

        plt.tight_layout()

        # Save individual plot
        output_path = plots_dir / f"{metric.replace('-', '_')}_plot.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to: {output_path}")
        plt.close()

    # Create combined UB and LB plot
    fig, ax = plt.subplots(figsize=(12, 7))

    # Define colors from seaborn palette
    lb_color = sns.color_palette("muted")[0]  # Blue
    ub_color = sns.color_palette("muted")[2]  # Green

    # Plot individual instances for LB with low alpha
    for instance_id, data in instances_data.items():
        ax.plot(
            data["transitions"],
            data["LB"],
            marker="",
            alpha=0.15,
            linewidth=1.0,
            color=lb_color,
        )

    # Plot individual instances for UB with low alpha
    for instance_id, data in instances_data.items():
        ax.plot(
            data["transitions"],
            data["UB"],
            marker="",
            alpha=0.15,
            linewidth=1.0,
            color=ub_color,
        )

    # Plot average LB with high alpha
    ax.plot(
        range(1, max_transitions + 1),
        avg_data["LB"],
        marker="",
        alpha=1.0,
        linewidth=3.0,
        color=lb_color,
        label="Average LB",
    )

    # Plot average UB with high alpha
    ax.plot(
        range(1, max_transitions + 1),
        avg_data["UB"],
        marker="",
        alpha=1.0,
        linewidth=3.0,
        color=ub_color,
        label="Average UB",
    )

    ax.set_xlabel("Transitions", fontsize=13)
    ax.set_ylabel("Probability", fontsize=13)
    ax.set_title(
        "Upper Bound and Lower Bound vs Transitions",
        fontsize=15,
        fontweight="bold",
        pad=20,
    )
    ax.legend(fontsize=11, frameon=True, shadow=True)
    ax.grid(True, alpha=0.25, linestyle="--")

    plt.tight_layout()

    # Save combined plot
    output_path = plots_dir / "UB_LB_combined_plot.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to: {output_path}")
    plt.close()

    print(f"\nAll plots saved to: {plots_dir}/")

    return plots_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Plot and summarize verification logs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example: python plot_logs.py src/logging/run_logs/logs_20251027213300 --threshold 0.85",
    )
    parser.add_argument("run_logs_folder", type=str, help="Path to the run logs folder")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.9,
        help="Threshold for constraint satisfaction (default: 0.9). Instances with final UB >= threshold are considered constraint-satisfied.",
    )
    parser.add_argument("--plots", type=bool, default=False)

    args = parser.parse_args()
    run_logs_folder = Path(args.run_logs_folder)
    threshold = args.threshold

    if not os.path.exists(run_logs_folder):
        print(f"Error: Folder {run_logs_folder} does not exist")
        import sys

        sys.exit(1)

    if not (0.0 <= threshold <= 1.0):
        print(f"Error: Threshold must be between 0.0 and 1.0, got {threshold}")
        import sys

        sys.exit(1)

    all_data = get_log_data(run_logs_folder)
    # all_profile_data = get_profile_data(run_logs_folder)

    # Print run_args.json if it exists
    run_args_path = run_logs_folder / "run_args.json"
    if run_args_path.exists():
        print(f"\n{'=' * 50}")
        print("Run Arguments (run_args.json):")
        print(f"{'=' * 50}")
        with open(run_args_path, "r") as f:
            run_args = json.load(f)
            print(json.dumps(run_args, indent=4))
        print(f"{'=' * 50}\n")
    else:
        print(f"\nWarning: run_args.json not found at {run_args_path}\n")

    summarize_log_data(all_data, run_logs_folder, threshold=threshold)
    summarize_profile_data(run_logs_folder)

    if args.plots:
        create_plots(all_data, run_logs_folder)
        # create_time_plots(all_data, all_profile_data, run_logs_folder)
