import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plot_logs import (
    create_plots,
    get_log_data,
    get_profile_data,
    read_log_file,
    summarize_log_data,
)


def format_table(headers, rows):
    """Simple table formatter without external dependencies."""
    # Calculate column widths
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(cell)))

    # Create separator
    separator = "+" + "+".join(["-" * (w + 2) for w in col_widths]) + "+"

    # Format header
    header_row = (
        "|" + "|".join([f" {h:{col_widths[i]}} " for i, h in enumerate(headers)]) + "|"
    )

    # Format rows
    formatted_rows = []
    for row in rows:
        formatted_row = (
            "|"
            + "|".join([f" {str(cell):{col_widths[i]}} " for i, cell in enumerate(row)])
            + "|"
        )
        formatted_rows.append(formatted_row)

    # Combine
    result = [separator, header_row, separator]
    result.extend(formatted_rows)
    result.append(separator)

    return "\n".join(result)


def get_profile_stats(run_logs_folder: Path, use_median: bool = False):
    """Get average/median timing profiles and return them as a dictionary."""
    profile_json_files = [
        f for f in os.listdir(run_logs_folder) if f.endswith(".profile.json")
    ]
    if len(profile_json_files) == 0:
        return None

    all_profiles = {}
    for file in profile_json_files:
        file_path = os.path.join(run_logs_folder, file)
        entries = read_log_file(file_path)
        for entry in entries:
            for key, value in entry.items():
                if key not in all_profiles:
                    all_profiles[key] = []
                all_profiles[key].append(value)

    agg_func = np.median if use_median else np.mean
    avg_profiles = {}
    total_time_per_idx = {}

    for key, values in all_profiles.items():
        avg_profiles[key] = agg_func(values)
        # Sum all times for each idx (across all instances)
        total_time_per_idx[key] = np.sum(values)

    # Calculate average/median time across all transition indices
    # This is: sum of all times for all indices / number of indices
    if total_time_per_idx:
        avg_time_all_transitions = agg_func(list(total_time_per_idx.values()))
    else:
        avg_time_all_transitions = 0.0

    return avg_profiles, total_time_per_idx, avg_time_all_transitions


def print_profile_comparison(profile_stats_list, log_names, use_median: bool = False):
    """Print a comparison table of profile statistics for multiple logs."""
    if all(ps is None for ps in profile_stats_list):
        print("\nNo profiling data available for any log.\n")
        return

    metric_label = "Median" if use_median else "Average"
    print("\n" + "=" * 100)
    print(f"PROFILING COMPARISON ({metric_label} Time per Transition Step in seconds)")
    print("=" * 100)

    # Extract profile data for each log
    all_avg_profiles = []
    all_avg_times = []

    for i, ps in enumerate(profile_stats_list):
        if ps is None:
            all_avg_profiles.append({})
            all_avg_times.append(0.0)
        else:
            avg_profiles, total_time, avg_time = ps
            all_avg_profiles.append(avg_profiles)
            all_avg_times.append(avg_time)

    # Get all unique timing categories
    all_keys = sorted(set().union(*[set(ap.keys()) for ap in all_avg_profiles]))

    if all_keys:
        # Create table data for per-category timing
        table_data = []
        for key in all_keys:
            row = [key]
            for avg_profiles in all_avg_profiles:
                val = avg_profiles.get(key, 0.0)
                row.append(f"{val:.6f}" if val > 0 else "N/A")
            table_data.append(row)

        headers = ["Timing Category"] + log_names
        print(format_table(headers, table_data))

    # Print average/median time for all transitions
    metric_label = "Median" if use_median else "Average"
    print("\n" + "=" * 100)
    print(f"{metric_label.upper()} TIME FOR ALL TRANSITIONS")
    print(f"(Sum of all times for each category, {metric_label.lower()} across categories)")
    print("=" * 100)

    table_data = []
    for i, log_name in enumerate(log_names):
        avg_time = all_avg_times[i]
        table_data.append([log_name, f"{avg_time:.6f}" if avg_time > 0 else "N/A"])

    print(format_table(["Log", f"{metric_label} Time (seconds)"], table_data))
    print()


def print_comparison_table(summaries, log_names, use_median: bool = False):
    """Print a comparison table for multiple logs."""
    metric_label = "Median" if use_median else "Avg"
    print("\n" + "=" * 100)
    print("LOG COMPARISON TABLE")
    print("=" * 100)

    # Define metrics to compare
    metrics = [
        ("Num Instances", "num_instances", "d"),
        (f"{metric_label} Incomplete", "avg_incomplete", ".6f"),
        (f"{metric_label} Complete", "avg_complete", ".6f"),
        (f"{metric_label} Transitions", "avg_transitions_to_completion", ".2f"),
        (f"{metric_label} UB", "avg_ub", ".6f"),
        (f"{metric_label} LB", "avg_lb", ".6f"),
        (f"{metric_label} UB-LB", "avg_ub_minus_lb", ".6f"),
        ("Max Transitions", "max_transitions", "d"),
        ("Min Transitions", "min_transitions", "d"),
    ]

    table_data = []
    for metric_name, metric_key, fmt in metrics:
        row = [metric_name]
        for summary in summaries:
            val = summary[metric_key]
            if fmt == "d":
                row.append(str(val))
            else:
                row.append(f"{val:{fmt}}")
        table_data.append(row)

    headers = ["Metric"] + log_names
    print(format_table(headers, table_data))

    # Add constraint satisfaction table if the data is available
    if summaries and "constraint_threshold" in summaries[0]:
        threshold = summaries[0]["constraint_threshold"]
        print(f"\n{'=' * 100}")
        print(f"CONSTRAINT SATISFACTION (threshold={threshold})")
        print("=" * 100)

        constraint_metrics = [
            (f"Satisfied (UB >= {threshold})", "num_constraint_satisfied", "d"),
            (f"Unsatisfied (UB < {threshold})", "num_constraint_unsatisfied", "d"),
            (f"% Satisfied", "pct_constraint_satisfied", ".1f"),
            (f"% Unsatisfied", "pct_constraint_unsatisfied", ".1f"),
        ]

        constraint_table_data = []
        for metric_name, metric_key, fmt in constraint_metrics:
            row = [metric_name]
            for summary in summaries:
                val = summary.get(metric_key, 0)
                if fmt == "d":
                    row.append(str(val))
                elif fmt == ".1f":
                    row.append(f"{val:.1f}%")
                else:
                    row.append(f"{val:{fmt}}")
            constraint_table_data.append(row)

        print(format_table(["Metric"] + log_names, constraint_table_data))

    print()


def create_combined_time_comparison_plots(
    all_data_list, all_profile_data_list, output_folder: Path, log_names, use_median: bool = False, max_time_limit: float = None
):
    """
    Create combined comparison plots for UB and LB vs time from multiple logs.
    Shows averages/medians from all logs on the same plot using profiling data.
    """
    plots_dir = output_folder / "comparison_plots"
    plots_dir.mkdir(exist_ok=True)

    agg_func = np.median if use_median else np.mean
    metric_label = "Median" if use_median else "Avg"

    def compute_time_averages(all_data, all_profile_data):
        """Compute average/median UB and LB across all instances over time."""
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

            profile_entries = all_profile_data[profile_file]

            # Ensure we have the same number of transition and profile entries
            if len(transition_entries) != len(profile_entries):
                continue

            # Extract metrics and time for each transition
            ub_values = []
            lb_values = []
            time_values = []
            cumulative_time = 0.0

            for transition_entry, profile_entry in zip(
                transition_entries, profile_entries
            ):
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
            return None, None, None

        # Apply max_time_limit if specified
        if max_time_limit is not None and max_time_limit > 0:
            max_time = min(max_time, max_time_limit)

        # Calculate and plot average across instances at regular time intervals
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

            avg_ub_samples.append(agg_func(ub_at_t))
            avg_lb_samples.append(agg_func(lb_at_t))

        return list(avg_ub_samples), list(avg_lb_samples), list(time_samples)

    # Compute averages for all logs
    all_time_averages = []
    max_time_global = 0.0

    for all_data, all_profile_data in zip(all_data_list, all_profile_data_list):
        if all_profile_data is None or len(all_profile_data) == 0:
            all_time_averages.append(None)
            continue

        avg_ub, avg_lb, time_samples = compute_time_averages(all_data, all_profile_data)

        if avg_ub is None:
            all_time_averages.append(None)
            continue

        all_time_averages.append((avg_ub, avg_lb, time_samples))
        if time_samples and len(time_samples) > 0:
            max_time_global = max(max_time_global, time_samples[-1])

    # Check if we have any valid data
    if all(avg is None for avg in all_time_averages):
        print("No profiling data available for time-based comparison plots")
        return

    # Create combined UB and LB vs time plot
    fig, ax = plt.subplots(figsize=(14, 8))

    # Define colors for different logs
    colors = [
        "blue",
        "red",
        "green",
        "orange",
        "purple",
        "brown",
        "pink",
        "gray",
        "olive",
        "cyan",
    ]

    # Plot averages for all logs
    for i, (log_name, avg_data) in enumerate(zip(log_names, all_time_averages)):
        if avg_data is None:
            continue

        avg_ub, avg_lb, time_samples = avg_data
        color = colors[i % len(colors)]

        ax.plot(
            time_samples,
            avg_ub,
            linewidth=2.5,
            color=color,
            label=f"{log_name} - UB {metric_label}",
            alpha=0.8,
        )
        ax.plot(
            time_samples,
            avg_lb,
            linewidth=2.5,
            color=color,
            label=f"{log_name} - LB {metric_label}",
            alpha=0.8,
            linestyle="--",
        )

    ax.set_xlabel("Time (seconds)", fontsize=12)
    ax.set_ylabel("Probability", fontsize=12)
    title = f"Combined UB and LB vs Time Comparison: {', '.join(log_names)}"
    if max_time_limit is not None:
        title += f" (clipped at {max_time_limit}s)"
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)

    # Set x-axis limit if max_time_limit is specified
    if max_time_limit is not None:
        ax.set_xlim(0, max_time_limit)

    plt.tight_layout()

    output_path = plots_dir / "combined_UB_LB_vs_time_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Combined time comparison plot saved to: {output_path}")
    plt.close()


def create_combined_comparison_plots(all_data_list, output_folder: Path, log_names, use_median: bool = False):
    """
    Create combined comparison plots for UB and LB from multiple logs.
    Shows averages/medians from all logs on the same plot.
    """
    plots_dir = output_folder / "comparison_plots"
    plots_dir.mkdir(exist_ok=True)

    agg_func = np.median if use_median else np.mean
    metric_label = "Median" if use_median else "Avg"

    def compute_averages(all_data):
        """Compute average/median UB and LB across all instances."""
        instances_data = {}
        max_transitions = 0

        for instance_id, entries in all_data.items():
            transition_entries = [e for e in entries if "transition" in e]
            if not transition_entries:
                continue

            num_transitions = len(transition_entries)
            max_transitions = max(max_transitions, num_transitions)

            ub_values = []
            lb_values = []

            for entry in transition_entries:
                incomplete = entry.get("incomplete prob sum", 0.0)
                complete = entry.get("complete prob sum", 0.0)
                ub = incomplete + complete
                lb = complete
                ub_values.append(ub)
                lb_values.append(lb)

            instances_data[instance_id] = {
                "transitions": list(range(1, num_transitions + 1)),
                "UB": ub_values,
                "LB": lb_values,
            }

        # Calculate averages
        avg_ub = []
        avg_lb = []

        for t in range(1, max_transitions + 1):
            ub_at_t = []
            lb_at_t = []

            for instance_id, data in instances_data.items():
                if t <= len(data["transitions"]):
                    ub_at_t.append(data["UB"][t - 1])
                    lb_at_t.append(data["LB"][t - 1])
                else:
                    ub_at_t.append(data["UB"][-1])
                    lb_at_t.append(data["LB"][-1])

            avg_ub.append(agg_func(ub_at_t))
            avg_lb.append(agg_func(lb_at_t))

        return avg_ub, avg_lb, max_transitions

    # Compute averages for all logs
    all_averages = []
    max_transitions_global = 0

    for all_data in all_data_list:
        avg_ub, avg_lb, max_trans = compute_averages(all_data)
        all_averages.append((avg_ub, avg_lb))
        max_transitions_global = max(max_transitions_global, max_trans)

    # Pad shorter sequences with their final value
    for i, (avg_ub, avg_lb) in enumerate(all_averages):
        if len(avg_ub) < max_transitions_global:
            avg_ub.extend([avg_ub[-1]] * (max_transitions_global - len(avg_ub)))
            avg_lb.extend([avg_lb[-1]] * (max_transitions_global - len(avg_lb)))
        all_averages[i] = (avg_ub, avg_lb)

    # Create combined UB and LB plot
    fig, ax = plt.subplots(figsize=(14, 8))

    transitions = range(1, max_transitions_global + 1)

    # Define colors for different logs
    colors = [
        "blue",
        "red",
        "green",
        "orange",
        "purple",
        "brown",
        "pink",
        "gray",
        "olive",
        "cyan",
    ]

    # Plot averages for all logs
    for i, (log_name, (avg_ub, avg_lb)) in enumerate(zip(log_names, all_averages)):
        color = colors[i % len(colors)]

        ax.plot(
            transitions,
            avg_ub,
            marker="o",
            markersize=4,
            linewidth=2.0,
            color=color,
            label=f"{log_name} - UB {metric_label}",
            alpha=0.7,
        )
        ax.plot(
            transitions,
            avg_lb,
            marker="s",
            markersize=4,
            linewidth=2.0,
            color=color,
            label=f"{log_name} - LB {metric_label}",
            alpha=0.7,
            linestyle="--",
        )

    ax.set_xlabel("Transitions", fontsize=12)
    ax.set_ylabel("Probability", fontsize=12)
    ax.set_title(
        f"Combined UB and LB Comparison: {', '.join(log_names)}",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    output_path = plots_dir / "combined_UB_LB_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Combined comparison plot saved to: {output_path}")
    plt.close()

    return plots_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare log files from multiple runs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare two logs with default names (folder names)
  python compare_logs.py dir1/logs_20251108234428 dir2/logs_20251109005253

  # Compare with custom legend names
  python compare_logs.py dir1/logs_20251108234428 dir2/logs_20251109005253 --names "Experiment A" "Experiment B"

  # Compare three logs with custom names
  python compare_logs.py log1 log2 log3 --names "Baseline" "Method 1" "Method 2"
        """,
    )

    parser.add_argument(
        "log_folders",
        nargs="+",
        type=str,
        help="Paths to log folders to compare (at least 2 required)",
    )

    parser.add_argument(
        "--names",
        nargs="+",
        type=str,
        help="Custom names for each log folder to use in plot legends (optional, must match number of log folders)",
    )

    parser.add_argument(
        "--median",
        action="store_true",
        help="Use median instead of mean for aggregating results across instances",
    )

    parser.add_argument(
        "--individual-plots",
        action="store_true",
        help="Generate individual plots for each log folder (default: only combined plots)",
    )

    parser.add_argument(
        "--max-time",
        type=float,
        default=None,
        help="Maximum time in seconds to display in combined time comparison plot (clips extended cases)",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.9,
        help="Threshold for constraint satisfaction (default: 0.9). Instances with final UB >= threshold are considered constraint-satisfied.",
    )

    args = parser.parse_args()

    if len(args.log_folders) < 2:
        print("Error: At least 2 log folders are required for comparison")
        sys.exit(1)

    log_folders = [Path(folder) for folder in args.log_folders]
    threshold = args.threshold

    # Validate threshold
    if not (0.0 <= threshold <= 1.0):
        print(f"Error: Threshold must be between 0.0 and 1.0, got {threshold}")
        sys.exit(1)

    # Validate all folders exist
    for log_folder in log_folders:
        if not os.path.exists(log_folder):
            print(f"Error: Folder {log_folder} does not exist")
            sys.exit(1)

    # Use custom names if provided, otherwise use folder names
    if args.names:
        if len(args.names) != len(log_folders):
            print(
                f"Error: Number of custom names ({len(args.names)}) must match number of log folders ({len(log_folders)})"
            )
            sys.exit(1)
        log_names = args.names
    else:
        # Use folder names as labels
        log_names = [folder.name for folder in log_folders]

    print("\n" + "=" * 100)
    print(f"COMPARING {len(log_folders)} LOGS: {', '.join(log_names)}")
    print("=" * 100 + "\n")

    # Process all logs
    all_data_list = []
    all_profile_data_list = []
    summaries = []
    profile_stats_list = []

    for i, (log_folder, log_name) in enumerate(zip(log_folders, log_names), 1):
        if args.individual_plots:
            print(f"\n{'=' * 100}")
            print(f"PROCESSING LOG {i}: {log_name}")
            print(f"{'=' * 100}")
        all_data = get_log_data(log_folder)
        all_profile_data = get_profile_data(log_folder)
        if args.individual_plots:
            summary = summarize_log_data(
                all_data, log_folder, use_median=args.median, threshold=threshold
            )
            profile_stats = get_profile_stats(log_folder, use_median=args.median)
        else:
            # Still compute summaries and profile stats, but don't print them
            import io
            import sys

            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
            summary = summarize_log_data(
                all_data, log_folder, use_median=args.median, threshold=threshold
            )
            profile_stats = get_profile_stats(log_folder, use_median=args.median)
            sys.stdout = old_stdout

        all_data_list.append(all_data)
        all_profile_data_list.append(all_profile_data)
        summaries.append(summary)
        profile_stats_list.append(profile_stats)

    # Only print comparison tables and generate individual plots if flag is set
    if args.individual_plots:
        # Print comparison table
        print_comparison_table(summaries, log_names, use_median=args.median)

        # Print profile comparison
        print_profile_comparison(profile_stats_list, log_names, use_median=args.median)

        # Generate individual plots for each log
        for i, (log_folder, log_name, all_data) in enumerate(
            zip(log_folders, log_names, all_data_list), 1
        ):
            print(f"\n{'=' * 100}")
            print(f"GENERATING PLOTS FOR LOG {i}: {log_name}")
            print(f"{'=' * 100}")
            create_plots(all_data, log_folder)
    else:
        # Just print that we're skipping individual analysis
        print(f"\n{'=' * 100}")
        print("SKIPPING INDIVIDUAL LOG ANALYSIS")
        print("(Use --individual-plots flag to enable)")
        print(f"{'=' * 100}")

    # Create combined comparison plots
    print(f"\n{'=' * 100}")
    print("GENERATING COMBINED COMPARISON PLOTS")
    print(f"{'=' * 100}")

    # Use current directory or a common location for comparison plots
    comparison_output = Path("./")
    create_combined_comparison_plots(all_data_list, comparison_output, log_names, use_median=args.median)
    create_combined_time_comparison_plots(
        all_data_list, all_profile_data_list, comparison_output, log_names, use_median=args.median, max_time_limit=args.max_time
    )

    print("\n" + "=" * 100)
    print("COMPARISON COMPLETE")
    print("=" * 100 + "\n")
