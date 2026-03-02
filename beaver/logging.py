import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set seaborn style for better-looking plots
sns.set_theme(style="whitegrid", palette="muted")
sns.set_context("notebook", font_scale=1.1)


def read_log_file(file_path: str):
    """Read a log file and parse JSON entries."""
    with open(file_path, "r") as f:
        content = f.read()
        # Split by newline-separated JSON objects
        parts = content.replace("}\n{", "}###SPLIT###{").split("###SPLIT###")
        entries = [json.loads(part) for part in parts]
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

        print("Average Timing Profiles over transitions (in seconds):")
        print(json.dumps(avg_profiles, indent=4))
        print("Average Timing Profiles over tasks (in seconds):")
        print(json.dumps(file_avg_profiles, indent=4))
        print("Max Timing Profiles over tasks (in seconds):")
        print(json.dumps(file_max_profiles, indent=4))

        return


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
    all_data, run_logs_folder: Path, use_median: bool = False, threshold: float = 0.9
):
    num_instances = len(all_data)
    # Filter to only transition entries
    entries = [
        [e for e in entries if "transition" in e] for entries in all_data.values()
    ]
    entries = [
        e for e in entries if len(e) > 0
    ]  # Filter out instances with no transitions]
    transitions = [e[-1]["transition"] for e in entries]

    # Collect final values for each instance
    final_incomplete = []
    final_complete = []
    final_ub = []
    final_lb = []
    final_ub_minus_lb = []

    for entry_list in entries:
        if entry_list:
            final_entry = entry_list[-1]
            incomplete = final_entry.get("incomplete prob sum", 0.0)
            complete = final_entry.get("complete prob sum", 0.0)

            final_incomplete.append(incomplete)
            final_complete.append(complete)
            final_ub.append(incomplete + complete)  # UB = incomplete + complete
            final_lb.append(complete)  # LB = complete
            final_ub_minus_lb.append(incomplete)  # UB - LB = incomplete

    # Calculate constraint satisfaction metrics
    num_satisfied = sum(1 for ub in final_ub if ub >= threshold)
    num_unsatisfied = sum(1 for ub in final_ub if ub < threshold)
    pct_satisfied = (num_satisfied / num_instances * 100) if num_instances > 0 else 0.0
    pct_unsatisfied = (
        (num_unsatisfied / num_instances * 100) if num_instances > 0 else 0.0
    )

    # Choose aggregation function
    agg_func = np.median if use_median else np.mean
    metric_label = "Median" if use_median else "Avg"

    # Create summary dictionary
    summary = {
        "num_instances": num_instances,
        "avg_incomplete": float(agg_func(final_incomplete)),
        "avg_complete": float(agg_func(final_complete)),
        "avg_transitions_to_completion": float(agg_func(transitions)),
        "avg_ub": float(agg_func(final_ub)),
        "avg_lb": float(agg_func(final_lb)),
        "avg_ub_minus_lb": float(agg_func(final_ub_minus_lb)),
        "max_transitions": int(max(transitions)) if transitions else 0,
        "min_transitions": int(min(transitions)) if transitions else 0,
        "constraint_threshold": threshold,
        "num_constraint_satisfied": num_satisfied,
        "num_constraint_unsatisfied": num_unsatisfied,
        "pct_constraint_satisfied": float(pct_satisfied),
        "pct_constraint_unsatisfied": float(pct_unsatisfied),
    }

    # Print summary
    print(f"\n{'=' * 50}")
    print(f"Summary for {num_instances} instances")
    print(f"{'=' * 50}")
    print(f"{metric_label} incomplete:              {summary['avg_incomplete']:.6f}")
    print(f"{metric_label} complete:                {summary['avg_complete']:.6f}")
    print(
        f"{metric_label} transitions to completion: {summary['avg_transitions_to_completion']:.2f}"
    )
    print(f"{metric_label} UB (incomplete + complete): {summary['avg_ub']:.6f}")
    print(f"{metric_label} LB (complete):            {summary['avg_lb']:.6f}")
    print(f"{metric_label} UB-LB:                   {summary['avg_ub_minus_lb']:.6f}")
    print(f"Max transitions:             {summary['max_transitions']}")
    print(f"Min transitions:             {summary['min_transitions']}")
    print(f"\nConstraint Satisfaction (threshold={threshold}):")
    print(
        f"  Satisfied (UB >= {threshold}):     {num_satisfied} ({pct_satisfied:.1f}%)"
    )
    print(
        f"  Unsatisfied (UB < {threshold}):    {num_unsatisfied} ({pct_unsatisfied:.1f}%)"
    )
    print(f"{'=' * 50}\n")

    # Save summary to JSON
    summary_path = run_logs_folder / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)
    print(f"Summary saved to: {summary_path}\n")

    return summary


def create_time_plots(all_data, all_profile_data, run_logs_folder: Path):
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

        profile_entries = all_profile_data[profile_file]

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
    # create_plots(all_data, run_logs_folder)
    # create_time_plots(all_data, all_profile_data, run_logs_folder)
    summarize_profile_data(run_logs_folder)
