"""Build a leaderboard CSV from batch_results summary.json files.

Usage:
    python make_leaderboard.py [RESULTS_DIR] [-o OUTPUT]

Walks every model/experiment directory under RESULTS_DIR (default:
``./batch_results``), finds the most recent ``summary.json``, and
assembles a single CSV where:

  - Each row is a model.
  - Columns are grouped per experiment, with sub-columns for the key
    metrics stored in summary.json.

The CSV uses a two-level header:  the first row is the experiment name
(spanning its metric columns) and the second row lists the metric names.
"""

import argparse
import csv
import json
from pathlib import Path


# Metrics to extract from each summary.json, in display order.
# (json_key, display_name_template)
#   {t} in the template is replaced with the threshold value.
METRICS = [
    ("num_instances", "count"),
    ("avg_incomplete", "avg_incomplete"),
    ("avg_complete", "avg_complete"),
    ("avg_transitions_to_completion", "N"),
    ("avg_ub", "avg_UB"),
    ("avg_lb", "avg_LB"),
    ("avg_ub_minus_lb", "avg_UB-LB"),
    ("num_constraint_satisfied", "satisfied(>={t})"),
    ("num_constraint_unsatisfied", "unsatisfied(<{t})"),
]


def _find_latest_summary(exp_dir: Path) -> Path | None:
    """Return the most recent summary.json under exp_dir/logs_*/.

    Scans logs_* directories in reverse-sorted order (newest first by
    timestamp suffix) so the first hit is the latest run.
    """
    if not exp_dir.is_dir():
        return None
    for logs_dir in sorted(exp_dir.iterdir(), reverse=True):
        if not logs_dir.is_dir() or not logs_dir.name.startswith("logs_"):
            continue
        summary = logs_dir / "summary.json"
        if summary.is_file():
            return summary
    return None


def _format_value(key: str, value) -> str:
    """Format a metric value for CSV output."""
    if value is None:
        return ""
    if key in ("num_instances", "num_constraint_satisfied", "num_constraint_unsatisfied"):
        return str(int(value))
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def build_leaderboard(results_dir: Path):
    """Scan results_dir and return CSV headers, data rows, and structured per-model data."""
    # Discover models and experiments
    model_dirs = sorted(
        [d for d in results_dir.iterdir() if d.is_dir()],
        key=lambda d: d.name.lower(),
    )
    if not model_dirs:
        raise SystemExit(f"No model directories found under {results_dir}")
    print(f"Model directories: {model_dirs}")

    # Collect all experiment names across all models (preserving order of
    # first appearance, but sorted for determinism).
    exp_names_set: set[str] = set()
    for model_dir in model_dirs:
        for child in model_dir.iterdir():
            if child.is_dir() and not child.name.startswith("."):
                exp_names_set.add(child.name)
    exp_names = sorted(exp_names_set)

    if not exp_names:
        raise SystemExit(f"No experiment directories found under {results_dir}")
    print(f"Experiment names: {exp_names}")

    # Determine threshold from the first available summary
    threshold = 0.9  # fallback
    for model_dir in model_dirs:
        for exp_name in exp_names:
            s = _find_latest_summary(model_dir / exp_name)
            if s:
                with open(s) as f:
                    threshold = json.load(f).get("constraint_threshold", 0.9)
                break
        else:
            continue
        break

    # Build metric display names
    metric_keys = [m[0] for m in METRICS]
    metric_names = [m[1].format(t=threshold) for m in METRICS]

    # Header row 1: "model" then experiment name spanning its metrics
    header1 = ["model"]
    for exp in exp_names:
        header1.append(exp)
        header1.extend([""] * (len(METRICS) - 1))

    # Header row 2: "" then metric names repeated per experiment
    header2 = [""]
    for _ in exp_names:
        header2.extend(metric_names)

    # Data rows + structured per-model data for the ASCII report
    rows: list[list[str]] = []
    model_exp_data: list[tuple[str, dict[str, dict[str, str]]]] = []
    for model_dir in model_dirs:
        model_name = model_dir.name.replace("--", "/")
        row = [model_name]
        exp_vals: dict[str, dict[str, str]] = {}
        for exp_name in exp_names:
            summary_path = _find_latest_summary(model_dir / exp_name)
            if summary_path is None:
                row.extend([""] * len(METRICS))
                continue
            with open(summary_path) as f:
                summary = json.load(f)
            cell_vals: dict[str, str] = {}
            for key in metric_keys:
                v = _format_value(key, summary.get(key))
                row.append(v)
                cell_vals[key] = v
            exp_vals[exp_name] = cell_vals
        rows.append(row)
        model_exp_data.append((model_name, exp_vals))

    return header1, header2, rows, exp_names, metric_keys, metric_names, model_exp_data


def main():
    parser = argparse.ArgumentParser(description="Build leaderboard CSV from batch results.")
    parser.add_argument(
        "results_dir",
        nargs="?",
        default="./logging/batch_results/model_leaderboard",
        help="Root directory containing model result folders (default: ./logging/batch_results/model_leaderboard)",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output CSV path (default: {results_dir}/leaderboard.csv)",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.is_dir():
        raise SystemExit(f"Results directory not found: {results_dir}")

    output_path = Path(args.output) if args.output else results_dir / "leaderboard.csv"

    header1, header2, rows, exp_names, metric_keys, metric_names, model_exp_data = (
        build_leaderboard(results_dir)
    )

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header1)
        writer.writerow(header2)
        writer.writerows(rows)

    ascii_path = output_path.with_suffix(".txt")
    report = _build_ascii_report(exp_names, metric_keys, metric_names, model_exp_data)
    ascii_path.write_text(report)

    exp_count = sum(1 for h in header1[1:] if h)
    print(f"Leaderboard written to: {output_path}")
    print(f"ASCII report written to: {ascii_path}")
    print(f"  Models:      {len(rows)}")
    print(f"  Experiments: {exp_count}")
    print()
    print(report)


def _build_ascii_report(
    exp_names: list[str],
    metric_keys: list[str],
    metric_names: list[str],
    model_exp_data: list[tuple[str, dict[str, dict[str, str]]]],
) -> str:
    """Build a human-readable ASCII report grouped by model.

    Each model gets a section header followed by a table of its experiment
    results (one row per experiment, one column per metric).
    """
    lines: list[str] = []

    # Column widths
    exp_col_w = max((len(e) for e in exp_names), default=12)
    metric_col_w = [max(len(m), 12) for m in metric_names]

    def hline(char="-"):
        parts = f"-+-".join(char * w for w in metric_col_w)
        return f"{char * exp_col_w}-+-{parts}"

    def metric_header():
        cols = " | ".join(m.ljust(metric_col_w[i]) for i, m in enumerate(metric_names))
        return f"{'experiment'.ljust(exp_col_w)} | {cols}"

    title = "LLM VERIFICATION LEADERBOARD"
    lines.append("=" * max(len(title) + 4, len(hline()) + 4))
    lines.append(f"  {title}")
    lines.append("=" * max(len(title) + 4, len(hline()) + 4))
    lines.append("")

    for model_name, exp_data in model_exp_data:
        lines.append(f"  Model: {model_name}")
        lines.append(f"  {hline('=')}")
        lines.append(f"  {metric_header()}")
        lines.append(f"  {hline('-')}")
        for exp in exp_names:
            vals = exp_data.get(exp)
            if vals is None:
                cells = " | ".join("-".ljust(metric_col_w[i]) for i in range(len(metric_names)))
            else:
                cells = " | ".join(
                    (vals.get(k) or "-").ljust(metric_col_w[i])
                    for i, k in enumerate(metric_keys)
                )
            lines.append(f"  {exp.ljust(exp_col_w)} | {cells}")
        lines.append(f"  {hline('-')}")
        lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    main()
