"""Dataset-agnostic TV grouping: turn ANY prompt dataset into a grouped dataset.

Unlike ``experiments/enron/build_tv_grouped_distribution.py`` (which *constructs*
Enron-specific prompts), this script does not build prompts at all. It takes a
dataset you already have and groups it for the cached-frontier verifier using the
exact same one-step TV pipeline the design relies on:

    load any dataset JSON  ->  exact one-step TV matrix  ->  complete-linkage
    diameter-cap clusters (every intra-group pair's TV <= ε)  ->  verify  ->
    write a verifier-ready grouped dataset (group / group_position / tv_index).

Any dataset works as long as each instance has a prompt: ``prompt`` is filled
from common aliases (``test_case_prompt`` / ``question``) if missing, and
``system_prompt`` / ``idx`` keys are ensured (the workers require them). Every
original field is preserved, so the output feeds ``cachedfrontier`` directly.

This is the dataset-agnostic core of the TV-grouping design
(see experiments/enron/TV_GROUPING_DESIGN.md). Use it to group enron,
secure_code, toxicity, or any custom dataset the same way — then compare
``cachedfrontier`` vs ``frontier`` with ``cost_analysis.py``.

Example:
    python -m beaver.grouping.tv_group_dataset \
        --dataset experiments/secure_code/data/autocomplete_ccpp_uniform_100.json \
        --output tv_out/secure_code_100/grouped_dataset_tv.json \
        --model Qwen/Qwen3-4B-Instruct-2507 --threshold 0.1 --plots

    # then, e.g.:
    #   frontier      over --output (uncached baseline)
    #   cachedfrontier over --output (TV-grouped cache)
    #   cost_analysis.py no_cache=... tv_grouped=...
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import List, Optional

import numpy as np


def load_dataset(
    path: Path,
    max_n: Optional[int] = None,
    sample_n: Optional[int] = None,
    seed: int = 0,
) -> List[dict]:
    """Load any dataset JSON (list of instances) and normalise it for TV grouping.

    Fills ``prompt`` from ``test_case_prompt`` / ``question`` when absent, ensures
    ``system_prompt`` and a fresh ``idx``, and keeps all other fields untouched.

    Subsampling (at most one of):
      * ``sample_n`` — a RANDOM sample of ``sample_n`` instances (seeded, without
        replacement); representative of the whole file.
      * ``max_n`` — the FIRST ``max_n`` instances in file order (deterministic
        head slice; biased if the file is ordered).
    ``idx`` is assigned after subsampling, so it is contiguous 0..n-1.
    """
    import random

    instances = json.load(open(path))
    if not isinstance(instances, list):
        raise SystemExit(f"{path}: expected a JSON list of instances")
    if sample_n and max_n:
        raise SystemExit("pass at most one of --sample-n (random) or --max-n (head slice)")
    if sample_n:
        if sample_n < len(instances):
            instances = random.Random(seed).sample(instances, sample_n)
    elif max_n:
        instances = instances[:max_n]
    for i, inst in enumerate(instances):
        inst.setdefault("prompt", inst.get("test_case_prompt", inst.get("question", "")))
        if not inst.get("prompt"):
            raise SystemExit(
                f"{path}: instance {i} has no 'prompt' (or 'test_case_prompt'/'question')"
            )
        inst.setdefault("system_prompt", "")
        inst["idx"] = i
    return instances


def group_dataset(
    instances: List[dict],
    model: str,
    threshold: float = 0.1,
    device: str = "cuda",
    dtype: str = "bfloat16",
) -> tuple[List[dict], np.ndarray, np.ndarray, dict]:
    """Compute one-step TV, diameter-cap cluster, and build a grouped dataset.

    Returns ``(grouped, D, labels, verification)`` where ``grouped`` carries the
    per-instance ``group`` / ``group_position`` / ``tv_index`` fields, ordered
    group-by-group (medoid-first), ready for the cached-frontier verifier.
    """
    from beaver.grouping.tv_distance import HFBackend, one_step_tv_matrix
    from beaver.grouping.tv_visualize import (
        build_grouped_dataset,
        diameter_clusters,
        verify_grouping,
    )

    backend = HFBackend(model, device=device, dtype=dtype)
    D = one_step_tv_matrix(instances=instances, backend=backend)
    labels = diameter_clusters(D, threshold)
    grouped = build_grouped_dataset(instances, D, labels)
    verification = verify_grouping(grouped, D, threshold)
    return grouped, D, labels, verification


def _report(labels: np.ndarray, n: int, threshold: float, v: dict) -> None:
    sizes = sorted(Counter(int(x) for x in labels).values(), reverse=True)
    singleton_prompts = sum(s for s in sizes if s == 1)
    print(f"\nTV grouping @ threshold={threshold}:  n={n}")
    print(f"  clusters: {len(sizes)}  (max={max(sizes)}, mean={np.mean(sizes):.1f})")
    print(f"  singletons: {sum(s == 1 for s in sizes)} clusters "
          f"({singleton_prompts}/{n} prompts = {100*singleton_prompts/n:.1f}%)")
    print(f"  in multi-member groups: {n - singleton_prompts}/{n} prompts")
    print(f"  sizes: {sizes}")
    print(f"  diameter check: max intra-group TV = {v['max_intra_distance']:.4f} "
          f"(<= {threshold}): {'PASS' if v['passed'] else 'FAIL'}")


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Group ANY prompt dataset by one-step TV distance "
                    "(dataset-agnostic).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--dataset", required=True, help="input dataset JSON (list of instances)")
    ap.add_argument("--output", required=True, help="output grouped dataset JSON (verifier-ready)")
    ap.add_argument("--model", required=True, help="HF model id (full-vocab next-token dists)")
    ap.add_argument("--threshold", type=float, default=0.1, help="TV diameter cap ε (default 0.1)")
    ap.add_argument("--sample-n", type=int, default=None,
                    help="RANDOM sample of N instances (seeded, representative)")
    ap.add_argument("--max-n", type=int, default=None,
                    help="first N instances (head slice; use --sample-n for a random subset)")
    ap.add_argument("--seed", type=int, default=0, help="random seed for --sample-n (default 0)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--artifacts-dir", default=None,
                    help="dir to save tv_distance.npy + tv_cluster_labels.npy "
                         "(default: output's dir)")
    ap.add_argument("--plots", action="store_true",
                    help="also write the tv_visualize plots (heatmap/hist/graph/mds)")
    args = ap.parse_args(argv)

    instances = load_dataset(
        Path(args.dataset), max_n=args.max_n, sample_n=args.sample_n, seed=args.seed
    )
    grouped, D, labels, v = group_dataset(
        instances, args.model, args.threshold, args.device, args.dtype
    )
    _report(labels, len(instances), args.threshold, v)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(grouped, open(out_path, "w"), indent=2)
    n_groups = len({d["group"] for d in grouped})
    print(f"\nwrote grouped dataset -> {out_path}  ({len(grouped)} instances, {n_groups} groups)")

    art = Path(args.artifacts_dir) if args.artifacts_dir else out_path.parent
    art.mkdir(parents=True, exist_ok=True)
    np.save(art / "tv_distance.npy", D)
    np.save(art / "tv_cluster_labels.npy", labels)
    print(f"artifacts -> {art}/tv_distance.npy, tv_cluster_labels.npy")

    if args.plots:
        from beaver.grouping.tv_visualize import (
            plot_heatmap, plot_histogram, plot_mds, plot_threshold_graph,
        )
        plot_heatmap(D, labels, art / "tv_heatmap.png", args.threshold)
        plot_histogram(D, art / "tv_hist.png", args.threshold)
        plot_threshold_graph(D, labels, art / "tv_graph.png", args.threshold)
        plot_mds(D, labels, art / "tv_mds.png", args.threshold)
        print(f"plots -> {art}/tv_heatmap.png, tv_hist.png, tv_graph.png, tv_mds.png")


if __name__ == "__main__":
    main()
