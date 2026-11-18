"""Visualize a one-step TV distance matrix and cluster it under a diameter cap.

Consumes the ``tv_distance.npy`` written by ``beaver.grouping.tv_distance`` and:

  * **Counts** how many prompt pairs are within a TV threshold (default 0.1) —
    the pairs cheap enough to share a cached frontier (TV < ε ⇒ δ-close).
  * **Clusters** so that *every* pair inside a group is within the threshold.
    This is a diameter cap, so we use **complete-linkage** agglomerative
    clustering (it bounds the max intra-cluster distance, not just the mean) and
    verify the guarantee holds.
  * **Plots** the matrix and the threshold graph four ways: a clustered heatmap,
    a pairwise-distance histogram, a threshold graph (nodes = prompts, edge =
    pair within the threshold, coloured by cluster), and a 2-D MDS embedding.

Example:
  python -m beaver.grouping.tv_visualize \
      --matrix tv_out/enron_all/tv_distance.npy --threshold 0.1 --out tv_out/enron_all
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

import numpy as np

import matplotlib
matplotlib.use("Agg")  # headless: write PNGs, never open a window
import matplotlib.pyplot as plt  # noqa: E402


# ── analysis ───────────────────────────────────────────────────────────────────
def count_pairs_within(D: np.ndarray, threshold: float) -> dict:
    """Number (and fraction) of unordered prompt pairs with TV <= threshold."""
    iu = np.triu_indices(len(D), k=1)
    pair = D[iu]
    within = int((pair <= threshold).sum())
    total = int(pair.size)
    return {
        "n_prompts": int(len(D)),
        "n_pairs": total,
        "n_within": within,
        "fraction_within": (within / total) if total else 0.0,
        "threshold": float(threshold),
    }


def diameter_clusters(D: np.ndarray, threshold: float) -> np.ndarray:
    """Complete-linkage clusters whose every intra-pair distance is <= threshold.

    Complete linkage cuts the dendrogram at ``threshold`` on the *max* inter-point
    distance, so each returned cluster has diameter <= threshold (verified by the
    caller). Returns an integer label per prompt.
    """
    from sklearn.cluster import AgglomerativeClustering

    if len(D) == 1:
        return np.zeros(1, dtype=int)
    model = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=threshold,
        metric="precomputed",
        linkage="complete",
    )
    return model.fit_predict(D)


def diameter_clusters_per_base(D: np.ndarray, threshold: float, base_ids: List) -> np.ndarray:
    """Like :func:`diameter_clusters`, but clusters each base prompt independently.

    Complete-linkage on the full matrix can still merge instances from two
    different base prompts if a short rollout depth (``k`` in
    ``tv_k_distance``) doesn't cover enough of the generation window for the
    constraint-relevant divergence to show up as TV distance. Clustering each
    base's own submatrix separately makes cross-base merging structurally
    impossible, independent of how tight ``k`` is.
    """
    n = len(D)
    labels = np.full(n, -1, dtype=int)
    next_label = 0
    base_arr = np.asarray(base_ids, dtype=object)
    for base in sorted(set(base_arr.tolist()), key=str):
        idx = np.where(base_arr == base)[0]
        sub = D[np.ix_(idx, idx)]
        sub_labels = diameter_clusters(sub, threshold)
        for local_g in sorted(set(int(x) for x in sub_labels)):
            labels[idx[sub_labels == local_g]] = next_label
            next_label += 1
    assert (labels >= 0).all(), "every instance must get a label"
    return labels


def max_intra_distance(D: np.ndarray, labels: np.ndarray) -> float:
    """Largest within-cluster pairwise distance (the realized diameter)."""
    worst = 0.0
    for g in set(int(x) for x in labels):
        idx = np.where(labels == g)[0]
        if len(idx) < 2:
            continue
        sub = D[np.ix_(idx, idx)]
        worst = max(worst, float(sub.max()))
    return worst


# ── grouped-dataset builder + verification ─────────────────────────────────────
def build_grouped_dataset(
    instances: List[dict], D: np.ndarray, labels: np.ndarray
) -> List[dict]:
    """Relabel instances by their TV cluster and emit them group-by-group.

    Row ``i`` of ``D`` must correspond to ``instances[i]``. Within each group the
    members are ordered medoid-first by a nearest-neighbour chain (the cache-warm
    order from :func:`cluster.order_within_group`), and each instance records
    ``tv_index`` (its row in ``D``) so the grouping can be verified against the
    matrix afterwards.
    """
    from beaver.grouping.cluster import order_within_group

    out: List[dict] = []
    for g in sorted(set(int(x) for x in labels)):
        members = [i for i in range(len(instances)) if int(labels[i]) == g]
        members = order_within_group(D, members)
        for pos, i in enumerate(members):
            inst = dict(instances[i])
            inst["group"] = int(g)
            inst["group_position"] = pos
            inst["tv_index"] = int(i)  # row in D, for verification
            out.append(inst)
    return out


def verify_grouping(grouped: List[dict], D: np.ndarray, threshold: float) -> dict:
    """Confirm every within-group prompt pair has actual TV distance <= threshold.

    Reads distances straight from ``D`` (the matrix entries *are* the actual
    one-step TV distances) via each instance's ``tv_index``. Returns the worst
    offending pair and whether the diameter cap holds for every group.
    """
    by_group: dict = {}
    for inst in grouped:
        by_group.setdefault(inst["group"], []).append(inst["tv_index"])

    worst = {"dist": 0.0, "group": None, "a": None, "b": None}
    n_pairs_checked = 0
    for g, rows in by_group.items():
        for a in range(len(rows)):
            for b in range(a + 1, len(rows)):
                d = float(D[rows[a], rows[b]])
                n_pairs_checked += 1
                if d > worst["dist"]:
                    worst = {"dist": d, "group": int(g), "a": rows[a], "b": rows[b]}
    return {
        "n_groups": len(by_group),
        "n_pairs_checked": n_pairs_checked,
        "max_intra_distance": worst["dist"],
        "worst_pair": (worst["a"], worst["b"]),
        "worst_group": worst["group"],
        "threshold": float(threshold),
        "passed": worst["dist"] <= threshold + 1e-9,
    }


# ── group summary (sizes, base/system-prompt breakdown, singletons) ────────────

# Candidate field names for "which underlying base prompt is this instance a
# variant of" — checked in order, first one present in the dataset wins.
# Covers secure_code (source_idx), toxicity (task_idx), enron (target_email),
# reasoning-chain datasets (prompt_group, see beaver.utils.sample_reasoning_chains).
_BASE_IDENTITY_FIELDS = ["source_idx", "task_idx", "target_email", "base_idx", "prompt_group"]


def _detect_base_field(instances: List[dict]) -> Optional[str]:
    if not instances:
        return None
    for field in _BASE_IDENTITY_FIELDS:
        if field in instances[0]:
            return field
    return None


def build_group_summary(
    instances: Optional[List[dict]], labels: np.ndarray, threshold: float
) -> dict:
    """Group-size distribution + (if instances given) base/system-prompt breakdown.

    Always includes the overview (sizes, singletons, max group) computed from
    ``labels`` alone. If ``instances`` is provided (same order as the distance
    matrix rows / ``labels``), each group entry also reports which base prompt
    (auto-detected via :data:`_BASE_IDENTITY_FIELDS`) and how many distinct
    system prompts its members come from, plus a dataset-wide base-prompt ->
    groups coverage table.
    """
    n = len(labels)
    group_ids = sorted(set(int(x) for x in labels))
    members_by_group = {g: [i for i in range(n) if int(labels[i]) == g] for g in group_ids}
    sizes = np.array([len(members_by_group[g]) for g in group_ids])

    overview = {
        "n_instances": int(n),
        "n_groups": len(group_ids),
        "threshold": float(threshold),
        "singleton_count": int((sizes == 1).sum()),
        "singleton_pct_of_groups": float((sizes == 1).sum() / len(group_ids) * 100) if group_ids else 0.0,
        "singleton_pct_of_instances": float((sizes == 1).sum() / n * 100) if n else 0.0,
        "max_group_size": int(sizes.max()) if len(sizes) else 0,
        "max_group_size_pct_of_instances": float(sizes.max() / n * 100) if n else 0.0,
        "mean_group_size": float(sizes.mean()) if len(sizes) else 0.0,
        "median_group_size": float(np.median(sizes)) if len(sizes) else 0.0,
        "size_histogram": {int(s): int((sizes == s).sum()) for s in sorted(set(sizes.tolist()), reverse=True)},
    }

    base_field = _detect_base_field(instances) if instances else None
    groups = []
    for g in sorted(group_ids, key=lambda g: -len(members_by_group[g])):
        members = members_by_group[g]
        entry = {"group": g, "size": len(members), "members_tv_index": members}
        if instances:
            if base_field:
                counts: dict = {}
                for i in members:
                    key = instances[i][base_field]
                    counts[key] = counts.get(key, 0) + 1
                entry["base_prompt_field"] = base_field
                entry["base_prompt_counts"] = counts
                entry["n_distinct_base_prompts"] = len(counts)
            if "system_prompt" in instances[0]:
                distinct = {instances[i]["system_prompt"] for i in members}
                entry["n_distinct_system_prompts"] = len(distinct)
        groups.append(entry)

    summary = {"overview": overview, "groups": groups}

    if instances and base_field:
        coverage: dict = {}
        for i in range(n):
            key = instances[i][base_field]
            coverage.setdefault(key, {"n_instances": 0, "groups": {}})
            coverage[key]["n_instances"] += 1
            g = int(labels[i])
            coverage[key]["groups"][g] = coverage[key]["groups"].get(g, 0) + 1
        for key, v in coverage.items():
            v["n_groups"] = len(v["groups"])
            v["group_sizes"] = sorted(v["groups"].values(), reverse=True)
        summary["base_prompt_field"] = base_field
        summary["base_prompt_coverage"] = coverage

    return summary


def write_group_summary_txt(summary: dict, path: Path) -> None:
    """Human-readable rendering of :func:`build_group_summary`'s output."""
    o = summary["overview"]
    lines = [
        "=== TV grouping summary ===",
        f"n_instances: {o['n_instances']}   n_groups: {o['n_groups']}   threshold: {o['threshold']}",
        f"singletons: {o['singleton_count']} "
        f"({o['singleton_pct_of_groups']:.1f}% of groups, "
        f"{o['singleton_pct_of_instances']:.1f}% of instances)",
        f"max group size: {o['max_group_size']} "
        f"({o['max_group_size_pct_of_instances']:.1f}% of instances)",
        f"mean group size: {o['mean_group_size']:.2f}   median: {o['median_group_size']:.1f}",
        "",
        "size distribution (size: n_groups):",
    ]
    for size, count in o["size_histogram"].items():
        lines.append(f"  size {size:>4}: {count:>4} group(s)")

    if "base_prompt_coverage" in summary:
        field = summary["base_prompt_field"]
        lines += ["", f"=== per-base coverage (field: {field}) ==="]
        for key, v in sorted(summary["base_prompt_coverage"].items(), key=lambda kv: -kv[1]["n_instances"]):
            lines.append(
                f"  {field}={key!r}: n_instances={v['n_instances']:>4}  "
                f"split into {v['n_groups']} group(s)  group_sizes={v['group_sizes']}"
            )

    lines += ["", "=== groups (largest first) ==="]
    for entry in summary["groups"]:
        parts = [f"group {entry['group']:>4}", f"size={entry['size']:>4}"]
        if "n_distinct_base_prompts" in entry:
            parts.append(f"n_bases={entry['n_distinct_base_prompts']}")
            parts.append(f"bases={entry['base_prompt_counts']}")
        if "n_distinct_system_prompts" in entry:
            parts.append(f"n_distinct_system_prompts={entry['n_distinct_system_prompts']}")
        lines.append("  " + "  ".join(parts))

    path.write_text("\n".join(lines) + "\n")


# ── plots ──────────────────────────────────────────────────────────────────────
def plot_heatmap(D: np.ndarray, labels: np.ndarray, path: Path, threshold: float):
    """TV matrix with rows/cols reordered by cluster so tight groups form blocks."""
    order = np.argsort(labels, kind="stable")
    Dord = D[np.ix_(order, order)]
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(Dord, cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_title(f"One-step TV distance (reordered by cluster, thr={threshold})")
    ax.set_xlabel("prompt (cluster-ordered)")
    ax.set_ylabel("prompt (cluster-ordered)")
    fig.colorbar(im, ax=ax, label="TV distance")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_histogram(D: np.ndarray, path: Path, threshold: float):
    """Distribution of all pairwise TV distances with the threshold marked."""
    iu = np.triu_indices(len(D), k=1)
    pair = D[iu]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(pair, bins=60, color="steelblue", edgecolor="white")
    ax.axvline(threshold, color="crimson", linestyle="--",
               label=f"threshold = {threshold}")
    within = int((pair <= threshold).sum())
    ax.set_title(f"Pairwise TV distances  ({within}/{pair.size} within threshold)")
    ax.set_xlabel("TV distance")
    ax.set_ylabel("# pairs")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_threshold_graph(D: np.ndarray, labels: np.ndarray, path: Path, threshold: float):
    """Graph with an edge for every pair within the threshold, nodes by cluster.

    This is the literal picture of "which prompts are within 0.1": connected
    components / dense cliques are the cacheable groups.
    """
    import networkx as nx

    n = len(D)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    iu = np.triu_indices(n, k=1)
    for i, j in zip(*iu):
        if D[i, j] <= threshold:
            G.add_edge(int(i), int(j), weight=float(D[i, j]))

    pos = nx.spring_layout(G, seed=0, k=1.0 / np.sqrt(max(n, 1)))
    fig, ax = plt.subplots(figsize=(9, 8))
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.2, width=0.5)
    nx.draw_networkx_nodes(
        G, pos, ax=ax, node_size=40, node_color=labels, cmap="tab20"
    )
    ax.set_title(
        f"TV threshold graph (edge if TV <= {threshold})  "
        f"{G.number_of_edges()} edges, {nx.number_connected_components(G)} components"
    )
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_mds(D: np.ndarray, labels: np.ndarray, path: Path, threshold: float):
    """2-D MDS embedding of the TV geometry, coloured by cluster."""
    from sklearn.manifold import MDS

    if len(D) < 3:
        return
    emb = MDS(
        n_components=2, dissimilarity="precomputed", random_state=0, n_init=4,
        normalized_stress="auto",
    ).fit_transform(D)
    fig, ax = plt.subplots(figsize=(8, 7))
    sc = ax.scatter(emb[:, 0], emb[:, 1], c=labels, cmap="tab20", s=30)
    ax.set_title(f"MDS of one-step TV distances (clusters @ thr={threshold})")
    ax.set_xlabel("MDS-1")
    ax.set_ylabel("MDS-2")
    fig.colorbar(sc, ax=ax, label="cluster")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ── CLI ────────────────────────────────────────────────────────────────────────
def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Cluster + visualize a one-step TV distance matrix."
    )
    ap.add_argument("--matrix", required=True, help="tv_distance.npy from tv_distance")
    ap.add_argument("--threshold", type=float, default=0.1,
                    help="diameter cap / pair threshold (default 0.1)")
    ap.add_argument("--out", default=None,
                    help="output dir for PNGs (default: matrix's dir)")
    ap.add_argument("--dataset", default=None,
                    help="dataset JSON whose instance order matches the matrix "
                    "rows; if given, write a TV-clustered grouped dataset")
    ap.add_argument("--dataset-out", default=None,
                    help="path for the grouped dataset JSON "
                    "(default: <out>/grouped_dataset_tv.json)")
    ap.add_argument("--per-base", action="store_true",
                    help="cluster each base prompt (identity field from "
                    "--base-field, or auto-detected via _BASE_IDENTITY_FIELDS) "
                    "independently, so no group ever spans two different bases. "
                    "Requires --dataset. Recommended whenever the rollout depth "
                    "(k in tv_k_distance) can't fully cover the constraint-relevant "
                    "part of generation, since the raw TV distance alone can "
                    "then merge unrelated bases.")
    ap.add_argument("--base-field", default=None,
                    help="explicit base-identity field name for --per-base, "
                    "overriding auto-detection (e.g. `group` for reasoning-chain "
                    "files predating the `prompt_group` field)")
    args = ap.parse_args(argv)

    D = np.load(args.matrix)
    out = Path(args.out) if args.out else Path(args.matrix).parent
    out.mkdir(parents=True, exist_ok=True)

    stats = count_pairs_within(D, args.threshold)

    instances = None
    if args.dataset:
        instances = json.load(open(args.dataset))
        if len(instances) != len(D):
            raise SystemExit(
                f"dataset has {len(instances)} instances but matrix is {len(D)}x"
                f"{len(D)} — they must be the same prompts in the same order."
            )

    if args.per_base:
        if not instances:
            raise SystemExit("--per-base requires --dataset (to detect the base-identity field)")
        base_field = args.base_field or _detect_base_field(instances)
        if not base_field:
            raise SystemExit(
                f"--per-base: no recognized base-identity field {_BASE_IDENTITY_FIELDS} "
                "in dataset -- pass --base-field explicitly"
            )
        if base_field not in instances[0]:
            raise SystemExit(f"--base-field {base_field!r} not present in dataset instances")
        base_ids = [inst[base_field] for inst in instances]
        labels = diameter_clusters_per_base(D, args.threshold, base_ids)
        print(f"per-base clustering: field={base_field!r}  ({len(set(base_ids))} bases)")
    else:
        labels = diameter_clusters(D, args.threshold)
    n_clusters = len(set(int(x) for x in labels))
    sizes = np.array([int((labels == g).sum()) for g in set(int(x) for x in labels)])
    realized_diam = max_intra_distance(D, labels)

    print(f"\nTV matrix: {args.matrix}  n={stats['n_prompts']}")
    print(f"threshold = {args.threshold}")
    print(f"pairs within threshold: {stats['n_within']} / {stats['n_pairs']}  "
          f"({stats['fraction_within']*100:.1f}%)")
    print(f"diameter-capped clusters (complete linkage): {n_clusters}")
    print(f"  sizes: min={sizes.min()} max={sizes.max()} "
          f"mean={sizes.mean():.1f}  singletons={(sizes==1).sum()}")
    print(f"  realized max intra-cluster distance = {realized_diam:.4f} "
          f"(must be <= {args.threshold}): "
          f"{'OK' if realized_diam <= args.threshold + 1e-9 else 'VIOLATED'}")

    # ── write + verify the grouped dataset ───────────────────────────────────
    if args.dataset:
        grouped = build_grouped_dataset(instances, D, labels)
        ds_out = Path(args.dataset_out) if args.dataset_out else out / "grouped_dataset_tv.json"
        json.dump(grouped, open(ds_out, "w"), indent=2)

        v = verify_grouping(grouped, D, args.threshold)
        print(f"\ngrouped dataset -> {ds_out}  ({v['n_groups']} groups)")
        print(f"verification: checked {v['n_pairs_checked']} within-group pairs")
        print(f"  max actual intra-group TV distance = {v['max_intra_distance']:.4f}  "
              f"(<= {args.threshold}): {'PASS' if v['passed'] else 'FAIL'}")
        if not v["passed"]:
            print(f"  WORST pair: rows {v['worst_pair']} in group {v['worst_group']}")

    # ── group details summary (sizes, base/system-prompt breakdown) ─────────
    summary = build_group_summary(instances, labels, args.threshold)
    summary_json = out / "group_summary.json"
    summary_txt = out / "group_summary.txt"
    json.dump(summary, open(summary_json, "w"), indent=2)
    write_group_summary_txt(summary, summary_txt)
    print(f"\ngroup summary -> {summary_json}  {summary_txt}")
    if not instances:
        print("  (no --dataset given: base/system-prompt breakdown omitted, sizes only)")
    elif not summary.get("base_prompt_field"):
        print(f"  (no recognized base-identity field {_BASE_IDENTITY_FIELDS} in dataset: "
              "base-prompt breakdown omitted)")

    # save labels + plots
    np.save(out / "tv_cluster_labels.npy", labels)
    plot_heatmap(D, labels, out / "tv_heatmap.png", args.threshold)
    plot_histogram(D, out / "tv_hist.png", args.threshold)
    plot_threshold_graph(D, labels, out / "tv_graph.png", args.threshold)
    plot_mds(D, labels, out / "tv_mds.png", args.threshold)
    print(f"\nwrote: {out}/tv_heatmap.png  tv_hist.png  tv_graph.png  tv_mds.png")
    print(f"       {out}/tv_cluster_labels.npy")


if __name__ == "__main__":
    main()
