"""Form groupings from a proxy distance matrix, and order instances within a group.

Two decisions, both driven by the same pairwise distances the proxies produce:

  * **Which instances share a group** (:func:`cluster_instances`) — agglomerative
    (average-linkage on the precomputed distances), k-means (on raw features), or
    greedy radius growing. Output is a label per instance.
  * **The order instances run within a group** (:func:`order_within_group`) — the
    cache is grown incrementally, so a good order starts at the group medoid and
    visits the nearest not-yet-seen instance each step (a cheap TSP-ish chain).
    This is the "cost optimisation via group reordering" lever: each instance
    warm-starts from a cache built on its closest predecessors.

:func:`assign_groups` ties both together, returning instance dicts with ``group``
overwritten and the list reordered so the cached-frontier verifier (which honours
``instance["group"]`` + list order) runs the chosen grouping + order verbatim.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np


def cluster_instances(
    distance: np.ndarray,
    n_clusters: Optional[int] = None,
    method: str = "agglomerative",
    threshold: Optional[float] = None,
    features: Optional[np.ndarray] = None,
    linkage: str = "average",
) -> np.ndarray:
    """Return an integer cluster label per instance.

    method:
      ``"agglomerative"`` — agglomerative clustering on the precomputed
          ``distance`` with the given ``linkage`` (default ``"average"``). Set
          either ``n_clusters`` or ``threshold`` (distance cut height).

          Use ``linkage="complete"`` together with ``threshold=δ`` for a
          **diameter cap**: complete linkage cuts the dendrogram on the *max*
          inter-point distance, so every returned cluster has diameter <= δ
          (i.e. every within-group pair is within δ of each other). This is the
          only linkage that gives that guarantee — ``"average"`` bounds only the
          mean inter-cluster distance, so two members of the same average-linkage
          cluster can be far more than δ apart.
      ``"kmeans"`` — k-means on ``features`` (``[n, d]``); needs ``n_clusters``.
      ``"greedy"`` — grow clusters by absorbing every point within ``threshold``
          of a seed (the lowest-index unassigned point). No sklearn needed.
          NOTE: this is *not* a diameter cap — two points each within ``threshold``
          of the seed can be up to ``2*threshold`` apart from each other. Use
          ``"agglomerative"`` + ``linkage="complete"`` for a real diameter cap.
    """
    n = distance.shape[0]
    if method == "agglomerative":
        from sklearn.cluster import AgglomerativeClustering

        if n_clusters is None and threshold is None:
            raise ValueError("agglomerative needs n_clusters or threshold")
        model = AgglomerativeClustering(
            n_clusters=n_clusters,
            distance_threshold=threshold if n_clusters is None else None,
            metric="precomputed",
            linkage=linkage,
        )
        return model.fit_predict(distance)

    if method == "kmeans":
        from sklearn.cluster import KMeans

        if features is None or n_clusters is None:
            raise ValueError("kmeans needs features and n_clusters")
        return KMeans(n_clusters=n_clusters, n_init=10, random_state=0).fit_predict(
            features
        )

    if method == "greedy":
        if threshold is None:
            raise ValueError("greedy needs threshold")
        labels = np.full(n, -1, dtype=int)
        cur = 0
        for i in range(n):
            if labels[i] != -1:
                continue
            members = np.where((distance[i] <= threshold) & (labels == -1))[0]
            labels[members] = cur
            cur += 1
        return labels

    raise ValueError(f"unknown method {method!r}")


def order_within_group(distance: np.ndarray, members: Sequence[int]) -> List[int]:
    """Nearest-neighbour chain over ``members`` starting at the group medoid.

    The medoid (smallest summed distance to the rest) seeds the cache; each
    subsequent instance is the unseen one closest to *any* already-placed member
    (so it warm-starts from the most similar predecessor's contribution to the
    shared cache). Returns ``members`` reordered.
    """
    members = list(members)
    if len(members) <= 2:
        return members
    sub = distance[np.ix_(members, members)]
    start = int(np.argmin(sub.sum(axis=1)))
    order = [start]
    remaining = set(range(len(members))) - {start}
    while remaining:
        best, best_d = None, np.inf
        for r in remaining:
            d = min(sub[r, p] for p in order)
            if d < best_d:
                best, best_d = r, d
        order.append(best)
        remaining.discard(best)
    return [members[i] for i in order]


def assign_groups(
    instances: Sequence[dict],
    distance: np.ndarray,
    labels: np.ndarray,
    reorder: bool = True,
) -> List[dict]:
    """Apply a grouping: copy each instance, overwrite ``group`` with its label,
    and (if ``reorder``) emit groups with members in cache-friendly order.

    The returned list is ordered group-by-group; feeding it to the cached-frontier
    verifier runs exactly this grouping + within-group order.
    """
    out: List[dict] = []
    for g in sorted(set(int(x) for x in labels)):
        members = [i for i in range(len(instances)) if int(labels[i]) == g]
        if reorder:
            members = order_within_group(distance, members)
        for pos, i in enumerate(members):
            inst = dict(instances[i])
            inst["group"] = int(g)
            inst["group_position"] = pos
            out.append(inst)
    return out
