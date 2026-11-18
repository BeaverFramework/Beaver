"""Group formation for cached distributional verification.

The cached-frontier verifier (``beaver.verifiers.cached_frontier_verifier``)
buckets instances by ``instance["group"]`` and processes each group
sequentially, every instance warm-starting from the token-trie cache the
previous ones in the group grew. Caching transfers cheaply when instances in a
group induce *similar output distributions*.

Modules:
  - ``tv_distance``         one-step total-variation distance matrix between prompts
  - ``tv_prefix_distance``  TV distance over the first k generated tokens
  - ``tv_visualize``        cluster a TV matrix into groups and write a grouped dataset
  - ``tv_group_dataset``    end-to-end: dataset JSON -> TV matrix -> grouped dataset JSON
  - ``cluster``             clustering + within-group ordering helpers
"""

from beaver.grouping.cluster import cluster_instances, order_within_group

__all__ = ["cluster_instances", "order_within_group"]
