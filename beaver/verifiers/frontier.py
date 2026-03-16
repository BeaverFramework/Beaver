"""Frontier data structure using a trie for managing candidate sequences during verification"""

import random
from typing import Dict, List, Optional, Set

import numpy as np
import torch

random.seed(42)


class FrontierElement:
    def __init__(
        self,
        element_id: int,
        token: int,
        tokens: List[int] = [],
        logprob: float = 0.0,
        is_completed: bool = False,
        parent: Optional["FrontierElement"] = None,
    ):
        self.element_id = element_id
        self.token = token
        self.tokens = tokens if tokens is not None else []
        self.logprob = logprob
        self.is_completed = is_completed
        self.parent = parent
        self.children: Dict[int, "FrontierElement"] = {}

    def add_child(self, token_id: int, child: "FrontierElement"):
        child.parent = self
        self.children[token_id] = child

    def get_child(self, token_id: int) -> Optional["FrontierElement"]:
        return self.children.get(token_id)

    def remove_child(self, token_id: int) -> Optional["FrontierElement"]:
        child = self.children.pop(token_id, None)
        if child:
            child.parent = None
        return child

    def is_leaf(self) -> bool:
        return len(self.children) == 0


class Frontier:
    def __init__(self, max_size: int, scoring_strategy: str = "highest-prob"):
        self.root = FrontierElement(
            token=-1,
            element_id=0,
        )
        self.total_elements = 1
        self._incomplete_leaves: Set[FrontierElement] = set([self.root])
        self._complete_leaves: Set[FrontierElement] = set()
        self.scoring_strategy = scoring_strategy
        self.max_gen = max_size

    def add_to_element(
        self,
        parent_element: FrontierElement,
        new_frontier_elements: List[FrontierElement],
    ):
        self._incomplete_leaves.discard(parent_element)
        for child in new_frontier_elements:
            parent_element.add_child(child.token, child)
            if child.is_completed:
                self._complete_leaves.add(child)
            else:
                self._incomplete_leaves.add(child)

    def pick_top_incomplete(self) -> Optional[FrontierElement]:
        if not self._incomplete_leaves:
            return None
        if self.scoring_strategy == "highest-prob":
            return max(self._incomplete_leaves, key=lambda x: x.logprob)
        elif self.scoring_strategy == "random-select":
            return random.choice(list(self._incomplete_leaves))
        elif self.scoring_strategy == "sample-select":
            comp_list = list(self._incomplete_leaves)
            dist = torch.exp(torch.tensor([x.logprob for x in comp_list]))
            sample = torch.multinomial(dist, 1)
            return comp_list[sample.item()]
        elif self.scoring_strategy == "length-bias":
            # Prioritize length in selection
            return max(
                self._incomplete_leaves,
                key=lambda x: (np.log(len(x.tokens) / self.max_gen) + x.logprob),
            )
        else:
            raise ValueError(f"Invalid scoring strategy: {self.scoring_strategy}")

    def prune_incomplete_leaves(self, topp: float = 1.0, topk: int = -1):
        """
        Prune incomplete leaves based on top-p and top-k criteria.
        Uses logprob (temperature 1.0) for pruning decisions.

        Args:
            topp: Top-p threshold (default 1.0 = no pruning)
            topk: Top-k threshold (default None = no pruning)
        """
        if len(self._incomplete_leaves) == 0:
            return 0.0

        # Convert to list for sorting
        incomplete_list = list(self._incomplete_leaves)

        # Apply top-k pruning
        if topk >= 0 and len(incomplete_list) > topk:
            # Sort by logprob (temp 1.0) in descending order
            incomplete_list.sort(key=lambda x: x.logprob, reverse=True)
            incomplete_list = incomplete_list[:topk]

        # Apply top-p pruning
        if topp < 1.0 and len(incomplete_list) > 0:
            incomplete_list.sort(key=lambda x: x.logprob, reverse=True)
            probs = np.array([np.exp(x.logprob) for x in incomplete_list])
            cumsum_probs = np.cumsum(probs)
            cutoff_idx = np.searchsorted(cumsum_probs, topp) + 1
            incomplete_list = incomplete_list[:cutoff_idx]

        pruned_set = set(incomplete_list)
        removed_leaves = self._incomplete_leaves - pruned_set

        if len(removed_leaves) > 0:
            pruned_prob = np.sum(
                np.exp(np.array([x.logprob for x in removed_leaves])), axis=0
            )
        else:
            pruned_prob = 0.0
        for leaf in removed_leaves:
            if leaf.parent is not None:
                leaf.parent.remove_child(leaf.token)

        self._incomplete_leaves = pruned_set
        return pruned_prob

    def calc_prob_sum(self):
        incomplete_sum = np.sum(
            np.array([np.exp(elem.logprob) for elem in self._incomplete_leaves]),
        )
        complete_sum = np.sum(
            np.array([np.exp(elem.logprob) for elem in self._complete_leaves])
        )
        return incomplete_sum, complete_sum

    def debug_frontier(self, tokenizer):
        print("Frontier Debug:")
        print("Root:", self.root.element_id)
        # get top 10 incomplete
        print(f"Incomplete Leaves ({len(self._incomplete_leaves)}):")
        top_incomplete = sorted(
            self._incomplete_leaves, key=lambda x: x.logprob, reverse=True
        )[:10]
        print("Top 10 Incomplete Leaves:")
        for leaf in top_incomplete:
            print(
                f"  {leaf.element_id}: {leaf.tokens} (logprob={leaf.logprob}) : {tokenizer.decode(leaf.tokens, skip_special_tokens=True)}"
            )

        # get top 10 complete
        print(f"Complete Leaves ({len(self._complete_leaves)}):")
        top_complete = sorted(
            self._complete_leaves, key=lambda x: x.logprob, reverse=True
        )[:10]
        print("Top 10 Complete Leaves:")
        for leaf in top_complete:
            print(
                f"  {leaf.element_id}: {leaf.tokens} (logprob={leaf.logprob}): {tokenizer.decode(leaf.tokens, skip_special_tokens=True)}"
            )
