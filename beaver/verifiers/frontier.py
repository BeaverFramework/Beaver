"""Frontier data structure using a trie for managing candidate sequences during verification"""

from typing import Dict, List, Optional, Set

import numpy as np


class FrontierElement:
    def __init__(
        self,
        element_id: int,
        token: int,
        logprob: float = 0.0,
        is_completed: bool = False,
        is_violation: bool = False,
        parent: Optional["FrontierElement"] = None,
    ):
        self.element_id = element_id
        self.token = token
        self.logprob = logprob
        assert not(is_completed and is_violation), "Node cannot be both violation and completion"
        self.is_completed = is_completed
        self.is_violation = is_violation
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

    @property
    def tokens(self) -> List[int]:
        token_list = []
        current = self

        while current is not None:
            if current.token == -1:
                break
            token_list.append(current.token)
            current = current.parent

        return token_list[::-1]


class Frontier:
    def __init__(self, max_size: int):
        self.root = FrontierElement(
            token=-1,
            element_id=0,
        )
        self.total_elements = 1
        self._violation_leaves: Set[FrontierElement] = set()
        self._incomplete_leaves: Set[FrontierElement] = set([self.root])
        self._complete_leaves: Set[FrontierElement] = set()
        self.max_gen = max_size

    def add_to_element(
        self,
        parent_element: FrontierElement,
        new_frontier_elements: List[FrontierElement],
    ):
        assert parent_element in self._incomplete_leaves
        self._incomplete_leaves.discard(parent_element)
        for child in new_frontier_elements:
            parent_element.add_child(child.token, child)
            if child.is_completed:
                self._complete_leaves.add(child)
            elif child.is_violation:
                self._violation_leaves.add(child)
            else:
                self._incomplete_leaves.add(child)

    def prune_incomplete_leaves(self, topp: float = 1.0, topk: int = -1):
        if len(self._incomplete_leaves) == 0:
            return 0.0

        incomplete_list = list(self._incomplete_leaves)

        if topk >= 0 and len(incomplete_list) > topk:
            # Sort by logprob (temp 1.0) in descending order
            incomplete_list.sort(key=lambda x: x.logprob, reverse=True)
            incomplete_list = incomplete_list[:topk]

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
        violation_sum = np.sum(
            np.array([np.exp(elem.logprob) for elem in self._violation_leaves])
        )
        return incomplete_sum, complete_sum, violation_sum

    def stats(self):
        """Summary statistics for the current frontier state (logged/printed at
        the end of an instance's verification)."""
        incomplete = self._incomplete_leaves
        if incomplete:
            max_incomplete_prob = float(np.exp(max(l.logprob for l in incomplete)))
        else:
            max_incomplete_prob = 0.0
        incomplete_sum, complete_sum, violation_sum = self.calc_prob_sum()
        return {
            "incomplete_size": len(self._incomplete_leaves),
            "complete_size": len(self._complete_leaves),
            "violation_size": len(self._violation_leaves),
            "total_elements": self.total_elements,
            "max_incomplete_prob": max_incomplete_prob,
            "incomplete_prob_sum": float(incomplete_sum),
            "complete_prob_sum": float(complete_sum),
            "violation_prob_sum": float(violation_sum),
        }

    def get_top_incomplete_seqs(self, scoring_func=None):
        if scoring_func is not None:
            top_incomplete = sorted(
                self._incomplete_leaves, key=lambda x: scoring_func(x), reverse=True
            )[:20]
        else:
            top_incomplete = sorted(
                self._incomplete_leaves, key=lambda x: x.logprob, reverse=True
            )[:20]

        return top_incomplete

    def debug_frontier(self, tokenizer, scoring_func=None):
        print("Frontier Debug:")
        print("Root:", self.root.element_id)
        print(f"Incomplete Leaves ({len(self._incomplete_leaves)}):")
        top_incomplete = sorted(
            self._incomplete_leaves, key=lambda x: x.logprob, reverse=True
        )[:10]
        print("Top 10 Incomplete Leaves (according to prob):")
        for leaf in top_incomplete:
            print(
                f"  {leaf.element_id}: {leaf.tokens} (logprob={leaf.logprob}) : {tokenizer.decode(leaf.tokens, skip_special_tokens=True)}"
            )
        print()
        if scoring_func is not None:
            top_incomplete = sorted(
                self._incomplete_leaves, key=lambda x: scoring_func(x), reverse=True
            )[:10]
            print("Top 10 Incomplete Leaves (according to score):")
            for leaf in top_incomplete:
                print(
                    f"  {leaf.element_id}: {leaf.tokens} (logprob={leaf.logprob}) (parent_id={leaf.parent.element_id}): {tokenizer.decode(leaf.tokens, skip_special_tokens=True)}"
                )
            print()

    def print_frontier(self, tokenizer=None):
        """Print full trie. Each node: token, logprob, decoded text, status flags."""
        print("Frontier Trie:")

        def _flags(node):
            f = []
            if node in self._incomplete_leaves:
                f.append("incomplete")
            if node.is_completed:
                f.append("complete")
            if node.is_violation:
                f.append("violation")
            return f" [{', '.join(f)}]" if f else ""

        def _walk(node, depth, sib_idx, n_sibs):
            if node.token == -1:
                label = "ROOT"
            else:
                text = tokenizer.decode([node.token]) if tokenizer is not None else ""
                label = f"{node.token!r}={text!r} logprob={node.logprob:.4f}"
            pos = f"L{depth} sib {sib_idx + 1}/{n_sibs}"
            print(f"{'  ' * depth}[{pos}] {label} (id={node.element_id}){_flags(node)}")
            child_tokens = sorted(node.children)
            for i, token_id in enumerate(child_tokens):
                _walk(node.children[token_id], depth + 1, i, len(child_tokens))

        _walk(self.root, 0, 0, 1)
        print()
