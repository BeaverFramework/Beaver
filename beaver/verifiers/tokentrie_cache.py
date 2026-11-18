from __future__ import annotations

import json
import math
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from beaver.verifiers.frontier import Frontier, FrontierElement
from beaver.model import (
    drain_model_timings as drain_http_timings,
    model_score_sequences_per_token,
)
from beaver.verifiers.worker_common import (
    apply_top_p_top_k,
    get_grammar_mask,
    update_frontier,
)


def _trie(seqs) -> dict:
    """Nested-dict token trie of ``seqs`` (token -> child dict)."""
    root: dict = {}
    for seq in seqs:
        node = root
        for tok in seq:
            node = node.setdefault(tok, {})
    return root


def _walk(trie: dict, seq) -> Optional[dict]:
    """The trie node at ``seq`` (``{}`` for a leaf), or None if ``seq`` is not a
    prefix of any inserted sequence."""
    node = trie
    for tok in seq:
        node = node.get(tok)
        if node is None:
            return None
    return node


class TokenTrieCache:
    def __init__(
        self,
        hit_prob: Optional[float] = None,
        evict_after: Optional[int] = None,
        max_size: Optional[int] = None,
    ):
        """``hit_prob`` / ``evict_after`` / ``max_size``: optional eviction of
        cached violation and completion leaves that carry little mass.

        Each warm start counts, for every cached leaf, the consecutive warm
        starts in which it was *missed* (its path probability under that
        instance's prompt was ``< hit_prob``, or the path was not rebuilt).
        A warm start never evicts. Eviction happens only when the horizon is
        updated (``merge_with_frontier``) and violations + completions then
        exceed ``max_size``: the leaves with the most misses are removed
        first, and a leaf with fewer than ``evict_after`` misses is never
        removed (the cache can stay above ``max_size``). Thus instances that
        the warm start settles do not shrink the cache. Incomplete leaves keep
        their own limit (``max_limit_incomplete``).

        Eviction never makes a bound unsound: ``build_frontier_from_cache``
        re-expands every sibling under the new prompt, so the mass of an
        evicted leaf comes back as a fresh leaf (or inside an ancestor) and
        the main loop explores it again. It only makes the warm start
        cheaper (fewer parent paths to score) and less complete."""
        self.violations: List[List[int]] = []
        self.completions: List[List[int]] = []
        self.incomplete: List[List[int]] = []

        self.violation_sources: Dict[Tuple[int, ...], object] = {}
        self.completion_sources: Dict[Tuple[int, ...], object] = {}

        # Incomplete leaves kept across merges (the shortest survive). Env
        # BEAVER_CACHE_MAX_INCOMPLETE overrides the default; "none" = no cap.
        cap = os.environ.get("BEAVER_CACHE_MAX_INCOMPLETE", "500")
        self.max_limit_incomplete = None if cap.lower() in ("none", "0", "inf") else int(cap)

        self.hit_prob = hit_prob
        self.evict_after = evict_after
        self.max_size = max_size
        self.track_hits = None not in (hit_prob, evict_after, max_size)
        # tuple(seq) -> consecutive warm starts in which the leaf was missed.
        self.misses: Dict[Tuple[int, ...], int] = {}
        self.last_evicted = 0
        self.evicted_total = 0

    def cache_violations(self, violation_seqs: List[List[int]], source=None) -> None:
        """Insert violation sequences, maintaining antichain invariant and trie labels."""
        for seq in violation_seqs:
            if self._check_antichain(seq, self.violations):
                self.violations.append(seq)
                self.violation_sources[tuple(seq)] = source
            else:
                # print(f"Found Antichain trying to be added in violations {seq} in {self.violations}")
                continue

    def cache_completions(self, completion_seqs: List[List[int]], source=None) -> None:
        """Insert completion sequences, maintaining antichain invariant and trie labels."""
        for seq in completion_seqs:
            if self._check_antichain(seq, self.completions):
                self.completions.append(seq)
                self.completion_sources[tuple(seq)] = source
            else:
                # print(f"Found Antichain trying to be added in completions {seq} in {self.completions}")
                continue

    def replace_incomplete(self, incomplete_seqs: List[List[int]]) -> None:
        """Merge this round's incomplete leaves into the cache.

        Per Proposition 4.5 (merge soundness), the merged incomplete set is
        ``I(H) ∪ I(H') \\ E`` — not a plain union — where ``E`` is every old
        incomplete leaf that got *extended* this round (its path is now a
        strict prefix of some node in the merge), because whatever it became
        is already correctly represented there instead.

        ``_build_large_antichain`` below already drops an old leaf that got
        extended into a *deeper incomplete* leaf (the extension lands back in
        this same list, so the prefix collapse catches it). It can't see an
        old leaf that got extended into a *violation or completion* instead —
        that extension lives in ``self.violations``/``self.completions``, a
        different list entirely — so that case needs an explicit check
        against those two lists (call ``merge_with_frontier``, not this
        directly, so violations/completions are already updated for this
        round by the time this runs).
        """
        combined = self._build_large_antichain(
            self.incomplete + [list(seq) for seq in incomplete_seqs]
        )
        resolved = _trie(self.violations + self.completions)
        self.incomplete = [
            seq for seq in combined if not resolved or _walk(resolved, seq) is None
        ]
        if (
            self.max_limit_incomplete is not None
            and len(self.incomplete) > self.max_limit_incomplete
        ):
            # sort with shortest first and keep top N
            self.incomplete = sorted(self.incomplete, key=len)[
                : self.max_limit_incomplete
            ]

    def _record_hits(self, nodes_by_tokens) -> None:
        """Update the miss counts of the cached violation and completion
        leaves after a warm start. A cached leaf that is not in the rebuilt
        trie (its path fell outside top-p/top-k under this prompt) is a miss.
        Leaves that left the cache by the antichain merge are forgotten."""
        log_hit = math.log(self.hit_prob)
        misses = {}
        for chain in (self.violations, self.completions):
            for seq in chain:
                key = tuple(seq)
                node = nodes_by_tokens.get(key)
                if node is not None and node.logprob >= log_hit:
                    misses[key] = 0
                else:
                    misses[key] = self.misses.get(key, 0) + 1
        self.misses = misses

    def _evict_overflow(self) -> int:
        """After a horizon update: while violations + completions exceed
        ``max_size``, remove the leaves with the most misses, but only leaves
        with at least ``evict_after`` misses. Returns the number removed."""
        over = len(self.violations) + len(self.completions) - self.max_size
        if over <= 0:
            return 0
        candidates = sorted(
            (k for k, v in self.misses.items() if v >= self.evict_after),
            key=lambda k: self.misses[k],
            reverse=True,
        )
        evict = set(candidates[:over])
        if not evict:
            return 0
        self.violations = [s for s in self.violations if tuple(s) not in evict]
        self.completions = [s for s in self.completions if tuple(s) not in evict]
        for key in evict:
            del self.misses[key]
            self.violation_sources.pop(key, None)
            self.completion_sources.pop(key, None)
        self.evicted_total += len(evict)
        return len(evict)

    def _check_antichain(self, seq: List[int], antichain: List[List[int]]) -> bool:
        seq_t = tuple(seq)
        for existing in antichain:
            ex_t = tuple(existing)
            if seq_t[: len(ex_t)] == ex_t:
                return False
            elif ex_t[: len(seq_t)] == seq_t:
                return False
        return True

    def _build_large_antichain(self, chain):
        """Keep only the *maximal* prefixes: drop any seq that is a prefix of
        another. Expanding along a kept (longer) path expands every node on it,
        so dropped prefixes are still materialized. Dedups + tolerates lists."""
        seqs = sorted({tuple(s) for s in chain}, key=len, reverse=True)
        trie = _trie(seqs)
        # A seq is a prefix of a longer one iff its trie node has children.
        return [list(s) for s in seqs if not _walk(trie, s)]

    def _score(
        self,
        instance: dict,
        seqs: List[List[int]],
    ) -> List[List[float]]:
        """Score an antichain, returning ``(per_token_logprobs, per_position_dists)``.

        ``per_position_dists[i][j]`` is the full next-token distribution at
        position ``j`` of ``seqs[i]`` (an ``[N, 2]`` ``[token_id, logprob]``
        array), reused by the rebuild to recover siblings without re-querying.
        """
        if not seqs:
            return []
        return model_score_sequences_per_token(instance, seqs)

    @staticmethod
    def _antichain_mass(per_tok: List[List[float]]) -> float:
        """``sum(exp(sum(per_token_logprobs_i)))`` over an antichain (matching
        ``Frontier.calc_prob_sum`` semantics)."""
        return sum(math.exp(sum(lps)) for lps in per_tok)

    def _get_distribution(self, instance, parents):
        # --- ORIGINAL: echo=True, one full-path scoring call per parent. Each
        # call re-echoes the whole shared prompt prefix, which vLLM must
        # recompute+serialize top-k logprobs for on every parent, every
        # instance -- expensive at scale. Uncomment these two lines (and
        # comment out the block below) to revert.
        _, distributions = model_score_sequences_per_token(
            instance, parents, return_dists=True
        )
        return distributions

    def build_frontier_from_cache(
        self, instance, frontier, early_termination_threshold
    ):
        """Warm-start a Frontier from this cache for ``instance``.

        Scores violations + completions first (cheap antichains). If either
        crosses the early-termination gate, returns those masses and skips the
        (potentially massive) incomplete set and sibling expansion. Otherwise
        scores incomplete too, builds the resumable trie, and — crucially —
        re-expands every on-path node's siblings under *this* prompt so that
        high-prob branches the cache never saw (sub-epsilon for earlier
        instances) are materialized as fresh incomplete leaves rather than
        silently dropped (which would understate the upper bound).

        Returns ``(violation_prob, completion_prob, incomplete_prob, pruned_prob,
        forward_passes, timings, semantic_attempted)``.

        ``pruned_prob`` seeds the caller's running pruned mass so the bound stays
        sound. On the early-gate path ``incomplete_prob`` is None and
        ``pruned_prob`` is 0.0. ``timings`` breaks the call down into its
        internal sections (seconds each) so callers can log/profile them.
        ``semantic_attempted`` is the gross number of semantic checks run during
        warm-start sibling classification (0 on the empty-cache path, which
        never runs ``_classify_leaves``).
        """

        if not self.violations and not self.completions and not self.incomplete:
            return 0.0, 0.0, 1.0, 0.0, 0, {}, 0

        timings = {}
        t0 = time.perf_counter()

        def _tick(label):
            nonlocal t0
            now = time.perf_counter()
            timings[label] = now - t0
            t0 = now


        # Parents = prefixes (drop last token) of every cached terminal. Walking
        # + expanding these paths rebuilds the trie and recovers siblings.
        all_parents = (
            [i[:-1] for i in self.incomplete]
            + [i[:-1] for i in self.violations]
            + [i[:-1] for i in self.completions]
        )
        parents = self._build_large_antichain(all_parents)
        _tick("build_antichain")

        forward_passes = len(parents)

        distributions = self._get_distribution(instance, parents)
        _tick("get_distribution")
        # Keys already carry their backend prefix ("http."/"local.") from
        # drain_model_timings -- don't add another one.
        for http_label, agg in drain_http_timings().items():
            for key, val in agg.items():
                timings[f"{http_label}.{key}"] = val

        leaves_to_check, pruned_prob = self._insert_parents(
            frontier, parents, distributions
        )
        _tick("insert_parents")

        nodes_by_tokens = (
            {tuple(n.tokens): n for n in leaves_to_check}
            if self.track_hits
            else None
        )
        violations = {tuple(s) for s in self.violations}
        completions = {tuple(s) for s in self.completions}
        incomplete = {tuple(s) for s in self.incomplete}
        for n in list(leaves_to_check):
            tokens = tuple(n.tokens)
            if tokens in violations:
                leaves_to_check.discard(n)
                self._flag_terminal(frontier, n, "violation")
            elif tokens in completions:
                leaves_to_check.discard(n)
                self._flag_terminal(frontier, n, "complete")
            elif tokens in incomplete:
                leaves_to_check.discard(n)
                self._flag_terminal(frontier, n, "incomplete")
        _tick("flag_terminals")

        if nodes_by_tokens is not None:
            self._record_hits(nodes_by_tokens)
            _tick("record_hits")

        warm_semantic_attempted = self._classify_leaves(
            frontier, instance, leaves_to_check
        )
        _tick("classify_leaves")

        inc_prob, comp_prob, viol_prob = frontier.calc_prob_sum()
        _tick("calc_prob_sum")

        return (
            float(viol_prob),
            float(comp_prob),
            float(inc_prob),
            float(pruned_prob),
            forward_passes,
            timings,
            warm_semantic_attempted,
        )

    def _insert_parents(
        self,
        frontier: Frontier,
        seqs: List[List[int]],
        dists: List[List],
    ):
        """Insert antichain paths into ``frontier``'s trie, flagging terminals.

        Each ``seq`` is walked from the root, creating/following child nodes.
        A node's ``logprob`` is the cumulative sum of per-token logprobs up to
        that token. The terminal node is flagged per ``kind`` and registered in
        the matching leaf set. Intermediate nodes carry no flag. Shared prefixes
        reuse existing nodes (the per-token score is identical under the same
        prompt, so we descend without overwriting).

        While walking, each node's children-distribution (``per_dist[depth]``,
        i.e. ``P(. | prompt, seq[:depth])``) is recorded in ``node_dists`` once
        per node, for the subsequent sibling-expansion pass.
        """
        leaves_to_check = set()
        expanded_nodes = set()
        culled_prob_sum = 0.0
        for seq, dist in zip(seqs, dists):
            node = frontier.root
            for depth, on_path_token in enumerate(seq):
                frontier._incomplete_leaves.discard(node)
                leaves_to_check.discard(node)
                if node not in expanded_nodes:
                    # Expand node once: attach every top-p/top-k child under this
                    # prompt's distribution. Unlabeled for now — classified later.
                    token_dist, _ = apply_top_p_top_k(dist[depth])
                    culled_prob_sum += np.exp(node.logprob) * max(
                        1.0 - float(np.sum(np.exp(token_dist[:, 1]))), 0.0
                    )

                    for tok, logprob in token_dist:
                        tok = int(tok)
                        child = FrontierElement(
                            element_id=frontier.total_elements,
                            token=tok,
                            logprob=node.logprob + float(logprob),
                        )
                        frontier.total_elements += 1
                        node.add_child(tok, child)
                        leaves_to_check.add(child)
                    expanded_nodes.add(node)

                # Descend the cached path. If the on-path token fell outside
                # top-p/top-k under *this* prompt the child is missing — the
                # cached path is not reproducible here (distribution shifted).
                child = node.get_child(on_path_token)
                node = child
                if node is None:
                    # print(
                        # f"on-path token {on_path_token} at depth {depth} fell outside top-p/top-k under this prompt; cached path {seq} not reproducible"
                    # )
                    break

            # add children of parents (since the sequeces are parents)
            if node is not None:
                frontier._incomplete_leaves.discard(node)
                leaves_to_check.discard(node)
                if node not in expanded_nodes:
                    frontier._incomplete_leaves.discard(node)
                    token_dist, _ = apply_top_p_top_k(dist[len(seq)])
                    culled_prob_sum += np.exp(node.logprob) * max(
                        1.0 - float(np.sum(np.exp(token_dist[:, 1]))), 0.0
                    )

                    for tok, logprob in token_dist:
                        tok = int(tok)
                        child = FrontierElement(
                            element_id=frontier.total_elements,
                            token=tok,
                            logprob=node.logprob + float(logprob),
                        )
                        frontier.total_elements += 1
                        node.add_child(tok, child)
                        leaves_to_check.add(child)
                    expanded_nodes.add(node)

        return leaves_to_check, culled_prob_sum

    def _classify_leaves(self, frontier, instance, leaves_to_check):
        """Label cache-unseen sibling leaves by running the real constraint.

        ``leaves_to_check`` are freshly materialized siblings whose full token
        path matched no cached antichain, so their label is unknown under this
        prompt. Group by parent and replay ``update_frontier`` (grammar mask +
        completion + semantic check) — the same path the main loop uses — to
        assign violation / complete / incomplete. Tokens dropped by the grammar
        mask are not valid continuations, so their leaves are removed from the
        trie.

        Returns ``n_semantic_attempted``, the gross number of semantic
        checks run while classifying these warm-start siblings, so the
        caller can attribute warm-start semantic cost the same way it
        attributes warm FP.
        """
        n_attempted = 0
        if not leaves_to_check:
            return n_attempted

        by_parent: Dict[FrontierElement, List[FrontierElement]] = defaultdict(list)
        for leaf in leaves_to_check:
            by_parent[leaf.parent].append(leaf)

        for parent, leaves in by_parent.items():
            # Reconstruct the [token_id, per-token logprob] rows update_frontier
            # expects (per-token logprob = child cumulative - parent cumulative).
            log_probs = np.array(
                [[float(leaf.token), leaf.logprob - parent.logprob] for leaf in leaves]
            )
            vocab_mask = get_grammar_mask(parent.tokens)
            new_elements, _pre, _sem, att = update_frontier(
                frontier, instance, parent, log_probs, vocab_mask, return_counts=True
            )
            n_attempted += att

            label = {}
            for e in new_elements:
                if e.is_violation:
                    label[e.token] = "violation"
                elif e.is_completed:
                    label[e.token] = "complete"
                else:
                    label[e.token] = "incomplete"

            for leaf in leaves:
                kind = label.get(leaf.token)
                if kind is None:
                    # Grammar-invalid: drop the leaf entirely.
                    frontier._incomplete_leaves.discard(leaf)
                    if leaf.parent is not None:
                        leaf.parent.remove_child(leaf.token)
                    continue
                self._flag_terminal(frontier, leaf, kind)

        return n_attempted

    def _flag_terminal(
        self, frontier: Frontier, node: FrontierElement, kind: str
    ) -> None:
        if kind == "violation":
            assert not node.is_completed, "Node cannot be both violation and completion"
            node.is_violation = True
            frontier._incomplete_leaves.discard(node)
            frontier._complete_leaves.discard(node)
            frontier._violation_leaves.add(node)
        elif kind == "complete":
            assert not node.is_violation, "Node cannot be both violation and completion"
            node.is_completed = True
            frontier._incomplete_leaves.discard(node)
            frontier._violation_leaves.discard(node)
            frontier._complete_leaves.add(node)
        elif kind == "incomplete":
            # Only an incomplete leaf if not already claimed as a terminal of
            # another set (a prefix can be a full path elsewhere).
            if (
                node not in frontier._violation_leaves
                and node not in frontier._complete_leaves
            ):
                frontier._incomplete_leaves.add(node)
        else:
            raise ValueError(f"Unknown terminal kind: {kind}")

    def merge_with_frontier(self, frontier, source=None) -> None:
        """Fold a frontier's three leaf sets into this cache -- H ⊔ H' where
        H is this cache and H' is ``frontier`` (Definition 4.2 / Proposition
        4.5: merge of constraint horizons).

        Violations and completions accumulate as antichains (each leaf's full
        token path ``leaf.tokens`` becomes an entry): ``V(H⊔H') = V(H)∪V(H')``,
        ``C(H⊔H') = C(H)∪C(H')``. Incomplete leaves merge as
        ``I(H⊔H') = I(H)∪I(H') \\ E`` (``replace_incomplete``, despite its
        name, does not just replace -- an old incomplete leaf only drops out
        if it was actually extended this round). Violations/completions must
        be folded in first so ``replace_incomplete`` can see this round's
        newly-resolved entries when computing ``E``. ``source`` (e.g. the
        instance idx/file) is recorded as provenance for debugging. Mutates
        the cache in place.
        """
        self.cache_violations(
            [leaf.tokens for leaf in frontier._violation_leaves], source
        )
        self.cache_completions(
            [leaf.tokens for leaf in frontier._complete_leaves], source
        )
        self.replace_incomplete([leaf.tokens for leaf in frontier._incomplete_leaves])
        self.last_evicted = self._evict_overflow() if self.track_hits else 0

    @staticmethod
    def _dump_sources(antichain, sources):
        """Serialize a ``tuple(seq) -> source`` map as a list aligned to
        ``antichain`` order (JSON has no tuple keys)."""
        return [sources.get(tuple(seq)) for seq in antichain]

    @staticmethod
    def _load_sources(antichain, source_list):
        """Rebuild the ``tuple(seq) -> source`` map from the aligned list."""
        if not source_list:
            return {}
        return {
            tuple(seq): src
            for seq, src in zip(antichain, source_list)
            if src is not None
        }

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "violations": self.violations,
            "completions": self.completions,
            "incomplete": self.incomplete,
            "violation_sources": self._dump_sources(
                self.violations, self.violation_sources
            ),
            "completion_sources": self._dump_sources(
                self.completions, self.completion_sources
            ),
        }
        with open(path, "w") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path: Path, **kwargs) -> "TokenTrieCache":
        """A cache saved by ``save``. ``kwargs`` go to ``__init__`` (eviction
        settings); the miss counts start at zero."""
        path = Path(path)
        with open(path, "r") as f:
            data = json.load(f)
        cache = cls(**kwargs)
        cache.violations = [list(v) for v in data["violations"]]
        completions = data.get("completions", data.get("satisfactions", []))
        cache.completions = [list(c) for c in completions]
        cache.incomplete = [list(i) for i in data.get("incomplete", [])]
        cache.violation_sources = cls._load_sources(
            cache.violations, data.get("violation_sources")
        )
        cache.completion_sources = cls._load_sources(
            cache.completions, data.get("completion_sources")
        )
        return cache

    def print_cache(self, tokenizer=None):
        print(f"Cache Statistics:")
        print(f"#Violations: {len(self.violations)}")
        print(f"#Completions: {len(self.completions)}")
        print(f"#Incomplete: {len(self.incomplete)}")
        if tokenizer:
            for sname, s in [
                ("Violations", self.violations),
                ("Completions", self.completions),
                ("Incomplete", self.incomplete),
            ]:
                pviol = [(v, tokenizer.decode(v)) for v in s]
                print(f"{sname}:")
                for k, v in pviol:
                    print(f'\t{k}, "{v}"')
