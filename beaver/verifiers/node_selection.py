"""Pluggable node-selection strategies for the shared frontier branch-and-bound
transition loop (frontier_step.py::frontier_transition).

Frontier (frontier.py) is a pure tree/leaf-set data structure -- it holds no
opinion on which incomplete leaf to expand next and stores no selector-owned
data (e.g. violation estimates). All of that lives here, in NodeSelector and
its subclasses.

Every strategy is implemented ONLY in batch form (`_pick_*_batch`), returning
up to k leaves. k=1 is not a special case: each batch picker degenerates to
exactly the single-node choice, so there is one code path to reason about and
no chance of the two drifting apart.

Expanding k>1 nodes per round is not identical to k sequential expansions: the
selector commits to a pre-round top-k and does not see round i's children
before choosing the rest of round i. A child is always less probable than its
parent so displacement is rare, but the search order does differ -- a
beam-style approximation traded for batched forward passes, not a free win.
"""

import heapq
import json
import random
from pathlib import Path

import torch
import numpy as np

from beaver.nvf import DirectionNVF, KNNEstimator, NonlinearProbe, ViolationProbe

random.seed(42)


def _decoder(tokenizer):
    """Decode helper that tolerates tokenizer=None (used in tests)."""

    def _decoded(x):
        if tokenizer is None:
            return None
        return tokenizer.decode(x.tokens, skip_special_tokens=False)

    return _decoded


def _base_debug(strategy, picked, extra=None):
    """Debug payload logged as ``pick_info`` on every transition.

    Emits both the batch-shaped fields and the singular ``picked_element_id`` /
    ``picked_score`` aliases, so log analysis written against the old
    single-node format keeps working at any k.
    """
    debug = {
        "pick_strategy": strategy,
        "batch_size": len(picked),
        "picked_element_ids": [x.element_id for x in picked],
        "picked_logprobs": [float(x.logprob) for x in picked],
        "picked_element_id": picked[0].element_id if picked else None,
    }
    if extra:
        debug.update(extra)
    return debug


class NodeSelector:
    """Probability-only selection over the frontier's incomplete leaves.

    selection_strategy:
      "max-mu"    -- the k highest-logprob incomplete leaves (default).
      "sample-mu" -- k distinct leaves sampled proportional to exp(logprob).
      "random"    -- k distinct uniformly random leaves.
      "ucb"       -- mu(leaf) x optimistic violation rate borrowed from the
                     nearest ancestor with evidence. See _pick_ucb_batch.
    """

    needs_hidden_states = False

    def __init__(self, selection_strategy="max-mu", ucb_c=1.0):
        self.selection_strategy = selection_strategy
        self.ucb_c = ucb_c
        # UCB statistics, keyed by FrontierElement (identity hash): node ->
        # (visits, resolved violation mass, resolved complete mass). Reset
        # whenever a new instance's Frontier shows up (element_ids restart at
        # 1 per instance, objects do not).
        self._ucb_root = None
        self._ucb_stats = {}

    # ── strategies ─────────────────────────────────────────────────────────

    def _pick_maxmu_batch(self, frontier, k, tokenizer=None):
        incomplete = frontier._incomplete_leaves
        picked = heapq.nlargest(k, incomplete, key=lambda x: x.logprob)
        return picked, _base_debug(
            "max-mu", picked, {"n_incomplete": len(incomplete)}
        )

    def _pick_sample_mu_batch(self, frontier, k, tokenizer=None):
        comp_list = list(frontier._incomplete_leaves)
        dist = torch.exp(torch.tensor([x.logprob for x in comp_list]))
        n = min(k, len(comp_list))
        # replacement=False so a batch never expands the same leaf twice --
        # add_to_element would assert on the second one.
        idx = torch.multinomial(dist, n, replacement=False).tolist()
        picked = [comp_list[i] for i in idx]
        return picked, _base_debug(
            "sample-mu", picked, {"n_incomplete": len(comp_list)}
        )

    def _pick_random_batch(self, frontier, k, tokenizer=None):
        comp_list = list(frontier._incomplete_leaves)
        picked = random.sample(comp_list, min(k, len(comp_list)))
        return picked, _base_debug(
            "random", picked, {"n_incomplete": len(comp_list)}
        )

    # A subtree's violation rate counts as evidence only once this fraction of
    # its mass has resolved; below it, a leaf borrows from further up.
    UCB_MIN_RESOLVED = 0.05

    def _pick_ucb_batch(self, frontier, k, tokenizer=None):
        """Top-k leaves by mass times an optimistic violation rate.

            score(leaf) = mu(leaf) * min(1, q(a) + ucb_c * sqrt(ln N_root / N(a)))

        a    -- the leaf's nearest proper ancestor whose resolved mass
                (V + C) is at least UCB_MIN_RESOLVED * mu(a). A leaf has never
                been expanded, so it has no evidence of its own; it borrows
                the violation rate of the closest subtree that does.
        q(a) -- V(a) / (V(a) + C(a)): the fraction of the mass already
                resolved under a that was violating.
        N    -- expansions inside a subtree (N_root = all expansions so far).

        With no evidence anywhere (early on, or before anything resolves) the
        factor is 1 and this is exactly max-mu. Leaves under prefixes whose
        resolved mass keeps coming out safe sink; leaves under prefixes that
        keep violating rise. The bonus shrinks as a subtree is explored, so a
        safe-looking subtree is still revisited now and then.

        Soundness does not depend on the order: every expansion moves exact
        mass, so this only changes which bounds tighten first.
        """
        if frontier.root is not self._ucb_root:
            self._ucb_root = frontier.root
            self._ucb_stats = {}
        stats = self._ucb_stats  # node -> (visits, violation mass, complete mass)
        incomplete = frontier._incomplete_leaves
        log_n_root = np.log(max(stats.get(frontier.root, (1, 0.0, 0.0))[0], 1))

        # eff[node] = (q, N) of the nearest ancestor-or-self of node with
        # evidence, or None. Filled top-down over expanded nodes only (a leaf
        # uses its parent's entry), so one pick costs O(tree), not O(leaves x
        # depth).
        eff = {}
        stack = [(frontier.root, None)]
        while stack:
            node, inherited = stack.pop()
            n, v, c = stats.get(node, (0, 0.0, 0.0))
            mu = float(np.exp(node.logprob))
            here = (v / (v + c), n) if v + c > 0 and v + c >= self.UCB_MIN_RESOLVED * mu else inherited
            eff[node] = here
            for child in node.children.values():
                if child.children:
                    stack.append((child, here))

        def _factor(leaf):
            src = eff.get(leaf.parent) if leaf.parent is not None else None
            if src is None:
                return 1.0, None
            q, n = src
            return min(1.0, q + self.ucb_c * np.sqrt(log_n_root / max(n, 1))), q

        scored = [(float(np.exp(x.logprob)) * _factor(x)[0], x) for x in incomplete]
        top = heapq.nlargest(k, scored, key=lambda t: t[0])
        picked = [x for _, x in top]
        by_mu = heapq.nlargest(1, incomplete, key=lambda x: x.logprob)
        root = stats.get(frontier.root, (0, 0.0, 0.0))
        return picked, _base_debug(
            "ucb",
            picked,
            {
                "n_incomplete": len(incomplete),
                "ucb_c": self.ucb_c,
                "picked_score": top[0][0] if top else None,
                "picked_scores": [s for s, _ in top],
                "picked_factors": [_factor(x)[0] for x in picked],
                "picked_q": [_factor(x)[1] for x in picked],
                # whether max-mu would have picked the same leaf (k=1 view)
                "same_as_maxmu": bool(picked and by_mu and picked[0] is by_mu[0]),
                "root_visits": root[0],
                "root_q": root[1] / (root[1] + root[2]) if root[1] + root[2] > 0 else None,
            },
        )

    # ── dispatch ───────────────────────────────────────────────────────────

    _STRATEGIES = {
        "max-mu": "_pick_maxmu_batch",
        "sample-mu": "_pick_sample_mu_batch",
        "random": "_pick_random_batch",
        "ucb": "_pick_ucb_batch",
    }

    def pick_batch(self, frontier, k, tokenizer=None):
        """Up to k incomplete leaves to expand together, best first.

        Returns ``([], None)`` when the frontier has no incomplete leaves, which
        the caller treats as termination.
        """
        if not frontier._incomplete_leaves:
            return [], None
        k = max(int(k), 1)
        name = self._STRATEGIES.get(self.selection_strategy)
        if name is None:
            raise ValueError(
                f"Invalid selection strategy: {self.selection_strategy!r}. "
                f"Expected one of {sorted(self._STRATEGIES)}."
            )
        return getattr(self, name)(frontier, k, tokenizer=tokenizer)

    def on_expand(self, element, new_elements, hidden_states, instance):
        if self.selection_strategy != "ucb":
            return
        # Mass this expansion resolved, credited to every ancestor.
        v = sum(float(np.exp(ch.logprob)) for ch in new_elements if ch.is_violation)
        c = sum(float(np.exp(ch.logprob)) for ch in new_elements if ch.is_completed)
        path = []
        node = element
        while node is not None:
            path.append(node)
            node = node.parent
        if path[-1] is not self._ucb_root:  # a new instance's tree
            self._ucb_root = path[-1]
            self._ucb_stats = {}
        for node in path:
            n, v0, c0 = self._ucb_stats.get(node, (0, 0.0, 0.0))
            self._ucb_stats[node] = (n + 1, v0 + v, c0 + c)


class RepresentationSelector(NodeSelector):
    """Scores the just-expanded node with a neural value function (NVF).

    The score (`violation_estimate`) is cached here -- keyed by `element_id` --
    rather than stored on the FrontierElement, since Frontier owns no
    selector-specific data. Because it is the *parent* that gets scored, all
    siblings share one value: the NVF discriminates between subtrees, not
    between siblings.

    nvf_type: "knn" | "linear" | "nonlinear" | "direction".

    selection_strategy:
      "gated"   -- max-mu restricted to nodes whose parent cleared a threshold
                   on the pooled-standardised NVF score. See
                   _pick_gated_batch.
      "fitness" -- exp(logprob) * parent NVF score. UNSAFE for any NVF whose
                   output can go negative (e.g. DirectionNVF's signed sum of
                   cosine similarities): max() over negative products prefers
                   the value nearest zero, i.e. the *least* probable child, and
                   any positive-scoring node outranks every negative-scoring
                   one no matter how improbable. Kept for the
                   probability-valued NVFs (knn/linear/nonlinear) it was
                   written for. Prefer "gated", which only ever compares the
                   score to a threshold.
      "max-mu"/"sample-mu"/"random" -- inherited, NVF unused.
    """

    needs_hidden_states = True

    def __init__(
        self,
        nvf_type="direction",
        probe_path=None,
        probe_layer=None,
        nvf_data_path=None,
        probe_gpu=None,
        knn_k=5,
        knn_normalize=True,
        selection_strategy="gated",
        direction_layers=None,
        direction_layer_weights=None,
        gate_theta=0.0,
        gate_pool_size=64,
    ):
        super().__init__(selection_strategy=selection_strategy)
        self.nvf_type = nvf_type
        self.probe_layer = probe_layer
        device = probe_gpu if probe_gpu is not None else "cpu"
        self._device = device
        self._violation_cache = {}  # element_id -> violation_estimate
        self.gate_theta = gate_theta
        self.gate_pool_size = gate_pool_size
        self._cur_idx = None
        self._pooled_stats = self._load_pooled_stats(probe_path, direction_layers)

        if nvf_type == "knn":
            if nvf_data_path is None:
                raise ValueError(
                    "nvf_data_path is required for KNN neural value function. "
                    "Pass nvf_data_path=<path to .pt file from extract_data.py>."
                )
            self._nvf = KNNEstimator.load(
                nvf_data_path, k=knn_k, normalize=knn_normalize, device=device
            )
        elif nvf_type == "linear":
            if probe_path is None:
                raise ValueError(
                    "probe_path is required for 'linear' neural value function."
                )
            self._nvf = ViolationProbe.load(probe_path, device=device)
        elif nvf_type == "nonlinear":
            if probe_path is None:
                raise ValueError(
                    "probe_path is required for 'nonlinear' neural value function."
                )
            self._nvf = NonlinearProbe.load(probe_path, device=device)
        elif nvf_type == "direction":
            if probe_path is None:
                raise ValueError(
                    "probe_path is required for 'direction' neural value "
                    "function (a directory containing directions.npz from "
                    "violation_direction_heretic.py)."
                )
            self._nvf = DirectionNVF.load(
                probe_path,
                device=device,
                layers=direction_layers,
                layer_weights=direction_layer_weights,
            )
        else:
            raise ValueError(
                f"Unknown nvf_type: '{nvf_type}'. "
                f"Choose 'knn', 'linear', 'nonlinear', or 'direction'."
            )

    # ── gate reference distribution ────────────────────────────────────────

    @staticmethod
    def _load_pooled_stats(probe_path, direction_layers):
        """(mu, sd) of the deployed direction's score over the whole fit set.

        POOLED, from the probe dir's gate_stats.json: the same two numbers are
        used for every instance, so the gate threshold is a property of the
        dataset rather than of whichever instance is being searched. Averaged
        across the scored layers when more than one is in play.

        Returns None when there is no gate_stats.json, which disables the gate
        (pick degenerates to max-mu) -- a loud no-op rather than a silent
        change of behaviour.
        """
        if not probe_path:
            return None
        path = Path(probe_path) / "gate_stats.json"
        if not path.exists():
            print(
                f"[gate] no gate_stats.json in {probe_path}; gate disabled "
                f"(falls back to max-mu). Run experiments.common.fit_gate_stats."
            )
            return None
        d = json.loads(path.read_text())
        # When the direction is scored with non-uniform (e.g. Fisher-LDA)
        # layer weights, the combined score's mu/sd are NOT the average of the
        # per-layer ones -- averaging them would silently mis-standardise and
        # theta would stop meaning what it was calibrated to mean.
        if "combined_mean" in d and "combined_std" in d:
            return (float(d["combined_mean"]), float(d["combined_std"]))
        layers = list(direction_layers) if direction_layers else d["layers"]
        mu = float(np.mean([d["mean"][l] for l in layers]))
        sd = float(np.mean([d["std"][l] for l in layers]))
        return (mu, sd)

    def _gate_stats(self):
        return self._pooled_stats

    def _cached_ve(self, element):
        if element is None:
            return 0.0
        return self._violation_cache.get(element.element_id, 0.0)

    def _z(self, element):
        """NVF score of the element's parent, standardised by the pooled stats."""
        stats = self._gate_stats()
        if stats is None:
            return 0.0
        mu, sd = stats
        return (self._cached_ve(element.parent) - mu) / (sd if sd > 1e-9 else 1.0)

    # ── strategies ─────────────────────────────────────────────────────────

    def _pick_gated_batch(self, frontier, k, tokenizer=None):
        """Top-k gate survivors by logprob, topped up from the pool.

        Three steps: take the top-``gate_pool_size`` leaves by logprob, keep
        those whose parent's standardised NVF score clears ``gate_theta``, and
        return the most probable survivors. If fewer than k clear it, the
        remainder are filled with the most probable non-survivors rather than
        shrinking the batch -- an under-full batch wastes the utilisation that
        batching exists to recover. If none clear it, this is exactly max-mu.

        The pool cap is the safety property: the pick can never leave the top-K
        by probability, so the selector cannot walk into the low-probability
        tail (which resolves no probability mass and so moves neither bound)
        nor fight the frontier's own top-K-by-logprob eviction in
        Frontier.prune_incomplete_leaves.
        """
        incomplete = frontier._incomplete_leaves
        pool = heapq.nlargest(
            max(self.gate_pool_size, k), incomplete, key=lambda x: x.logprob
        )
        stats = self._gate_stats()
        gated, rest = [], []
        if stats is None:
            rest = list(pool)
        else:
            for x in pool:
                (gated if self._z(x) >= self.gate_theta else rest).append(x)

        picked = (gated + rest)[:k]
        gated_ids = {x.element_id for x in gated}
        _decoded = _decoder(tokenizer)
        extra = {
            "picked_score": float(self._z(picked[0])) if picked else None,
            "picked_in_gate": [x.element_id in gated_ids for x in picked],
            # gate configuration + the pooled reference it thresholds against,
            # logged every transition so a run is auditable from its own logs
            # without needing the probe dir.
            "gate_theta": self.gate_theta,
            "gate_mu": stats[0] if stats else None,
            "gate_sd": stats[1] if stats else None,
            "gate_pool_size": self.gate_pool_size,
            "gate_active": bool(gated),
            "n_pool": len(pool),
            "n_gated": len(gated),
            "n_from_gate": sum(1 for x in picked if x.element_id in gated_ids),
            "n_incomplete": len(incomplete),
            # Top of the POOL (not the gated subset), each tagged with whether
            # it cleared the gate -- so the log shows what was rejected as well
            # as what was kept.
            "top_candidates": [
                {
                    "element_id": x.element_id,
                    "logprob": float(x.logprob),
                    "violation_estimate": self._cached_ve(x.parent),
                    "violation_z": float(self._z(x)),
                    "in_gate": x.element_id in gated_ids,
                    "decoded": _decoded(x),
                }
                for x in pool[:10]
            ],
        }
        return picked, _base_debug("gated", picked, extra)

    def _pick_fitness_batch(self, frontier, k, tokenizer=None):
        """Top-k by exp(logprob) * parent NVF score.

        See the class docstring: this ordering inverts whenever the NVF can go
        negative, which is why "gated" exists. Retained for probability-valued
        NVFs only.
        """
        incomplete = frontier._incomplete_leaves

        def _fitness(x):
            return np.exp(x.logprob) * (self._cached_ve(x.parent) + 1e-6)

        ranked = heapq.nlargest(max(k, 10), incomplete, key=_fitness)
        picked = ranked[:k]
        _decoded = _decoder(tokenizer)
        extra = {
            "picked_score": float(_fitness(picked[0])) if picked else None,
            "n_incomplete": len(incomplete),
            "top_candidates": [
                {
                    "element_id": x.element_id,
                    "logprob": float(x.logprob),
                    "violation_estimate": self._cached_ve(x.parent),
                    "fitness": float(_fitness(x)),
                    "decoded": _decoded(x),
                }
                for x in ranked[:10]
            ],
        }
        return picked, _base_debug("fitness", picked, extra)

    _STRATEGIES = dict(
        NodeSelector._STRATEGIES,
        gated="_pick_gated_batch",
        fitness="_pick_fitness_batch",
    )

    # ── NVF bookkeeping ────────────────────────────────────────────────────

    def on_expand(self, element, new_elements, hidden_states, instance):
        super().on_expand(element, new_elements, hidden_states, instance)
        # One selector instance is built per worker *process* and reused across
        # every instance that worker handles, but Frontier restarts
        # total_elements at 1 for each instance -- so element_ids collide across
        # instances and the cache must be dropped at each boundary.
        idx = instance.get("idx") if isinstance(instance, dict) else None
        if idx != self._cur_idx:
            self._cur_idx = idx
            self._violation_cache = {}

        if not hidden_states:
            self._violation_cache[element.element_id] = 0.0
            return
        if self.nvf_type == "direction":
            estimate, _ = self._nvf.predict(
                hidden_states, len(element.tokens), return_info=False
            )
        elif self.nvf_type == "knn":
            hs = next(iter(hidden_states.values()))
            estimate, _ = self._nvf.predict(hs, return_info=False)
        else:
            hs = next(iter(hidden_states.values()))
            with torch.no_grad():
                estimate = self._nvf(hs.float().to(self._device)).item()
        self._violation_cache[element.element_id] = estimate


def build_selector(node_selection, config):
    """Factory: build the NodeSelector configured by a worker config dict.

    Called once per worker process from worker_common.init_worker_state().
    """
    selection_strategy = config.get("frontier_scoring_strategy") or "max-mu"
    if not node_selection:
        return NodeSelector(
            selection_strategy=selection_strategy,
            ucb_c=config.get("ucb_c", 1.0),
        )
    if node_selection == "representation":
        return RepresentationSelector(
            nvf_type=config.get("nvf_type", "knn"),
            probe_path=config.get("probe_path"),
            probe_layer=config.get("probe_layer"),
            nvf_data_path=config.get("nvf_data_path"),
            probe_gpu=config.get("probe_gpu"),
            knn_k=config.get("knn_k", 5),
            knn_normalize=config.get("knn_normalize", True),
            selection_strategy=selection_strategy,
            direction_layers=config.get("direction_layers"),
            direction_layer_weights=config.get("direction_layer_weights"),
            gate_theta=config.get("gate_theta", 0.0),
            gate_pool_size=config.get("gate_pool_size", 64),
        )
    raise ValueError(f"Unknown node_selection: {node_selection!r}")
