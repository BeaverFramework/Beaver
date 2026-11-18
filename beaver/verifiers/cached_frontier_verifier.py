"""Frontier verifier with in-memory token-trie cache grown across a distribution.

Instances are processed sequentially. After each instance the explored trie and
violation antichain are merged into a running cache that warm-starts the next
instance. The cache is only written to disk (if cache_path is set) at the very
end of the run.
"""

import json
import multiprocessing as mp
import os
import time
import traceback
from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

from beaver.utils import log_json
from beaver.verifiers.base_verifier import BaseVerifier
from beaver.verifiers.frontier import Frontier
from beaver.verifiers.frontier_step import (
    build_instance_complete,
    check_termination,
    frontier_transition,
)
from beaver.verifiers.tokentrie_cache import TokenTrieCache
from beaver.verifiers.worker_common import (
    _w,
    init_worker_state,
    log_profiling,
    worker_setup,
)

# Progress counters, set once per worker process by _init_worker (Value
# objects under the spawn start method can only be shared via a Pool's
# initializer/initargs, not as regular per-task arguments — see _init_worker).
_n_done = None
_total_fp = None


def _init_worker(config, n_done, total_fp):
    global _n_done, _total_fp
    _n_done = n_done
    _total_fp = total_fp
    init_worker_state(config)


def _process_instance(
    instance, run_log_dir, cache: TokenTrieCache, group_position=None, return_frontier=False
):
    """Process one instance, warm-starting from the given cache. Updates cache in process

    ``group_position`` is the 0-based order in which this instance ran within its
    group (instances of a group run sequentially, each warm-starting from the
    cache the previous ones grew). Logged so plots can use the true order instead
    of inferring it from warm-cache size.

    Returns:
        result_dict for this instance. If ``return_frontier`` is True, it
        returns ``(result_dict, frontier)``. IterativeVerifier uses the
        frontier leaves in the mutator prompt.
    """
    instance, log_file, profile_log_file = worker_setup((instance, run_log_dir))

    instance_start_time = time.time()

    cache_violations_before = len(cache.violations)
    cache_completions_before = len(cache.completions)

    frontier = Frontier(max_size=_w.gen_length)

    # Warm-start the frontier (and prob masses) from the cached trie. The
    # rebuild re-expands on-path siblings under this prompt; warm_pruned is the
    # mass it culled during that expansion, used to seed pruned_prob_sum below.
    warm_start_begin = time.time()
    (
        violation_prob_sum,
        complete_prob_sum,
        incomplete_prob_sum,
        warm_pruned,
        warm_start_forward_passes,
        warm_start_section_timings,
        warm_semantic_attempted,
    ) = cache.build_frontier_from_cache(
        instance, frontier, _w.early_termination_threshold
    )
    warm_start_time = time.time() - warm_start_begin

    # print("frontier stats:")
    # frontier.debug_frontier(_w.tokenizer)
    # frontier.print_frontier(_w.tokenizer)
    # breakpoint()

    warm_stats = {
        "group": instance.get("group"),
        "group_position": group_position,
        "cache_violations": cache_violations_before,
        "cache_completions": cache_completions_before,
        "cache_incomplete": len(cache.incomplete),
        "cache_total": (
            cache_violations_before + cache_completions_before + len(cache.incomplete)
        ),
        "warm_violation_prob": violation_prob_sum,
        "warm_completion_prob": complete_prob_sum,
        "warm_incomplete_prob": incomplete_prob_sum,
        "warm_start_time": warm_start_time,
        "warm_start_forward_passes": warm_start_forward_passes,
        "warm_start_semantic_checks": warm_semantic_attempted,
    }
    log_json({"event": "warm_start", **warm_stats}, log_file)
    log_profiling(
        {
            "event": "warm_start",
            "idx": instance.get("idx"),
            "group": instance.get("group"),
            "group_position": group_position,
            "warm_start_time": warm_start_time,
            **warm_start_section_timings,
        },
        profile_log_file,
    )

    if _w.verbose:
        print(f"\n[warm_start] idx={instance['idx']} ")
        print(json.dumps(warm_stats))

    pruned_prob_sum = warm_pruned
    transitions = warm_start_forward_passes
    running_results = {}

    # Seeded from the warm start; mutated in place by each main-loop
    # transition (see frontier_transition below). Must exist before the
    # first check_termination call, not just the main loop -- it's also
    # used to decide whether warm-start alone already resolved the instance
    # (the warm_stop path below).
    state = {
        "incomplete": incomplete_prob_sum,
        "complete": complete_prob_sum,
        "pruned": pruned_prob_sum,
        "violation": violation_prob_sum,
    }

    warm_stop, warm_reason = check_termination(
        state,
        frontier,
        transitions,
    )

    # IterativeVerifier: stop when the instance cannot enter the beam, i.e.
    # its violation upper bound (1 - certified safe mass) is below the
    # smallest violation lower bound in the beam. Its lower bound stays sound,
    # it is only not tight.
    beam_threshold = instance.get("beam_entry_threshold")

    def _beam_unreachable():
        return beam_threshold is not None and 1.0 - state["complete"] < beam_threshold

    if not warm_stop and _beam_unreachable():
        warm_stop, warm_reason = True, "beam_unreachable"

    if warm_stop:
        cache_extra = {
            **warm_stats,
            "warm_skip": True,
            "violation_antichain_size": len(cache.violations),
            "completion_antichain_size": len(cache.completions),
            "incomplete_antichain_size": len(cache.incomplete),
            # Warm-skip does no main-loop search, so the grand total is just the
            # warm-start sibling-classification checks.
            "total_semantic_checks": warm_semantic_attempted,
        }
        result = build_instance_complete(
            instance,
            frontier,
            termination_reason=warm_reason,
            state=state,
            transitions=transitions,
            instance_run_time=time.time() - instance_start_time,
            cache_extra=cache_extra,
        )
        log_json(result, log_file)
        return (result, frontier) if return_frontier else result

    # ── Main exploration loop ─────────────────────────────────────────────
    # Grand-total semantic checks (gross) for this instance, seeded with the
    # warm-start sibling classifications and grown by each main-loop transition.
    total_semantic_attempted = warm_semantic_attempted
    while transitions < _w.max_iterations:
        status, step_results, reason = frontier_transition(
            frontier, instance, log_file, profile_log_file, state, transitions
        )
        if status == "empty":
            termination_reason = reason
            break

        running_results = step_results
        transitions += 1
        total_semantic_attempted += step_results.get("num_semantic_checks", 0)

        if status != "stop" and _beam_unreachable():
            status, reason = "stop", "beam_unreachable"

        if status == "stop":
            termination_reason = reason
            if _w.verbose:
                print(
                    f"Ending frontier analysis {instance['idx']} "
                    f"reason={termination_reason}"
                )
            break
    else:
        termination_reason = "max_iterations"

    cache.merge_with_frontier(frontier, source=instance.get("file", instance["idx"]))

    cache_extra = {
        **warm_stats,
        "warm_skip": False,
        # Leaves evicted by this horizon update (TokenTrieCache max_size).
        "cache_evicted": cache.last_evicted,
        "violation_antichain_size": len(cache.violations),
        "completion_antichain_size": len(cache.completions),
        "incomplete_antichain_size": len(cache.incomplete),
        # Grand total: warm-start sibling checks + every main-loop transition's.
        "total_semantic_checks": total_semantic_attempted,
    }
    result = build_instance_complete(
        instance,
        frontier,
        termination_reason=termination_reason,
        state=state,
        transitions=transitions,
        instance_run_time=time.time() - instance_start_time,
        last_transition=running_results or None,
        cache_extra=cache_extra,
    )
    log_json(result, log_file)
    # print(f"[frontier_stats] idx={instance['idx']} {json.dumps(result)}")

    if _w.verbose:
        print(
            f"\n[instance_complete] idx={instance['idx']}  transitions={transitions}\n"
            f"  new_violations={result['new_violations_found']}  "
            f"violation_antichain={result['violation_antichain_size']}  "
            f"new_completions={result['new_completions_found']}  "
            f"completion_antichain={result['completion_antichain_size']}  "
            f"incomplete_antichain={result['incomplete_antichain_size']}\n"
            f"  final: incomplete={state['incomplete']:.4f}  "
            f"complete={state['complete']:.4f}  "
            f"violation={state['violation']:.4f}  "
            f"pruned={state['pruned']:.4f}"
        )

    # The returned record carries the same uniform keys as every other path
    # (result) plus the last transition's running detail.
    result = {**result, **running_results}
    return (result, frontier) if return_frontier else result


def _initial_cache(group) -> TokenTrieCache:
    """The cache a group starts from: empty, or -- when BEAVER_CACHE_INIT_DIR
    is set -- the trie an earlier run saved for this group (<dir>/<group>.json),
    e.g. a model with the same tokenizer on the same grouped dataset. The warm
    start re-scores every inherited sequence under the current model and
    prompt, and leaf labels depend only on the text and the group's (shared)
    constraint target, so bounds stay sound."""
    init_dir = os.environ.get("BEAVER_CACHE_INIT_DIR")
    if init_dir:
        path = Path(init_dir) / f"{group}.json"
        if path.exists():
            return TokenTrieCache.load(path)
    return TokenTrieCache()


def _worker_process_group(args):
    """Top-level function for multiprocessing — processes one group's
    instances sequentially in this worker process.

    Only used by the multi-worker Pool path (one group per Pool task); the
    single-process fallback in ``__call__`` runs a flat per-instance loop
    directly instead (see there for why: it doesn't need a picklable
    per-group task function, so it can just update the progress bar after
    every real instance without this indirection).

    A group's cache never needs to be visible to any other group or worker
    (each instance only warm-starts from its own group's prior members), so
    it's created fresh, process-local, right here — no cross-process sharing
    required. This is what lets groups run under a real ``multiprocessing.Pool``
    (one OS process + GIL per worker) instead of threads sharing one GIL.

    The module-level ``_n_done``/``_total_fp`` counters (set by _init_worker)
    are bumped after every *instance*, so the caller can poll them for a live
    progress bar without waiting for the whole group (which may be dozens of
    instances) to return over the Pool's IPC.
    """
    group, instances, run_log_dir, cache_path = args
    cache = _initial_cache(group)
    group_results = []
    for pos, instance in enumerate(instances):
        try:
            result = _process_instance(
                instance, run_log_dir, cache, group_position=pos
            )
        except Exception as e:
            tb = traceback.format_exc()
            print(
                f"[CachedFrontierVerifier] Instance "
                f"{instance.get('idx')} failed: {e}\n{tb}"
            )
            result = {
                "idx": instance.get("idx"),
                "total_transitions": 0,
                "instance_run_time": 0.0,
                "error": str(e),
            }
        group_results.append(result)
        cache.save(Path(cache_path) / f"{group}.json")
        if _n_done is not None:
            with _n_done.get_lock():
                _n_done.value += 1
                _total_fp.value += result.get("total_transitions", 0)
    return group, group_results


class CachedFrontierVerifier(BaseVerifier):
    def __init__(self, model, dataset, server_addr, **kwargs):
        super().__init__(model, dataset, server_addr, **kwargs)
        self.frontier_topp = kwargs.get("max_frontier_prob", 1.0)
        self.frontier_topk = kwargs.get("max_frontier_size", -1)
        self.node_selection = kwargs.get("node_selection", None)
        default_strategy = (
            "fitness" if self.node_selection == "representation" else "max-mu"
        )
        self.frontier_scoring_strategy = kwargs.get(
            "frontier_scoring_strategy", default_strategy
        )
        self.cache_path = kwargs.get("cache_path", None)
        # node_selection="representation" config -- unused otherwise.
        self.nvf_type = kwargs.get("nvf_type", "knn")
        self.probe_path = kwargs.get("probe_path", None)
        self.probe_layer = kwargs.get("probe_layer", None)
        self.nvf_data_path = kwargs.get("nvf_data_path", None)
        self.probe_gpu = kwargs.get("probe_gpu", None)
        self.knn_k = kwargs.get("knn_k", 5)
        self.knn_normalize = kwargs.get("knn_normalize", True)
        # "gated" scoring-strategy knobs -- unused by the other strategies.
        self.direction_layers = kwargs.get("direction_layers", None)
        self.direction_layer_weights = kwargs.get("direction_layer_weights", None)
        self.expand_batch_size = kwargs.get("expand_batch_size", 1)
        self.stopping_criterion = kwargs.get("stopping_criterion", "max_seq_prob")
        self.gate_theta = kwargs.get("gate_theta", 0.0)
        self.gate_pool_size = kwargs.get("gate_pool_size", 64)
        self.gate_warmup = kwargs.get("gate_warmup", 20)
        self.ucb_c = kwargs.get("ucb_c", 1.0)

    def _build_worker_config(self):
        # Add the frontier settings to the base config.
        # Subclasses (IterativeVerifier) use this method and get the same config.
        config = super()._build_worker_config()
        config["frontier_topp"] = self.frontier_topp
        config["frontier_topk"] = self.frontier_topk
        config["frontier_scoring_strategy"] = self.frontier_scoring_strategy
        config["node_selection"] = self.node_selection
        config["nvf_type"] = self.nvf_type
        config["probe_path"] = self.probe_path
        config["probe_layer"] = self.probe_layer
        config["nvf_data_path"] = self.nvf_data_path
        config["probe_gpu"] = self.probe_gpu
        config["knn_k"] = self.knn_k
        config["knn_normalize"] = self.knn_normalize
        config["direction_layers"] = self.direction_layers
        config["direction_layer_weights"] = self.direction_layer_weights
        config["expand_batch_size"] = self.expand_batch_size
        config["stopping_criterion"] = self.stopping_criterion
        config["gate_theta"] = self.gate_theta
        config["gate_pool_size"] = self.gate_pool_size
        config["gate_warmup"] = self.gate_warmup
        config["ucb_c"] = self.ucb_c
        return config

    def __call__(self, dataset, run_log_dir):
        config = self._build_worker_config()

        for i, inst in enumerate(dataset):
            if "origin_code" in inst:
                dataset[i]["origin_code_ids"] = self.tokenizer.encode(
                    inst["origin_code"], add_special_tokens=False
                )
            else:
                dataset[i]["origin_code_ids"] = []

        if self.cache_path is None:
            self.cache_path = run_log_dir / "frontier_cache"
        Path(self.cache_path).mkdir(parents=True, exist_ok=True)

        # ── Bucket instances by group (preserving order within a group) ──
        missing = [inst.get("idx") for inst in dataset if "group" not in inst]
        if missing:
            raise ValueError(
                f"{type(self).__name__} needs a 'group' key on every instance "
                f"(instances in a group share one token-trie cache); "
                f"{len(missing)} instance(s) have none, e.g. idx={missing[0]}. "
                f"Build a grouped dataset with beaver.grouping.tv_group_dataset, "
                f"or use verifier='frontier'."
            )
        groups = OrderedDict()
        for instance in dataset:
            groups.setdefault(instance["group"], []).append(instance)

        results_by_group = defaultdict(list)
        bar = tqdm(total=len(dataset), desc="Processing instances (cached)")

        def _refresh_bar(n_done, total_fp):
            bar.n = n_done
            bar.set_postfix_str(f"total_fp={total_fp}  avg={total_fp / max(n_done, 1):.1f}/inst")
            bar.refresh()

        if self.max_workers and self.max_workers > 1 and len(groups) > 1:
            ctx = mp.get_context("spawn")
            # Value objects can only cross a spawn Pool boundary via
            # initializer/initargs (special-cased "inheritance" pickling), not
            # as regular per-task arguments — see _init_worker.
            n_done = ctx.Value("i", 0)
            total_fp = ctx.Value("l", 0)
            work_items = [
                (g, insts, run_log_dir, self.cache_path) for g, insts in groups.items()
            ]

            with ctx.Pool(
                processes=min(self.max_workers, len(groups)),
                initializer=_init_worker,
                initargs=(config, n_done, total_fp),
            ) as pool:
                # chunksize=1: a free worker takes the next group.
                async_result = pool.map_async(
                    _worker_process_group, work_items, chunksize=1
                )
                while not async_result.ready():
                    async_result.wait(0.2)
                    _refresh_bar(n_done.value, total_fp.value)
                for group, group_results in async_result.get():
                    results_by_group[group] = group_results
            _refresh_bar(n_done.value, total_fp.value)
        else:
            # Single-process: just walk the dataset in order, one cache per
            # group (looked up by group id, not by pre-bucketing into
            # contiguous runs -- correct regardless of whether groups happen
            # to be interleaved in the input). Updates the bar after every
            # real instance directly, no Pool/Value-counter machinery needed.
            init_worker_state(config)
            caches = {}
            group_positions = defaultdict(int)
            n_done, total_fp = 0, 0
            for instance in dataset:
                g = instance["group"]
                if g not in caches:
                    caches[g] = _initial_cache(g)
                pos = group_positions[g]
                group_positions[g] += 1
                try:
                    result = _process_instance(
                        instance, run_log_dir, caches[g], group_position=pos
                    )
                except Exception as e:
                    tb = traceback.format_exc()
                    print(
                        f"[CachedFrontierVerifier] Instance "
                        f"{instance.get('idx')} failed: {e}\n{tb}"
                    )
                    result = {
                        "idx": instance.get("idx"),
                        "total_transitions": 0,
                        "instance_run_time": 0.0,
                        "error": str(e),
                    }
                results_by_group[g].append(result)
                caches[g].save(Path(self.cache_path) / f"{g}.json")
                n_done += 1
                total_fp += result.get("total_transitions", 0)
                _refresh_bar(n_done, total_fp)

        bar.close()

        # ── Flatten results back to the original group order ─────────────
        results = []
        for g in groups:
            results.extend(results_by_group.get(g, []))

        return results
