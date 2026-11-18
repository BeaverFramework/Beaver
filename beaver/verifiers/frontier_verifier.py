"""Frontier (branch-and-bound) verifier for LLM verification."""

import json
import time

from beaver.utils import log_json
from beaver.verifiers.base_verifier import BaseVerifier
from beaver.verifiers.frontier import Frontier
from beaver.verifiers.frontier_step import build_instance_complete, frontier_transition
from beaver.verifiers.worker_common import (
    _w,
    init_worker_state,
    safe_worker,
    worker_setup,
)


@safe_worker
def _worker_process_instance(args):
    """Top-level function for multiprocessing — processes a single instance.

    Cold-start variant: no warm-start cache. The per-transition expansion,
    logging, and termination check are shared with CachedFrontierVerifier via
    ``frontier_transition`` so both verifiers emit identical transition logs.
    """
    instance, log_file, profile_log_file = worker_setup(args)

    instance_start_time = time.time()
    transitions = 0
    frontier = Frontier(max_size=_w.gen_length)
    # Probability masses carried across transitions (mutated in place).
    state = {"incomplete": 1.0, "complete": 0.0, "pruned": 0.0, "violation": 0.0}

    running_results = {}
    termination_reason = "max_iterations"

    while transitions < _w.max_iterations:
        status, step_results, reason = frontier_transition(
            frontier, instance, log_file, profile_log_file, state, transitions
        )
        if status == "empty":
            termination_reason = reason
            break

        running_results = step_results
        transitions += 1

        if status == "stop":
            termination_reason = reason
            if _w.verbose:
                print(f"Ending frontier analysis {instance['idx']}")
            break

    instance_end_time = time.time()

    result = build_instance_complete(
        instance,
        frontier,
        termination_reason=termination_reason,
        state=state,
        transitions=transitions,
        instance_run_time=instance_end_time - instance_start_time,
        last_transition=running_results or None,
    )
    log_json(result, log_file)
    # print(f"[frontier_stats] idx={instance['idx']} {json.dumps(result)}")

    if _w.verbose:
        print(
            f"\n[instance_complete] idx={instance['idx']}  transitions={transitions}  "
            f"reason={termination_reason}\n"
            f"  new_violations={result['new_violations_found']}  "
            f"new_completions={result['new_completions_found']}  "
            f"new_incomplete={result['new_incomplete_found']}\n"
            f"  final: incomplete={state['incomplete']:.4f}  "
            f"complete={state['complete']:.4f}  "
            f"violation={state['violation']:.4f}  "
            f"pruned={state['pruned']:.4f}"
        )

    return {**result, **running_results}


class FrontierVerifier(BaseVerifier):
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

    def __call__(self, dataset, run_log_dir):
        config = self._build_worker_config()
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

        # dataset = self._tokenize_dataset(dataset)

        # Handle origin_code field (secure_code dataset)
        for i in range(len(dataset)):
            if "origin_code" in dataset[i]:
                dataset[i]["origin_code_ids"] = self.tokenizer.encode(
                    dataset[i]["origin_code"], add_special_tokens=False
                )
            else:
                dataset[i]["origin_code_ids"] = []

        return self._run_pool(
            dataset,
            run_log_dir,
            worker_fn=_worker_process_instance,
            init_fn=init_worker_state,
            config=config,
        )
