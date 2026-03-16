"""Frontier (branch-and-bound) verifier for LLM verification."""

import time
import torch
import json
import numpy as np

from beaver.constraints.base_constraints import (
    check_semantic_call,
    enforce_semantic_constraint,
)
from beaver.utils import log_json
from beaver.verifiers.base_verifier import BaseVerifier
from beaver.verifiers.frontier import Frontier, FrontierElement
from beaver.verifiers.worker_common import (
    _w,
    apply_top_p_top_k,
    get_grammar_mask,
    init_worker_state,
    log_profiling,
    model_generate_next_token_logprobs,
    safe_worker,
    worker_setup,
)


@safe_worker
def _worker_process_instance(args):
    """Top-level function for multiprocessing — processes a single instance."""
    instance, log_file, profile_log_file = worker_setup(args)

    def update_frontier(frontier, previous_element, log_probs, bit_mask):
        """Expand a frontier element, filtering by grammar + semantics.

        Args:
            frontier: The Frontier object
            previous_element: FrontierElement being expanded
            log_probs: numpy array of shape [N, 2] with [token_id, logprob] pairs
            bit_mask: Grammar validity mask (torch.Tensor or np.ndarray)

        Returns:
            (new_elements, delta_incomplete_prob_sum, delta_complete_prob_sum,
             presemantic_check_time, semantic_check_time)
        """

        # Convert bit_mask to numpy if it's a torch tensor
        if isinstance(bit_mask, torch.Tensor):
            bit_mask = bit_mask.cpu().numpy()

        # Extract token IDs from logprobs array and filter by grammar mask
        all_token_ids = log_probs[:, 0].astype(int)
        in_vocab_mask = all_token_ids < len(bit_mask)
        valid_indices = all_token_ids[in_vocab_mask]

        valid_mask = bit_mask[valid_indices]
        valid_indices = valid_indices[valid_mask]

        if _w.verbose:
            print(f"in_vocab_mask sum: {np.sum(in_vocab_mask)} / {len(in_vocab_mask)}")
            print(f"Valid mask sum: {np.sum(valid_mask)} / {len(valid_mask)}")
            print(f"Valid indices {len(valid_indices)}")

        if len(valid_indices) == 0:
            return [], 0, 0, 0, 0

        # Create a dict for fast logprob lookup: {token_id: logprob}
        logprobs_dict = {
            int(log_probs[i, 0]): log_probs[i, 1] for i in range(len(log_probs))
        }

        # Decode tokens individually
        decoded_tokens = np.array(
            [
                (
                    ""
                    if i in _w.eos_tokens
                    else _w.tokenizer.decode([i], skip_special_tokens=True)
                )
                for i in valid_indices
            ]
        )

        # Construct full decoded sequences
        current_decoded = _w.tokenizer.decode(
            previous_element.tokens, skip_special_tokens=True
        )
        decoded_sequences = np.array([current_decoded + tok for tok in decoded_tokens])

        # Construct full token lists
        token_lists = np.array(
            [
                previous_element.tokens + [int(valid_indices[idx])]
                for idx in range(len(valid_indices))
            ],
            dtype=object,
        )

        # Determine completion flags
        if len(previous_element.tokens) >= _w.gen_length - 1:
            complete_flag = np.ones(len(decoded_sequences), dtype=bool)
        else:
            complete_flag = np.array(
                [i in _w.eos_tokens for i in valid_indices], dtype=bool
            )

        if _w.verbose:
            print(
                f"Complete flag: {sum(complete_flag)}: "
                f"{[i for i in valid_indices[complete_flag]]}"
            )
            print(f"Worker eos tokens: {_w.eos_tokens}")

        presemantic_check_time = time.time()

        # Semantic checking
        semantic_check_mask = np.logical_or(
            check_semantic_call(
                _w.dataset_name, instance, decoded_sequences, token_lists
            ),
            complete_flag,
        )
        semantic_check_indices = np.where(semantic_check_mask)[0]

        semantic_correct_indices = np.array([], dtype=np.intp)
        if len(semantic_check_indices) > 0:
            sequences_to_check = decoded_sequences[semantic_check_indices]
            semantic_correctness_mask = enforce_semantic_constraint(
                _w.dataset_name, instance, sequences_to_check, use_cache=_w.use_cache
            )
            semantic_correct_indices = semantic_check_indices[semantic_correctness_mask]

        semantic_check_time = time.time()

        if _w.verbose:
            print(f"Check mask: {sum(semantic_check_mask)}")
            print(f"Check indices: {valid_indices[semantic_check_indices]}")
            print(f"Correct indices: {valid_indices[semantic_correct_indices]}")

        violations = set(valid_indices[semantic_check_indices]) - set(
            valid_indices[semantic_correct_indices]
        )
        non_violations = set(valid_indices) - set(violations)

        total_violation_prob = np.sum(np.exp(np.array([previous_element.logprob + logprobs_dict[v] for v in violations]))).item()

        # Build new frontier elements
        new_elements = []

        for idx in range(len(non_violations)):

            token_id = int(valid_indices[idx])
            new_tokens = previous_element.tokens + [token_id]
            new_elem = FrontierElement(
                element_id=frontier.total_elements,
                token=token_id,
                tokens=new_tokens,
                logprob=previous_element.logprob + logprobs_dict[token_id],
                is_completed=complete_flag[idx].item(),
            )

            frontier.total_elements += 1
            new_elements.append(new_elem)

        return (
            new_elements,
            presemantic_check_time,
            semantic_check_time,
            len(violations),
            total_violation_prob,
        )

    # ── Main processing logic ────────────────────────────────────────

    instance_start_time = time.time()
    transitions = 0
    frontier = Frontier(
        max_size=_w.gen_length,
        scoring_strategy=_w.frontier_scoring_strategy,
    )
    incomplete_prob_sum = 1.0
    complete_prob_sum = 0.0
    pruned_prob_sum = 0.0
    violation_prob_sum = 0.0

    running_results = {}

    while transitions < _w.max_iterations:

        # each step of frontier verification
        #
        # pick the top incomplete element from the frontier
        start_step_time = time.time()
        element = frontier.pick_top_incomplete()
        if element is None:
            if _w.verbose:
                print(f"Frontier is empty at transition {transitions}")
            break

        model_generate_time = time.time()
        model_logprobs, final_prompt = model_generate_next_token_logprobs(instance, element.tokens)
        logprobs, reduced_logprobs = apply_top_p_top_k(model_logprobs)

        # Calculate pruned prob (from tokens that are not counted)
        culled_prob_sum = np.exp(element.logprob) * max(
            1 - np.sum(np.exp(logprobs[:, 1])), 0.0
        )
        check_validity_time = time.time()

        vocab_mask = get_grammar_mask(element.tokens)

        (
            new_elements,
            presemantic_check_time,
            semantic_check_time,
            num_violations,
            total_violation_prob,
        ) = update_frontier(frontier, element, logprobs, vocab_mask)

        frontier.add_to_element(element, new_elements)

        frontier_pruned_prob = frontier.prune_incomplete_leaves(
            topp=_w.frontier_topp, topk=_w.frontier_topk
        )

        update_results_time = time.time()

        incomplete_prob_sum -= np.exp(element.logprob)
        incomplete_prob_sum -= frontier_pruned_prob
        for elem in new_elements:
            if elem.is_completed:
                complete_prob_sum += np.exp(elem.logprob)
            else:
                incomplete_prob_sum += np.exp(elem.logprob)

        incomplete_prob_sum = max(
            incomplete_prob_sum, 0.0
        )  # Guard against negative probabilities
        complete_prob_sum = min(
            complete_prob_sum, 1.0
        )  # Guard against probabilities > 1.0

        pruned_prob_sum += culled_prob_sum + frontier_pruned_prob

        violation_prob_sum += total_violation_prob

        upper_bound = incomplete_prob_sum + complete_prob_sum + pruned_prob_sum

        lower_bound = complete_prob_sum

        end_step_time = time.time()
        running_results = {
            "transition": transitions,
            "expanded element": element.tokens,
            "decoded element": _w.tokenizer.decode(
                element.tokens, skip_special_tokens=False
            ),
            "exact_prompt": final_prompt,
            "num_violations": num_violations,
            "total_violation_prob": total_violation_prob,
            "num_new_elements": len(new_elements),
            "incomplete_size": len(frontier._incomplete_leaves),
            "complete_size": len(frontier._complete_leaves),
            "incomplete prob sum": incomplete_prob_sum,
            "complete prob sum": complete_prob_sum,
            "pruned prob sum": pruned_prob_sum,
            "upper_bound": upper_bound,
            "lower_bound": lower_bound,
        }
        log_json(running_results, log_file)
        profiling_data = {
            "element_selection_time": model_generate_time - start_step_time,
            "model_generate": check_validity_time - model_generate_time,
            "grammar_mask": presemantic_check_time - check_validity_time,
            "semantic_check": semantic_check_time - presemantic_check_time,
            "frontier_add": update_results_time - semantic_check_time,
            "check_validity": update_results_time - check_validity_time,
            "update_results": end_step_time - update_results_time,
            "total_time": end_step_time - start_step_time,
        }
        log_profiling(
            profiling_data,
            profile_log_file,
        )

        if _w.verbose:
            print(json.dumps(running_results, indent=2))
            frontier.debug_frontier(_w.tokenizer)


        transitions += 1

        if pruned_prob_sum > 10 * _w.epsilon:
            raise RuntimeError(
                f"Error: Pruned probability sum {pruned_prob_sum} exceeds reasonable threshold {10 * _w.epsilon} at transition {transitions}"
            )

        if incomplete_prob_sum < _w.epsilon:
            if _w.verbose:
                print(
                    f"Ending frontier analysis {instance['idx']} "
                    f"since incomplete probability is below epsilon"
                )
            break


    instance_end_time = time.time()
    return {
        "idx": instance["idx"],
        **running_results,
        "instance_run_time" : instance_end_time - instance_start_time,
    }


class FrontierVerifier(BaseVerifier):
    def __init__(self, model, dataset, server_addr, **kwargs):
        super().__init__(model, dataset, server_addr, **kwargs)
        self.frontier_topp = kwargs.get("max_frontier_prob", 1.0)
        self.frontier_topk = kwargs.get("max_frontier_size", -1)
        self.frontier_scoring_strategy = kwargs.get(
            "frontier_scoring_strategy", "highest-prob"
        )

    def __call__(self, dataset, run_log_dir):
        config = self._build_worker_config()
        config["frontier_topp"] = self.frontier_topp
        config["frontier_topk"] = self.frontier_topk
        config["frontier_scoring_strategy"] = self.frontier_scoring_strategy

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
