"""Frontier (branch-and-bound) verifier for LLM verification."""

import time
import torch

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
    instance, prompt_ids, log_file, profile_log_file = worker_setup(args)

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
            return [], 0.0, 0.0, 0, 0, 0

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

        # Build new frontier elements
        new_elements = []
        delta_incomplete_prob_sum = 0.0
        delta_complete_prob_sum = 0.0

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
            if complete_flag[idx].item():
                delta_complete_prob_sum += np.exp(new_elem.logprob)
            else:
                delta_incomplete_prob_sum += np.exp(new_elem.logprob)
            frontier.total_elements += 1
            new_elements.append(new_elem)

        return (
            new_elements,
            delta_incomplete_prob_sum,
            delta_complete_prob_sum,
            presemantic_check_time,
            semantic_check_time,
            len(violations),
        )

    def expand_sequence(sequence, frontier):
        model_start = time.time()
        model_logprobs = model_generate_next_token_logprobs(
            prompt_ids, sequence.tokens, system_prompt=instance.get("system_prompt")
        )
        logprobs, reduced_logprobs = apply_top_p_top_k(model_logprobs)
        model_end = time.time()
        # Calculate culled probability: tokens not in the returned logprobs

        culled_prob_sum = np.exp(sequence.logprob) * max(
            1 - np.sum(np.exp(logprobs[:, 1])), 0.0
        )

        vocab_mask = get_grammar_mask(sequence.tokens)
        grammar_decode_end = time.time()
        (
            new_elements,
            delta_incomplete_prob_sum,
            delta_complete_prob_sum,
            presemantic_check_time,
            semantic_check_time,
            num_violations,
        ) = update_frontier(frontier, sequence, logprobs, vocab_mask)

        prefrontier_add_time = time.time()

        frontier.add_to_element(sequence, new_elements)
        update_frontier_end = time.time()

        frontier_pruned_prob = frontier.prune_incomplete_leaves(
            topp=_w.frontier_topp, topk=_w.frontier_topk
        )
        prune_frontier_end = time.time()

        delta_incomplete_prob_sum -= np.exp(sequence.logprob)
        delta_incomplete_prob_sum += culled_prob_sum
        culled_prob_sum = frontier_pruned_prob + culled_prob_sum

        log_profiling(
            {
                "model_generate": model_end - model_start,
                "grammar_decode": grammar_decode_end - model_end,
                "presemantic_check": presemantic_check_time - grammar_decode_end,
                "semantic_check": semantic_check_time - presemantic_check_time,
                "prefrontier_add": prefrontier_add_time - semantic_check_time,
                "update_frontier": update_frontier_end - grammar_decode_end,
                "prune_frontier": prune_frontier_end - update_frontier_end,
                "total_time": prune_frontier_end - model_start,
            },
            profile_log_file,
        )

        if _w.verbose:
            frontier.debug_frontier(_w.tokenizer)
            breakpoint()

        return (
            delta_incomplete_prob_sum,
            delta_complete_prob_sum,
            culled_prob_sum,
            num_violations,
            len(new_elements),
        )

    # ── Main processing logic ────────────────────────────────────────
    instance_start = time.time()
    transitions = 0
    frontier = Frontier(
        max_size=_w.gen_length,
        scoring_strategy=_w.frontier_scoring_strategy,
    )
    incomplete_prob_sum = 1.0
    complete_prob_sum = 0.0
    pruned_prob_sum = 0.0

    while transitions < _w.max_iterations:
        element = frontier.pick_top_incomplete()

        if element is None:
            if _w.verbose:
                print(f"Frontier is empty at transition {transitions}")
            break

        start_time = time.time()
        (
            delta_incomplete_prob_sum,
            delta_complete_prob_sum,
            delta_pruned_prob_sum,
            num_violations,
            num_new_elements,
        ) = expand_sequence(element, frontier)

        incomplete_prob_sum += delta_incomplete_prob_sum
        incomplete_prob_sum = max(
            incomplete_prob_sum, 0.0
        )  # Guard against negative probabilities
        complete_prob_sum += delta_complete_prob_sum
        pruned_prob_sum += delta_pruned_prob_sum

        end_time = time.time()
        transitions += 1

        # Log transition results (always to file, print only in verbose)
        running_results = {
            "transition": transitions,
            "expanded element": element.tokens,
            "decoded element": _w.tokenizer.decode(
                element.tokens, skip_special_tokens=False
            ),
            "incomplete prob sum": incomplete_prob_sum,
            "complete prob sum": complete_prob_sum,
            "pruned prob sum": pruned_prob_sum,
            "num_violations": num_violations,
            "num_new_elements": num_new_elements,
            "incomplete_size": len(frontier._incomplete_leaves),
            "complete_size": len(frontier._complete_leaves),
        }
        log_json(running_results, log_file)

        if _w.verbose:
            iter_time = end_time - start_time
            print(
                f"Instance {instance['idx']} transition: {transitions} "
                f"incomplete_prob: {incomplete_prob_sum} "
                f"complete_prob: {complete_prob_sum} "
                f"({iter_time:.3f}s)"
            )
            for k, v in running_results.items():
                print(f"\t{k}: {v}")

            breakpoint()

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

    return {
        "idx": instance["idx"],
        "transitions": transitions,
        "time_s": time.time() - instance_start,
        "incomplete_prob": incomplete_prob_sum,
        "complete_prob": complete_prob_sum,
        "pruned_prob": pruned_prob_sum,
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

        dataset = self._tokenize_dataset(dataset)

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
