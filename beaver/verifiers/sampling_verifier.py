"""Sampling (full-sequence) verifier for LLM verification."""

import time

import numpy as np

from beaver.constraints.base_constraints import enforce_semantic_constraint
from beaver.utils import log_json
from beaver.verifiers.base_verifier import BaseVerifier
from beaver.verifiers.worker_common import (
    _w,
    init_worker_state,
    log_profiling,
    model_sample_sequence,
    safe_worker,
    worker_setup,
)


@safe_worker
def _worker_process_instance(args):
    """Top-level function for multiprocessing — processes a single instance."""
    instance, prompt_ids, log_file, profile_log_file = worker_setup(args)

    # No need for sample_token function anymore - server does the sampling!

    # ── Main processing logic ────────────────────────────────────────
    instance_start = time.time()
    total_transitions = 0
    upper_bound = 1.0
    lower_bound = 0.0
    sequences_completed = 0
    seen_sequences = set()

    if _w.verbose:
        print(f"Starting Sampling Verification for instance {instance['idx']}")

    while total_transitions < _w.max_iterations:
        transitions_remaining = _w.max_iterations - total_transitions

        model_start = time.time()
        token_ids, token_logprobs = model_sample_sequence(
            prompt_ids,
            min(_w.gen_length, transitions_remaining),
            system_prompt=instance.get("system_prompt"),
        )

        total_transitions += len(token_ids)
        model_end = time.time()
        if not token_ids:
            print(
                f"Model failed on sampling for sample {sequences_completed} instance {instance['idx']}"
            )
            break
        elif tuple(token_ids) in seen_sequences:
            # repeated sequence, skip probability update but count transitions
            if _w.verbose:
                print(
                    f"Instance {instance['idx']} - Repeated sequence encountered, skipping probability update."
                )
            continue
        else:
            seen_sequences.add(tuple(token_ids))

        total_seq_prob = np.exp(np.sum(token_logprobs)).item()

        # Remove EOS token from decoded text if present
        if token_ids[-1] in _w.eos_tokens:
            token_ids = token_ids[:-1]

        decoded_text = _w.tokenizer.decode(token_ids, skip_special_tokens=True)

        # Check if we've seen this sequence before
        semantic_correctness_mask = enforce_semantic_constraint(
            _w.dataset_name, instance, [decoded_text], use_cache=_w.use_cache
        ).item()

        if semantic_correctness_mask:
            lower_bound += total_seq_prob
        else:
            upper_bound -= total_seq_prob

        sequences_completed += 1

        bound_update_end = time.time()

        # Log transition
        transition_info = {
            "transition": total_transitions + len(token_ids),
            "expanded element": token_ids,
            "decoded element": decoded_text,
            "upper_bound": upper_bound,
            "lower_bound": lower_bound,
            "incomplete prob sum": upper_bound - lower_bound,
            "complete prob sum": lower_bound,
            "complete_size": sequences_completed,
            "unique_sequences": len(seen_sequences),
            "current_sequence_prob": total_seq_prob,
            "validity_of_current_sequence": semantic_correctness_mask,
        }
        log_json(transition_info, log_file)

        log_profiling(
            {
                "model_sample": model_end - model_start,
                "bound_update": bound_update_end - model_end,
                "total_time": bound_update_end - model_start,
            },
            profile_log_file,
        )

        if _w.verbose:
            print(
                f"Instance {instance['idx']} transition: "
                f"{total_transitions + len(token_ids)} "
                f"incomplete_prob: {upper_bound - lower_bound} "
                f"complete_prob: {lower_bound} "
                f"current_sequence_prob: {total_seq_prob}"
            )

        gap = upper_bound - lower_bound
        if gap < _w.epsilon:
            break

    return {
        "idx": instance["idx"],
        "transitions": total_transitions,
        "time_s": time.time() - instance_start,
        "sequences": sequences_completed,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
    }


class SamplingVerifier(BaseVerifier):
    def __init__(self, model, dataset, server_addr, **kwargs):
        super().__init__(model, dataset, server_addr, **kwargs)

    def __call__(self, dataset, run_log_dir):
        config = self._build_worker_config()
        dataset = self._tokenize_dataset(dataset)
        return self._run_pool(
            dataset,
            run_log_dir,
            worker_fn=_worker_process_instance,
            init_fn=init_worker_state,
            config=config,
        )
