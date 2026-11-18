"""Sampling (full-sequence) verifier for LLM verification."""

import time

import numpy as np

from beaver.constraints.base_constraints import enforce_semantic_constraint
from beaver.utils import log_json
from beaver.verifiers.base_verifier import BaseVerifier
from beaver.verifiers.frontier_step import build_instance_complete, check_termination
from beaver.model import model_sample_sequence
from beaver.verifiers.worker_common import (
    _w,
    init_worker_state,
    log_profiling,
    safe_worker,
    worker_setup,
)


def _log_skipped_draw(log_file, instance, transition, n_tokens, state, why):
    """A draw that moved no mass (repeated sequence / cut short by the budget)
    still cost ``n_tokens`` forward passes; record it so the per-transition fp
    sum matches ``total_transitions``."""
    log_json(
        {
            "event": "transition",
            "idx": instance.get("idx"),
            "transition": transition,
            "batch_size": n_tokens,
            "skipped": why,
            "upper_bound": state["incomplete"] + state["complete"],
            "lower_bound": state["complete"],
            "incomplete prob sum": state["incomplete"],
            "complete prob sum": state["complete"],
            "pruned prob sum": state["pruned"],
            "violation prob sum": state["violation"],
        },
        log_file,
    )


@safe_worker
def _worker_process_instance(args):
    """Top-level function for multiprocessing — processes a single instance."""
    instance, log_file, profile_log_file = worker_setup(args)

    # No need for sample_token function anymore - server does the sampling!

    # ── Main processing logic ────────────────────────────────────────
    instance_start = time.time()
    total_transitions = 0
    state = {"incomplete": 1.0, "complete": 0.0, "pruned": 0.0, "violation": 0.0}
    sequences_completed = 0
    seen_sequences = set()
    termination_reason = "max_iterations"

    if _w.verbose:
        print(f"Starting Sampling Verification for instance {instance['idx']}")

    while total_transitions < _w.max_iterations:
        transitions_remaining = _w.max_iterations - total_transitions

        model_start = time.time()
        token_ids, token_logprobs = model_sample_sequence(
            instance,
            min(_w.gen_length, transitions_remaining),
        )

        n_tokens = len(token_ids)
        total_transitions += n_tokens
        model_end = time.time()
        if not token_ids:
            print(
                f"Model failed on sampling for sample {sequences_completed} instance {instance['idx']}"
            )
            termination_reason = "model_sample_failure"
            break
        elif token_ids[-1] not in _w.eos_tokens and n_tokens < _w.gen_length:
            # Draw cut short by the remaining budget: not a full sequence, so
            # its (prefix) mass must not be moved. Tokens still count as cost.
            _log_skipped_draw(log_file, instance, total_transitions, n_tokens, state, "truncated")
            continue
        elif tuple(token_ids) in seen_sequences:
            # repeated sequence, skip probability update but count transitions
            if _w.verbose:
                print(
                    f"Instance {instance['idx']} - Repeated sequence encountered, skipping probability update."
                )
            _log_skipped_draw(log_file, instance, total_transitions, n_tokens, state, "repeated")
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

        state["incomplete"] -= total_seq_prob
        if semantic_correctness_mask:
            state["complete"] += total_seq_prob
        else:
            state["violation"] += total_seq_prob

        sequences_completed += 1

        bound_update_end = time.time()

        upper_bound = state["incomplete"] + state["complete"]
        lower_bound = state["complete"]

        # Log transition
        transition_info = {
            "event": "transition",
            "transition": total_transitions,
            # forward passes charged this round = generated tokens; summed by
            # logging.py (absent == 1 would count sequences, not tokens)
            "batch_size": n_tokens,
            "expanded element": token_ids,
            "decoded element": decoded_text,
            "upper_bound": upper_bound,
            "lower_bound": lower_bound,
            "incomplete prob sum": state["incomplete"],
            "complete prob sum": state["complete"],
            "pruned prob sum": state["pruned"],
            "violation prob sum": state["violation"],
            "complete_size": sequences_completed,
            "unique_sequences": len(seen_sequences),
            "seq_prob": total_seq_prob,
            "validity_of_current_sequence": semantic_correctness_mask,
        }
        log_json(transition_info, log_file)

        log_profiling(
            {
                "event": "transition",
                "batch_size": n_tokens,
                "model_sample": model_end - model_start,
                "bound_update": bound_update_end - model_end,
                "total_time": bound_update_end - model_start,
            },
            profile_log_file,
        )

        if _w.verbose:
            print(
                f"Instance {instance['idx']} transition: "
                f"{total_transitions} "
                f"incomplete_prob: {state['incomplete']} "
                f"complete_prob: {lower_bound} "
                f"current_sequence_prob: {total_seq_prob}"
            )


        should_stop, reason = check_termination(state, None, total_transitions)
        if should_stop:
            termination_reason = reason
            break

        if (
            _w.early_termination_threshold is not None
            and upper_bound < _w.early_termination_threshold
        ):
            if _w.verbose:
                print(
                    f"Ending sampling analysis {instance['idx']} "
                    f"since upper_bound {upper_bound} < "
                    f"early_termination_threshold {_w.early_termination_threshold}"
                )
            termination_reason = "early_termination"
            break

    instance_run_time = time.time() - instance_start
    result = build_instance_complete(
        instance,
        None,
        termination_reason=termination_reason,
        state=state,
        transitions=total_transitions,
        instance_run_time=instance_run_time,
    )
    log_json(result, log_file)

    return {
        "idx": instance["idx"],
        "transitions": total_transitions,
        "time_s": instance_run_time,
        "sequences": sequences_completed,
        "lower_bound": state["complete"],
        "upper_bound": state["incomplete"] + state["complete"],
    }


class SamplingVerifier(BaseVerifier):
    def __init__(self, model, dataset, server_addr, **kwargs):
        super().__init__(model, dataset, server_addr, **kwargs)

    def __call__(self, dataset, run_log_dir):
        config = self._build_worker_config()
        # dataset = self._tokenize_dataset(dataset)
        return self._run_pool(
            dataset,
            run_log_dir,
            worker_fn=_worker_process_instance,
            init_fn=init_worker_state,
            config=config,
        )
