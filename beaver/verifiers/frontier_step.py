"""Shared frontier step + termination logic.

Both ``FrontierVerifier`` and ``CachedFrontierVerifier`` run the same
branch-and-bound expansion loop; the only difference is that the cached variant
warm-starts the frontier from a token-trie cache and emits warm_start /
instance_complete events. The per-transition work — expand an element, update
the probability masses, log the transition + profiling, and decide whether to
stop — is identical and lives here so the two verifiers stay in sync.
"""

import json
import time

import numpy as np

from beaver.utils import log_json
from beaver.model import model_generate_next_token_logprobs_batch
from beaver.model import (
    drain_model_timings as drain_http_timings,
)
from beaver.verifiers.worker_common import (
    _w,
    apply_top_p_top_k,
    get_grammar_mask,
    log_profiling,
    update_frontier,
)


def check_termination(state, frontier, transitions):
    """Evaluate the stop conditions. Returns (should_stop, reason).

    Two selectable criteria, chosen by ``_w.stopping_criterion``:

    "max_seq_prob" (DEFAULT, the historical behaviour)
        Stop when the most-probable incomplete leaf drops below epsilon. Note
        this bounds the largest *single* leaf, not the unresolved *mass*: with
        gen_length 32 it fires around depth 10, so with prefix_closed=False
        nothing ever completes, the constraint is never called, and every run
        reports zero violations by construction. Pair it with prefix_closed,
        or use "bounds" below.

    "bounds"
        Stop as soon as the instance is actually decided, i.e. any of
            UB <= threshold      -> risky verdict reached
            LB >  threshold      -> safe verdict reached
            UB - LB <= epsilon   -> bounds tight enough to be conclusive
        UB = 1 - proven-violation mass, LB = proven-safe mass, matching
        logging.py's three-way verdict, so "resolved here" and "resolved in
        the summary" mean the same thing.

    ``frontier`` may be None for verifiers with no token-level search tree
    (e.g. SamplingVerifier). Termination is then decided purely from ``state``.
    """
    # Verifiers without a search tree (SamplingVerifier) pass frontier=None:
    # there are no leaves to test, so only the budget / bounds rules apply.
    # (Returning "frontier_empty" here, as an earlier revision did, stopped
    # the sampling verifier after its very first draw.)
    if frontier is None:
        incomplete = None
    else:
        incomplete = frontier._incomplete_leaves
        if not incomplete:
            return True, "frontier_empty"

    criterion = getattr(_w, "stopping_criterion", "max_seq_prob") or "max_seq_prob"

    if criterion == "bounds":
        upper = state["incomplete"] + state["complete"] + state["pruned"]
        lower = state["complete"]
        threshold = getattr(_w, "early_termination_threshold", None)
        if threshold is not None:
            if upper <= threshold:
                return True, "ub_below_threshold"
            if lower > threshold:
                return True, "lb_above_threshold"
        if (upper - lower) <= _w.epsilon:
            return True, "gap_below_epsilon"
        return False, None

    if criterion != "max_seq_prob":
        raise ValueError(
            f"Unknown stopping_criterion {criterion!r}; "
            f"expected 'max_seq_prob' or 'bounds'."
        )

    if incomplete is None:
        return False, None

    max_incomplete_prob = float(np.exp(max(leaf.logprob for leaf in incomplete)))
    if max_incomplete_prob < _w.epsilon:
        return True, "max_incomplete_below_epsilon"

    return False, None


def frontier_transition(frontier, instance, log_file, profile_log_file, state, transitions):
    """Run one expansion step of the frontier search.

    Pops the top incomplete element, expands it under the model + grammar +
    semantic constraints, updates the probability masses carried in ``state``
    (keys: ``incomplete``, ``complete``, ``pruned``, ``violation``), logs the
    transition and profiling entries, and evaluates the stop conditions.

    ``transitions`` is the 0-based index of this step (used as the logged
    ``transition`` value); termination is checked against ``transitions + 1``
    (the count of completed steps), matching the original loops.

    Returns ``(status, running_results, reason)`` where status is one of:
        "empty"    — frontier exhausted, nothing expanded (caller should stop)
        "stop"     — a termination condition fired
        "continue" — keep going
    ``running_results`` is the logged transition dict (None when "empty") and
    ``reason`` is the termination-reason string (for "empty"/"stop"; None when
    "continue").
    """
    start_step_time = time.time()

    k = getattr(_w, "expand_batch_size")
    elements, pick_debug = _w.selector.pick_batch(frontier, k, tokenizer=_w.tokenizer)
    if not elements:
        if _w.verbose:
            print(f"Frontier is empty at transition {transitions}")
        return "empty", None, "frontier_empty"

    model_generate_time = time.time()

    request_hidden_states = bool(_w.selector.needs_hidden_states)
    rows = model_generate_next_token_logprobs_batch(
        instance,
        [e.tokens for e in elements],
        return_hidden_states=request_hidden_states,
    )
    http_timings = drain_http_timings()
    check_validity_time = time.time()

    culled_prob_sum = 0.0
    total_new = 0
    total_semantic = 0
    num_violations = 0
    violation_texts = []
    expanded_info = []
    grammar_mask_time = check_validity_time
    presemantic_check_time = check_validity_time
    semantic_check_time = check_validity_time


    for element, row in zip(elements, rows):
        if request_hidden_states:
            model_logprobs, hidden_states, _prompt = row
        else:
            model_logprobs, _prompt = row
            hidden_states = {}
        logprobs, _ = apply_top_p_top_k(model_logprobs)
        culled_prob_sum += np.exp(element.logprob) * max(
            1 - np.sum(np.exp(logprobs[:, 1])), 0.0
        )
        vocab_mask = get_grammar_mask(element.tokens)
        grammar_mask_time = time.time()
        (
            new_elements,
            presemantic_check_time,
            semantic_check_time,
            n_semantic_attempted,
        ) = update_frontier(
            frontier, instance, element, logprobs, vocab_mask, return_counts=True
        )
        _w.selector.on_expand(element, new_elements, hidden_states, instance)
        frontier.add_to_element(element, new_elements)

        state["incomplete"] -= np.exp(element.logprob)
        total_new += len(new_elements)
        total_semantic += n_semantic_attempted
        n_viol_here = 0
        for elem in new_elements:
            if elem.is_violation:
                state["violation"] += np.exp(elem.logprob)
                num_violations += 1
                n_viol_here += 1
                violation_texts.append(
                    {
                        "decoded": _w.tokenizer.decode(
                            elem.tokens, skip_special_tokens=True
                        ),
                        "seq_prob": np.exp(elem.logprob),
                    }
                )
            elif elem.is_completed:
                state["complete"] += np.exp(elem.logprob)
            else:
                state["incomplete"] += np.exp(elem.logprob)
        expanded_info.append(
            {
                "tokens": element.tokens,
                "seq_prob": float(np.exp(element.logprob)),
                "decoded": _w.tokenizer.decode(element.tokens, skip_special_tokens=False),
                "num_new_elements": len(new_elements),
                "num_violations": n_viol_here,
                "num_semantic_checks": int(n_semantic_attempted),
            }
        )


    # One prune for the whole round.
    frontier_pruned_prob = frontier.prune_incomplete_leaves(
        topp=_w.frontier_topp, topk=_w.frontier_topk
    )
    state["incomplete"] -= frontier_pruned_prob
    update_results_time = time.time()

    state["incomplete"] = max(state["incomplete"], 0.0)
    state["complete"] = min(state["complete"], 1.0)
    state["pruned"] += culled_prob_sum + frontier_pruned_prob

    upper_bound = state["incomplete"] + state["complete"] + state["pruned"]
    assert np.abs(upper_bound - (1 - state["violation"])) <= 1e-4, (
        f"upper_bound={upper_bound} violation_prob_sum={state['violation']}"
    )
    lower_bound = state["complete"]

    end_step_time = time.time()
    running_results = {
        "event": "transition",
        "transition": transitions,
        "batch_size": len(elements),
        "expanded elements": expanded_info,
        "pick_info": pick_debug,
        "num_violations": num_violations,
        "violation_texts": violation_texts,
        "num_semantic_checks": total_semantic,
        "num_new_elements": total_new,
        "incomplete_size": len(frontier._incomplete_leaves),
        "complete_size": len(frontier._complete_leaves),
        "incomplete prob sum": state["incomplete"],
        "complete prob sum": state["complete"],
        "pruned prob sum": state["pruned"],
        "violation prob sum": state["violation"],
        "upper_bound": upper_bound,
        "lower_bound": lower_bound,
    }
    log_json(running_results, log_file)
    log_profiling(
        {
            "event": "transition",
            "idx": instance.get("idx"),
            "transition": transitions,
            "batch_size": len(elements),
            "element_selection_time": model_generate_time - start_step_time,
            "model_generate": check_validity_time - model_generate_time,
            "grammar_mask": 0.0,
            "decode_and_complete_check": presemantic_check_time - grammar_mask_time,
            "semantic_check": semantic_check_time - presemantic_check_time,
            "frontier_add": update_results_time - semantic_check_time,
            "update_results": end_step_time - update_results_time,
            "total_time": end_step_time - start_step_time,
            **http_timings,
        },
        profile_log_file,
    )

    should_stop, reason = check_termination(state, frontier, transitions + 1)
    return ("stop" if should_stop else "continue"), running_results, reason


def build_instance_complete(
    instance,
    frontier,
    *,
    termination_reason,
    state,
    transitions,
    instance_run_time,
    last_transition=None,
    cache_extra=None,
):
    """Build the uniform ``instance_complete`` record logged by every frontier
    verifier strategy, so the event has the same shape (and can be found the
    same way) regardless of which verifier produced it.

    ``state`` carries the final probability masses (keys: ``incomplete``,
    ``complete``, ``pruned``, ``violation`` — ``incomplete`` may be ``None``
    when a warm-start gate cut the incomplete set before it was computed).
    ``last_transition`` is the final ``frontier_transition`` step's logged
    dict (``None`` if the search loop never ran, e.g. an immediate
    warm-skip) — nested under its own key rather than flattened, so it can't
    collide with the frontier-stats keys below.

    ``cache_extra`` is supplied only by ``CachedFrontierVerifier``: warm-start
    and antichain fields that don't apply to the cold-start verifier, merged
    in last so they're additive rather than required.

    ``frontier`` may be None for verifiers with no token-level search tree
    (e.g. SamplingVerifier) — the frontier-derived fields (leaf counts,
    ``frontier.stats()``) are then omitted rather than required.
    """
    record = {
        "event": "instance_complete",
        "idx": instance["idx"],
        "group": instance.get("group"),
        "termination_reason": termination_reason,
        "final_incomplete_prob": state["incomplete"],
        "final_complete_prob": state["complete"],
        "final_violation_prob": state["violation"],
        "final_pruned_prob": state["pruned"],
        "total_transitions": transitions,
        "instance_run_time": instance_run_time,
        "last_transition": last_transition,
    }
    if frontier is not None:
        record["new_violations_found"] = len(frontier._violation_leaves)
        record["new_completions_found"] = len(frontier._complete_leaves)
        record["new_incomplete_found"] = len(frontier._incomplete_leaves)
        record.update(frontier.stats())
    if cache_extra:
        record.update(cache_extra)
    return record
