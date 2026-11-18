"""Shared worker state and utilities for verification workers.

All multiprocessing worker state lives here as a single SimpleNamespace (`_w`),
replacing the 17+ individual `_worker_*` globals that were duplicated across
frontier_verifier.py and sampling_verifier.py.
"""

import functools
import threading
import time
import traceback
import types

import httpx
import numpy as np
import torch
from openai import OpenAI
from transformers import AutoTokenizer

from beaver.constraints.base_constraints import (
    check_complete_call,
    check_semantic_call,
    enforce_semantic_constraint,
)
from beaver.utils import log_json
from beaver.verifiers.decoding import decode_distribution
from beaver.verifiers.frontier import FrontierElement

# ---------------------------------------------------------------------------
# Single global worker state — populated by init_worker_state() in each
# spawned process.  Every field that was previously a _worker_* global
# becomes an attribute of _w.
# ---------------------------------------------------------------------------
_w = types.SimpleNamespace()


def init_worker_state(config_dict):
    """Initialize worker process state from a config dictionary.

    Called as the multiprocessing Pool initializer.  *config_dict* must be
    pickleable (strings, numbers, booleans, top-level function references).

    NOTE: We clear and repopulate _w in-place (rather than reassigning) so
    that modules which imported _w via ``from worker_common import _w``
    keep a reference to the same object.
    """
    # Clear any previous state without reassigning the object
    _w.__dict__.clear()

    # Set CUDA_VISIBLE_DEVICES before any GPU operations
    gpu_visible_devices = config_dict.get("gpu_visible_devices")
    if gpu_visible_devices is not None:
        import os

        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_visible_devices)

    from llguidance.hf import from_tokenizer

    _w.model_name = config_dict["model_name"]  # Store model name for API calls
    _w.tokenizer = AutoTokenizer.from_pretrained(config_dict["model_name"])
    _w.lltokenizer = from_tokenizer(_w.tokenizer)
    _w.vocab_size = len(_w.tokenizer)  # Store total vocab size including special tokens
    _w.server_addr = config_dict["server_addr"]
    _w.ebnf = config_dict["ebnf"]
    _w.dataset_name = config_dict["dataset_name"]
    _w.use_cache = config_dict.get("use_cache", True)
    _w.prefix_closed = config_dict.get("prefix_closed", False)
    _w.temperature = config_dict.get("temperature", 1.0)
    _w.top_p = config_dict.get("top_p", 1.0)
    _w.top_k = config_dict.get("top_k", -1)
    # True: bounds are w.r.t. the *decoded* distribution (temperature, top-k,
    # top-p applied and renormalised; see beaver/verifiers/decoding.py).
    # False: top-p/top-k are an exploration cutoff whose dropped mass is
    # carried as unresolved, temperature is ignored -- the historical mode.
    _w.decoded_dist = bool(config_dict.get("decoded_dist", False))
    _w.eos_tokens = config_dict.get("eos_tokens", [])
    _w.stop_tokens = [
        _w.tokenizer.decode([et], skip_special_tokens=False) for et in _w.eos_tokens
    ]
    _w.gen_length = config_dict.get("gen_length", 128)
    _w.epsilon = config_dict.get("epsilon", 0.01)
    _w.early_termination_threshold = config_dict.get(
        "early_termination_threshold", None
    )
    _w.verbose = config_dict.get("verbose", False)
    _w.max_iterations = config_dict.get("max_iterations", 100)
    _w.semantic_symbol = config_dict.get("semantic_symbol", None)
    _w.num_logprobs = config_dict.get("num_logprobs", 100)
    # Max sequences per scoring request. Scoring re-sends the whole cached
    # antichain under a shared prefix; one giant request defeats vLLM prefix
    # caching (siblings in the same scheduler step recompute the prefix) and
    # blows the per-request read timeout. Chunking sends them sequentially so
    # the first chunk commits the prefix KV and the rest hit the cache.
    _w.score_chunk_size = config_dict.get("score_chunk_size", 64)
    _w.use_grammar = config_dict.get("use_grammar", True)
    _w.chat_mode = config_dict.get("chat_mode", True)
    _w.system_message = config_dict.get("system_message", None)
    _w.fewshot_messages = config_dict.get("fewshot_messages", [])
    _w.enable_thinking = config_dict.get("enable_thinking", False)
    _w.chat_template_kwargs = config_dict.get("chat_template_kwargs", None)

    # Re-register the constraint so _REGISTRY is populated in this worker process.
    # With spawn start method, workers start fresh and don't inherit the main
    # process's _REGISTRY, so we pass the functions through config_dict.
    if "check_fn" in config_dict:
        from beaver.constraints.base_constraints import register_constraint

        register_constraint(
            config_dict["dataset_name"],
            check_call_fn=config_dict["check_call_fn"],
            instance_context_fn=config_dict["instance_context_fn"],
            check_fn=config_dict["check_fn"],
            complete_fn=config_dict.get("complete_fn", None),
        )

    # Frontier-specific — only set when present in config
    if "frontier_topp" in config_dict:
        _w.frontier_topp = config_dict["frontier_topp"]
        _w.frontier_topk = config_dict["frontier_topk"]
        _w.frontier_scoring_strategy = config_dict["frontier_scoring_strategy"]

    # Representation verifier — neural value function config
    _w.probe_path = config_dict.get("probe_path", None)
    _w.probe_layer = config_dict.get("probe_layer", None)
    _w.nvf_type = config_dict.get("nvf_type", "knn")
    _w.nvf_data_path = config_dict.get("nvf_data_path", None)
    _w.probe_gpu = config_dict.get("probe_gpu", None)
    _w.gpu_visible_devices = config_dict.get("gpu_visible_devices", None)
    _w.knn_k = config_dict.get("knn_k", 5)
    _w.knn_normalize = config_dict.get("knn_normalize", True)

    # Pluggable node-selection strategy for frontier_step.py::frontier_transition.
    _w.node_selection = config_dict.get("node_selection", None)
    from beaver.verifiers.node_selection import build_selector

    _w.selector = build_selector(_w.node_selection, config_dict)

    # Surrogate model server config (distribution / iterative verifiers)
    _w.surrogate_server_addr = config_dict.get("surrogate_server_addr", None)
    _w.surrogate_model = config_dict.get("surrogate_model", None)
    _w.surrogate_temperature = config_dict.get("surrogate_temperature", 1.0)
    _w.surrogate_top_p = config_dict.get("surrogate_top_p", 1.0)
    _w.surrogate_top_k = config_dict.get("surrogate_top_k", -1)
    _w.surrogate_client = None
    if _w.surrogate_server_addr is not None:
        _w.surrogate_client = OpenAI(
            base_url=f"{_w.surrogate_server_addr}/v1",
            api_key="dummy",
            timeout=httpx.Timeout(connect=5.0, read=30.0, write=5.0, pool=5.0),
        )
        try:
            served_ids = [m.id for m in _w.surrogate_client.models.list().data]
            if served_ids and _w.surrogate_model not in served_ids:
                if len(served_ids) == 1:
                    _w.surrogate_model = served_ids[0]
                else:
                    matched = [
                        m
                        for m in served_ids
                        if (_w.surrogate_model or "") in m
                        or m in (_w.surrogate_model or "")
                    ]
                    if matched:
                        _w.surrogate_model = matched[0]
        except Exception:
            pass

    # Which model backend the calls in beaver/model/ route to:
    #   "vllm"       -> OpenAI-compatible HTTP to a `vllm serve` process
    #   "hf"         -> OpenAI-compatible HTTP to beaver.hf_server
    #   "vllm-local" -> in-process vllm.LLM engine, no server
    _w.backend = config_dict.get("backend", "vllm")

    # Offline-engine knobs; only read when backend == "vllm-local".
    _w.vllm_max_model_len = config_dict.get("vllm_max_model_len", None)
    _w.vllm_gpu_memory_utilization = config_dict.get(
        "vllm_gpu_memory_utilization", None
    )
    # False lets the engine read the prefix cache for prompt_logprobs requests.
    # This is the whole reason the local backend exists -- vLLM's HTTP layer
    # hardcodes it to True (see beaver/model/local_vllm.py).
    _w.vllm_skip_prefix_cache_read = config_dict.get(
        "vllm_skip_prefix_cache_read", False
    )

    # hf-local (TransformerLens) knobs; only read when backend == "hf-local".
    _w.hf_local_dtype = config_dict.get("hf_local_dtype", "bfloat16")
    # TL-registered model whose conversion recipe fits this checkpoint's
    # architecture (e.g. "Qwen/Qwen3-4B" for "Qwen/Qwen3-4B-Instruct-2507").
    _w.hf_local_tl_recipe = config_dict.get("hf_local_tl_recipe", None)
    _w.hf_local_hidden_layers = config_dict.get("hf_local_hidden_layers", None)
    # Which TL activation to capture: "resid_post" (residual stream, matches
    # hf_server's layer forward-hook) or "mlp_post" (post-activation MLP
    # neurons, d_mlp-wide).
    _w.hf_local_hook = config_dict.get("hf_local_hook", "resid_post")

    # How many frontier nodes to expand per round. 1 = the original
    # single-node loop, unchanged. >1 batches them into one forward pass.
    _w.expand_batch_size = int(config_dict.get("expand_batch_size", 1) or 1)

    # Which stop rule frontier_step.check_termination applies.
    #   "max_seq_prob" (default) -- historical: max incomplete leaf < epsilon
    #   "bounds"                 -- UB<=threshold | LB>threshold | UB-LB<=eps
    _w.stopping_criterion = config_dict.get("stopping_criterion") or "max_seq_prob"

    _w.http_timeout = httpx.Timeout(connect=5.0, read=30.0, write=5.0, pool=5.0)
    _w.http_thread_local = threading.local()

    # Persistent OpenAI-compatible client. Only used to resolve the served
    # model name below (and by the surrogate path); the actual completion
    # calls go through beaver.model's own raw-httpx client. The local backend
    # has no server to ask, so it keeps the configured name as-is.
    _w.client = None
    if _w.backend in ("vllm", "hf"):
        _w.client = OpenAI(
            base_url=f"{_w.server_addr}/v1",
            api_key="dummy",
            timeout=httpx.Timeout(connect=5.0, read=30.0, write=5.0, pool=5.0),
        )
        # Resolve the model name actually registered on the server.
        try:
            served_ids = [m.id for m in _w.client.models.list().data]
            if served_ids and _w.model_name not in served_ids:
                if len(served_ids) == 1:
                    _w.model_name = served_ids[0]
                else:
                    matched = [
                        m
                        for m in served_ids
                        if _w.model_name in m or m in _w.model_name
                    ]
                    if matched:
                        _w.model_name = matched[0]
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Helper function to build prompt with optional chat template
# ---------------------------------------------------------------------------


_W_FALLBACK = object()  # sentinel: pull this arg from the worker global `_w`


def build_prompt(
    instance,
    continuation,
    tokenizer=None,
    chat_mode=_W_FALLBACK,
    system_message=_W_FALLBACK,
    fewshot_messages=_W_FALLBACK,
    chat_template_kwargs=_W_FALLBACK,
):
    """Build the final prompt string sent to the vLLM completions API.

    Args:
        instance: Dict with at least ``prompt`` (string). May also contain
            ``system_prompt`` and ``fewshot_messages`` (per-instance overrides),
            and ``rendered_prompt`` (see below).
        continuation: List of token IDs generated so far (the partial sequence
            being extended).
        tokenizer, chat_mode, system_message, fewshot_messages,
            chat_template_kwargs: when None, fall back to the worker global
            ``_w`` (the normal in-pool path). Pass them explicitly to call from
            outside a worker process (e.g. probe_ranker in the main process),
            where ``_w`` is not initialized.

    Returns:
        The prompt string ready for the completions API.
    """

    if tokenizer is None:
        tokenizer = _w.tokenizer
    if chat_mode is _W_FALLBACK:
        chat_mode = _w.chat_mode
    if system_message is _W_FALLBACK:
        system_message = _w.system_message
    if fewshot_messages is _W_FALLBACK:
        fewshot_messages = _w.fewshot_messages
    if chat_template_kwargs is _W_FALLBACK:
        chat_template_kwargs = getattr(_w, "chat_template_kwargs", None)
    if chat_template_kwargs is None:
        chat_template_kwargs = {"enable_thinking": getattr(_w, "enable_thinking", False)}

    prompt_text = instance["prompt"]

    rendered = instance.get("rendered_prompt")
    if rendered is None and instance.get("reasoning_chain_tokens"):
        raise ValueError(
            f"instance idx={instance.get('idx')} carries reasoning_chain_tokens "
            f"but no rendered_prompt. The chain is the conditioning prefix; "
            f"without the pre-rendered prompt that contains it, verification "
            f"would silently condition on the bare prompt instead. Regenerate "
            f"the chains file so each instance carries rendered_prompt."
        )
    if rendered is not None:
        if continuation:
            return rendered + tokenizer.decode(continuation, skip_special_tokens=False)
        return rendered

    if not chat_mode:
        if continuation:
            return prompt_text + tokenizer.decode(
                continuation, skip_special_tokens=False
            )
        return prompt_text

    # Build messages list for chat template
    messages = []

    # System message: per-instance > global > none
    system = instance.get("system_prompt") or system_message
    if system:
        messages.append({"role": "system", "content": system})

    # Few-shot: per-instance > global > none
    fewshot = instance.get("fewshot_messages") or fewshot_messages or []
    for msg in fewshot:
        if "prompt" in msg and "response" in msg:
            messages.append({"role": "user", "content": msg["prompt"]})
            messages.append({"role": "assistant", "content": msg["response"]})
        elif "role" in msg and "content" in msg:
            messages.append(msg)
        else:
            raise ValueError(
                f"Invalid fewshot message format: {msg}. "
                "Expected {{'prompt': ..., 'response': ...}} "
                "or {{'role': ..., 'content': ...}}"
            )

    messages.append({"role": "user", "content": prompt_text})

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        **chat_template_kwargs,
    )
    if continuation:
        prompt += tokenizer.decode(continuation, skip_special_tokens=False)

    return prompt


# ---------------------------------------------------------------------------
# Model calls live in beaver/model/ now
# ---------------------------------------------------------------------------
#
# _completion_with_retry, the model_* operations, the response parsers, and the
# per-thread timing accumulator moved to the beaver.model package so a single
# `backend` setting can route them to a vllm server, beaver.hf_server, or an
# in-process vLLM engine. Import them from there:
#
#     from beaver.model import model_generate_next_token_logprobs, drain_model_timings
#
# This module keeps the pieces those calls depend on -- the `_w` worker state,
# init_worker_state(), and build_prompt() -- plus the grammar/constraint and
# frontier-update logic below. beaver.model imports from here, so nothing is
# re-exported back (that would be an import cycle).


# ---------------------------------------------------------------------------
# worker_setup — shared boilerplate at the start of _worker_process_instance
# ---------------------------------------------------------------------------


def worker_setup(args):
    """Common setup for worker process instances.

    Returns (instance, log_file, profile_log_file).
    """
    instance, run_log_dir = args

    assert _w.tokenizer is not None, "Worker tokenizer not initialized"
    assert _w.dataset_name is not None, "Worker dataset_name not initialized"

    log_file = run_log_dir / f"{instance['idx']}.jsonl"
    profile_log_file = run_log_dir / f"{instance['idx']}.profile.json"

    setup_info = {
        "event": "instance_start",
        "idx": instance["idx"],
        "start_time": time.time(),
        # .get, not [] -- system_prompt is optional (build_prompt reads it with
        # .get too). Hard-indexing it here killed every worker with
        # KeyError: 'system_prompt' on any dataset that omits it, e.g.
        # secure_code's default instruct-v2.json.
        "system_prompt": instance.get("system_prompt"),
        "prompt": instance["prompt"],
        "max_iterations": _w.max_iterations,
        "use_grammar": _w.use_grammar,
    }
    if "group" in instance:
        setup_info["group"] = instance["group"]
    # IterativeVerifier search fields: lineage and the beam-entry threshold.
    for key in ("search_round", "parent_idx", "based_on", "beam_entry_threshold",
                "tree", "tree_new", "tree_distance"):
        if key in instance:
            setup_info[key] = instance[key]
    log_json(setup_info, log_file)

    return instance, log_file, profile_log_file


# ---------------------------------------------------------------------------
# apply_top_p_top_k — parameterized pruning shared by both verifiers
# ---------------------------------------------------------------------------


def apply_top_p_top_k(log_probs):
    """Turn a raw ``[N, 2]`` ``[token_id, logprob]`` slice into the children
    the verifier attaches, plus the mass it leaves unresolved.

    In decoded mode (``_w.decoded_dist``) the slice is mapped to the decoded
    distribution for the run's temperature / top-p / top-k and the returned
    log-probs are *decoded* ones (see beaver/verifiers/decoding.py).

    Otherwise this is the historical cutoff: rows outside top-p / top-k are
    dropped, their raw mass is returned as ``culled_prob_sum`` (carried as
    unresolved by the caller) and the kept rows keep their raw log-probs.

    Args:
        log_probs: np.array of shape [N, 2] with columns [token_id, logprob],
                   assumed sorted by logprob descending.
    Returns:
        (filtered_log_probs, culled_prob_sum) where filtered_log_probs has the
        same [N', 2] shape with pruned rows removed.
    """
    if getattr(_w, "decoded_dist", False):
        return decode_distribution(log_probs, _w.temperature, _w.top_p, _w.top_k)
    if _w.top_p >= 1.0 and _w.top_k < 0:
        return log_probs, 0.0

    # Sort by logprob descending (in case input isn't sorted)
    order = np.argsort(log_probs[:, 1])[::-1]
    sorted_lp = log_probs[order]

    keep = len(sorted_lp)
    culled_prob_sum = 0.0

    # Top-k: keep only the k highest-logprob entries
    if _w.top_k > 0 and keep > _w.top_k:
        culled_prob_sum += np.exp(sorted_lp[_w.top_k :, 1]).sum()
        sorted_lp = sorted_lp[: _w.top_k]
        keep = _w.top_k

    # Top-p: keep smallest set whose cumulative prob >= top_p
    if _w.top_p < 1.0:
        probs = np.exp(sorted_lp[:, 1])
        cumsum = np.cumsum(probs)
        cutoff = np.searchsorted(cumsum, _w.top_p, side="right") + 1
        if cutoff < keep:
            culled_prob_sum += probs[cutoff:].sum()
            sorted_lp = sorted_lp[:cutoff]

    return sorted_lp, culled_prob_sum


def logprobs_dict_to_tensor(logprobs_dict, vocab_size=None):
    """Convert logprobs dict to tensor for existing code compatibility.

    Args:
        logprobs_dict: Dict mapping {token_id: logprob}
        vocab_size: Optional vocab size. If None, uses _w.vocab_size.

    Returns:
        torch.Tensor: Log probabilities tensor of shape (vocab_size,)
                      with -inf for missing tokens
    """
    if vocab_size is None:
        vocab_size = _w.vocab_size

    log_probs = torch.full((vocab_size,), float("-inf"))
    for token_id, logprob in logprobs_dict.items():
        log_probs[token_id] = logprob

    return log_probs


# ---------------------------------------------------------------------------
# get_grammar_mask — grammar constraint mask used by both verifiers
# ---------------------------------------------------------------------------


def get_grammar_mask(tokens):
    """Get grammar validity mask for next tokens.

    Args:
        tokens: List of token IDs
        logprobs_dict: Dict {token_id: logprob} (not used - kept for compatibility)

    Returns:
        boolean numpy array of shape (vocab_size,).
    """
    if _w.use_grammar:
        from beaver.verifiers.llguidance_grammar import get_next_token_bool_mask

        return get_next_token_bool_mask(tokens, _w.lltokenizer, _w.ebnf)

    # Return all True for vocab_size
    return np.ones(_w.vocab_size, dtype=bool)


# ---------------------------------------------------------------------------
# log_profiling — write profiling data, optionally print in verbose mode
# ---------------------------------------------------------------------------


def log_profiling(profiling_data, profile_log_file):
    """Log profiling data to file; print if verbose."""
    log_json(profiling_data, profile_log_file)
    if _w.verbose:
        print(profiling_data)


# ---------------------------------------------------------------------------
# safe_worker — decorator that catches exceptions so a worker crash returns
# an error dict instead of killing the process silently.
# ---------------------------------------------------------------------------


def safe_worker(fn):
    """Wrap a worker function so exceptions produce an error result dict."""

    @functools.wraps(fn)
    def wrapper(args):
        try:
            return fn(args)
        except Exception as e:
            try:
                idx = args[0]["idx"]
            except Exception:
                idx = "unknown"
            tb = traceback.format_exc()
            print(f"[Worker ERROR] Instance {idx} failed: {e}\n{tb}")
            return {
                "idx": idx,
                "total_transitions": 0,
                "instance_run_time": 0.0,
                "transitions": 0,
                "time_s": 0.0,
                "error": f"{type(e).__name__}: {e}",
            }

    return wrapper


def update_frontier(
    frontier,
    instance,
    previous_element,
    log_probs,
    bit_mask,
    element_cls=FrontierElement,
    return_violations=False,
    return_counts=False,
):
    """Expand a frontier element, filtering by grammar + semantics.

    Args:
        frontier: The Frontier object
        previous_element: FrontierElement being expanded
        log_probs: numpy array of shape [N, 2] with [token_id, logprob] pairs
        bit_mask: Grammar validity mask (torch.Tensor or np.ndarray)
        element_cls: Class to use for new elements (default: FrontierElement)
        return_violations: If True, append violation_seqs as 6th return value.
            violation_seqs is a List[List[int]] of full token paths that violated.
        return_counts: If True, append ``n_semantic_attempted`` as the 4th
            return value — the number of sequences *sent to* the semantic
            constraint this expansion (i.e. the gross semantic-check count
            before the cache/dedup optimizations reduce it to actual check_fn
            calls). This is the "everything checked before caching" cost.

    A sequence counts as "complete" (and gets semantically checked) either
    because the model emitted EOS, the dataset's own complete_fn says so
    (e.g. a detected refusal), or it hit the gen_length cap with no token
    budget left to keep expanding.

    Returns:
        (new_elements, presemantic_check_time, semantic_check_time
            [, n_semantic_attempted])
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
        if return_counts:
            return [], 0, 0, 0
        return [], 0, 0

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
        complete_flag = np.ones(len(valid_indices), dtype=bool)
    else:
        eos_flag = np.array([i in _w.eos_tokens for i in valid_indices], dtype=bool)
        refusal_flag = check_complete_call(
            _w.dataset_name, instance, decoded_sequences, token_lists
        )
        complete_flag = np.logical_or(eos_flag, refusal_flag)

    if _w.verbose:
        print(
            f"Complete flag: {sum(complete_flag)}: "
            f"{[i for i in valid_indices[complete_flag]]}"
        )
        print(f"Worker eos tokens: {_w.eos_tokens}")

    presemantic_check_time = time.time()

    # Semantic checking. When prefix_closed is False, only ever resolve a
    # sequence's violation/completion status once it is actually complete —
    # never on an early, incomplete prefix.
    if _w.prefix_closed:
        semantic_check_mask = np.logical_or(
            check_semantic_call(_w.dataset_name, instance, decoded_sequences, token_lists),
            complete_flag,
        )
    else:
        semantic_check_mask = complete_flag
    semantic_check_indices = np.where(semantic_check_mask)[0]

    # Semantic-check cost (gross): the number of sequences sent to the semantic
    # constraint this expansion, counted *before* the constraint's internal
    # dedup + cache reduce it to actual check_fn calls. This is the
    # "everything checked before caching / optimizations" cost.
    n_semantic_attempted = int(len(semantic_check_indices))

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
    new_elements = []

    for idx in range(len(valid_indices)):
        token_id = int(valid_indices[idx])
        seq_logprob = previous_element.logprob + logprobs_dict[token_id]

        is_violation = token_id in violations
        new_elem = element_cls(
            element_id=frontier.total_elements,
            token=token_id,
            logprob=seq_logprob,
            is_completed=False if is_violation else complete_flag[idx].item(),
            is_violation=is_violation,
        )

        frontier.total_elements += 1
        new_elements.append(new_elem)

    if return_counts:
        return (
            new_elements,
            presemantic_check_time,
            semantic_check_time,
            n_semantic_attempted,
        )
    if return_violations:
        return (
            new_elements,
            presemantic_check_time,
            semantic_check_time,
        )
    return (
        new_elements,
        presemantic_check_time,
        semantic_check_time,
    )
