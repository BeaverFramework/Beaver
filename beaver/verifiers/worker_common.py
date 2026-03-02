"""Shared worker state and utilities for verification workers.

All multiprocessing worker state lives here as a single SimpleNamespace (`_w`),
replacing the 17+ individual `_worker_*` globals that were duplicated across
frontier_verifier.py and sampling_verifier.py.
"""

import time
import functools
import traceback
import types

from openai import OpenAI
import httpx
import numpy as np
import torch

from beaver.utils import log_json

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

    from llguidance.hf import from_tokenizer
    from transformers import AutoTokenizer

    _w.model_name = config_dict["model_name"]  # Store model name for API calls
    _w.tokenizer = AutoTokenizer.from_pretrained(config_dict["model_name"])
    _w.lltokenizer = from_tokenizer(_w.tokenizer)
    _w.vocab_size = len(_w.tokenizer)  # Store total vocab size including special tokens
    _w.server_addr = config_dict["server_addr"]
    _w.ebnf = config_dict["ebnf"]
    _w.dataset_name = config_dict["dataset_name"]
    _w.use_cache = config_dict.get("use_cache", True)
    _w.temperature = config_dict.get("temperature", 1.0)
    _w.top_p = config_dict.get("top_p", 1.0)
    _w.top_k = config_dict.get("top_k", -1)
    _w.eos_tokens = config_dict.get("eos_tokens", [])
    _w.stop_tokens = [
        _w.tokenizer.decode([et], skip_special_tokens=False) for et in _w.eos_tokens
    ]
    _w.gen_length = config_dict.get("gen_length", 128)
    _w.epsilon = config_dict.get("epsilon", 0.01)
    _w.verbose = config_dict.get("verbose", False)
    _w.max_iterations = config_dict.get("max_iterations", 100)
    _w.semantic_symbol = config_dict.get("semantic_symbol", None)
    _w.num_logprobs = config_dict.get("num_logprobs", 100)
    _w.use_grammar = config_dict.get("use_grammar", True)
    _w.chat_mode = config_dict.get("chat_mode", False)
    _w.system_message = config_dict.get("system_message", None)
    _w.fewshot_messages = config_dict.get("fewshot_messages", [])

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
        )

    # Frontier-specific — only set when present in config
    if "frontier_topp" in config_dict:
        _w.frontier_topp = config_dict["frontier_topp"]
        _w.frontier_topk = config_dict["frontier_topk"]
        _w.frontier_scoring_strategy = config_dict["frontier_scoring_strategy"]

    # Persistent OpenAI client — reuses TCP connections across calls
    from openai import OpenAI

    _w.client = OpenAI(
        base_url=f"{_w.server_addr}/v1",
        api_key="dummy",
        timeout=httpx.Timeout(connect=5.0, read=30.0, write=5.0, pool=5.0),
        # max_retries=0,  # We handle retries ourselves
    )

    # Resolve the model name actually registered on the server.
    # vLLM may serve the model under a different ID (e.g. a local path or a
    # shortened HuggingFace name) than what was requested.
    try:
        served_ids = [m.id for m in _w.client.models.list().data]
        if served_ids and _w.model_name not in served_ids:
            if len(served_ids) == 1:
                _w.model_name = served_ids[0]
            else:
                # Pick the first model whose ID contains or is contained by the requested name
                matched = [
                    m for m in served_ids if _w.model_name in m or m in _w.model_name
                ]
                if matched:
                    _w.model_name = matched[0]
    except Exception:
        pass  # Keep original; the API call will surface a clear 404 if still wrong


# ---------------------------------------------------------------------------
# Helper function to build prompt with optional chat template
# ---------------------------------------------------------------------------


def build_prompt_with_chat_template(input_ids, continuation, system_prompt=None):
    if not _w.chat_mode:
        # No chat template, just decode the input_ids
        return _w.tokenizer.decode(input_ids + continuation, skip_special_tokens=False)

    # Build messages list for chat template
    messages = []

    # Use per-instance system_prompt if provided, otherwise fall back to global
    effective_system = system_prompt if system_prompt else _w.system_message
    if effective_system:
        messages.append({"role": "system", "content": effective_system})

    # Add few-shot examples if present
    if _w.fewshot_messages:
        for msg in _w.fewshot_messages:
            # Support two formats:
            # 1. {"question": "...", "response": "..."} - convert to user/assistant pairs
            # 2. {"role": "...", "content": "..."} - use directly
            if "question" in msg and "response" in msg:
                messages.append({"role": "user", "content": msg["question"]})
                messages.append({"role": "assistant", "content": msg["response"]})
            elif "role" in msg and "content" in msg:
                messages.append(msg)
            else:
                raise ValueError(
                    f"Invalid fewshot message format: {msg}. "
                    "Expected either {{'question': ..., 'response': ...}} "
                    "or {{'role': ..., 'content': ...}}"
                )

    # Add current user input
    current_content = _w.tokenizer.decode(input_ids, skip_special_tokens=False)
    messages.append({"role": "user", "content": current_content})

    # Apply chat template
    prompt = _w.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    if continuation:
        prompt += _w.tokenizer.decode(continuation, skip_special_tokens=False)

    return prompt


# ---------------------------------------------------------------------------
# model_generate — get logprobs from vLLM OpenAI-compatible API
# ---------------------------------------------------------------------------


_MODEL_MAX_RETRIES = 5
_MODEL_RETRY_DELAY = 2  # base seconds between retries (doubles each attempt)


def _reset_client():
    """Recreate the OpenAI client (e.g. after a timeout kills the connection)."""

    _w.client = OpenAI(
        base_url=f"{_w.server_addr}/v1",
        api_key="dummy",
        timeout=httpx.Timeout(connect=5.0, read=30.0, write=5.0, pool=5.0),
        # max_retries=0,
    )


def model_generate_next_token_logprobs(input_ids, continuation, system_prompt=None):
    """Get next-token logprobs from vLLM server using OpenAI-compatible API.

    Returns a numpy array of shape [N, 2] with [token_id, logprob] pairs.
    Uses temperature, top_p, top_k from worker config (_w).

    Always uses completions API. If chat_mode=True, applies chat template manually.

    Args:
        input_ids: List of token IDs for the prompt

    Returns:
        np.ndarray: Array of shape [N, 2] with [token_id, logprob] pairs

    Raises:
        Exception: If server request fails after retries
    """
    for attempt in range(_MODEL_MAX_RETRIES):
        try:
            start_prompt_time = time.time()
            prompt = build_prompt_with_chat_template(
                input_ids, continuation, system_prompt=system_prompt
            )
            end_prompt_time = time.time()

            if _w.verbose:
                print(
                    f"[Worker] model_generate prompt time: {end_prompt_time - start_prompt_time:.3f}s"
                )
                print(f"prompt: {prompt}")

            response = _w.client.completions.create(
                model=_w.model_name,
                prompt=prompt,
                max_tokens=1,
                temperature=_w.temperature,
                logprobs=min(_w.num_logprobs, _w.vocab_size),
                # stop=[],  # Disable automatic stop tokens
                extra_body={
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            )
            end_request_time = time.time()
            if _w.verbose:
                print(
                    f"[Worker] model_generate request time: {end_request_time - end_prompt_time:.3f}s"
                )

            # Extract logprobs from response
            logprobs_obj = response.choices[0].logprobs

            # Completions API: logprobs.top_logprobs is a list of dicts
            if (
                logprobs_obj is None
                or not hasattr(logprobs_obj, "top_logprobs")
                or logprobs_obj.top_logprobs is None
            ):
                raise ValueError(
                    "No logprobs returned from server. "
                    "Make sure server was started with --max-logprobs -1"
                )
            top_logprobs_dict = logprobs_obj.top_logprobs[0]

            # Convert dict {token: logprob} to [[token_id, logprob], ...]
            logprobs = []
            for key, logprob in top_logprobs_dict.items():
                if key.startswith("token_id:"):
                    token_id = int(key.split(":")[1])
                    logprobs.append([token_id, logprob])

            if not logprobs:
                raise ValueError(
                    "No token_id logprobs found in response. "
                    "Check if server was started with --max-logprobs flag."
                )
            end_parse_time = time.time()

            if _w.verbose:
                print(f"model response: {response}")
                print(f"model response logprobs: {logprobs}")
                print(
                    f"[Worker] model_generate parse time: {end_parse_time - end_request_time:.3f}s"
                )

            return np.array(logprobs)

        except Exception:
            if attempt == _MODEL_MAX_RETRIES - 1:
                print(
                    f"[Worker] model_generate failed after {_MODEL_MAX_RETRIES} attempts: {traceback.format_exc()}"
                )
                raise

            delay = _MODEL_RETRY_DELAY * (2**attempt)
            print(
                f"[Worker] model_generate retry {attempt + 1}/{_MODEL_MAX_RETRIES} "
                f"(waiting {delay}s): {traceback.format_exc()}"
            )
            _reset_client()
            time.sleep(delay)


def model_sample_sequence(input_ids, max_tokens, system_prompt=None):
    """Sample multiple tokens from vLLM server.

    For sampling verifier - generates a sequence of tokens with their logprobs.
    Server applies temperature/top_p/top_k before sampling.

    Always uses completions API. If chat_mode=True, applies chat template manually.

    Args:
        input_ids: List of token IDs for the prompt
        max_tokens: Maximum number of tokens to generate

    Returns:
        tuple: (token_ids_list, token_logprobs_list) where:
            - token_ids_list: List of generated token IDs
            - token_logprobs_list: List of logprobs for each token

    Raises:
        Exception: If server request fails after retries
    """
    for attempt in range(_MODEL_MAX_RETRIES):
        try:
            # Build prompt (applies chat template if chat_mode=True)
            start_prompt_time = time.time()
            prompt = build_prompt_with_chat_template(
                input_ids, [], system_prompt=system_prompt
            )
            end_prompt_time = time.time()

            if _w.verbose:
                print(
                    f"[Worker] model_generate prompt time: {end_prompt_time - start_prompt_time:.3f}s"
                )
                print(f"prompt: {prompt}")

            # Request sequence generation with logprobs using completions API
            response = _w.client.completions.create(
                model=_w.model_name,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=_w.temperature,
                top_p=_w.top_p,
                logprobs=1,  # Just need each sampled token's logprob
                stop=[],  # Disable automatic stop tokens
                extra_body={
                    "top_k": _w.top_k,
                },
            )

            end_request_time = time.time()
            if _w.verbose:
                print(
                    f"[Worker] model_generate request time: {end_request_time - end_prompt_time:.3f}s"
                )

            # Extract the sampled tokens and their logprobs
            choice = response.choices[0]
            logprobs_obj = choice.logprobs

            # Completions API
            if logprobs_obj is None or not logprobs_obj.token_logprobs:
                return [], []

            # Parse token IDs from 'token_id:123' format
            token_ids = []
            for token_str in logprobs_obj.tokens:
                if token_str.startswith("token_id:"):
                    token_id = int(token_str.split(":")[1])
                    token_ids.append(token_id)
                else:
                    # Shouldn't happen with --max-logprobs set
                    # Fall back to encoding the token string
                    enc_ids = _w.tokenizer.encode(token_str, add_special_tokens=False)
                    if enc_ids:
                        token_ids.append(enc_ids[0])

            token_logprobs = logprobs_obj.token_logprobs
            end_parse_time = time.time()

            if _w.verbose:
                print(f"model response: {response}")
                print(f"model response logprobs: {token_ids, token_logprobs}")
                print(
                    f"[Worker] model_generate parse time: {end_parse_time - end_request_time:.3f}s"
                )

            return token_ids, token_logprobs

        except Exception:
            if attempt == _MODEL_MAX_RETRIES - 1:
                print(
                    f"[Worker] model_sample_sequence failed after {_MODEL_MAX_RETRIES} attempts: {traceback.format_exc()}"
                )
                raise

            delay = _MODEL_RETRY_DELAY * (2**attempt)
            print(
                f"[Worker] model_sample_sequence retry {attempt + 1}/{_MODEL_MAX_RETRIES} "
                f"(waiting {delay}s): {traceback.format_exc()}"
            )
            _reset_client()
            time.sleep(delay)


# ---------------------------------------------------------------------------
# worker_setup — shared boilerplate at the start of _worker_process_instance
# ---------------------------------------------------------------------------


def worker_setup(args):
    """Common setup for worker process instances.

    Returns (instance, prompt_ids, log_file, profile_log_file).
    """
    instance, run_log_dir = args

    assert _w.tokenizer is not None, "Worker tokenizer not initialized"
    assert _w.dataset_name is not None, "Worker dataset_name not initialized"

    prompt_ids = instance["prompt_ids"]
    log_file = run_log_dir / f"{instance['idx']}.jsonl"
    profile_log_file = run_log_dir / f"{instance['idx']}.profile.json"

    setup_info = {
        "idx": instance["idx"],
        "prompt_length": len(prompt_ids),
        "prompt_ids": prompt_ids,
        "decoded_prompt": _w.tokenizer.decode(prompt_ids, skip_special_tokens=True),
        "max_iterations": _w.max_iterations,
        "use_grammar": _w.use_grammar,
    }
    log_json(setup_info, log_file)

    return instance, prompt_ids, log_file, profile_log_file


# ---------------------------------------------------------------------------
# apply_top_p_top_k — parameterized pruning shared by both verifiers
# ---------------------------------------------------------------------------


def apply_top_p_top_k(log_probs):
    """Apply top-p and top-k pruning, removing filtered token entries.

    Args:
        log_probs: np.array of shape [N, 2] with columns [token_id, logprob],
                   assumed sorted by logprob descending.
    Returns:
        (filtered_log_probs, culled_prob_sum) where filtered_log_probs has the
        same [N', 2] shape with pruned rows removed.
    """
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
                "transitions": 0,
                "time_s": 0.0,
                "error": f"{type(e).__name__}: {e}",
            }

    return wrapper
