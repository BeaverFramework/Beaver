"""Backend-agnostic model operations.

Every function here goes through ``beaver.model.completion``, which dispatches
on ``_w.backend``. Because all backends return the same choice shape (see
parsing.py), these bodies are unchanged from their previous home in
beaver/verifiers/worker_common.py.
"""

import time
import traceback

import httpx
from openai import OpenAI
from transformers import AutoTokenizer

from beaver.model.parsing import (
    parse_hidden_states,
    parse_logprobs,
    parse_sample_choice,
    top_logprobs_to_array,
)
from beaver.verifiers.worker_common import _w, build_prompt

_MODEL_MAX_RETRIES = 5
_MODEL_RETRY_DELAY = 2


def _completion(*args, **kwargs):
    # Imported lazily to keep beaver.model.__init__ free to import this module.
    from beaver.model import completion

    return completion(*args, **kwargs)


def _expansion_sampling_kwargs():
    """Sampling params for a next-token-distribution request.

    Decoded mode wants the *raw* top-N slice: temperature / top-p / top-k are
    applied in Python (``apply_top_p_top_k`` -> ``decode_distribution``) so the
    treatment is identical across backends and identical to the cache-rebuild
    path, whose echo scores are raw on every backend anyway. Otherwise the
    worker globals apply, as they always have.
    """
    if getattr(_w, "decoded_dist", False):
        return dict(temperature=1.0, top_p=1.0, top_k=-1)
    return {}


def model_generate_next_token_logprobs(
    instance, continuation, return_hidden_states=False
):
    prompt = build_prompt(instance, continuation)
    choices = _completion(
        prompt,
        max_tokens=1,
        logprobs=min(_w.num_logprobs, _w.vocab_size),
        return_hidden_states=return_hidden_states,
        label="model_generate_next_token_logprobs",
        **_expansion_sampling_kwargs(),
    )
    choice = choices[0]
    logprobs_array = parse_logprobs(choice)[0]

    if return_hidden_states:
        hs_raw = getattr(choice, "hidden_states", None)
        if hs_raw is None and hasattr(choice, "model_extra"):
            hs_raw = (choice.model_extra or {}).get("hidden_states")
        hidden_states = parse_hidden_states(hs_raw) if hs_raw else {}
        return logprobs_array, hidden_states, prompt

    return logprobs_array, prompt


def model_generate_next_token_logprobs_batch(
    instance, continuations, return_hidden_states=False
):
    """Batched form of model_generate_next_token_logprobs.

    Sends N continuations of the same instance as ONE request so the backend
    can pad them into a single forward pass. This is the whole point of
    expanding several frontier nodes per round: a single-row forward pass
    leaves most of the GPU idle, and hf-local cannot use worker parallelism at
    all (max_workers is clamped to 1), so batching here is its only route to
    utilisation.

    Both backends already accept multiple rows: hf_server declares
    ``prompt: list[list[int]]`` and fans rows into its batching queue, and
    hf_local's ``_as_token_ids`` normalises a list of ID rows.

    Returns a list, one entry per continuation, in the order given:
        Without hidden states: ``[(logprobs_array, prompt), ...]``
        With hidden states:    ``[(logprobs_array, hidden_states, prompt), ...]``
    """
    if not continuations:
        return []
    prompts = [build_prompt(instance, c) for c in continuations]
    choices = _completion(
        prompts,
        max_tokens=1,
        logprobs=min(_w.num_logprobs, _w.vocab_size),
        return_hidden_states=return_hidden_states,
        label="model_generate_next_token_logprobs",
        **_expansion_sampling_kwargs(),
    )
    if len(choices) != len(prompts):
        raise RuntimeError(
            f"batched generate asked for {len(prompts)} rows but the backend "
            f"returned {len(choices)} choices"
        )
    # Backends may return choices out of submission order; `index` is the
    # submission index, so sort by it rather than trusting arrival order.
    choices = sorted(choices, key=lambda c: getattr(c, "index", 0))

    out = []
    for prompt, choice in zip(prompts, choices):
        logprobs_array = parse_logprobs(choice)[0]
        if return_hidden_states:
            hs_raw = getattr(choice, "hidden_states", None)
            if hs_raw is None and hasattr(choice, "model_extra"):
                hs_raw = (choice.model_extra or {}).get("hidden_states")
            out.append(
                (logprobs_array, parse_hidden_states(hs_raw) if hs_raw else {}, prompt)
            )
        else:
            out.append((logprobs_array, prompt))
    return out


def model_score_sequences_per_token(instance, sequences, return_dists=False):
    """Per-token log-probs of each token sequence under the instance prompt.

    The shared prompt prefix and each candidate sequence are concatenated as
    *token IDs* and sent with ``echo=True``, which echoes the log-prob of every
    prompt token. The prefix is then sliced off.

    Sampling is forced to temperature=1.0 / top_p=1.0 / top_k=-1 so the scores
    are the model's true conditional log-probs. This matters because backends
    run with processed_logprobs: any non-trivial temperature or top-p/top-k
    would otherwise rescale (or -inf-mask) the prompt logprobs.

    In decoded mode (``_w.decoded_dist``) the per-token log-probs returned are
    the *decoded* ones: each position's raw top-N slice is passed through
    ``apply_top_p_top_k`` and the on-path token looked up in it (``-inf`` if
    the sampler cannot emit it there), so a cached sequence's mass matches
    what the frontier expansion would assign it. ``per_position_dists`` stay
    raw; callers decode them at the point of use.

    Concatenating at the token-ID level (rather than decoding to text and
    re-tokenizing) preserves the exact candidate tokens.

    Args:
        sequences: list of token-ID lists.
        return_dists: when True, also return the full top-``num_logprobs``
            next-token distribution at each on-path position. Position ``j`` of
            ``seq_i`` is ``P(. | prompt, seq_i[:j])`` — the children-candidate
            distribution for expanding the depth-``j`` node — used by the cache
            rebuild to recover high-prob siblings without extra round-trips.

    Returns:
        Without dists: ``list[list[float]]`` where
            ``result[i][j] = log p(seq_i[j] | prompt, seq_i[:j])``.
        With dists: ``(per_token_logprobs, per_position_dists)`` where
            ``per_position_dists[i][j]`` is an ``[N, 2]`` array of
            ``[token_id, logprob]`` for the distribution at position ``j``.
    """
    if not sequences:
        return ([], []) if return_dists else []

    prefix_ids = _w.tokenizer.encode(
        build_prompt(instance, []), add_special_tokens=False
    )
    start = len(prefix_ids)

    decoded = getattr(_w, "decoded_dist", False)
    # Decoded mode needs the same slice width as expansions so the support
    # (and any unresolved remainder) is judged identically on both paths.
    # Without return_dists only the on-path log-probs (token_logprobs) are
    # read, so the top-N per position is not requested (logprobs=0): with
    # echo=True it would come back for every prompt position.
    if decoded:
        logprobs = max(50, min(_w.num_logprobs, _w.vocab_size))
    else:
        logprobs = 50 if return_dists else 0

    # Sent in score_chunk_size-sized batches (one multi-prompt request per
    # chunk) rather than one sequence per request: vLLM schedules a
    # multi-prompt request as concurrent sequences internally, sharing the
    # already-committed prefix cache across them, so batching mainly
    # amortizes fixed per-request overhead rather than reducing GPU compute.
    choices = []
    for i in range(0, len(sequences), _w.score_chunk_size):
        chunk = sequences[i : i + _w.score_chunk_size]
        chunk_choices = _completion(
            [prefix_ids + list(seq) for seq in chunk],
            max_tokens=1,  # nothing is sampled; sliced off below
            # logprobs=0 echoes only the on-path token logprob; bumping to
            # num_logprobs additionally echoes the top-N distribution per
            # position (needed for sibling recovery on rebuild).
            logprobs=logprobs,
            echo=True,
            temperature=1.0,
            top_p=1.0,
            top_k=-1,
            timeout=120.0,
            label="model_score_sequences_per_token",
        )
        choices.extend(chunk_choices)

    if decoded:
        from beaver.verifiers.worker_common import apply_top_p_top_k

        per_token = []
        for choice, seq in zip(choices, sequences):
            lps = []
            for j, tok in enumerate(seq):
                dist = top_logprobs_to_array(choice.logprobs.top_logprobs[start + j])
                if dist.shape[0] == 0:
                    # hf_server only keeps --echo-tail-k trailing positions;
                    # a miss here would silently score the sequence at 0.
                    raise RuntimeError(
                        f"no next-token distribution echoed at continuation "
                        f"position {j} of a {len(seq)}-token sequence; raise "
                        f"the server's --echo-tail-k above gen_length"
                    )
                kept, _ = apply_top_p_top_k(dist)
                hit = kept[kept[:, 0] == tok]
                lps.append(float(hit[0, 1]) if len(hit) else float("-inf"))
            per_token.append(lps)
    else:
        per_token = [
            choice.logprobs.token_logprobs[start : start + len(seq)]
            for choice, seq in zip(choices, sequences)
        ]
    if not return_dists:
        return per_token

    per_position_dists = [
        [
            top_logprobs_to_array(d)
            for d in choice.logprobs.top_logprobs[start : start + len(seq) + 1]
        ]
        for choice, seq in zip(choices, sequences)
    ]
    return per_token, per_position_dists


def model_score_sequences(instance, sequences):
    """Summed log-prob of each token sequence under the instance prompt.

    Args:
        sequences: list of token-ID lists.

    Returns:
        list[float]: one summed log-prob per sequence.
    """
    return [sum(lps) for lps in model_score_sequences_per_token(instance, sequences)]


def model_sample_sequence(instance, max_tokens):
    """Sample a full sequence from the backend. Returns (token_ids, token_logprobs)."""
    prompt = build_prompt(instance, [])
    choices = _completion(
        prompt,
        max_tokens=max_tokens,
        n=1,
        logprobs=1,
        label="model_sample_sequence",
    )
    return parse_sample_choice(choices[0], _w.tokenizer)


def model_sample_continuations(instance, continuation, max_tokens, n=1):
    """Sample n continuations from a prefix. Returns list of (token_ids, logprobs)."""
    prompt = build_prompt(instance, continuation)
    choices = _completion(
        prompt,
        max_tokens=max_tokens,
        n=n,
        logprobs=1,
        label="model_sample_continuations",
    )
    results = [parse_sample_choice(c, _w.tokenizer) for c in choices]

    if _w.verbose:
        print(f"model_sample_continuations: {n} samples, {len(results)} parsed")

    return results


def model_sample_continuations_surrogate(instance, continuation, max_tokens, n=1):
    """Sample n continuations from a prefix via the *surrogate* model server.

    Uses _w.surrogate_client and _w.surrogate_* sampling params. Returns
    ``(list_of_text_strings, prompt)`` — note this returns decoded text, NOT
    (token_ids, logprobs) like model_sample_continuations.

    Always goes over HTTP: the surrogate is a separate, smaller model served
    alongside the main one, independent of the main backend choice.
    """
    if _w.surrogate_client is None:
        raise RuntimeError(
            "surrogate_client not initialized; pass surrogate_server_addr "
            "and surrogate_model to the verifier."
        )

    tokenizer = AutoTokenizer.from_pretrained(_w.surrogate_model)
    for attempt in range(_MODEL_MAX_RETRIES):
        try:
            prompt = build_prompt(instance, continuation, tokenizer)

            response = _w.surrogate_client.completions.create(
                model=_w.surrogate_model,
                prompt=prompt,
                max_tokens=max_tokens,
                n=n,
                temperature=_w.surrogate_temperature,
                top_p=_w.surrogate_top_p,
                extra_body={
                    "top_k": _w.surrogate_top_k,
                    "chat_template_kwargs": {"enable_thinking": _w.enable_thinking},
                },
            )

            results = [choice.text.strip() for choice in response.choices]

            if _w.verbose:
                print(
                    f"model_sample_continuations_surrogate: {n} samples, "
                    f"{len(results)} parsed"
                )

            return results, prompt

        except Exception:
            if attempt == _MODEL_MAX_RETRIES - 1:
                print(
                    f"[Worker] model_sample_continuations_surrogate failed after "
                    f"{_MODEL_MAX_RETRIES} attempts: {traceback.format_exc()}"
                )
                raise

            delay = _MODEL_RETRY_DELAY * (2**attempt)
            print(
                f"[Worker] model_sample_continuations_surrogate retry "
                f"{attempt + 1}/{_MODEL_MAX_RETRIES} "
                f"(waiting {delay}s): {traceback.format_exc()}"
            )
            _w.surrogate_client = OpenAI(
                base_url=f"{_w.surrogate_server_addr}/v1",
                api_key="dummy",
                timeout=httpx.Timeout(connect=5.0, read=30.0, write=5.0, pool=5.0),
            )
            time.sleep(delay)
