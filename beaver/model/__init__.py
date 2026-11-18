"""Model-calling layer for beaver verifiers.

One public entry point, ``completion()``, dispatches to a backend chosen by
``_w.backend``:

    "vllm"        OpenAI-compatible HTTP -> a `vllm serve` process
    "hf"          OpenAI-compatible HTTP -> beaver.hf_server
    "vllm-local"  in-process vllm.LLM engine, no server (see local_vllm.py)
    "hf-local"    in-process TransformerLens model, no server; the only backend
                  giving hidden states AND prefix caching (see hf_local.py)

Every backend returns the same list-of-choices shape, so the higher-level
operations in ops.py are backend-agnostic.

Timing for every call is accumulated per thread and drained by
``drain_model_timings()``, which verifiers fold into their profile logs.
"""

import types

from beaver.model.ops import (
    model_generate_next_token_logprobs,
    model_generate_next_token_logprobs_batch,
    model_sample_continuations,
    model_sample_continuations_surrogate,
    model_sample_sequence,
    model_score_sequences,
    model_score_sequences_per_token,
)
from beaver.model.parsing import (
    parse_hidden_states,
    parse_logprobs,
    parse_sample_choice,
    top_logprobs_to_array,
)
from beaver.verifiers.worker_common import _w

BACKENDS = ("vllm", "hf", "vllm-local", "hf-local")
HTTP_BACKENDS = ("vllm", "hf")
# Backends able to return transformer hidden states, which the
# "representation" verifier's NVF needs.
#
# vllm-local qualifies via vLLM 0.21's extract_hidden_states spec-decoding
# method plus an in-process KV connector (beaver/model/vllm_hidden_states.py).
# The "vllm" HTTP backend still does not: its EngineCore is a separate process
# from the API server and the completions response has nowhere to carry the
# states, so they would need an out-of-band side channel keyed by request id.
HIDDEN_STATE_BACKENDS = ("hf", "hf-local", "vllm-local")

__all__ = [
    "BACKENDS",
    "HTTP_BACKENDS",
    "HIDDEN_STATE_BACKENDS",
    "completion",
    "drain_model_timings",
    "model_generate_next_token_logprobs",
    "model_generate_next_token_logprobs_batch",
    "model_sample_continuations",
    "model_sample_continuations_surrogate",
    "model_sample_sequence",
    "model_score_sequences",
    "model_score_sequences_per_token",
    "parse_hidden_states",
    "parse_logprobs",
    "parse_sample_choice",
    "top_logprobs_to_array",
]


def _record_timing(label, *, prefix, net_s, parse_s, n_rows, resp_bytes, **extra):
    """Append one call's timing to this thread's accumulator.

    ``prefix`` namespaces the emitted profile keys by backend family: HTTP
    backends keep the historical ``http.*`` keys so runs stay comparable
    against earlier logs, while the local engine emits ``local.*`` (and its own
    ``num_cached_tokens``, which no HTTP backend can report).
    """
    timings = getattr(_w.http_thread_local, "timings", None)
    if timings is None:
        timings = []
        _w.http_thread_local.timings = timings
    timings.append(
        types.SimpleNamespace(
            label=label, prefix=prefix, net_s=net_s, parse_s=parse_s,
            n_rows=n_rows, resp_bytes=resp_bytes, extra=extra,
        )
    )


def drain_model_timings():
    """Pop and summarize this thread's accumulated call timings, grouped by
    ``"{prefix}.{label}"``.

    Returns ``{key: {"net_s", "parse_s", "n_rows", "resp_bytes", "n_requests",
    ...}}`` (all sums except n_requests, which is the call count). Calling this
    clears the accumulator, so repeated calls only report requests made since
    the previous drain.
    """
    timings = getattr(_w.http_thread_local, "timings", None)
    if not timings:
        return {}
    _w.http_thread_local.timings = []
    summary = {}
    for t in timings:
        agg = summary.setdefault(
            f"{t.prefix}.{t.label}",
            {"net_s": 0.0, "parse_s": 0.0, "n_rows": 0, "resp_bytes": 0,
             "n_requests": 0},
        )
        agg["net_s"] += t.net_s
        agg["parse_s"] += t.parse_s
        agg["n_rows"] += t.n_rows
        agg["resp_bytes"] += t.resp_bytes
        agg["n_requests"] += 1
        for k, v in t.extra.items():
            agg[k] = agg.get(k, 0) + v
    return summary


# Kept as an alias: several call sites and every historical profile key still
# use the http-era name.
drain_http_timings = drain_model_timings


def completion(
    prompt,
    *,
    max_tokens,
    n=1,
    logprobs=None,
    echo=False,
    temperature=None,
    top_p=None,
    top_k=None,
    timeout=None,
    return_hidden_states=False,
    label="model_call",
):
    """Dispatch one completion call to the configured backend.

    Sampling params default to the worker globals (``_w.temperature`` /
    ``_w.top_p`` / ``_w.top_k``); pass them explicitly to override. Scoring
    overrides them to 1.0 / 1.0 / -1 so the returned logprobs are the model's
    true conditional logprobs rather than the processed distribution.

    Returns a list of choice objects (see parsing.py for the shape).
    """
    temperature = _w.temperature if temperature is None else temperature
    top_p = _w.top_p if top_p is None else top_p
    top_k = _w.top_k if top_k is None else top_k

    backend = getattr(_w, "backend", "vllm")
    if backend == "vllm-local":
        from beaver.model import local_vllm as impl
    elif backend == "hf-local":
        from beaver.model import hf_local as impl
    elif backend in HTTP_BACKENDS:
        from beaver.model import http_backend as impl
    else:
        raise ValueError(
            f"Unknown backend {backend!r}. Expected one of {BACKENDS}."
        )

    return impl.completion(
        prompt,
        max_tokens=max_tokens,
        n=n,
        logprobs=logprobs,
        echo=echo,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        timeout=timeout,
        return_hidden_states=return_hidden_states,
        label=label,
        record_timing=_record_timing,
    )
