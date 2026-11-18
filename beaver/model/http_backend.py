"""HTTP backend: OpenAI-compatible ``/v1/completions`` over the network.

Serves both ``backend="vllm"`` (a ``vllm serve`` process) and ``backend="hf"``
(``beaver.hf_server``) -- the wire format is identical, so one implementation
covers both.

Moved from beaver/verifiers/worker_common.py with no behavior change.
"""

import time
import traceback
import types

import httpx
import orjson

from beaver.verifiers.worker_common import _w

_MODEL_MAX_RETRIES = 5
_MODEL_RETRY_DELAY = 2  # base seconds between retries (doubles each attempt)


def _get_http_client():
    """Return this thread's raw httpx client, creating it on first use.

    Each thread gets its own client/connection pool (see init_worker_state) so
    one thread's _reset_client() can't close sockets another thread is reading.
    """
    client = getattr(_w.http_thread_local, "client", None)
    if client is None:
        client = httpx.Client(
            base_url=f"{_w.server_addr}/v1", timeout=_w.http_timeout
        )
        _w.http_thread_local.client = client
    return client


def _reset_client():
    """Recreate the calling thread's httpx client (e.g. after a timeout kills
    the connection)."""
    client = getattr(_w.http_thread_local, "client", None)
    if client is not None:
        try:
            client.close()
        except Exception:
            pass
    _w.http_thread_local.client = httpx.Client(
        base_url=f"{_w.server_addr}/v1", timeout=_w.http_timeout
    )


def _wrap_choice(choice_dict):
    """Wrap a raw completions ``choice`` JSON dict in an attribute-accessible
    object mirroring the OpenAI client's response shape.

    Callers read ``choice.index``, ``choice.text``, ``choice.hidden_states``,
    and ``choice.logprobs.{tokens,token_logprobs,top_logprobs}``. The nested
    ``logprobs`` dict (plain ``token_id:N -> float`` maps) is exposed as a
    namespace so the shared parsers in parsing.py work unchanged.

    ``local_vllm`` builds this same shape by hand -- keep them in sync.
    """
    lp = choice_dict.get("logprobs")
    logprobs_obj = types.SimpleNamespace(**lp) if lp is not None else None
    return types.SimpleNamespace(
        index=choice_dict.get("index", 0),
        text=choice_dict.get("text", ""),
        logprobs=logprobs_obj,
        hidden_states=choice_dict.get("hidden_states"),
        # "stop" (EOS / stop string) vs "length" (hit max_tokens); the
        # sampling verifier uses it to tell a complete draw from a truncated one.
        finish_reason=choice_dict.get("finish_reason"),
    )


def completion(
    prompt,
    *,
    max_tokens,
    n=1,
    logprobs=None,
    echo=False,
    temperature,
    top_p,
    top_k,
    timeout=None,
    return_hidden_states=False,
    label="model_call",
    record_timing,
):
    """Single OpenAI-compatible completions call with retry + client reset.

    Sampling params are resolved by the caller (see ``model.completion``), not
    defaulted here. Scoring passes 1.0 / 1.0 / -1 so the returned logprobs are
    the model's true conditional logprobs rather than the
    temperature/top-p/top-k-processed distribution.

    Returns ``response.choices`` as wrapped namespaces.
    """
    # Body matches what the OpenAI client sent (extra_body merged top-level), so
    # the wire format is identical for both vLLM and beaver.hf_server.
    body = {
        "model": _w.model_name,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "n": n,
        "temperature": temperature,
        "top_p": top_p,
        "echo": echo,
        "stop": [],
        "top_k": top_k,
        "chat_template_kwargs": {"enable_thinking": _w.enable_thinking},
        # build_prompt() already applies the model's chat template, so the prompt
        # string carries the model's own special tokens (incl. BOS). Tell vLLM not
        # to prepend another set — otherwise BOS-using models (e.g. Llama-3.1) get
        # a double <|begin_of_text|> and go off-distribution. No-op for Qwen
        # (no BOS token), so it is safe for every model.
        "add_special_tokens": False,
    }
    if return_hidden_states:
        body["return_hidden_states"] = True
    if logprobs is not None:
        body["logprobs"] = logprobs

    req_timeout = _w.http_timeout if timeout is None else timeout

    for attempt in range(_MODEL_MAX_RETRIES):
        try:
            if _w.verbose:
                print(f"prompt: {prompt}\nmax_tokens: {max_tokens}")

            # Raw POST + orjson parse: bypasses the OpenAI client's pydantic
            # validation of large top_logprobs payloads (serialization hotspot).
            _t_net = time.perf_counter()
            resp = _get_http_client().post(
                "/completions",
                content=orjson.dumps(body),
                headers={"content-type": "application/json"},
                timeout=req_timeout,
            )
            resp.raise_for_status()
            net_s = time.perf_counter() - _t_net

            _t_parse = time.perf_counter()
            data = orjson.loads(resp.content)
            choices = [_wrap_choice(c) for c in data["choices"]]
            parse_s = time.perf_counter() - _t_parse

            record_timing(
                label,
                prefix="http",
                net_s=net_s,
                parse_s=parse_s,
                n_rows=len(prompt) if isinstance(prompt, list) else 1,
                resp_bytes=len(resp.content),
            )

            if _w.verbose:
                print(f"response: {data}")

            return choices

        except Exception:
            if attempt == _MODEL_MAX_RETRIES - 1:
                print(
                    f"[Worker] {label} failed after {_MODEL_MAX_RETRIES} attempts: "
                    f"{traceback.format_exc()}"
                )
                raise

            delay = _MODEL_RETRY_DELAY * (2**attempt)
            print(
                f"[Worker] {label} retry {attempt + 1}/{_MODEL_MAX_RETRIES} "
                f"(waiting {delay}s): {traceback.format_exc()}"
            )
            _reset_client()
            time.sleep(delay)
