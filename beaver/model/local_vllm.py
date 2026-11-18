"""Offline (in-process) vLLM backend.

Why this exists
---------------
`CachedFrontierVerifier`'s warm-start scores an antichain of cached parent
paths that all share one long prompt prefix. The algorithm assumes that prefix
is served from vLLM's prefix cache rather than recomputed per call.

Over the OpenAI HTTP server that assumption silently fails: ``echo=True +
logprobs=N`` implies ``prompt_logprobs``, which makes
``SamplingParams.__post_init__`` default ``skip_reading_prefix_cache=True``
(vllm/sampling_params.py:421-425), and the KV cache manager then skips
prefix-cache reads entirely. ``CompletionRequest.to_sampling_params()`` never
exposes that field -- not directly, not via ``vllm_xargs`` -- so it is
unreachable over HTTP. Constructing ``SamplingParams`` ourselves, which only
the offline API allows, is the entire point of this backend.

Measured (Qwen3-4B, 177-token shared prefix, 4 sequences): with
``skip_reading_prefix_cache=False`` the engine reports ``num_cached_tokens=176``
versus ``0`` with the HTTP-forced default, and continuation-region logprobs
agree with the recompute path to within the engine's own run-to-run
nondeterminism (0.161 observed vs a 0.281 noise floor) while being bit-for-bit
self-consistent across repeat calls.

Two non-obvious requirements
----------------------------
1. ``skip_tokenizer_init=True`` is **mandatory**. With prefix-cache reads on,
   cached prompt positions come back with empty logprob entries whose token-id
   tensors can hold out-of-range values; vLLM's detokenizer then dies with
   ``OverflowError`` in ``logprobs.py::_update_prompt_logprobs``. Skipping the
   engine-side tokenizer bypasses that path. Beaver already owns a tokenizer
   (``_w.tokenizer``), so text output is produced client-side here instead.
   This is safe for our use because the corrupt region is exactly the shared
   prefix, which every caller slices off (verified clean in the continuation
   region: 0 out-of-range ids over 554 entries).
2. The ``L+1``'th distribution must come from the *completion* output.
   ``ops.model_score_sequences_per_token`` reads
   ``top_logprobs[start : start+len(seq)+1]``; ``prompt_logprobs`` only covers
   prompt positions ``0..L-1``. That final entry -- the children-candidate
   distribution for the deepest node, consumed by
   ``TokenTrieCache._insert_parents`` -- exists only in
   ``outputs[0].logprobs``.

Note: ``vllm.LLM`` spawns an ``EngineCore`` subprocess that re-imports the
entrypoint module, so any script that builds an engine needs an
``if __name__ == "__main__":`` guard.
"""

import time
import types

import torch

from beaver.verifiers.worker_common import _w

# Lazy module-level singleton, keyed by model name. Deliberately module-level
# rather than on `_w`: init_worker_state() clears `_w.__dict__`, and reloading
# a multi-GB engine per run would dwarf everything this backend saves.
_ENGINE = None
_ENGINE_KEY = None


def _hidden_layers():
    """Layers whose hidden states this engine must capture (possibly empty).

    Reuses the same ``_w`` field the hf-local backend reads, which api.py
    already derives from ``direction_layers``/``probe_layer`` -- so pointing a
    representation run at vllm-local needs no new flag.
    """
    layers = getattr(_w, "vllm_hidden_layers", None) or getattr(
        _w, "hf_local_hidden_layers", None
    )
    return sorted({int(x) for x in layers}) if layers else []


def _vllm_capture_layers(layers):
    """Translate beaver layer indices into vLLM's aux-hidden-state indices.

    vLLM's ``eagle_aux_hidden_state_layer_ids`` counts the *input* to block N,
    whereas beaver (and hf_server, and TransformerLens ``blocks.N.hook_resid_
    post``) mean the *output* of block N. So vLLM index N == beaver layer N-1,
    and asking vLLM for beaver layer L means requesting L+1.

    Verified empirically in float32 on Qwen3-4B: TransformerLens
    ``blocks.10.hook_resid_post`` matches vLLM aux layer 11 at cosine
    1.000000, while TL layer 11 matches it at only 0.881. Without this shift a
    direction fitted at layer L is silently scored against layer L-1 -- no
    error, plausible-looking logprobs, wrong scores.
    """
    out = [int(l) + 1 for l in layers]
    if any(l < 1 for l in out):
        raise ValueError(f"hidden layer indices must be >= 0, got {layers}")
    return out


def _engine_key():
    return (
        _w.model_name,
        getattr(_w, "vllm_max_model_len", None),
        getattr(_w, "vllm_gpu_memory_utilization", None),
        getattr(_w, "vllm_dtype", None),
        # Capturing hidden states changes the computation graph (it adds a
        # CacheOnlyAttentionLayer), so an engine built without it cannot serve
        # a request that needs it.
        tuple(_vllm_capture_layers(_hidden_layers())) if _hidden_layers() else (),
    )


def get_engine():
    """Return the process-wide vLLM engine, constructing it on first use."""
    global _ENGINE, _ENGINE_KEY

    key = _engine_key()
    if _ENGINE is not None and _ENGINE_KEY == key:
        return _ENGINE
    if _ENGINE is not None:
        raise RuntimeError(
            "A local vLLM engine is already loaded for a different "
            f"configuration ({_ENGINE_KEY} != {key}). One engine per process; "
            "run the other configuration in a separate process."
        )

    from vllm import LLM

    kwargs = dict(
        model=_w.model_name,
        enable_prefix_caching=True,
        # -1 == "no cap", matching how the vllm serve baselines are launched.
        max_logprobs=-1,
        # See module docstring, requirement (1) -- not optional.
        skip_tokenizer_init=True,
        # Matches `vllm serve --logprobs-mode processed_logprobs`. Scoring
        # calls pass temperature=1/top_p=1/top_k=-1, so processed == raw there.
        logprobs_mode="processed_logprobs",
    )
    # Explicit dtype: needed to cross-validate this backend against hf-local /
    # hf_server, whose agreement can only be judged in float32 (bf16 noise is
    # ~0.3 logprob, which hides real disagreement).
    if getattr(_w, "vllm_dtype", None):
        kwargs["dtype"] = _w.vllm_dtype
    if getattr(_w, "vllm_max_model_len", None) is not None:
        kwargs["max_model_len"] = _w.vllm_max_model_len
    if getattr(_w, "vllm_gpu_memory_utilization", None) is not None:
        kwargs["gpu_memory_utilization"] = _w.vllm_gpu_memory_utilization

    from beaver.model import vllm_hidden_states as vhs

    # Always pin EngineCore into this process, capture or not. With the
    # default subprocess EngineCore the module-level engine is never shut
    # down at interpreter exit and the process hangs after the run finishes
    # (a 3.4-minute pilot sat for 4 h until SLURM killed it). In-process is
    # also ~20% faster per call here (no IPC on the request path).
    vhs.ensure_in_process()

    layers = _hidden_layers()
    if layers:
        capture = _vllm_capture_layers(layers)
        from vllm.config import KVTransferConfig

        # Must happen before LLM(...) so the connector name is resolvable;
        # in-process EngineCore (above) is also what makes its captures
        # readable from here rather than landing in a subprocess.
        vhs.register()
        kwargs["speculative_config"] = {
            "method": "extract_hidden_states",
            "num_speculative_tokens": 1,
            "draft_model_config": {
                "hf_config": {"eagle_aux_hidden_state_layer_ids": capture}
            },
        }
        kwargs["kv_transfer_config"] = KVTransferConfig(
            kv_connector=vhs.CONNECTOR_NAME,
            kv_role="kv_both",
        )

    print(f"[beaver.model.local_vllm] loading engine: {kwargs}")
    t0 = time.time()
    _ENGINE = LLM(**kwargs)
    _ENGINE_KEY = key
    print(f"[beaver.model.local_vllm] engine ready in {time.time() - t0:.1f}s")
    return _ENGINE


def _as_token_ids(prompt):
    """Normalize a prompt into a list of token-ID rows.

    Accepts the same shapes the HTTP path accepts: a text string, a
    pretokenized ``list[int]``, or a batch (``list[str]`` / ``list[list[int]]``).

    Text is tokenized with ``add_special_tokens=False`` because
    ``build_prompt()`` has already applied the chat template (which carries the
    model's own BOS). Adding another set would push BOS-using models off
    distribution -- the same reason the HTTP body sends
    ``add_special_tokens: False``.
    """
    if isinstance(prompt, str):
        return [_w.tokenizer.encode(prompt, add_special_tokens=False)]
    if isinstance(prompt, list) and prompt and isinstance(prompt[0], str):
        return [_w.tokenizer.encode(p, add_special_tokens=False) for p in prompt]
    if isinstance(prompt, list) and prompt and isinstance(prompt[0], list):
        return [[int(t) for t in row] for row in prompt]
    # A bare list[int]: one pretokenized prompt.
    return [[int(t) for t in prompt]]


def _logprob_dict(entry):
    """vLLM ``{token_id: Logprob}`` -> the wire-format ``{"token_id:N": float}``
    dict that parsing.py expects. ``None``/empty (e.g. position 0, or a
    prefix-cache-hit position) becomes ``{}``, matching how the HTTP path
    represents "no distribution here"."""
    if not entry:
        return {}
    return {f"token_id:{tid}": lp.logprob for tid, lp in entry.items()}


def _b64(t):
    """float32 little-endian base64, matching hf_server's wire encoding."""
    import base64

    return base64.b64encode(
        t.detach().to("cpu", torch.float32).contiguous().numpy().tobytes()
    ).decode()


def _collect_hidden(req_id, layers):
    """Captured last-token states for ``req_id`` as parse_hidden_states dicts."""
    from beaver.model import vllm_hidden_states as vhs

    last = vhs.pop_last(req_id)          # [n_layers, d_model] or None
    if last is None:
        raise RuntimeError(
            f"vllm-local: no hidden states captured for request {req_id!r}. "
            "The engine was built without extract_hidden_states, or the "
            "connector did not run."
        )
    last = last.reshape(len(layers), -1)
    return [
        {"layer": layer, "data": _b64(last[i]),
         "shape": [int(last.shape[-1])], "dtype": "float32"}
        for i, layer in enumerate(layers)
    ]


def _build_choice(index, out, echo, want_text, hs_out=None):
    """Assemble one ``RequestOutput`` into the shared choice shape.

    Mirrors ``http_backend._wrap_choice``'s output exactly -- see the module
    docstring in parsing.py for why every backend must agree on this shape.
    """
    comp = out.outputs[0]
    gen_ids = list(comp.token_ids)

    # Distribution over the token that follows the whole prompt. This is the
    # "+1" entry described in requirement (2) of the module docstring.
    gen_dicts = [_logprob_dict(d) for d in (comp.logprobs or [])]

    if echo:
        prompt_ids = list(out.prompt_token_ids)
        prompt_entries = list(out.prompt_logprobs or [])
        # Defensive: the length invariant held in every validation run, but a
        # silent mismatch here would corrupt every downstream slice.
        if len(prompt_entries) != len(prompt_ids):
            raise RuntimeError(
                "local vLLM returned "
                f"{len(prompt_entries)} prompt_logprobs for {len(prompt_ids)} "
                "prompt tokens; ops.py slicing assumes 1:1 correspondence."
            )

        # token_logprobs[j] = log p(prompt_ids[j] | prompt_ids[:j]), with None
        # at position 0 (no context) -- the same convention the OpenAI echo
        # response and hf_server both use.
        token_logprobs = []
        for pos, entry in enumerate(prompt_entries):
            lp = entry.get(prompt_ids[pos]) if entry else None
            token_logprobs.append(getattr(lp, "logprob", None) if lp else None)

        top_logprobs = [_logprob_dict(e) for e in prompt_entries]
        # Append the post-prompt distribution so the list has L+1 entries.
        top_logprobs.extend(gen_dicts[:1])

        tokens = [f"token_id:{t}" for t in prompt_ids] + [
            f"token_id:{t}" for t in gen_ids[:1]
        ]
        token_logprobs.extend([None] * (len(tokens) - len(token_logprobs)))
    else:
        tokens = [f"token_id:{t}" for t in gen_ids]
        token_logprobs = []
        for i, tid in enumerate(gen_ids):
            entry = comp.logprobs[i] if comp.logprobs and i < len(comp.logprobs) else None
            lp = entry.get(tid) if entry else None
            token_logprobs.append(getattr(lp, "logprob", None) if lp else None)
        top_logprobs = gen_dicts

    logprobs_obj = types.SimpleNamespace(
        tokens=tokens,
        token_logprobs=token_logprobs,
        top_logprobs=top_logprobs,
        text_offset=[],
    )
    # skip_tokenizer_init=True means the engine returns no text; decode with
    # beaver's own tokenizer only when a caller actually wants it.
    text = (
        _w.tokenizer.decode(gen_ids, skip_special_tokens=False)
        if (want_text and gen_ids)
        else ""
    )
    return types.SimpleNamespace(
        index=index,
        text=text,
        logprobs=logprobs_obj,
        hidden_states=hs_out,
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
    timeout=None,          # accepted for signature parity; no sockets here
    return_hidden_states=False,
    label="model_call",
    record_timing,
):
    """Run one batch of prompts through the in-process engine.

    Returns the same list-of-choices shape as ``http_backend.completion``.
    """
    layers = _hidden_layers()
    if return_hidden_states and not layers:
        raise ValueError(
            "backend='vllm-local' was asked for hidden states but no layers "
            "are configured. Set --direction_layers/--probe_layer (api.py "
            "derives the capture list from them)."
        )

    from vllm import SamplingParams

    engine = get_engine()
    rows = _as_token_ids(prompt)

    sp = SamplingParams(
        max_tokens=max_tokens,
        n=n,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        logprobs=logprobs,
        # echo=True over HTTP is exactly "give me prompt_logprobs too".
        prompt_logprobs=logprobs if echo else None,
        # The whole reason this backend exists -- unreachable over HTTP.
        skip_reading_prefix_cache=getattr(
            _w, "vllm_skip_prefix_cache_read", False
        ),
        detokenize=False,
    )

    _t = time.perf_counter()
    outs = engine.generate(rows, sp, use_tqdm=False)
    net_s = time.perf_counter() - _t

    if layers and not return_hidden_states:
        # The connector captured states for every request in this batch
        # regardless; only _collect_hidden pops them. Drop them now or they
        # accumulate on the GPU for the life of the process.
        from beaver.model import vllm_hidden_states as vhs

        vhs.clear()

    _t_parse = time.perf_counter()
    want_text = not echo  # scoring never reads .text; sampling does
    choices = [
        _build_choice(
            i, o, echo, want_text,
            hs_out=_collect_hidden(o.request_id, layers) if return_hidden_states else None,
        )
        for i, o in enumerate(outs)
    ]
    parse_s = time.perf_counter() - _t_parse

    record_timing(
        label,
        prefix="local",
        net_s=net_s,
        parse_s=parse_s,
        n_rows=len(rows),
        resp_bytes=0,  # no serialization on this path -- that is the point
        # Direct prefix-cache-hit measurement. Neither HTTP backend can report
        # this; it is how we verify the cache is actually being read.
        num_cached_tokens=sum(getattr(o, "num_cached_tokens", 0) or 0 for o in outs),
    )
    return choices
