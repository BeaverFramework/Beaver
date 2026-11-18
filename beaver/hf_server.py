"""Minimal HuggingFace model server with an OpenAI-compatible /v1/completions
endpoint, plus an optional hook for returning hidden states from a specific
layer.

Usage::

    python -m beaver.hf_server --model openai/gpt-oss-20b --port 8082 \
        --hidden-state-layers 18

The server is intentionally tiny: one ``model.generate()`` call per request,
plus one forward hook per requested layer to capture the hidden state at the
last input position (used to predict the first new token).

Optimizations over baseline:
  - Async request batching: parallel requests are accumulated over a short
    window and executed as a single padded batch forward pass.
  - For max_tokens=1, uses a direct model() call instead of generate(),
    avoiding KV-cache + sampling loop overhead.
  - logits_to_keep=1 on the forward pass: the LM head projects only the last
    position, saving O(B * seq_len * vocab) of activation memory per batch.
    echo=True requests (used by worker_common.py's model_score_sequences_per_
    token to warm-start the cached-frontier verifier) widen this to the last
    _ECHO_TAIL_K positions instead of every prompt position, since every row
    is left-padded and one scalar keep-count applies to the whole batch.
  - Vectorized sampling + logprob extraction across the batch, with a single
    GPU->CPU sync at the end (was O(B) syncs from per-row .item() calls).
  - Hook captures stay on GPU in native dtype; moved to CPU *after* the
    forward pass completes, then cast to fp32 on CPU to avoid briefly
    doubling the tensor's GPU footprint.
  - Attention impl configurable via --attn-impl. Default is "sdpa", which is
    typically faster than flash_attention_2 for this workload (short-ish
    sequences, small batches, single decode step → FA2 unpad/repad overhead
    is not amortized).
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import logging
import math
import os
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any
import inspect
import numpy as np
import torch
import uvicorn
import orjson
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel

from concurrent.futures import ThreadPoolExecutor

_INFER_EXECUTOR = ThreadPoolExecutor(max_workers=1)
# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class CompletionRequest(BaseModel):
    model: str
    # Text (the normal generate path, via build_prompt()), a pretokenized
    # token-ID list (the scoring path, which must send exact token IDs rather
    # than text — see model_score_sequences_per_token's docstring on why
    # decode-then-retokenize is not safe), or a batch of either — list[str]
    # (N separate text prompts) or list[list[int]] (N separate pretokenized
    # prompts, e.g. tokentrie_cache.py's warm-start cache-rebuild, which
    # scores many cached trie nodes in one call).
    prompt: str | list[str] | list[int] | list[list[int]]
    max_tokens: int = 1
    temperature: float = 1.0
    top_p: float = 1.0
    # Sampling-path extras (max_tokens > 1, _generate): top_k <= 0 disables
    # top-k (HF generate would otherwise apply the checkpoint's
    # generation_config top_k, e.g. 20 for Qwen3-4B-Instruct-2507, so a
    # "raw" temperature-1 draw was not raw); n = sequences per prompt; seed
    # seeds torch's RNG before generate() for reproducible draws.
    top_k: int = -1
    n: int = 1
    seed: int | None = None
    logprobs: int | None = None
    echo: bool = False
    stop: list[str] | str | None = None
    return_hidden_states: bool = False
    extra_body: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Server state
# ---------------------------------------------------------------------------

_state: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# FastAPI app with lifespan (replaces deprecated on_event("startup"))
# ---------------------------------------------------------------------------

from contextlib import asynccontextmanager


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    # Kick off the batch worker on startup. Kept as a task so it's cancelled
    # automatically when the app shuts down.
    worker = asyncio.create_task(_batch_worker())
    try:
        yield
    finally:
        worker.cancel()
        try:
            await worker
        except (asyncio.CancelledError, Exception):
            pass


app = FastAPI(title="beaver-hf-server", lifespan=_lifespan)

# ---------------------------------------------------------------------------
# Batching queue
# ---------------------------------------------------------------------------

# Each queued item is (prompt_str, request, asyncio.Future)
_batch_queue: asyncio.Queue[tuple[str, CompletionRequest, asyncio.Future]] = (
    asyncio.Queue()
)
_BATCH_MAX_SIZE = 16
_BATCH_WAIT_MS = 20  # ms to wait for more requests before dispatching

# Trailing positions to keep real logits for on echo=True requests. The server
# never sees the prefix/continuation boundary (only the client knows that), so
# this must just comfortably exceed the longest continuation any run will ever
# score. Bump this if a run uses a much larger gen_length than ~64 tokens.
_ECHO_TAIL_K = 64

# ---------------------------------------------------------------- prefix KV
# Shared-prompt KV cache.
#
# Every scoring batch is prompt+path rows that share a long identical prompt.
# Without this the server re-prefills that prompt on every request -- the one
# thing hf-local had over hf_server (see beaver/model/hf_local.py's header).
#
# Keyed on the exact prefix token tuple, which is what makes reuse safe: a
# different prompt is a different key, so a stale entry can never be served to
# the wrong sequence. Same content-addressing property vLLM gets from
# chain-hashed blocks, just at whole-prefix granularity. The server therefore
# never needs to be told where the prompt ends -- it discovers the boundary as
# the longest common prefix of the rows in the batch.
#
# Stored as per-layer batch-1 (key, value) tensors rather than a live
# DynamicCache: a DynamicCache is mutated in place by the forward pass that
# uses it, so a shared one would be corrupted after first use. Rebuilding a
# throwaway DynamicCache per batch from expanded views costs nothing (torch.cat
# in Cache.update allocates a new tensor and leaves these untouched).
_PREFIX_KV: dict[tuple, list] = {}
_PREFIX_ORDER: list[tuple] = []
_PREFIX_ENABLED = False
# Below this a prefill is cheaper than the cache bookkeeping.
_MIN_PREFIX_TOKENS = 64
_MAX_PREFIX_ENTRIES = 8
# Cache effectiveness counters, exposed at /v1/prefix_cache_stats.
#   hits/misses      -- prefix KV reused vs rebuilt
#   evictions        -- entries dropped by the LRU (the "breakage" signal:
#                       steady eviction means the working set exceeds
#                       --prefix-cache-entries and prefixes are rebuilt)
#   batches_cached   -- forwards that served >0 tokens from cache
#   batches_uncached -- forwards where the batch shared no usable prefix
#                       (the other breakage signal: batching mixed prompts
#                       from different instances, so the LCP collapsed)
_PREFIX_STATS = {
    "hits": 0, "misses": 0, "evictions": 0,
    "cached_tokens": 0, "fed_tokens": 0,
    "batches_cached": 0, "batches_uncached": 0,
    "prefix_build_s": 0.0,
}


def _prefix_cache_bytes() -> int:
    """Resident GPU bytes held by the prefix KV cache."""
    total = 0
    for kv in _PREFIX_KV.values():
        for k, v in kv:
            total += k.numel() * k.element_size() + v.numel() * v.element_size()
    return total


def prefix_cache_stats() -> dict:
    hits, misses = _PREFIX_STATS["hits"], _PREFIX_STATS["misses"]
    bc, bu = _PREFIX_STATS["batches_cached"], _PREFIX_STATS["batches_uncached"]
    fed, cached = _PREFIX_STATS["fed_tokens"], _PREFIX_STATS["cached_tokens"]
    return {
        **_PREFIX_STATS,
        "enabled": _PREFIX_ENABLED,
        "entries": len(_PREFIX_KV),
        "max_entries": _MAX_PREFIX_ENTRIES,
        "min_prefix_tokens": _MIN_PREFIX_TOKENS,
        "bytes": _prefix_cache_bytes(),
        "mib": round(_prefix_cache_bytes() / 2**20, 1),
        "entry_tokens": [len(k) for k in _PREFIX_ORDER],
        # Of the prefix lookups, how many avoided a rebuild.
        "hit_rate": round(hits / (hits + misses), 4) if (hits + misses) else None,
        # Of the forwards, how many got any cache benefit at all. This is the
        # number that exposes the batching-vs-caching conflict.
        "batch_cached_rate": round(bc / (bc + bu), 4) if (bc + bu) else None,
        # Fraction of prompt tokens that never had to be recomputed.
        "token_reuse_rate": round(cached / (cached + fed), 4) if (cached + fed) else None,
    }


def _common_prefix_len(rows: list[list[int]]) -> int:
    """Longest token prefix shared by every row."""
    if not rows:
        return 0
    shortest = min(len(r) for r in rows)
    first = rows[0]
    n = 0
    while n < shortest and all(r[n] == first[n] for r in rows):
        n += 1
    return n


def _prefix_cache_len(rows: list[list[int]], keep_k: int) -> int:
    """How many leading tokens this batch may serve from the prefix cache.

    Bounded by ``keep_k`` so every row still has at least ``logits_to_keep``
    real positions left in its tail. That preserves both invariants the rest
    of the pipeline relies on: ``logits[:, -1]`` is the last *real* token, and
    the echo window's absolute-position mapping (``p_abs = L - K + m`` in
    _finalize_batch) still lands inside the recomputed region.
    """
    if not _PREFIX_ENABLED or len(rows) < 1:
        return 0
    shortest = min(len(r) for r in rows)
    n = min(_common_prefix_len(rows), shortest - keep_k)
    return n if n >= _MIN_PREFIX_TOKENS else 0


def _prefix_kv_tensors(model, prefix_ids: list[int]):
    """Per-layer (key, value) batch-1 tensors for ``prefix_ids``, cached LRU."""
    from transformers import DynamicCache

    key = tuple(prefix_ids)
    hit = _PREFIX_KV.get(key)
    if hit is not None:
        _PREFIX_ORDER.remove(key)
        _PREFIX_ORDER.append(key)
        _PREFIX_STATS["hits"] += 1
        return hit

    _PREFIX_STATS["misses"] += 1
    ids = torch.tensor([prefix_ids], device=model.device, dtype=torch.long)
    cache = DynamicCache()
    # The hidden-state hook must not fire here: these positions are prompt
    # prefix, never the last real token any request reads, and letting it
    # write would clobber the capture the real forward is about to make.
    prev_capture = _state.get("capture_hidden", False)
    _state["capture_hidden"] = False
    try:
        with torch.inference_mode():
            model(input_ids=ids, use_cache=True, past_key_values=cache,
                  logits_to_keep=1)
    finally:
        _state["capture_hidden"] = prev_capture

    kv = [(layer.keys.detach(), layer.values.detach()) for layer in cache.layers]
    _PREFIX_KV[key] = kv
    _PREFIX_ORDER.append(key)
    while len(_PREFIX_ORDER) > _MAX_PREFIX_ENTRIES:
        _PREFIX_KV.pop(_PREFIX_ORDER.pop(0), None)
        _PREFIX_STATS["evictions"] += 1
    return kv


def _expand_prefix_cache(kv, batch_size: int):
    """A throwaway DynamicCache viewing the batch-1 prefix KV as batch B."""
    from transformers import DynamicCache

    return DynamicCache([
        (k.expand(batch_size, -1, -1, -1), v.expand(batch_size, -1, -1, -1))
        for k, v in kv
    ])


async def _batch_worker():
    """Background task: drain the queue, pad prompts, run one forward pass."""
    loop = asyncio.get_running_loop()
    while True:
        # Wait for at least one item
        first = await _batch_queue.get()
        batch: list[tuple[str, CompletionRequest, asyncio.Future]] = [first]

        # Collect more items that arrived in the wait window
        deadline = loop.time() + _BATCH_WAIT_MS / 1000
        while len(batch) < _BATCH_MAX_SIZE:
            remaining = deadline - loop.time()
            if remaining <= 0:
                break
            try:
                item = await asyncio.wait_for(_batch_queue.get(), timeout=remaining)
                batch.append(item)
            except asyncio.TimeoutError:
                break

        # Run batched inference in a thread to avoid blocking the event loop
        await loop.run_in_executor(_INFER_EXECUTOR, _run_batch, batch)


def _encode_prompt(prompt: str | list[int], tokenizer) -> list[int]:
    """Return a request's prompt as a list of token IDs.

    ``prompt`` is either raw text (the normal generate path, via
    ``build_prompt()``) or an already-tokenized ID list (the scoring path,
    which must preserve exact token boundaries — see
    ``model_score_sequences_per_token``'s docstring). ``add_special_tokens``
    is always False: text prompts already carry the model's chat-template
    special tokens baked in, so re-adding them here would double them up
    (harmless for Qwen, which has no BOS, but wrong for BOS-using models).
    """
    if isinstance(prompt, list):
        return [int(t) for t in prompt]
    return tokenizer.encode(prompt, add_special_tokens=False)


def _pad_rows(
    ids_per_row: list[list[int]], pad_id: int, device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Left-pad variable-length ID rows into a single [B, max_len] batch.

    Left-padding keeps every row's real last token aligned in the final
    column — required for causal LMs and for the hidden-state hook's
    ``h[:, -1, :]`` extraction, and also what lets a single scalar
    ``logits_to_keep`` value apply correctly to every row at once (see
    ``_run_batch``).
    """
    max_len = max(len(ids) for ids in ids_per_row)
    B = len(ids_per_row)
    input_ids = torch.full((B, max_len), pad_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((B, max_len), dtype=torch.long, device=device)
    for i, ids in enumerate(ids_per_row):
        if ids:
            input_ids[i, max_len - len(ids) :] = torch.tensor(
                ids, dtype=torch.long, device=device
            )
            attention_mask[i, max_len - len(ids) :] = 1
    return input_ids, attention_mask


def _run_fast_subbatch(
    items: list[tuple[str, CompletionRequest, asyncio.Future]],
    model,
    tokenizer,
    hook_outputs: dict[int, torch.Tensor],
) -> None:
    """Tokenize + forward-pass + finalize one *homogeneous-echo* sub-batch.

    Split out of ``_run_batch`` so echo=True rows (which need
    ``logits_to_keep=_ECHO_TAIL_K``) never share a forward pass with echo=False
    rows (which only need ``logits_to_keep=1``) — see the profiling note below
    for why that mattered.
    """
    prompts = [p for p, _r, _f in items]
    requests = [r for _p, r, _f in items]
    futures = [f for _p, _r, f in items]

    any_echo = any(r.echo for r in requests)

    # --- Tokenize & left-pad into a batch ---
    _t_tok0 = time.perf_counter()
    ids_per_row = [_encode_prompt(p, tokenizer) for p in prompts]
    pad_id = tokenizer.pad_token_id
    # K is needed before padding now: it bounds how much of the shared prompt
    # the prefix cache may swallow (see _prefix_cache_len).
    K = _ECHO_TAIL_K if any_echo else 1
    cache_len = _prefix_cache_len(ids_per_row, K)
    # Only the uncached tail is fed to the forward. Left-padding the tails
    # keeps every row's last real token at index -1 exactly as before; the pad
    # run simply sits *between* the cached prefix and the tail, which the
    # attention mask handles like any other masked span.
    tails = [ids[cache_len:] for ids in ids_per_row] if cache_len else ids_per_row
    input_ids, tail_mask = _pad_rows(tails, pad_id, model.device)
    _t_tok1 = time.perf_counter()

    _t_pfx0 = time.perf_counter()
    past_key_values = None
    if cache_len:
        kv = _prefix_kv_tensors(model, ids_per_row[0][:cache_len])
        past_key_values = _expand_prefix_cache(kv, len(tails))
        prefix_mask = torch.ones(
            (len(tails), cache_len), dtype=tail_mask.dtype, device=tail_mask.device
        )
        attention_mask = torch.cat([prefix_mask, tail_mask], dim=1)
        _PREFIX_STATS["cached_tokens"] += cache_len * len(tails)
        _PREFIX_STATS["batches_cached"] += 1
    else:
        attention_mask = tail_mask
        _PREFIX_STATS["batches_uncached"] += 1
    _t_pfx1 = time.perf_counter()
    _PREFIX_STATS["prefix_build_s"] += _t_pfx1 - _t_pfx0
    _PREFIX_STATS["fed_tokens"] += int(input_ids.numel())

    # ---- FAST PATH: single forward pass, no generate() ----
    # Always clear before the forward pass. The hook's `idx not in`
    # guard then only matters for the slow path's multi-step generate;
    # here the hook always writes exactly once per batch. This avoids
    # stale CPU tensors lingering from prior batches.
    hook_outputs.clear()
    # Compute position_ids from attention_mask so left-padded rows get
    # the correct RoPE positions for their real tokens (otherwise the
    # last real token of a shorter prompt gets RoPE position seq_len-1
    # instead of its actual prompt length - 1, which shifts its hidden
    # state away from the single-prompt / right-padded reference).
    want_hidden = any(r.return_hidden_states for r in requests)
    _state["capture_hidden"] = want_hidden
    # Computed over the FULL mask (cached prefix included) so the tail's RoPE
    # positions continue from the prefix rather than restarting at 0, then
    # sliced to the positions actually being fed to the forward.
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 1)
    if cache_len:
        position_ids = position_ids[:, cache_len:]

    # With logits_to_keep=1 (the common case) the LM head only projects
    # the last position, so shape is [B, 1, vocab] (not [B, seq_len,
    # vocab]) — for a 20B model with 200K vocab at B=16 seq=2048 this
    # saves ~13 GB. echo=True requests need more than the last position
    # (to recover per-position distributions along the scored
    # continuation), so they widen the kept window to _ECHO_TAIL_K. This
    # sub-batch is always echo-homogeneous (see _run_batch), so K applies
    # correctly to every row here -- no cross-contamination of unrelated
    # echo=False rows paying for a window they never asked for (profiled:
    # 212/221 mixed batches before this split, at production scale, each
    # forcing K=64 onto ordinary single-token live-search rows).
    _t_fwd0 = time.perf_counter()
    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=bool(cache_len),
            logits_to_keep=K,
        )
    if torch.cuda.is_available():
        # model(...) only queues GPU kernels -- force completion here so
        # _t_fwd1 - _t_fwd0 is real forward-pass wall time, not just
        # Python dispatch time. (Profiling-only cost: a few ms per batch.)
        torch.cuda.synchronize()
    _t_fwd1 = time.perf_counter()
    _state["capture_hidden"] = False
    logits = outputs.logits  # [B, K, vocab]
    print(
        f"[hf-server-prof] t={time.time():.3f} batch B={len(items)} any_echo={any_echo} K={K} "
        f"tokenize_s={_t_tok1 - _t_tok0:.4f} forward_s={_t_fwd1 - _t_fwd0:.4f} "
        f"prefix_s={_t_pfx1 - _t_pfx0:.4f} cached_tokens={cache_len * len(items)} "
        f"fed_tokens={int(input_ids.numel())}",
        flush=True,
    )

    # Move hook captures off GPU. Do .cpu() BEFORE .float() so we don't
    # briefly allocate a 2x-sized fp32 copy on-device.
    hook_outputs_cpu: dict[int, torch.Tensor] | None = None
    if any(r.return_hidden_states for r in requests) and hook_outputs:
        hook_outputs_cpu = {k: v.cpu().float() for k, v in hook_outputs.items()}

    try:
        results = _finalize_batch(
            logits,
            requests,
            ids_per_row,
            attention_mask,
            hook_outputs_cpu,
        )
        for i, res in enumerate(results):
            futures[i].get_loop().call_soon_threadsafe(futures[i].set_result, res)
    except Exception as e:
        for fut in futures:
            if not fut.done():
                fut.get_loop().call_soon_threadsafe(fut.set_exception, e)


def _prefix_groups(items, tokenizer):
    """Split a batch into prefix-homogeneous sub-batches.

    Why this is needed
    ------------------
    The batch worker coalesces whatever arrives inside its window, and with N
    concurrent workers those requests come from N *different* instances with
    different prompts. The longest common prefix across such a batch is
    ~nothing, so the prefix cache never fires: measured on a 16-worker
    secure_code run, 101 of 103 batches had cache_len == 0 purely because 90 of
    them were the full 64-row mixed batch.

    Grouping on the leading _MIN_PREFIX_TOKENS tokens puts rows from the same
    instance back together, which is the only arrangement where a shared prefix
    exists to cache. Rows that share those leading tokens are guaranteed an LCP
    of at least _MIN_PREFIX_TOKENS, i.e. exactly the threshold below which
    _prefix_cache_len would decline to cache anyway.

    The trade-off is real and is why this is tied to --prefix-cache rather than
    always on: it converts one wide forward into several narrow ones, paying
    more fixed per-forward overhead to avoid recomputing the shared prompt.
    Which side wins depends on prompt length vs batch width, so measure with
    /v1/prefix_cache_stats rather than assuming.
    """
    if not _PREFIX_ENABLED or len(items) <= 1:
        return [items] if items else []
    groups: dict[tuple, list] = {}
    for item in items:
        # Free for the scoring path: _encode_prompt returns pretokenized
        # ID lists unchanged, so this re-derives nothing.
        ids = _encode_prompt(item[0], tokenizer)
        groups.setdefault(tuple(ids[:_MIN_PREFIX_TOKENS]), []).append(item)
    return list(groups.values())


def _run_batch(batch: list[tuple[str, CompletionRequest, asyncio.Future]]):
    """Execute a batch of prompts, one forward pass per echo-homogeneous group."""
    print(f"[hf-server] Running batch of {len(batch)} request(s)")
    try:
        model = _state["model"]
        tokenizer = _state["tokenizer"]
        hook_outputs: dict[int, torch.Tensor] = _state["hook_outputs"]

        # Multi-token (generate()) requests can't share the fast path below —
        # split them out so an echo=True scoring request never gets silently
        # routed through _generate(), which has no echo handling at all.
        # Scoring calls always send max_tokens=1, so this only ever splits
        # off unrelated sampling-style requests that happened to land in the
        # same batching window.
        fast_items = [item for item in batch if item[1].max_tokens <= 1]
        slow_items = [item for item in batch if item[1].max_tokens > 1]

        for prompt, request, future in slow_items:
            try:
                result = _generate(prompt, request)
                future.get_loop().call_soon_threadsafe(future.set_result, result)
            except Exception as e:
                future.get_loop().call_soon_threadsafe(future.set_exception, e)

        if not fast_items:
            return

        # Split by echo so a single logits_to_keep value never has to cover
        # both an echo=True row (needs _ECHO_TAIL_K) and an echo=False row
        # (needs only 1) in the same forward pass.
        echo_items = [item for item in fast_items if item[1].echo]
        noecho_items = [item for item in fast_items if not item[1].echo]

        for group in _prefix_groups(echo_items, tokenizer):
            _run_fast_subbatch(group, model, tokenizer, hook_outputs)
        for group in _prefix_groups(noecho_items, tokenizer):
            _run_fast_subbatch(group, model, tokenizer, hook_outputs)
    except Exception as e:
        # If the whole batch fails, propagate to all futures
        for _p, _r, fut in batch:
            if not fut.done():
                fut.get_loop().call_soon_threadsafe(fut.set_exception, e)


def _finalize_batch(
    logits: torch.Tensor,  # [B, K, V] on GPU (K=1 unless the batch has echo rows)
    requests: list[CompletionRequest],
    ids_per_row: list[list[int]],
    attention_mask: torch.Tensor,  # [B, seq_len] on GPU
    hook_outputs_cpu: dict[int, torch.Tensor] | None,
) -> list[dict]:
    """Vectorized sampling + logprob extraction for a whole batch.

    Does all sampling and logprob math on GPU in batched form, then issues
    exactly one round of GPU->CPU syncs at the end to build per-row payloads
    in pure Python. Replaces the old per-row _process_single_token loop that
    issued ~4 `.item()` syncs per request.

    Non-echo rows only ever read the last (K-1) position, identical to the
    pre-echo behavior. echo=True rows additionally get, for every position in
    the K-window, a full top-`logprobs` distribution — mirroring what
    vLLM's ``echo=True`` completions return — so callers like
    ``model_score_sequences_per_token`` (worker_common.py) work unchanged
    regardless of which server they're pointed at.
    """
    _t0 = time.perf_counter()
    tokenizer = _state["tokenizer"]
    vocab_size = _state["vocab_size"]
    B, K, _V = logits.shape
    device = logits.device

    # Clean non-finite logits to prevent multinomial CUDA asserts.
    logits_f = logits.float()
    if not torch.isfinite(logits_f).all():
        logits_f = torch.nan_to_num(
            logits_f,
            nan=0.0,
            posinf=torch.finfo(logits_f.dtype).max,
            neginf=torch.finfo(logits_f.dtype).min,
        )
    last_logits = logits_f[:, -1, :]  # [B, V] — same "next token" position as before

    temps = [r.temperature for r in requests]
    top_ps = [r.top_p for r in requests]
    logprobs_ns = [min(r.logprobs or 0, vocab_size) for r in requests]
    max_num_top = max(logprobs_ns) if logprobs_ns else 0
    uniform_temp = all(t == temps[0] for t in temps)
    uniform_top_p = all(p == top_ps[0] for p in top_ps)

    # --- Sampling (vectorized if all requests share params) ---------------
    if uniform_temp and uniform_top_p:
        T = temps[0]
        P = top_ps[0]
        if T > 0:
            probs = torch.softmax(last_logits / T, dim=-1)  # [B, V]
            if P < 1.0:
                sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
                cumsum = sorted_probs.cumsum(dim=-1)
                mask = cumsum - sorted_probs > P  # keep first token that crosses P
                sorted_probs = sorted_probs.masked_fill(mask, 0.0)
                sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
                probs = torch.zeros_like(probs).scatter_(1, sorted_idx, sorted_probs)
            sampled_ids = torch.multinomial(probs, 1).squeeze(-1)  # [B]
        else:
            sampled_ids = last_logits.argmax(dim=-1)  # [B]
    else:
        # Rare: mixed per-row sampling params. Still avoid syncs inside loop.
        sampled_ids = torch.empty(B, dtype=torch.long, device=device)
        for i in range(B):
            row = last_logits[i]
            T = temps[i]
            P = top_ps[i]
            if T > 0:
                p = torch.softmax(row / T, dim=-1)
                if P < 1.0:
                    sp, si = torch.sort(p, descending=True)
                    cs = sp.cumsum(dim=0)
                    m = cs - sp > P
                    sp = sp.masked_fill(m, 0.0)
                    sp = sp / sp.sum()
                    p = torch.zeros_like(p).scatter_(0, si, sp)
                sampled_ids[i] = torch.multinomial(p, 1).squeeze()
            else:
                sampled_ids[i] = row.argmax()

    # --- Logprobs (fully batched) -----------------------------------------
    log_norm = last_logits.logsumexp(dim=-1)  # [B]
    token_lps = (
        last_logits.gather(1, sampled_ids.unsqueeze(1)).squeeze(1) - log_norm
    )  # [B]

    # --- Per-position distributions across the whole K-window -------------
    # For pure-generate batches K==1 so this is identical (and identically
    # cheap) to the old last-position-only computation. Echo rows widen K
    # (see _run_batch), so this also covers every position needed to
    # reconstruct per-token distributions along the scored continuation.
    log_norm_K = logits_f.logsumexp(dim=-1)  # [B, K]
    is_echo = [bool(r.echo) for r in requests]
    L_list = [len(ids) for ids in ids_per_row]

    # Note: L (row length) includes the shared prompt prefix, which is always
    # much longer than --echo-tail-k by design -- the server never sees the
    # prompt/continuation boundary, so an "L > K" check can't distinguish
    # "truncating the (unused) prompt prefix, fine" from "truncating real
    # continuation data, wrong" and would false-positive on every request.
    # Correctness here is the caller's responsibility: --echo-tail-k must be
    # >= the longest continuation (parent-path length + 1) any run scores,
    # i.e. >= gen_length for tokentrie_cache.py's warm-start callers.

    # On-path target token id at each kept position, for echo rows only (0
    # elsewhere — unused there). Position m keeps absolute logits position
    # p_abs = L-K+m, which predicts token ids[p_abs+1]; j=p_abs+1 is only a
    # valid echoed on-path token when 1 <= j <= L-1 (j==0 has no preceding
    # context — echoed as None/{} same as vLLM; j==L is the "generated"
    # slot beyond the given sequence, no on-path target exists there).
    target_ids_list = [[0] * K for _ in range(B)]
    for i in range(B):
        if not is_echo[i]:
            continue
        L = L_list[i]
        for m in range(K):
            j = L - K + m + 1
            if 1 <= j <= L - 1:
                target_ids_list[i][m] = ids_per_row[i][j]
    target_ids = torch.tensor(target_ids_list, dtype=torch.long, device=device)
    target_lps_K = (
        logits_f.gather(2, target_ids.unsqueeze(-1)).squeeze(-1) - log_norm_K
    )  # [B, K]

    if max_num_top > 0:
        topk_vals_K, topk_ids_K = torch.topk(logits_f, max_num_top, dim=-1)  # [B,K,N]
        topk_lps_K = topk_vals_K - log_norm_K.unsqueeze(-1)
    else:
        topk_ids_K = None
        topk_lps_K = None

    prompt_lengths_gpu = attention_mask.sum(dim=1)  # [B]

    # --- Single batched GPU->CPU sync -------------------------------------
    sampled_ids_cpu = sampled_ids.tolist()
    token_lps_cpu = token_lps.tolist()
    target_lps_K_cpu = target_lps_K.tolist()  # [B][K]
    prompt_lengths_cpu = prompt_lengths_gpu.tolist()
    if topk_ids_K is not None:
        topk_ids_K_cpu = topk_ids_K.tolist()  # [B][K][N]
        topk_lps_K_cpu = topk_lps_K.tolist()
    else:
        topk_ids_K_cpu = None
        topk_lps_K_cpu = None
    _t_gpu = time.perf_counter()

    # --- Build per-row response dicts (pure Python, no GPU) ---------------
    results: list[dict] = []
    for i in range(B):
        tid = int(sampled_ids_cpu[i])
        token_lp = float(token_lps_cpu[i])
        # orjson (unlike stdlib json) refuses to serialize inf/-inf/nan at
        # all, so a non-finite value must become None (JSON null) here
        # rather than -inf, matching how the echo branch already represents
        # "no valid value" for a position (see token_logprobs[j] above).
        token_lp_out = token_lp if math.isfinite(token_lp) else None
        generated_text = tokenizer.decode([tid], skip_special_tokens=False)
        n_top = logprobs_ns[i]

        if is_echo[i]:
            L = L_list[i]
            ids = ids_per_row[i]
            token_logprobs: list[float | None] = [None] * L
            top_logprobs_list: list[dict] = [{} for _ in range(L + 1)]
            for m in range(K):
                p_abs = L - K + m
                if p_abs < 0:
                    continue  # row shorter than the kept window
                j = p_abs + 1
                if n_top > 0 and topk_ids_K_cpu is not None:
                    step_dict = {
                        f"token_id:{int(idx)}": float(lp)
                        for idx, lp in zip(
                            topk_ids_K_cpu[i][m][:n_top], topk_lps_K_cpu[i][m][:n_top]
                        )
                        if math.isfinite(lp)
                    }
                else:
                    step_dict = {}
                if 1 <= j <= L - 1:
                    tlp = float(target_lps_K_cpu[i][m])
                    if math.isfinite(tlp):
                        step_dict.setdefault(f"token_id:{ids[j]}", tlp)
                        token_logprobs[j] = tlp
                top_logprobs_list[j] = step_dict
            tokens_list = [f"token_id:{t}" for t in ids] + [f"token_id:{tid}"]
            logprobs_payload = {
                "tokens": tokens_list,
                "token_logprobs": token_logprobs,
                "top_logprobs": top_logprobs_list,
                "text_offset": [],
            }
        else:
            if n_top > 0 and topk_ids_K_cpu is not None:
                step_dict = {
                    f"token_id:{int(idx)}": float(lp)
                    for idx, lp in zip(
                        topk_ids_K_cpu[i][-1][:n_top], topk_lps_K_cpu[i][-1][:n_top]
                    )
                    if math.isfinite(lp)
                }
                if math.isfinite(token_lp):
                    step_dict.setdefault(f"token_id:{tid}", token_lp)
                top_logprobs_list = [step_dict]
            else:
                top_logprobs_list = [{}]

            logprobs_payload = {
                "tokens": [f"token_id:{tid}"],
                "token_logprobs": [token_lp_out],
                "top_logprobs": top_logprobs_list,
                "text_offset": [],
            }

        hidden_states_out = None
        if requests[i].return_hidden_states and hook_outputs_cpu:
            hidden_states_out = [
                {
                    "layer": layer_idx,
                    "data": _b64(hook_outputs_cpu[layer_idx][i]),
                    "shape": list(hook_outputs_cpu[layer_idx][i].shape),
                    "dtype": "float32",
                }
                for layer_idx in sorted(hook_outputs_cpu.keys())
            ]

        results.append(
            {
                "index": i,
                "text": generated_text,
                "logprobs": logprobs_payload,
                "finish_reason": "length",
                "hidden_states": hidden_states_out,
                "prompt_tokens": int(prompt_lengths_cpu[i]),
            }
        )

    _t_python = time.perf_counter()
    n_echo = sum(is_echo)
    print(
        f"[hf-server-prof] t={time.time():.3f} finalize B={B} K={K} n_echo={n_echo} "
        f"gpu_math_s={_t_gpu - _t0:.4f} python_build_s={_t_python - _t_gpu:.4f}"
    )
    return results


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_model(
    model_name: str,
    hidden_state_layers: list[int] | None = None,
    dtype: str = "auto",
    attn_impl: str = "sdpa",
):
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    if dtype == "auto":
        # Use the dtype the model was trained with (from its HF config)
        model_config = AutoConfig.from_pretrained(model_name)
        config_dtype = getattr(model_config, "torch_dtype", None)
        if config_dtype is not None and isinstance(config_dtype, torch.dtype):
            torch_dtype = config_dtype
        elif config_dtype is not None and isinstance(config_dtype, str):
            torch_dtype = getattr(torch, config_dtype, torch.bfloat16)
        else:
            torch_dtype = torch.float16
        dtype = str(torch_dtype).replace("torch.", "")
        print(f"[hf-server] Auto-detected dtype={dtype} from model config")
    else:
        torch_dtype = getattr(torch, dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Set once here rather than per-batch (safer across threads, cheaper).
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    print(f"[hf-server] Loading {model_name} (dtype={dtype}, attn_impl={attn_impl})")
    # transformers >=4.50 renamed torch_dtype -> dtype. Pick whichever the
    # installed version supports so we don't emit a deprecation warning and
    # remain compatible with older installs.

    _sig = inspect.signature(AutoModelForCausalLM.from_pretrained)
    _dtype_kwarg = "dtype" if "dtype" in _sig.parameters else "torch_dtype"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        attn_implementation=attn_impl,
        **{_dtype_kwarg: torch_dtype},
    )
    model.eval()

    # Sanity-check: if accelerate silently offloaded any layers to CPU/disk,
    # inference will be *extremely* slow (10–100x). Surface this loudly.
    try:
        hf_device_map = getattr(model, "hf_device_map", None)
        if hf_device_map:
            devices = set(str(v) for v in hf_device_map.values())
            print(f"[hf-server] Device map uses: {devices}")
            offloaded = {
                k: v
                for k, v in hf_device_map.items()
                if str(v) in ("cpu", "disk") or "cpu" in str(v).lower()
            }
            if offloaded:
                print(
                    f"[hf-server] WARNING: {len(offloaded)} module(s) offloaded off-GPU. "
                    f"Model likely doesn't fit in VRAM — inference will be slow. "
                    f"Offloaded (sample): {list(offloaded.items())[:3]}"
                )
    except Exception as e:
        print(f"[hf-server] Could not inspect device map: {e}")
    # Locate transformer layers for the optional hidden-state hook
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        layers = getattr(getattr(model, "transformer", None), "h", None)

    hook_outputs: dict[int, torch.Tensor] = {}
    if hidden_state_layers and layers is not None:
        for layer_idx in hidden_state_layers:
            if layer_idx >= len(layers):
                print(
                    f"[hf-server] WARNING: layer {layer_idx} >= {len(layers)}, skipping"
                )
                continue

            def _make_hook(idx):
                def _hook(_module, _inp, output):
                    if not _state.get("capture_hidden", False):
                        return  # no-op unless explicitly capturing
                    h = output[0] if isinstance(output, tuple) else output
                    if idx not in hook_outputs:
                        hook_outputs[idx] = h[:, -1, :].detach().clone()

                return _hook

            layers[layer_idx].register_forward_hook(_make_hook(layer_idx))
            print(f"[hf-server] Hook registered on layer {layer_idx}")

    _state.update(
        model=model,
        tokenizer=tokenizer,
        model_name=model_name,
        hidden_state_layers=hidden_state_layers or [],
        hook_outputs=hook_outputs,
        vocab_size=len(tokenizer),
    )
    print(f"[hf-server] Ready. Vocab size: {_state['vocab_size']}")


# ---------------------------------------------------------------------------
# Inference (sequential fallback for multi-token generation)
# ---------------------------------------------------------------------------


def _b64(tensor: torch.Tensor) -> str:
    return base64.b64encode(tensor.numpy().astype(np.float32).tobytes()).decode("ascii")


def _multi_sequence_result(out, prompt_len: int, request: CompletionRequest, tokenizer) -> dict:
    """n sampled sequences for one prompt (request.n > 1): the first as the
    result dict, the rest under ``extra_choices`` (create_completion flattens
    them into consecutive choices, as the OpenAI API does for ``n``).

    generate() pads finished rows with pad_token_id after their EOS, so each
    row is cut after its first EOS (kept, finish_reason "stop"); rows that
    ran to max_new_tokens get "length". Per-step log-probs come from the
    processed scores, which equal the raw log-softmax when no warper is
    active (temperature 1, top_p 1, top_k 0).
    """
    eos_ids = set()
    for v in (tokenizer.eos_token_id, getattr(getattr(_state.get("model"), "generation_config", None), "eos_token_id", None)):
        for t in (v if isinstance(v, (list, tuple)) else [v]):
            if t is not None:
                eos_ids.add(int(t))
    num_top = min(request.logprobs or 0, _state["vocab_size"])
    scores = out.scores if hasattr(out, "scores") and out.scores else []
    choices = []
    for row in range(out.sequences.shape[0]):
        ids = out.sequences[row, prompt_len:].tolist()
        finish = "length"
        for j, t in enumerate(ids):
            if t in eos_ids:
                ids = ids[: j + 1]
                finish = "stop"
                break
        token_lps, toks, tops = [], [], []
        for step in range(len(ids)):
            tid = ids[step]
            toks.append(f"token_id:{tid}")
            if step < len(scores):
                logits = scores[step][row].float()
                log_norm = logits.logsumexp(dim=-1)
                lp = float((logits[tid] - log_norm).item())
                token_lps.append(lp if math.isfinite(lp) else None)
                if num_top > 0:
                    vals, idx = torch.topk(logits, num_top)
                    d = {f"token_id:{int(i)}": float(v) for i, v in zip(idx.tolist(), (vals - log_norm).tolist()) if math.isfinite(v)}
                    if math.isfinite(lp):
                        d.setdefault(f"token_id:{tid}", lp)
                    tops.append(d)
                else:
                    tops.append({})
            else:
                token_lps.append(None)
                tops.append({})
        choices.append({
            "index": row,
            "text": tokenizer.decode(ids, skip_special_tokens=False),
            "logprobs": {"tokens": toks, "token_logprobs": token_lps, "top_logprobs": tops, "text_offset": []},
            "finish_reason": finish,
            "hidden_states": None,
            "prompt_tokens": prompt_len,
        })
    first = choices[0]
    first["extra_choices"] = choices[1:]
    return first


def _generate(prompt: str, request: CompletionRequest) -> dict:
    """Sequential single-prompt generate() — used as fallback when max_tokens > 1."""
    model = _state["model"]
    tokenizer = _state["tokenizer"]
    hook_outputs: dict[int, torch.Tensor] = _state["hook_outputs"]
    hook_outputs.clear()

    encoded = tokenizer(prompt, return_tensors="pt", return_attention_mask=True)
    input_ids = encoded["input_ids"].to(model.device)
    attention_mask = encoded["attention_mask"].to(model.device)
    prompt_len = input_ids.shape[1]

    do_sample = request.temperature > 0
    n_seq = max(1, int(request.n or 1))
    gen_kwargs = dict(
        max_new_tokens=max(1, request.max_tokens),
        do_sample=do_sample,
        temperature=request.temperature if do_sample else 1.0,
        top_p=request.top_p if do_sample else 1.0,
        # 0 disables HF's top-k warper (the checkpoint's generation_config
        # may set one; vLLM's -1 convention maps to 0 here)
        top_k=(request.top_k if request.top_k > 0 else 0) if do_sample else 0,
        num_return_sequences=n_seq if do_sample else 1,
        return_dict_in_generate=True,
        output_scores=bool(request.logprobs),
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    if request.seed is not None:
        torch.manual_seed(int(request.seed))

    with torch.inference_mode():
        out = model.generate(input_ids, attention_mask=attention_mask, **gen_kwargs)

    # Move hook captures off GPU. .cpu() first, then .float() to avoid
    # briefly doubling the tensor's GPU footprint.
    for k in hook_outputs:
        hook_outputs[k] = hook_outputs[k].cpu().float()

    if n_seq > 1 and do_sample:
        return _multi_sequence_result(out, prompt_len, request, tokenizer)

    new_token_ids = out.sequences[0, prompt_len:].tolist()
    generated_text = tokenizer.decode(new_token_ids, skip_special_tokens=False)

    stops = request.stop
    if isinstance(stops, str):
        stops = [stops]
    if stops:
        cut = min(
            (generated_text.find(s) for s in stops if s and s in generated_text),
            default=-1,
        )
        if cut != -1:
            generated_text = generated_text[:cut]
            kept = len(tokenizer(generated_text, add_special_tokens=False)["input_ids"])
            new_token_ids = new_token_ids[:kept]
            finish_reason = "stop"

    # --- Per-step logprobs --------------------------------------------------
    num_top = min(request.logprobs or 0, _state["vocab_size"])
    top_logprobs_per_step: list[dict[str, float]] = []
    token_logprobs_per_step: list[float] = []
    tokens_per_step: list[str] = []

    scores = out.scores if hasattr(out, "scores") and out.scores else []
    for step, score in enumerate(scores):
        logits = score[0].float()
        log_norm = logits.logsumexp(dim=-1)
        tid = new_token_ids[step]
        token_lp = float((logits[tid] - log_norm).item())
        # orjson refuses to serialize inf/-inf/nan -- store None (JSON null)
        # instead, same convention as the fast path in _finalize_batch.
        token_logprobs_per_step.append(token_lp if math.isfinite(token_lp) else None)
        tokens_per_step.append(f"token_id:{tid}")

        if num_top > 0:
            topk_vals, topk_ids = torch.topk(logits, num_top)
            topk_lps = topk_vals - log_norm
            step_dict = {
                f"token_id:{int(i)}": float(v)
                for i, v in zip(topk_ids.tolist(), topk_lps.tolist())
                if math.isfinite(v)
            }
            if math.isfinite(token_lp):
                step_dict.setdefault(f"token_id:{tid}", token_lp)
            top_logprobs_per_step.append(step_dict)
        else:
            top_logprobs_per_step.append({})

    logprobs_payload = {
        "tokens": tokens_per_step,
        "token_logprobs": token_logprobs_per_step,
        "top_logprobs": top_logprobs_per_step,
        "text_offset": [],
    }

    # --- Hidden states ------------------------------------------------------
    hidden_states_out = None
    if request.return_hidden_states and hook_outputs:
        hidden_states_out = [
            {
                "layer": layer_idx,
                "data": _b64(hook_outputs[layer_idx].squeeze(0)),
                "shape": list(hook_outputs[layer_idx].squeeze(0).shape),
                "dtype": "float32",
            }
            for layer_idx in sorted(hook_outputs.keys())
        ]

    return {
        "index": 0,
        "text": generated_text,
        "logprobs": logprobs_payload,
        "finish_reason": "length",
        "hidden_states": hidden_states_out,
        "prompt_tokens": prompt_len,
    }


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
async def health():
    if "model" not in _state:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ok"}


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [{"id": _state.get("model_name", "unknown"), "object": "model"}],
    }


@app.get("/v1/prefix_cache_stats")
def prefix_cache_stats_endpoint():
    """Prefix-cache effectiveness counters.

    The three rates are the ones worth watching:
      hit_rate           -- prefix lookups that avoided a rebuild
      batch_cached_rate  -- forwards that got ANY cache benefit. A low value
                            here with a high hit_rate means batching is mixing
                            prompts from different instances, so the shared
                            prefix collapses before the cache is consulted.
      token_reuse_rate   -- prompt tokens never recomputed. This is the one
                            that should track the wall-clock saving.
    ``evictions`` rising steadily means the working set exceeds
    --prefix-cache-entries and prefixes are being rebuilt after eviction.
    """
    return prefix_cache_stats()


@app.post("/v1/completions")
async def create_completion(request: CompletionRequest):
    if "model" not in _state:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # OpenAI convention: prompt is a string (one prompt), a list of strings (N
    # separate prompts), a list of ints (one pretokenized prompt — the
    # scoring path, which must be kept as a single row, not split per-token),
    # or a list of int-lists (N separate pretokenized prompts — the batched
    # scoring path, e.g. tokentrie_cache.py's cache-rebuild). Distinguish
    # "N separate prompts" from "one pretokenized prompt" by the element
    # type: str or list means N prompts; int means this list IS the one
    # pretokenized prompt.
    if (
        isinstance(request.prompt, list)
        and request.prompt
        and isinstance(request.prompt[0], (str, list))
    ):
        prompts = request.prompt
    else:
        prompts = [request.prompt]

    # Submit all prompts to the batching queue and await results
    _t_req0 = time.perf_counter()
    futures = []
    loop = asyncio.get_running_loop()
    for prompt in prompts:
        fut = loop.create_future()
        await _batch_queue.put((prompt, request, fut))
        futures.append(fut)

    results = await asyncio.gather(*futures)
    _t_gathered = time.perf_counter()

    choices = []
    total_prompt_tokens = 0
    total_completion_tokens = 0
    for result in results:
        # request.n > 1 (sampling path): the extra sequences ride along on
        # the first result and become consecutive choices here.
        for choice in [result] + (result.pop("extra_choices", None) or []):
            choice["index"] = len(choices)
            total_prompt_tokens += choice.pop("prompt_tokens", 0)
            total_completion_tokens += len(choice["logprobs"]["tokens"])
            choices.append(choice)

    payload = {
        "id": f"cmpl-{uuid.uuid4().hex[:12]}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": _state.get("model_name", "unknown"),
        "choices": choices,
        "usage": {
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "total_tokens": total_prompt_tokens + total_completion_tokens,
        },
    }

    # Encode with orjson directly (skips FastAPI's default jsonable_encoder +
    # stdlib json.dumps path, which profiling showed costs ~0.25s/request on
    # large echo/warm-start responses -- ~17% of total request time at
    # production scale). `payload` is plain dict/list/str/float/int/None
    # throughout, so orjson can serialize it with no encoder step needed.
    _t_encode0 = time.perf_counter()
    body = orjson.dumps(payload)
    _t_encode1 = time.perf_counter()
    print(
        f"[hf-server-prof] t={time.time():.3f} request rows={len(choices)} "
        f"queue_wait_and_compute_s={_t_gathered - _t_req0:.4f} "
        f"encode_probe_s={_t_encode1 - _t_encode0:.4f} resp_bytes={len(body)}"
    )

    return Response(content=body, media_type="application/json")


# ---------------------------------------------------------------------------
# Programmatic start/stop
# ---------------------------------------------------------------------------


def start_server(
    model_id: str,
    port: int = 8082,
    gpu_visible_devices: str | None = None,
    hidden_state_layers: list[int] | None = None,
    startup_timeout: int = 600,
    health_check_interval: int = 5,
    dtype: str = "bfloat16",
    attn_impl: str = "sdpa",
    log_dir: Path | str | None = None,
    quiet: bool = False,
    extra_args: list[str] | None = None,
) -> subprocess.Popen | None:
    """Spawn a ``beaver.hf_server`` subprocess and wait until it's healthy.

    ``extra_args`` are appended to the command line (e.g.
    ``["--echo-tail-k", "32"]``)."""
    import requests as req

    cmd = [
        sys.executable,
        "-m",
        "beaver.hf_server",
        "--model",
        model_id,
        "--port",
        str(port),
        "--dtype",
        dtype,
        "--attn-impl",
        attn_impl,
    ]
    if gpu_visible_devices is not None:
        cmd.extend(["--gpu-visible-devices", str(gpu_visible_devices)])
    if hidden_state_layers:
        for layer in hidden_state_layers:
            cmd.extend(["--hidden-state-layers", str(layer)])
    if quiet:
        cmd.append("--quiet")
    cmd.extend(extra_args or [])

    env = os.environ.copy()
    if gpu_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_visible_devices)
        print(f"CUDA_VISIBLE_DEVICES={gpu_visible_devices}")

    print(f"[beaver] Starting HF server for {model_id} on port {port}")
    print(f"[beaver]   Command: {' '.join(cmd)}")

    server_log: Path | None = None
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        with open(log_dir / "server_config.json", "w") as f:
            json.dump(
                {
                    "backend": "hf",
                    "model": model_id,
                    "port": port,
                    "dtype": dtype,
                    "attn_impl": attn_impl,
                    "hidden_state_layers": hidden_state_layers or [],
                    "gpu_visible_devices": gpu_visible_devices,
                    "quiet": quiet,
                },
                f,
                indent=2,
            )
        with open(log_dir / "server_cmd.txt", "w") as f:
            f.write(" ".join(cmd) + "\n")
        server_log = log_dir / "server.log"
        with open(server_log, "w") as fh:
            proc = subprocess.Popen(
                cmd,
                env=env,
                stdout=fh,
                stderr=subprocess.STDOUT,
                preexec_fn=os.setsid,
            )
    else:
        proc = subprocess.Popen(
            cmd, env=env, stderr=subprocess.STDOUT, preexec_fn=os.setsid
        )

    health_url = f"http://localhost:{port}/health"
    print(f"[beaver] Waiting for server (timeout: {startup_timeout}s) ...")
    t0 = time.time()
    try:
        while time.time() - t0 < startup_timeout:
            if proc.poll() is not None:
                print(f"[beaver] Server process exited with code {proc.returncode}")
                if server_log:
                    print(f"[beaver] Check log: {server_log}")
                return None
            try:
                if req.get(health_url, timeout=5).status_code == 200:
                    print(f"[beaver] Server healthy ({time.time() - t0:.1f}s)")
                    return proc
            except req.exceptions.RequestException:
                pass
            time.sleep(health_check_interval)
    except KeyboardInterrupt:
        print("\n[beaver] Interrupted — stopping server ...")
        stop_server(proc)
        raise

    print(f"[beaver] Server failed to start within {startup_timeout}s")
    if server_log:
        print(f"[beaver] Check log: {server_log}")
    stop_server(proc)
    return None


def stop_server(proc: subprocess.Popen | None) -> None:
    """Gracefully stop a server subprocess."""
    if proc is None:
        return
    print("[beaver] Stopping server ...")
    try:
        pgid = os.getpgid(proc.pid)
        os.killpg(pgid, signal.SIGTERM)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(pgid, signal.SIGKILL)
            proc.wait(timeout=5)
    except ProcessLookupError:
        pass
    print("[beaver] Server stopped")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Beaver HF model server")
    parser.add_argument("--model", required=True)
    parser.add_argument("--port", type=int, default=8082)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--hidden-state-layers", type=int, nargs="+", default=None)
    parser.add_argument("--gpu-visible-devices", default=None)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument(
        "--tf32",
        action="store_true",
        help="With --dtype float32, run matmuls/convolutions in TF32 (tensor "
        "cores, 10-bit mantissa) instead of full fp32. Faster, less exact.",
    )
    parser.add_argument(
        "--attn-impl",
        default="sdpa",
        choices=["sdpa", "flash_attention_2", "eager"],
        help="Attention implementation. sdpa is typically faster for "
        "short-sequence / single-token batched workloads than FA2, which "
        "pays unpad/repad overhead per call.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Silence uvicorn access logs and transformers info warnings.",
    )
    parser.add_argument(
        "--batch-wait-ms",
        type=int,
        default=10,
        help="Max ms to wait for more requests before dispatching a batch.",
    )
    parser.add_argument(
        "--batch-max-size",
        type=int,
        default=64,
        help="Max requests per batch.",
    )
    parser.add_argument(
        "--echo-tail-k",
        type=int,
        default=64,
        help=(
            "Trailing positions to keep real logits for on echo=True requests "
            "(the shared prompt prefix is always discarded by the caller, "
            "regardless of this value). Must be >= the longest continuation "
            "any run scores -- i.e. >= gen_length. Set it to match gen_length "
            "(rather than leaving the default 64) to cut the widened "
            "logits-to-keep compute/serialization cost on echo calls."
        ),
    )
    parser.add_argument(
        "--prefix-cache",
        action="store_true",
        help=(
            "Reuse a KV cache for the shared prompt prefix across requests. "
            "The boundary is discovered as the longest common prefix of the "
            "rows in a batch, so the client never has to declare it. Biggest "
            "win on scoring batches, where every row is the same long prompt "
            "plus a short differing path."
        ),
    )
    parser.add_argument(
        "--prefix-cache-min",
        type=int,
        default=64,
        help="Shortest shared prefix worth caching (tokens).",
    )
    parser.add_argument(
        "--prefix-cache-entries",
        type=int,
        default=8,
        help="Max distinct prompt prefixes held in the LRU KV cache.",
    )
    args = parser.parse_args()

    global _PREFIX_ENABLED, _MIN_PREFIX_TOKENS, _MAX_PREFIX_ENTRIES
    _PREFIX_ENABLED = bool(args.prefix_cache)
    _MIN_PREFIX_TOKENS = int(args.prefix_cache_min)
    _MAX_PREFIX_ENTRIES = int(args.prefix_cache_entries)

    if args.gpu_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_visible_devices

    if args.quiet:
        os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
        try:
            from transformers import logging as hf_logging

            hf_logging.set_verbosity_error()
        except Exception:
            pass
        logging.getLogger("uvicorn").setLevel(logging.WARNING)
        logging.getLogger("uvicorn.error").setLevel(logging.WARNING)
        logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

    # Update batch config from CLI args
    global _BATCH_WAIT_MS, _BATCH_MAX_SIZE, _ECHO_TAIL_K
    _BATCH_WAIT_MS = args.batch_wait_ms
    _BATCH_MAX_SIZE = args.batch_max_size
    _ECHO_TAIL_K = args.echo_tail_k

    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    print(f"[hf-server] matmul tf32={torch.backends.cuda.matmul.allow_tf32}")

    load_model(
        args.model,
        hidden_state_layers=args.hidden_state_layers,
        dtype=args.dtype,
        attn_impl=args.attn_impl,
    )
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="warning" if args.quiet else "info",
        access_log=not args.quiet,
    )


if __name__ == "__main__":
    main()
