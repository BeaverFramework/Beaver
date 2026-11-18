"""
Base semantic constraint interface.

Each dataset registers into _REGISTRY via register_constraint():
    check_call_fn(instance, decoded_sequences, token_lists) -> np.ndarray[bool]
        Pre-filter: True means this sequence needs the full check.
    instance_context_fn(instance) -> str
        Returns instance-specific fields that affect the check result
        (e.g. expected answer, language). Empty string if result depends
        only on the sequence text.
        Only used when use_cache=True.
    check_fn(instance, sequences: list[str]) -> list[bool]
        The actual check, batched: one bool per sequence. May be slow.
    complete_fn(instance, decoded_sequences, token_lists) -> np.ndarray[bool]
        Optional. True marks a sequence as complete before EOS / gen_length.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Callable

import sqlite3

import numpy as np

from .semantic_constraint_cache import _get_cache, SemanticConstraintCache

DEFAULT_CONSTRAINT_TIMEOUT: float = 300.0

_REGISTRY: dict[str, tuple] = {}


def register_constraint(
    dataset_name: str,
    check_call_fn: Callable,
    instance_context_fn: Callable,
    check_fn: Callable,
    complete_fn: Callable | None = None,
):
    _REGISTRY[dataset_name] = (check_call_fn, instance_context_fn, check_fn, complete_fn)


def check_semantic_call(dataset_name, instance, decoded_sequences, token_lists):
    check_call_fn, _, _, _ = _REGISTRY[dataset_name]
    return check_call_fn(instance, decoded_sequences, token_lists)


def check_complete_call(dataset_name, instance, decoded_sequences, token_lists):
    _, _, _, complete_fn = _REGISTRY[dataset_name]
    if complete_fn is None:
        return np.zeros(len(decoded_sequences), dtype=bool)
    return complete_fn(instance, decoded_sequences, token_lists)


def enforce_semantic_constraint(
    dataset_name: str,
    instance: dict,
    decoded_sequences,
    timeout: float = DEFAULT_CONSTRAINT_TIMEOUT,
    use_cache: bool = True,
) -> np.ndarray:
    _, instance_context_fn, check_fn, _ = _REGISTRY[dataset_name]
    if not use_cache:
        return _run_checks(
            instance, list(np.asarray(decoded_sequences)), check_fn, timeout
        )
    try:
        cache = _get_cache(dataset_name)
    except sqlite3.OperationalError as e:
        # The cache is a sqlite file on a network filesystem; 16 workers
        # opening it at once can trip "database is locked" despite the busy
        # timeout. A lost cache is a slowdown, a lost instance is a hole in
        # the results, so run the checks uncached.
        _warn_cache_once(f"open failed ({e}); running uncached")
        return _run_checks(
            instance, list(np.asarray(decoded_sequences)), check_fn, timeout
        )
    instance_context = instance_context_fn(instance)
    return _enforce(
        instance, decoded_sequences, cache, instance_context, check_fn, timeout
    )


_CACHE_WARNED = False


def _warn_cache_once(msg: str):
    global _CACHE_WARNED
    if not _CACHE_WARNED:
        print(f"[semantic_constraint_cache] {msg}")
        _CACHE_WARNED = True


def _enforce(
    instance,
    decoded_sequences,
    cache: SemanticConstraintCache,
    instance_context,
    check_fn,
    timeout,
):
    decoded_sequences = np.asarray(decoded_sequences)
    if len(decoded_sequences) == 0:
        return np.array([], dtype=bool)

    # Dedup and key on the RAW text. Stripping here (as before) handed
    # check_fn whitespace-stripped strings, which silently defeated any Phi
    # that looks at trailing whitespace -- secure_code's line-complete gate
    # never saw a trailing newline, so semgrep never ran under the cache and
    # every semgrep-only violation was cached as "satisfied"
    # (dist300_*_seccode_pc_eps1e-3_d2500 runs, 2026-09-16).
    unique_seqs, inverse = np.unique(decoded_sequences, return_inverse=True)

    keys = [cache.make_key(s, instance_context) for s in unique_seqs.tolist()]
    try:
        cached = cache.get_batch(keys)
    except sqlite3.OperationalError as e:
        _warn_cache_once(f"read failed ({e}); treating as uncached")
        cached = [None] * len(keys)
    cached_arr = np.array(cached, dtype=object)
    uncached_mask = cached_arr == None  # noqa: E711

    if uncached_mask.any():
        uncached_seqs = unique_seqs[uncached_mask].tolist()
        uncached_keys = [k for k, m in zip(keys, uncached_mask) if m]
        new_results = _run_checks(instance, uncached_seqs, check_fn, timeout)
        try:
            cache.set_batch(uncached_keys, new_results.tolist())
        except sqlite3.OperationalError as e:
            _warn_cache_once(f"write failed ({e}); result not cached")
        cached_arr[uncached_mask] = new_results

    return cached_arr.astype(bool)[inverse]


def _run_checks(instance, sequences, check_fn, timeout):
    results = np.asarray(check_fn(instance, sequences), dtype=bool)
    if results.shape != (len(sequences),):
        raise TypeError(
            "constraint_fn must be batched: constraint_fn(instance, "
            "sequences: list[str]) -> list[bool], one bool per sequence. "
            f"Got a result of shape {results.shape} for {len(sequences)} "
            "sequence(s). (The per-sequence signature "
            "constraint_fn(instance, seq) -> bool is no longer supported.)"
        )
    return results
