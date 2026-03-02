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
    check_fn(instance, sequence: str) -> bool
        The actual check. May be slow.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Callable

import numpy as np

from .semantic_constraint_cache import _get_cache, SemanticConstraintCache

DEFAULT_CONSTRAINT_TIMEOUT: float = 300.0

_REGISTRY: dict[str, tuple] = {}


def register_constraint(
    dataset_name: str,
    check_call_fn: Callable,
    instance_context_fn: Callable,
    check_fn: Callable,
):
    _REGISTRY[dataset_name] = (check_call_fn, instance_context_fn, check_fn)


def check_semantic_call(dataset_name, instance, decoded_sequences, token_lists):
    check_call_fn, _, _ = _REGISTRY[dataset_name]
    return check_call_fn(instance, decoded_sequences, token_lists)


def enforce_semantic_constraint(
    dataset_name: str,
    instance: dict,
    decoded_sequences,
    timeout: float = DEFAULT_CONSTRAINT_TIMEOUT,
    use_cache: bool = True,
) -> np.ndarray:
    _, instance_context_fn, check_fn = _REGISTRY[dataset_name]
    if not use_cache:
        return _run_checks(instance, list(np.asarray(decoded_sequences)), check_fn, timeout)
    cache = _get_cache(dataset_name)
    instance_context = instance_context_fn(instance)
    return _enforce(instance, decoded_sequences, cache, instance_context, check_fn, timeout)


def _enforce(instance, decoded_sequences, cache: SemanticConstraintCache, instance_context, check_fn, timeout):
    decoded_sequences = np.asarray(decoded_sequences)
    if len(decoded_sequences) == 0:
        return np.array([], dtype=bool)

    stripped = np.char.strip(decoded_sequences)
    unique_seqs, inverse = np.unique(stripped, return_inverse=True)

    keys = [cache.make_key(s, instance_context) for s in unique_seqs.tolist()]
    cached = cache.get_batch(keys)
    cached_arr = np.array(cached, dtype=object)
    uncached_mask = cached_arr == None  # noqa: E711

    if uncached_mask.any():
        uncached_seqs = unique_seqs[uncached_mask].tolist()
        uncached_keys = [k for k, m in zip(keys, uncached_mask) if m]
        new_results = _run_checks(instance, uncached_seqs, check_fn, timeout)
        cache.set_batch(uncached_keys, new_results.tolist())
        cached_arr[uncached_mask] = new_results

    return cached_arr.astype(bool)[inverse]


def _run_checks(instance, sequences, check_fn, timeout):
    results = []
    with ThreadPoolExecutor(max_workers=1) as executor:
        for seq in sequences:
            future = executor.submit(check_fn, instance, seq)
            try:
                results.append(bool(future.result(timeout=timeout)))
            except FuturesTimeoutError:
                print(f"[Constraint] check_fn timed out for: {seq[:80]!r} — defaulting True")
                results.append(True)
            except Exception as e:
                print(f"[Constraint] check_fn raised {type(e).__name__}: {e} — defaulting True")
                results.append(True)
    return np.array(results, dtype=bool)
