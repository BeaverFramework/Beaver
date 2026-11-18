"""Decoded next-token distribution from a raw top-N logprob slice.

Two ways to read a run's ``temperature`` / ``top_p`` / ``top_k``:

* **cutoff** (the historical behaviour, ``apply_top_p_top_k``): top-p / top-k
  decide which children the verifier *explores*; the mass they drop is
  carried as unresolved, so the bounds remain bounds on the violation
  probability under the *raw* distribution. Temperature is not applied.
* **decoded** (this module): the bounds are on the violation probability
  under the distribution a sampler configured with (T, p, k) actually draws
  from, ``renorm(top_p(top_k(softmax(logits / T))))``. Tokens that the
  sampler can never emit have probability zero, not "unknown".

The backend only ever hands us the ``N`` highest raw log-probs at a position
(``num_logprobs``), so the full-vocabulary tempered normaliser is not known
exactly for ``T != 1``. For ``T <= 1`` it is bracketed: every tail token has
raw probability at most ``p_min`` (the smallest in the slice), hence tempered
mass at most ``p_j * p_min ** (1/T - 1)``, so the tail's tempered mass lies in
``[0, m * p_min ** (1/T - 1)]`` with ``m`` the raw tail mass. We always take
the conservative end (larger normaliser, smaller probabilities), so any
slack lands in the unresolved mass and the bounds stay sound. At ``T == 1``
the bracket collapses and everything is exact. ``T > 1`` is refused: the
tail can then dominate and no useful bracket exists from a top-N slice.

Once the kept set is fixed the renormalisation ``q_i / sum_kept q`` cancels
the normaliser, so the only place the bracket matters is the nucleus
boundary test, and only when the boundary falls inside the slack.
"""

from __future__ import annotations

import numpy as np


def _logsumexp(x: np.ndarray) -> float:
    m = float(np.max(x))
    return m + float(np.log(np.sum(np.exp(x - m))))


def decode_distribution(
    log_probs: np.ndarray,
    temperature: float,
    top_p: float,
    top_k: int,
):
    """Turn a raw ``[N, 2]`` ``[token_id, logprob]`` slice into the decoded one.

    Returns ``(kept, unresolved)``: ``kept`` is ``[M, 2]`` with the *decoded*
    log-probabilities of the tokens the sampler can emit, sorted descending;
    ``unresolved`` is the probability mass (under the decoded distribution)
    that this slice cannot account for -- zero whenever the sampler's
    support fits inside the slice, which is the normal case.

    ``temperature == 0`` is greedy decoding (identical to ``top_k == 1``).
    """
    log_probs = np.asarray(log_probs, dtype=np.float64)
    if log_probs.size == 0:
        return log_probs.reshape(0, 2), 1.0
    if temperature < 0 or temperature > 1.0:
        raise ValueError(
            f"decoded_dist supports 0 <= temperature <= 1, got {temperature}"
        )
    # Descending log-prob, ties broken by token id, so the kept set is
    # deterministic across backends and runs.
    order = np.lexsort((log_probs[:, 0], -log_probs[:, 1]))
    lp = log_probs[order]
    if temperature == 0.0:
        top_k = 1
        temperature = 1.0
    raw_p = np.exp(lp[:, 1])
    raw_tail = max(1.0 - float(raw_p.sum()), 0.0)

    # Tempered, unnormalised weights, scaled so the largest is 1 for stability.
    scaled = (lp[:, 1] - lp[0, 1]) / temperature
    w = np.exp(scaled)
    z_slice = float(w.sum())
    # Tail bracket (see module docstring); zero at T == 1 when the slice is
    # complete, and exactly the raw tail mass at T == 1 otherwise.
    p_min = float(raw_p[-1])
    tail_hi = raw_tail * (p_min ** (1.0 / temperature - 1.0)) if raw_tail > 0 else 0.0
    # tail_hi is in raw-probability units at T == 1; convert to the same scale
    # as ``w`` (which is exp(lp/T) / exp(lp_0/T)).
    tail_hi_scaled = tail_hi / np.exp(lp[0, 1] / temperature) if tail_hi > 0 else 0.0
    z_hi = z_slice + tail_hi_scaled
    q_lo = w / z_hi  # conservative (lower) tempered probabilities

    keep = len(lp)
    if top_k is not None and top_k > 0:
        keep = min(keep, int(top_k))
    fits = keep < len(lp)  # a top-k cut inside the slice: support is known
    if top_p is not None and top_p < 1.0:
        cum = np.cumsum(q_lo[:keep])
        # HF / vLLM convention: drop token i iff the cumulative mass *before*
        # it already exceeds p, i.e. keep the first token that crosses p.
        before = cum - q_lo[:keep]
        n_keep = int(np.sum(before <= top_p))
        if n_keep < keep:
            keep = n_keep
            fits = True
        elif top_k is not None and 0 < top_k < len(lp):
            fits = True  # top-k already bounded the support
        else:
            # Nucleus runs past the slice (or exactly to its end): support
            # not fully known. Only "fits" if the slice is the whole vocab.
            fits = raw_tail == 0.0
    elif not fits:
        fits = raw_tail == 0.0  # no truncation at all: slice must be complete

    kept_s = scaled[:keep]
    if fits:
        # Support known: renormalise within it. The normaliser cancels, so
        # this is exact given the membership decision above. Log-space so a
        # far-below-the-top token underflows to a finite, not -inf, value.
        dec_lp = kept_s - _logsumexp(kept_s)
        unresolved = 0.0
    else:
        # Support extends past the slice. True decoded prob of a kept token is
        # q_i / M with M the support's tempered mass; M <= p_eff + q_boundary
        # where the boundary token has q <= q_min(slice). Lower-estimate.
        p_eff = top_p if (top_p is not None and top_p < 1.0) else 1.0
        m_hi = min(1.0, p_eff + float(q_lo[keep - 1]))
        dec_lp = kept_s - np.log(z_hi) - np.log(m_hi)
        unresolved = max(1.0 - float(np.exp(dec_lp).sum()), 0.0)
    kept = np.column_stack([lp[:keep, 0], dec_lp])
    return kept, unresolved
