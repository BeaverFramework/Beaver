"""Shared utilities for the activation-direction / probe-training scripts.

Both `violation_direction_detection.py` (TransformerLens, multi-position) and
`violation_direction_heretic.py` (Heretic-native, get_residuals_batched) import
these. The functions live here so the two scripts reuse one copy instead of
each carrying their own.
"""

import numpy as np
from scipy.stats import rankdata


def _auc(scores, y):
    """Mann-Whitney AUC with correct tie handling.

    Average ranks (scipy.rankdata) so tied scores contribute 0.5, as the
    statistic requires. This matters a lot for 'last_prompt'/'risk_prior', where
    every sample within a prompt has an identical score (identical features) --
    argsort ranks would break those ties by array position and bias the AUC.
    scores [n], y [n] in {0,1} -> scalar AUC (np.nan if one class only).
    """
    y = np.asarray(y)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return np.nan
    ranks = rankdata(scores)  # average ranks; ties -> mean rank
    rank_pos = ranks[y == 1].sum()
    return (rank_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def _proj_auc(proj, ins):
    """Mann-Whitney AUC of projection scores vs labels, vectorized over layers.
    proj [N, L], ins [N] bool -> [L]."""
    N = proj.shape[0]
    ranks = proj.argsort(0).argsort(0) + 1.0
    n_pos = ins.sum()
    n_neg = N - n_pos
    rank_pos = ranks[ins].sum(0)
    return (rank_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
