"""Per-layer direction computation + in-sample separability metrics.

Backend-agnostic: operates on a numpy residual tensor [N, L, d] regardless of
whether it came from TransformerLens or Heretic. Shared by both scripts.
"""

import numpy as np

from beaver.activation_detection import _auc


def analyze_layers(acts, y, orthogonalize: bool = False):
    """Per layer: diff-of-means direction + cheap (in-sample) metrics.

    acts [N, L, d], y [N] (1=violating, 0=non). Returns (dirs [L, d] unit, rows).
      r_norm    |mean_viol - mean_non|  (grows with depth; NOT a quality metric)
      fisher    gap^2 / pooled within-class var along r  (1-D separability)
      proj_auc  AUC of the diff-of-means projection (IN-SAMPLE -> optimistic),
                computed with correct tie handling (matters for risk_prior, where
                every sample within a prompt shares an identical score).

    Direction = normalize(bad_means - good_means)  (Heretic main.py:464). With
    orthogonalize=True, the component parallel to good_means is removed first
    (projected abliteration, main.py:466-476).
    """
    y = np.asarray(y)
    ins = y == 1
    good_means = acts[~ins].mean(0)  # [L, d] non-violating
    bad_means = acts[ins].mean(0)    # [L, d] violating
    r = bad_means - good_means       # [L, d]
    rnorm = np.linalg.norm(r, axis=1)  # [L]
    dirs = r / (rnorm[:, None] + 1e-8)  # [L, d] unit directions

    if orthogonalize:
        # https://huggingface.co/blog/grimjim/projected-abliteration
        good_dirs = good_means / (np.linalg.norm(good_means, axis=1, keepdims=True) + 1e-8)
        proj = np.sum(dirs * good_dirs, axis=1, keepdims=True)
        dirs = dirs - proj * good_dirs
        dirs = dirs / (np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-8)

    proj = np.einsum("nld,ld->nl", acts, dirs)  # [N, L]
    pi, ps = proj[ins], proj[~ins]
    gap = pi.mean(0) - ps.mean(0)
    fisher = gap**2 / (pi.var(0) + ps.var(0) + 1e-8)
    auc = np.array([_auc(proj[:, l], ins) for l in range(acts.shape[1])])

    rows = []
    for l in range(acts.shape[1]):
        rows.append(
            dict(
                layer=l,
                r_norm=float(rnorm[l]),
                fisher=float(fisher[l]),
                proj_auc=float(auc[l]),
            )
        )
        print(
            f"layer {l:2d}  |r|={rnorm[l]:8.3f}  fisher={fisher[l]:.3f}  "
            f"proj_auc={auc[l]:.3f}"
        )
    return dirs, rows
