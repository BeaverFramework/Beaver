"""Probe / direction TESTING toolkit, shared by the activation-direction scripts.

Everything here evaluates a candidate direction or probe -- it does not compute
residuals or directions, it scores them. Both `violation_direction_detection.py`
(TransformerLens) and `violation_direction_heretic.py` (Heretic-native) import
these so the held-out evaluation is identical across the two residual backends.

Contents:
  _fit_probe              StandardScaler (+ optional PCA) + LogisticRegression.
  _batched_logistic_fit   one logistic regression per layer, fit jointly on GPU.
  grouped_cv_auc          per-layer grouped (by-prompt) held-out AUC, mean +/- std
                          across folds, for diff-of-means and probe detectors.
  embedding_floor         diff-of-means AUC at the embedding layer (lexical floor).
  bag_of_words_baseline   grouped-CV TF-IDF logistic on raw generated text.
  per_cwe_auc             held-out probe AUC restricted to each CWE.

The single-vector AUC primitive `_auc` lives in the package __init__.
"""

import numpy as np
import torch
from tqdm import tqdm

from beaver.activation_detection import _auc

# Defaults shared by the testing functions. The scripts may keep their own copies
# of these as user-facing config; these are the values the functions default to.
SEED = 0
N_CV_SPLITS = 5
# Optional PCA before the logistic probe: large speedup + mild regularization,
# little AUC loss. None = no PCA.
PROBE_PCA_DIM = 256


# %% ----------------------------------------------- grouped held-out eval
def _fit_probe(Xtr, ytr, pca_dim=PROBE_PCA_DIM):
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    steps = [("sc", StandardScaler())]
    if pca_dim and pca_dim < Xtr.shape[1]:
        steps.append(("pca", PCA(n_components=pca_dim, random_state=SEED)))
    steps.append(("clf", LogisticRegression(max_iter=2000, C=1.0)))
    pipe = Pipeline(steps).fit(Xtr, ytr)
    return pipe


def _batched_logistic_fit(X, y, l2=1.0, max_iter=60):
    """Fit one logistic regression PER LAYER simultaneously on the GPU.

    X [N, L, d] standardized, y [N] in {0,1}. Returns (W [L,d], b [L]).
    Objective per layer matches sklearn LogisticRegression(C=1/l2): sum of
    sample BCE + 0.5*l2*||w||^2 (bias unregularized). Layers are independent,
    so one LBFGS over the stacked params optimizes all 36 at once.
    """
    N, L, d = X.shape
    W = torch.zeros(L, d, device=X.device, requires_grad=True)
    b = torch.zeros(L, device=X.device, requires_grad=True)
    yL = y.view(N, 1).expand(N, L)
    bce = torch.nn.functional.binary_cross_entropy_with_logits
    opt = torch.optim.LBFGS(
        [W, b], lr=1.0, max_iter=max_iter, line_search_fn="strong_wolfe"
    )

    def closure():
        opt.zero_grad()
        logits = torch.einsum("nld,ld->nl", X, W) + b  # [N, L]
        loss = bce(logits, yL, reduction="sum") + 0.5 * l2 * (W * W).sum()
        loss = loss / (N * L)  # scale for LBFGS
        loss.backward()
        return loss

    opt.step(closure)
    return W.detach(), b.detach()


def grouped_cv_auc(acts, y, groups, n_splits=5, seed=0, l2=1.0):
    """Per-layer grouped (by prompt) held-out AUC, GPU-accelerated.

    Return structure:
      {'diffmeans': {'mean':[L],'std':[L]}, 'probe': {'mean':[L],'std':[L]}}
    Per-fold AUC, then mean +/- std across folds.
    """
    from sklearn.model_selection import StratifiedGroupKFold

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    y = np.asarray(y)
    N, L, _ = acts.shape
    X_all = torch.from_numpy(np.ascontiguousarray(acts)).float().to(dev)  # [N,L,d]
    y_all = torch.from_numpy(y.astype(np.float32)).to(dev)

    sgkf = StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed)
    folds = list(sgkf.split(np.zeros(N), y, groups))
    dm = np.full((n_splits, L), np.nan)
    pr = np.full((n_splits, L), np.nan)

    for fi, (tr, te) in enumerate(tqdm(folds, desc="grouped-cv folds")):
        tr_t = torch.as_tensor(tr, device=dev)
        te_t = torch.as_tensor(te, device=dev)
        Xtr, Xte, ytr = X_all[tr_t], X_all[te_t], y_all[tr_t]
        mu = Xtr.mean(0)
        sd = Xtr.std(0).clamp_min(1e-6)  # [L, d]
        Xtr_s, Xte_s = (Xtr - mu) / sd, (Xte - mu) / sd

        pos = ytr == 1  # diff-of-means
        r = Xtr_s[pos].mean(0) - Xtr_s[~pos].mean(0)
        r = r / (r.norm(dim=1, keepdim=True) + 1e-8)
        s_dm = torch.einsum("nld,ld->nl", Xte_s, r).cpu().numpy()

        W, b = _batched_logistic_fit(Xtr_s, ytr, l2=l2)  # probe
        s_pr = (torch.einsum("nld,ld->nl", Xte_s, W) + b).cpu().numpy()

        yte = y[te]
        for li in range(L):
            dm[fi, li] = _auc(s_dm[:, li], yte)
            pr[fi, li] = _auc(s_pr[:, li], yte)

    return {
        "diffmeans": dict(mean=np.nanmean(dm, 0), std=np.nanstd(dm, 0)),
        "probe": dict(mean=np.nanmean(pr, 0), std=np.nanstd(pr, 0)),
    }


# %% ------------------------------------------------- lexical-floor controls
def embedding_floor(emb, y):
    """AUC of the diff-of-means projection at the EMBEDDING layer (resid_pre.0)
    for one read position. High here == the signal is lexical."""
    ins = np.asarray(y) == 1
    r = emb[ins].mean(0) - emb[~ins].mean(0)
    r = r / (np.linalg.norm(r) + 1e-8)
    return float(_auc(emb @ r, ins))


def bag_of_words_baseline(labels, y, groups, n_splits=N_CV_SPLITS, seed=SEED):
    """The decisive control: grouped-CV TF-IDF logistic regression on the raw
    generated TEXT. If the activation probe doesn't clearly beat this, your
    'direction' is a keyword detector, not a learned concept."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.pipeline import make_pipeline

    texts = np.array([l["generation"] for l in labels])
    y = np.asarray(y)
    sgkf = StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed)
    oof = np.full(len(y), np.nan)
    for tr, te in sgkf.split(texts, y, groups):
        pipe = make_pipeline(
            TfidfVectorizer(min_df=2), LogisticRegression(max_iter=2000)
        )
        pipe.fit(texts[tr], y[tr])
        oof[te] = pipe.decision_function(texts[te])
    auc = _auc(oof, y)
    print(f"bag-of-words baseline (grouped CV): AUC={auc:.3f}")
    return float(auc)


# %% --------------------------------------------------------- per-CWE
def per_cwe_auc(acts, y, cwes, groups, layer, min_per_class=15):
    """Held-out probe AUC restricted to each CWE (violating vs non-violating
    within that CWE) at one layer. Pooling CWEs blurs the direction; this shows
    which vulnerability classes are actually linearly legible."""
    y, cwes = np.asarray(y), np.asarray(cwes)
    out = {}
    for c in sorted(set(cwes)):
        m = cwes == c
        yc = y[m]
        if yc.sum() < min_per_class or (~yc.astype(bool)).sum() < min_per_class:
            continue
        if len(set(groups[m])) < N_CV_SPLITS:  # too few prompts to group-split
            continue
        cv = grouped_cv_auc(acts[m][:, layer : layer + 1, :], yc, groups[m])
        out[c] = float(cv["probe"]["mean"][0])
        print(
            f"  CWE {c:>10}  n={m.sum():4d} ({int(yc.sum())} viol)  "
            f"probe_auc={out[c]:.3f}"
        )
    return out
