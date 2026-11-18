"""Calibrate a violation-direction projection into a P(violation) estimate.

The probe gives a raw scalar per prompt: proj = residual @ direction. It ranks
well but is unbounded -- not a probability. A calibrator is a 1-D monotonic map
proj -> p in [0, 1], fit against each prompt's empirical violation fraction
(#violating generations / #generations) from the detection run.

fit_calibrator returns a small picklable object with .predict(proj) -> p, plus
fit metrics (Brier, leave-one-out Brier) so you can tell whether the mapping is
trustworthy. Logistic (Platt) is the default: 2 params (slope, bias), robust when
there are only a few distinct prompts. Isotonic is available for many-prompt
regimes but overfits with few points.
"""

import numpy as np


class Calibrator:
    """Picklable wrapper around a fitted sklearn 1-D probability map.

    predict(proj) accepts a scalar or array of projections and returns p in [0,1].
    `meta` carries the fit method, layer/read_pos it was fit at, and metrics.
    """

    def __init__(self, kind, model, meta):
        self.kind = kind          # "logistic" | "isotonic"
        self.model = model        # the fitted sklearn estimator
        self.meta = dict(meta)    # method, layer, read_pos, brier, loo_brier, ...

    def predict(self, proj):
        proj = np.atleast_1d(np.asarray(proj, dtype=np.float64))
        if self.kind == "logistic":
            p = self.model.predict_proba(proj.reshape(-1, 1))[:, 1]
        else:  # isotonic
            p = self.model.predict(proj)
        return p


def _fit_logistic(proj, frac, n):
    """Platt scaling against soft labels. Each prompt contributes its n samples:
    round(frac*n) positives and the rest negatives at the same proj value, so the
    fit is naturally weighted by how many generations backed each fraction."""
    from sklearn.linear_model import LogisticRegression

    X, y = [], []
    for pr, fr, ni in zip(proj, frac, n):
        ni = max(int(round(ni)), 1)
        pos = int(round(fr * ni))
        X.extend([pr] * ni)
        y.extend([1] * pos + [0] * (ni - pos))
    X = np.asarray(X, dtype=np.float64).reshape(-1, 1)
    y = np.asarray(y)
    if y.min() == y.max():  # one class only -> degenerate; constant predictor
        clf = _ConstantProba(float(y.mean()))
    else:
        clf = LogisticRegression(max_iter=2000, C=1e6).fit(X, y)  # ~unregularized
    return clf


def _fit_isotonic(proj, frac, n):
    from sklearn.isotonic import IsotonicRegression

    return IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(
        np.asarray(proj, dtype=np.float64), np.asarray(frac, dtype=np.float64),
        sample_weight=np.asarray(n, dtype=np.float64),
    )


class _ConstantProba:
    """Fallback when all training prompts share one label (all 0 or all 1)."""

    def __init__(self, p):
        self.p = float(np.clip(p, 1e-6, 1 - 1e-6))

    def predict_proba(self, X):
        col = np.full(len(X), self.p)
        return np.stack([1 - col, col], axis=1)


def _brier(p, frac, n):
    """Sample-weighted Brier score against the empirical fractions."""
    p, frac, n = np.asarray(p), np.asarray(frac), np.asarray(n, dtype=float)
    return float(np.sum(n * (p - frac) ** 2) / max(n.sum(), 1))


def fit_calibrator(proj, frac, n, method="logistic", read_pos=None, layer=None):
    """Fit proj -> P(violation). Returns a Calibrator.

    proj   [P] projection of each prompt's residual onto the direction.
    frac   [P] empirical violation fraction per prompt in [0,1].
    n      [P] #generations behind each fraction (fit + Brier weight).
    method "logistic" (default) or "isotonic".

    meta includes in-sample Brier and leave-one-out Brier (refit dropping each
    prompt) -- the honest number with so few prompts. Lower is better; compare to
    the trivial baseline of predicting the global mean fraction.
    """
    proj = np.asarray(proj, dtype=np.float64)
    frac = np.asarray(frac, dtype=np.float64)
    n = np.asarray(n, dtype=np.float64)
    P = len(proj)
    fitter = _fit_logistic if method == "logistic" else _fit_isotonic
    kind = method

    model = fitter(proj, frac, n)
    cal = Calibrator(kind, model, dict(method=method, read_pos=read_pos, layer=layer))
    p_in = cal.predict(proj)
    brier = _brier(p_in, frac, n)

    # leave-one-out Brier (cheap at small P).
    loo = np.empty(P)
    for i in range(P):
        m = np.arange(P) != i
        if not m.any():
            loo[i] = frac[i]
            continue
        mdl = fitter(proj[m], frac[m], n[m])
        loo[i] = Calibrator(kind, mdl, {}).predict(proj[i])[0]
    loo_brier = _brier(loo, frac, n)
    baseline = _brier(np.full(P, np.average(frac, weights=n)), frac, n)

    cal.meta.update(
        n_prompts=int(P),
        brier=brier,
        loo_brier=loo_brier,
        baseline_brier=baseline,  # predict-the-mean; calibrator should beat this
    )
    return cal


def prompt_fractions(y, groups):
    """Per-prompt (group) empirical violation fraction + sample count.

    y [N] in {0,1} (1=violating), groups [N] prompt ids. Returns aligned arrays
    (group_ids [P], frac [P], n [P]) sorted by group id.
    """
    y = np.asarray(y)
    groups = np.asarray(groups)
    gids = np.unique(groups)
    frac = np.array([y[groups == g].mean() for g in gids], dtype=np.float64)
    n = np.array([(groups == g).sum() for g in gids], dtype=np.float64)
    return gids, frac, n
