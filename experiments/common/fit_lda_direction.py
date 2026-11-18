"""Fisher-LDA layer weights + matching pooled gate statistics.

Per-layer directions each give one cosine score, so a sample is a vector
    x = [cos(h^(0), d_0), ..., cos(h^(L-1), d_{L-1})]
and any way of collapsing x to a scalar is a weight vector w, scoring w.x.

Uniform (w = 1/L) and AUC-proportional weights both judge each layer *in
isolation*, which double-counts redundancy: adjacent residual layers correlate
~0.95, so twenty near-duplicate mid-layers outvote one layer carrying
independent signal. Fisher's discriminant

    w = Sigma^-1 (mu_1 - mu_0)          Sigma = within-class covariance of x

divides the signal by how much each layer co-varies with the rest, i.e. it
asks what a layer *adds*. Measured held-out (prompt-level 5-fold), LDA beat
both on every cell tried, by up to +0.058 AUC.

Sigma is near-singular across correlated layers, so it is shrunk:
Sigma + eps*trace(Sigma)/L * I.

Also writes the gate reference for the COMBINED score. This matters: the gate
standardises (score - mu)/sd, and with non-uniform weights the combined score's
mu/sd are NOT the average of the per-layer ones. Storing them separately is
what lets theta keep its meaning.
"""

import argparse
import json

import numpy as np


def auc(scores, labels):
    labels = np.asarray(labels, dtype=bool)
    n1 = int(labels.sum())
    n0 = len(labels) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = np.argsort(np.argsort(scores)) + 1
    return (r[labels].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def per_layer_scores(R, D):
    """[n, L] cosine of each sample's layer-l state against direction d_l."""
    n, L, _ = R.shape
    S = np.zeros((n, L), dtype=np.float32)
    for l in range(L):
        H = R[:, l, :]
        S[:, l] = (H @ D[l]) / (np.linalg.norm(H, axis=1) + 1e-8)
    return S


def lda_weights(X, y, eps=1e-3):
    m1, m0 = X[y].mean(0), X[~y].mean(0)
    C = np.cov(X.T)
    C = C + np.eye(C.shape[0]) * eps * np.trace(C) / C.shape[0]
    w = np.linalg.solve(C, m1 - m0)
    return w / (np.linalg.norm(w) + 1e-12)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reps", required=True)
    ap.add_argument("--directions", required=True)
    ap.add_argument("--out", required=True, help="lda.json")
    ap.add_argument("--gate-stats", required=True, help="gate_stats.json to update")
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    z = np.load(args.reps)
    R, y, idx = z["reps"], z["y"].astype(bool), z["idx"]
    dz = np.load(args.directions)
    D = dz[dz.files[0]][1:]
    D = D / (np.linalg.norm(D, axis=1, keepdims=True) + 1e-8)
    L = R.shape[1]
    assert D.shape[0] == L, f"{D.shape[0]} directions vs {L} layers"

    S = per_layer_scores(R, D)

    # Held-out quality, folds split BY PROMPT so the 10 completions of one
    # prompt never straddle train/test (they are highly correlated; splitting
    # by sample would let the fit memorise prompt identity).
    prompts = np.unique(idx)
    rng = np.random.RandomState(0)
    folds = np.array_split(rng.permutation(len(prompts)), args.folds)
    pred_lda = np.zeros(len(y))
    pred_uni = np.zeros(len(y))
    for f in folds:
        te = np.isin(idx, prompts[f])
        tr = ~te
        if y[tr].sum() == 0 or (~y[tr]).sum() == 0:
            continue
        pred_lda[te] = S[te] @ lda_weights(S[tr], y[tr])
        pred_uni[te] = S[te].mean(1)
    cv_lda, cv_uni = auc(pred_lda, y), auc(pred_uni, y)
    best_single = max(auc(S[:, l], y) for l in range(L))

    # Deployed weights: fit on everything (the CV above is the honest estimate
    # of how this will generalise).
    w = lda_weights(S, y)
    # UNIT MISMATCH, and it is load-bearing. DirectionNVF normalises the weight
    # vector by its L1 norm before scoring (nvf/__init__.py: `total =
    # np.abs(w).sum(); self.layer_weights = w / total`), so the score it hands
    # the gate is (S @ w) / sum|w| -- NOT S @ w. lda_weights returns a unit-L2
    # vector whose L1 norm is ~4.8 here, so storing the undivided statistics
    # standardises a compressed score against an uncompressed reference and
    # every runtime z lands ~0.4-0.5 too low. Observed consequence: with
    # theta=0 the gate rejected all 10000 transitions of a pilot, because even
    # a maximally-violating completion only reaches z = -0.09 on that scale.
    # Store the statistics of the quantity DirectionNVF actually produces.
    w_l1 = float(np.abs(w).sum())
    combined_raw = S @ w
    combined = combined_raw / (w_l1 if w_l1 > 0 else 1.0)
    out = {
        "weights": [float(v) for v in w],
        "layers": list(range(L)),
        "cv_auc_lda": float(cv_lda),
        "cv_auc_uniform": float(cv_uni),
        "pooled_auc_best_single_layer": float(best_single),
        # These two are what the gate standardises against, so they are on
        # DirectionNVF's scale. NOTE DirectionNVF also adds a depth term
        # (b/sum|w|)/token_count, ~0.002/token_count -- a small offset that
        # decays with depth and is not modelled here.
        "combined_mean": float(combined.mean()),
        "combined_std": float(combined.std()),
        "weights_l1_norm": w_l1,
        # The undivided statistics, kept for traceability against runs fitted
        # before this scaling was corrected.
        "combined_mean_unnormalized": float(combined_raw.mean()),
        "combined_std_unnormalized": float(combined_raw.std()),
        "violation_rate": float(y.mean()),
        "n_samples": int(len(y)),
    }
    json.dump(out, open(args.out, "w"), indent=1)

    # Fold the combined reference into gate_stats so the selector can
    # standardise the weighted score correctly.
    gs = json.load(open(args.gate_stats))
    gs["combined_mean"] = out["combined_mean"]
    gs["combined_std"] = out["combined_std"]
    gs["combined_weights"] = out["weights"]
    json.dump(gs, open(args.gate_stats, "w"), indent=1)

    frac = float((((combined - combined.mean()) / combined.std()) >= 0).mean())
    print(f"  LDA cv_auc={cv_lda:.4f}  uniform={cv_uni:.4f}  best-1layer={best_single:.4f}")
    print(f"  combined mu={out['combined_mean']:+.4f} sd={out['combined_std']:.4f}")
    print(f"  theta=0 would pass {frac:.1%} of fit samples")
    print(f"  -> {args.out}")


if __name__ == "__main__":
    main()
