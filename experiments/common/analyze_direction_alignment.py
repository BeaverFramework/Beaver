"""How well does the harmfulness direction align with violation probability?

Three questions, in increasing order of what the NVF actually needs to be true:

  1. SEPARATION. Does the combined (Fisher-LDA weighted) direction score
     separate violating from non-violating completions at all? This is the AUC
     the fitting stage already reports; recomputed here on the deployed
     weights so the number in this report is the deployed one.

  2. CALIBRATION. A verifier does not consume a ranking, it consumes a
     *probability*. Binning samples by score and reading off the observed
     violation fraction per bin says whether the score can be read as one.
     A high AUC with a flat calibration curve would still be useless to the
     search.

  3. PROMPT-LEVEL PREDICTION. Each prompt has 10 completions, so it carries an
     empirical P(violation) in {0.0, 0.1, ..., 1.0}. Does the mean score over
     a prompt's completions track that rate? This is the closest thing the
     fit data can say about "score at a point predicts violation downstream",
     because a prompt IS the length-0 prefix of its own completions.

Caveat that matters for reading (3): these activations are captured at the
LAST token of the FULL generation, not at a prefix. So (3) measures agreement
between an after-the-fact score and a prompt's violation propensity, NOT the
verifier's actual runtime setting -- scoring a partial continuation to predict
whether it will end up violating. For that see the --prefix mode of
extract_activations_vllm-based runs; it needs states captured at truncated
generations, which is a separate extraction.

Memory: reps is [n, L, d] float32 -- 2.2 GB at 6000x36x2560. Per-layer scores
are computed one layer at a time and the big array is never copied.

Example:
    python -m experiments.common.analyze_direction_alignment \\
        --reps repr_out/qwen3_4b/secure_code/gen_last_reps.npz \\
        --directions repr_out/qwen3_4b/secure_code/gen_last/directions.npz \\
        --lda repr_out/qwen3_4b/secure_code/gen_last/lda.json \\
        --out repr_out/qwen3_4b/secure_code/gen_last/alignment.json
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
    return float((r[labels].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def per_layer_scores(R, D):
    """[n, L] cosine of each sample's layer-l state against direction d_l."""
    n, L, _ = R.shape
    S = np.zeros((n, L), dtype=np.float32)
    for l in range(L):
        H = R[:, l, :]
        S[:, l] = (H @ D[l]) / (np.linalg.norm(H, axis=1) + 1e-8)
    return S


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--reps", required=True)
    ap.add_argument("--directions", required=True)
    ap.add_argument("--lda", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bins", type=int, default=10)
    args = ap.parse_args()

    z = np.load(args.reps)
    R, y, idx = z["reps"], z["y"].astype(bool), z["idx"]
    dz = np.load(args.directions)
    # Row 0 of the stack is a zero pad -- DirectionNVF reads row i+1 as layer i.
    D = dz[dz.files[0]][1:]
    D = D / (np.linalg.norm(D, axis=1, keepdims=True) + 1e-8)
    w = np.asarray(json.load(open(args.lda))["weights"], dtype=np.float32)

    S = per_layer_scores(R, D)
    assert S.shape[1] == len(w), f"{S.shape[1]} layers vs {len(w)} weights"
    combined = S @ w

    out = {
        "n_samples": int(len(y)),
        "n_prompts": int(len(set(idx.tolist()))),
        "violation_rate": float(y.mean()),
        "combined_auc_in_sample": auc(combined, y),
        "per_layer_auc": [auc(S[:, l], y) for l in range(S.shape[1])],
        "score_violating_mean": float(combined[y].mean()),
        "score_violating_std": float(combined[y].std()),
        "score_safe_mean": float(combined[~y].mean()),
        "score_safe_std": float(combined[~y].std()),
    }
    # Cohen's d: how many pooled SDs apart the two classes sit. AUC saturates
    # near 1 and stops discriminating between good and very good; d does not.
    pooled = np.sqrt((combined[y].var() + combined[~y].var()) / 2)
    out["cohens_d"] = float((combined[y].mean() - combined[~y].mean()) / (pooled + 1e-12))

    # ── 2. calibration: equal-count bins, observed violation fraction each ────
    order = np.argsort(combined)
    chunks = np.array_split(order, args.bins)
    out["calibration"] = [
        {
            "bin": i,
            "n": int(len(c)),
            "score_mean": float(combined[c].mean()),
            "score_lo": float(combined[c].min()),
            "score_hi": float(combined[c].max()),
            "observed_violation_rate": float(y[c].mean()),
        }
        for i, c in enumerate(chunks)
    ]

    # ── 3. prompt level: mean score vs empirical P(violation) over 10 samples ─
    prompts = sorted(set(idx.tolist()))
    pm, pr, pn = [], [], []
    for p in prompts:
        m = idx == p
        pm.append(float(combined[m].mean()))
        pr.append(float(y[m].mean()))
        pn.append(int(m.sum()))
    pm, pr = np.asarray(pm), np.asarray(pr)

    def pearson(a, b):
        a, b = a - a.mean(), b - b.mean()
        return float((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

    def spearman(a, b):
        ra = np.argsort(np.argsort(a)).astype(float)
        rb = np.argsort(np.argsort(b)).astype(float)
        return pearson(ra, rb)

    out["prompt_level"] = {
        "n_prompts": len(prompts),
        "samples_per_prompt": sorted(set(pn)),
        "pearson_score_vs_violation_rate": pearson(pm, pr),
        "spearman_score_vs_violation_rate": spearman(pm, pr),
        # How many prompts are unanimous? A dataset where every prompt is 0/10
        # or 10/10 makes the prompt-level correlation a restatement of the
        # sample-level AUC rather than an independent check.
        "prompts_all_safe": int((pr == 0).sum()),
        "prompts_all_violating": int((pr == 1).sum()),
        "prompts_mixed": int(((pr > 0) & (pr < 1)).sum()),
        "rate_histogram": {
            f"{k/10:.1f}": int((np.abs(pr - k / 10) < 1e-9).sum()) for k in range(11)
        },
    }
    # Same calibration idea, one level up: bin prompts by mean score and read
    # the average empirical violation rate.
    porder = np.argsort(pm)
    pchunks = np.array_split(porder, args.bins)
    out["prompt_calibration"] = [
        {
            "bin": i,
            "n_prompts": int(len(c)),
            "score_mean": float(pm[c].mean()),
            "observed_violation_rate": float(pr[c].mean()),
        }
        for i, c in enumerate(pchunks)
    ]

    json.dump(out, open(args.out, "w"), indent=1)
    print(f"n={out['n_samples']} prompts={out['n_prompts']} "
          f"violation_rate={out['violation_rate']:.3f}")
    print(f"combined AUC (in-sample) = {out['combined_auc_in_sample']:.4f}  "
          f"Cohen's d = {out['cohens_d']:.2f}")
    print(f"best single layer AUC    = {max(out['per_layer_auc']):.4f} "
          f"(layer {int(np.argmax(out['per_layer_auc']))})")
    pl = out["prompt_level"]
    print(f"prompt-level: pearson={pl['pearson_score_vs_violation_rate']:.4f} "
          f"spearman={pl['spearman_score_vs_violation_rate']:.4f}")
    print(f"  prompts all-safe={pl['prompts_all_safe']} "
          f"mixed={pl['prompts_mixed']} all-violating={pl['prompts_all_violating']}")
    print("\ncalibration (equal-count bins by score):")
    for b in out["calibration"]:
        print(f"  bin{b['bin']:>2}  score~{b['score_mean']:+.4f}  "
              f"observed P(violation)={b['observed_violation_rate']:.3f}  n={b['n']}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
