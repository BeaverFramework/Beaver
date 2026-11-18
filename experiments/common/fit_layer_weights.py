"""Pick the direction's layer(s) by WITHIN-PROMPT ranking ability.

Why within-prompt: node selection only ever compares continuations *of the
same prompt*. A layer that separates violating from safe samples mostly by
reading which prompt/template it is scores well pooled and is useless at
search time, because that component is constant across every candidate the
selector is choosing between. On the toxicity fit set, template_id alone
explains 21.5% of label variance, and the pooled-AUC-best layer (16) is only
the 12th-best layer within-prompt.

Emits, for each layer:
  pooled_auc   -- AUC over all samples (what a naive fit optimises)
  wp_auc       -- mean per-prompt AUC over prompts with mixed labels
  template_auc -- AUC separating high- from low-violation-rate templates,
                  i.e. how much of the layer's signal is the confound
and picks:
  best_single_layer -- argmax wp_auc
  layers/weights    -- weights proportional to (wp_auc-.5) - (template_auc-.5),
                       normalised to sum to 1

Directions are refit per fold on training prompts only (grouped 5-fold by
prompt), so the AUCs are held-out rather than in-sample.

Example:
    python -m experiments.common.fit_layer_weights \
        --reps repr_out/llama32_3b/secure_code/gen_last_reps.npz \
        --out  repr_out/llama32_3b/secure_code/gen_last/layer_weights.json
"""

import argparse
import collections
import json

import numpy as np


def auc(scores, labels):
    labels = np.asarray(labels, dtype=bool)
    n1 = int(labels.sum())
    n0 = len(labels) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    ranks = np.argsort(np.argsort(scores)) + 1
    return (ranks[labels].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--reps", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    z = np.load(args.reps)
    R, y, idx = z["reps"], z["y"].astype(bool), z["idx"]
    tid = z["template_id"] if "template_id" in z else np.full(len(y), -1)
    n, L, _ = R.shape
    print(f"{n} samples, {L} layers, violation rate {y.mean():.1%}")

    prompts = np.unique(idx)
    rng = np.random.RandomState(0)
    folds = np.array_split(rng.permutation(len(prompts)), args.folds)
    scores = np.zeros((n, L), dtype=np.float32)
    for f in folds:
        test_p = set(prompts[f].tolist())
        te = np.array([i in test_p for i in idx])
        tr = ~te
        if y[tr].sum() == 0 or (~y[tr]).sum() == 0:
            continue
        for l in range(L):
            H = R[tr, l, :]
            d = H[y[tr]].mean(0) - H[~y[tr]].mean(0)
            d /= np.linalg.norm(d) + 1e-8
            He = R[te, l, :]
            scores[te, l] = (He @ d) / (np.linalg.norm(He, axis=1) + 1e-8)

    by = collections.defaultdict(list)
    for i, p in enumerate(idx):
        by[int(p)].append(i)
    mixed = [np.array(v) for v in by.values() if 0 < y[np.array(v)].sum() < len(v)]
    print(f"prompts with mixed labels (usable within-prompt): {len(mixed)}/{len(by)}")
    if not mixed:
        raise SystemExit(
            "No prompt has both a violating and a non-violating sample, so "
            "within-prompt ranking cannot be measured. Sample more per prompt, "
            "or raise temperature."
        )

    uniq_t = np.unique(tid)
    if len(uniq_t) > 1:
        vr = {int(t): y[tid == t].mean() for t in uniq_t}
        hi = np.array([vr[int(t)] for t in tid]) > np.median(list(vr.values()))
    else:
        hi = None

    pooled = np.array([auc(scores[:, l], y) for l in range(L)])
    wp = np.array(
        [np.nanmean([auc(scores[m, l], y[m]) for m in mixed]) for l in range(L)]
    )
    tauc = (
        np.array([auc(scores[:, l], hi) for l in range(L)])
        if hi is not None
        else np.full(L, 0.5)
    )

    raw = np.clip((wp - 0.5) - (tauc - 0.5), 0, None)
    if raw.sum() <= 0:
        raise SystemExit("No layer beats chance within-prompt; direction is unusable.")
    keep = np.where(raw > 0)[0]
    w = raw[keep] / raw[keep].sum()
    best = int(np.nanargmax(wp))

    print(f"\n{'layer':>5} {'pooled':>8} {'within':>8} {'template':>9} {'weight':>8}")
    wmap = dict(zip(keep.tolist(), w.tolist()))
    for l in range(L):
        print(
            f"{l:5d} {pooled[l]:8.4f} {wp[l]:8.4f} {tauc[l]:9.4f} "
            f"{wmap.get(l, 0.0):8.4f}"
        )
    print(f"\nbest single layer (within-prompt): {best}  wp_auc={wp[best]:.4f}")
    print(f"pooled-best layer would have been: {int(np.nanargmax(pooled))}")

    out = {
        "best_single_layer": best,
        "layers": [int(x) for x in keep],
        "weights": [float(x) for x in w],
        "pooled_auc": [float(x) for x in pooled],
        "wp_auc": [float(x) for x in wp],
        "template_auc": [float(x) for x in tauc],
        "n_mixed_prompts": len(mixed),
        "violation_rate": float(y.mean()),
    }
    json.dump(out, open(args.out, "w"), indent=1)
    print(f"\nwrote {args.out}")
    print("--direction_layers " + ",".join(str(int(x)) for x in keep))
    print("--direction_layer_weights " + ",".join(f"{x:.5f}" for x in w))


if __name__ == "__main__":
    main()
