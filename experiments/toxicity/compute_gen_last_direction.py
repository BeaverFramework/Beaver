"""Compute the gen_last harmful direction from all sampled data, and test it.

direction[layer] = normalize(mean(reps[violating]) - mean(reps[non-violating])),
per layer, using every sample (paper's Eq. 3 diff-of-means, no per-fold
refitting for the deployed vector itself).

Testing, same methodology as the earlier 100-prompt audit so results are
directly comparable:
  - held-out grouped (by prompt) 5-fold CV AUC per layer (a FRESH direction
    refit per fold, not the deployed one -- generalization estimate)
  - template confound: project the deployed direction onto every sample,
    eta^2 against the true label vs against template_id, per layer

Example:
    python -m experiments.toxicity.compute_gen_last_direction \
        --reps logging/tox_gen_last_reps_5000.npz \
        --out-direction logging/tox_gen_last_direction_500x10.npz \
        --out-report logging/tox_gen_last_direction_500x10_report.json
"""

import argparse
import json

import numpy as np


def eta_sq(values, group_labels):
    values = np.asarray(values)
    grand_mean = values.mean()
    ss_total = ((values - grand_mean) ** 2).sum()
    ss_between = 0.0
    for gl in set(group_labels.tolist()):
        idx = group_labels == gl
        if idx.sum() < 2:
            continue
        ss_between += idx.sum() * (values[idx].mean() - grand_mean) ** 2
    return float(ss_between / ss_total) if ss_total > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reps", required=True, help="npz from extract_gen_last_representations.py")
    ap.add_argument("--out-direction", required=True, help="directions.npz-compatible output")
    ap.add_argument("--out-report", required=True, help="per-layer diagnostics JSON")
    args = ap.parse_args()

    z = np.load(args.reps, allow_pickle=True)
    reps = z["reps"]  # [N, 28, 3072]
    y = z["y"]
    groups = z["idx"]  # prompt id
    template_ids = z["template_id"]
    N, L, D = reps.shape
    print(f"{N} samples, {L} layers, {D} dim -- {int(y.sum())} positive / {int((y==0).sum())} negative")

    # ---- 1. Deployed direction: diff-of-means over ALL data ----
    ins = y == 1
    good_means = reps[~ins].mean(0)  # [L, D]
    bad_means = reps[ins].mean(0)
    r = bad_means - good_means
    rnorm = np.linalg.norm(r, axis=1)
    dirs = r / (rnorm[:, None] + 1e-8)  # [L, D] unit directions, hf_server layer order

    # Store in heretic-index-compatible form (row 0 = unused embedding
    # placeholder, DirectionNVF.load drops it via directions[1:] anyway) so
    # this file loads through the existing DirectionNVF.load pipeline
    # unchanged.
    dirs_heretic_shape = np.concatenate([np.zeros((1, D), dtype=dirs.dtype), dirs], axis=0)  # [29, D]
    np.savez(args.out_direction, gen_last=dirs_heretic_shape)
    print(f"wrote direction -> {args.out_direction}  shape={dirs_heretic_shape.shape}")

    # ---- 2. Held-out grouped CV AUC per layer ----
    from beaver.activation_detection.probes import grouped_cv_auc
    cv = grouped_cv_auc(reps, y, groups, n_splits=5, seed=0)
    heldout_auc = cv["diffmeans"]["mean"]

    # ---- 3. Template confound: project deployed direction, eta^2 ----
    rows = []
    for l in range(L):
        proj = reps[:, l, :] @ dirs[l]
        el = eta_sq(proj, y)
        et = eta_sq(proj, template_ids)
        rows.append({
            "layer": l,
            "heldout_auc": round(float(heldout_auc[l]), 4),
            "label_eta2": round(el, 4),
            "template_eta2": round(et, 4),
            "confounded": bool(et > el),
        })

    n_confounded = sum(r["confounded"] for r in rows)
    print(f"\nlayers where template_eta2 > label_eta2: {n_confounded} / {L}")
    print(f"{'layer':>5} {'heldout_auc':>11} {'label_eta2':>10} {'template_eta2':>13}  flag")
    for r_ in rows:
        flag = "<-- confounded" if r_["confounded"] else ""
        print(f"{r_['layer']:>5} {r_['heldout_auc']:>11.3f} {r_['label_eta2']:>10.4f} {r_['template_eta2']:>13.4f}  {flag}")

    report = {
        "n_samples": int(N),
        "n_positive": int(y.sum()),
        "n_negative": int((y == 0).sum()),
        "n_prompts": int(len(set(groups.tolist()))),
        "n_layers_confounded": int(n_confounded),
        "layers": rows,
    }
    json.dump(report, open(args.out_report, "w"), indent=2)
    print(f"\nwrote report -> {args.out_report}")


if __name__ == "__main__":
    main()
