"""Pick a gate_theta: drop confidently-safe nodes, keep nearly all violations.

The gate thresholds a STANDARDISED score, not a probability:

    z = (combined_score - combined_mean) / combined_std

Fisher-LDA does NOT bound its output to [0, 1]. `lda_weights` returns a unit
L2-norm w, and the score is w . s over per-layer cosines s in [-1, 1], so the
combined score is an unbounded projection -- empirically a band a few
hundredths wide around combined_mean. That is why gate_stats.json ships
combined_mean/combined_std and why theta is quoted in standard deviations.

Consequence for theta=0: it means "at the fit-set mean". With a 23.3% violation
rate the mean sits deep inside the safe mass, so theta=0 is a permissive
threshold, not a neutral one -- it is far below the violating cluster and
passes most of the safe cluster too.

This sweeps theta over the labelled fit set and reports, for each value, what
fraction of violating and of safe samples clear it. The threshold you want is
the one that keeps violation recall high (nothing dangerous gets pruned) while
cutting the safe mass hard (that is the whole point of gating).

Asymmetry to keep in mind: a violation wrongly pruned is a missed detection the
search can never recover; a safe node wrongly kept only costs a transition. So
choose from the recall side, not by maximising balanced accuracy.

Example:
    python -m experiments.common.calibrate_gate_theta \\
        --reps repr_out/qwen3_4b/secure_code/gen_last_reps.npz \\
        --directions repr_out/qwen3_4b/secure_code/gen_last/directions.npz \\
        --lda repr_out/qwen3_4b/secure_code/gen_last/lda.json \\
        --gate-stats repr_out/qwen3_4b/secure_code/gen_last/gate_stats.json \\
        --out repr_out/qwen3_4b/secure_code/gen_last/theta_sweep.json
"""

import argparse
import json

import numpy as np


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--reps", required=True)
    ap.add_argument("--directions", required=True)
    ap.add_argument("--lda", required=True)
    ap.add_argument("--gate-stats", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--targets", default="0.999,0.99,0.95,0.90",
                    help="violation recall levels to solve theta for")
    args = ap.parse_args()

    z = np.load(args.reps)
    R, y = z["reps"], z["y"].astype(bool)
    dz = np.load(args.directions)
    D = dz[dz.files[0]][1:]                      # row 0 is the zero pad
    D = D / (np.linalg.norm(D, axis=1, keepdims=True) + 1e-8)
    w = np.asarray(json.load(open(args.lda))["weights"], dtype=np.float32)

    n, L, _ = R.shape
    S = np.zeros((n, L), dtype=np.float32)
    for l in range(L):
        H = R[:, l, :]
        S[:, l] = (H @ D[l]) / (np.linalg.norm(H, axis=1) + 1e-8)
    # Match DirectionNVF's scale: it normalises the weight vector by its L1
    # norm before scoring, so the quantity the gate thresholds is
    # (S @ w)/sum|w|, not S @ w. Thresholds derived from the undivided score
    # are off by that factor (~4.8 here) and do not transfer to a run.
    w_l1 = float(np.abs(w).sum())
    combined = (S @ w) / (w_l1 if w_l1 > 0 else 1.0)

    gs = json.load(open(args.gate_stats))
    mu, sd = float(gs["combined_mean"]), float(gs["combined_std"])
    zs = (combined - mu) / (sd if sd > 1e-9 else 1.0)

    zv, zsafe = zs[y], zs[~y]
    out = {
        "n": int(n), "n_violating": int(y.sum()), "n_safe": int((~y).sum()),
        "combined_mean": mu, "combined_std": sd,
        # Evidence for "LDA is not bounded to [0,1]".
        "raw_score_min": float(combined.min()),
        "raw_score_max": float(combined.max()),
        "raw_score_range": float(combined.max() - combined.min()),
        "w_l2_norm": float(np.linalg.norm(w)),
        "w_l1_norm": w_l1,
        "z_violating": {"mean": float(zv.mean()), "std": float(zv.std()),
                        "p05": float(np.quantile(zv, .05)),
                        "p10": float(np.quantile(zv, .10)),
                        "p50": float(np.quantile(zv, .50))},
        "z_safe": {"mean": float(zsafe.mean()), "std": float(zsafe.std()),
                   "p50": float(np.quantile(zsafe, .50)),
                   "p90": float(np.quantile(zsafe, .90)),
                   "p99": float(np.quantile(zsafe, .99))},
    }

    grid = np.round(np.arange(-1.5, 3.01, 0.1), 2)
    sweep = []
    for t in grid:
        keep_v = float((zv >= t).mean())
        keep_s = float((zsafe >= t).mean())
        sweep.append({
            "theta": float(t),
            "violation_recall": keep_v,       # violations the gate keeps
            "safe_pass_rate": keep_s,         # safe nodes it fails to prune
            "safe_pruned": 1.0 - keep_s,
            "overall_pass_rate": float((zs >= t).mean()),
        })
    out["sweep"] = sweep

    # theta achieving each target recall: the largest theta still keeping that
    # fraction of violations, i.e. the corresponding quantile of the violating z.
    rec = {}
    for tgt in [float(x) for x in args.targets.split(",")]:
        t = float(np.quantile(zv, 1.0 - tgt))
        rec[f"{tgt:.3f}"] = {
            "theta": round(t, 3),
            "violation_recall": float((zv >= t).mean()),
            "safe_pass_rate": float((zsafe >= t).mean()),
            "safe_pruned": float((zsafe < t).mean()),
            "overall_pass_rate": float((zs >= t).mean()),
        }
    out["recommended"] = rec
    json.dump(out, open(args.out, "w"), indent=1)

    print(f"n={n}  violating={int(y.sum())}  safe={int((~y).sum())}")
    print(f"score range on DirectionNVF's scale: [{combined.min():+.5f}, {combined.max():+.5f}]  "
          f"(NOT [0,1]; ||w||_2={np.linalg.norm(w):.4f}, sum|w|={w_l1:.4f})")
    print(f"standardisation: mu={mu:+.5f} sd={sd:.5f}")
    print(f"z violating: mean={zv.mean():+.3f} sd={zv.std():.3f} p05={np.quantile(zv,.05):+.3f}")
    print(f"z safe     : mean={zsafe.mean():+.3f} sd={zsafe.std():.3f} p90={np.quantile(zsafe,.90):+.3f}")
    print()
    print(f"{'theta':>6} {'viol kept':>10} {'safe kept':>10} {'safe pruned':>12} {'overall':>8}")
    for r in sweep:
        if abs(r["theta"] * 10 % 2) < 1e-6:      # every 0.2
            print(f"{r['theta']:>6.1f} {r['violation_recall']:>10.3f} "
                  f"{r['safe_pass_rate']:>10.3f} {r['safe_pruned']:>12.3f} "
                  f"{r['overall_pass_rate']:>8.3f}")
    print("\nto keep this fraction of violations:")
    for k, v in rec.items():
        print(f"  recall {k}: theta={v['theta']:+.3f}  "
              f"safe pruned={v['safe_pruned']:.3f}  overall pass={v['overall_pass_rate']:.3f}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
