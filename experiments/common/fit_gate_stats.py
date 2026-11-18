"""Pooled gate statistics: mean/std of the deployed direction's score.

The gate needs a reference distribution to threshold against. This computes
it ONCE over the whole fit set (all 5000 samples, all prompts pooled) using
the deployed direction, and writes per-layer mean/std to gate_stats.json.

The selector then uses these fixed numbers for every instance, so the gate
threshold is a property of the dataset, not of the instance being searched.

Example:
    python -m experiments.common.fit_gate_stats \
        --reps repr_out/llama32_3b/toxicity/gen_last_reps_5000.npz \
        --directions repr_out/llama32_3b/toxicity/gen_last_500x10/directions.npz \
        --out repr_out/llama32_3b/toxicity/gen_last_500x10/gate_stats.json
"""

import argparse
import json

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reps", required=True)
    ap.add_argument("--directions", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--key", default=None)
    args = ap.parse_args()

    z = np.load(args.reps)
    R = z["reps"]                      # [n, L, d]
    dz = np.load(args.directions)
    key = args.key or dz.files[0]
    D = dz[key]                        # [L+1, d], row 0 = embedding pad
    D = D[1:]
    D = D / (np.linalg.norm(D, axis=1, keepdims=True) + 1e-8)

    n, L, d = R.shape
    assert D.shape[0] == L, f"direction stack has {D.shape[0]} layers, reps have {L}"

    mean, std = [], []
    for l in range(L):
        H = R[:, l, :]
        s = (H @ D[l]) / (np.linalg.norm(H, axis=1) + 1e-8)
        mean.append(float(s.mean()))
        std.append(float(s.std()))

    out = {
        "n_samples": int(n),
        "layers": list(range(L)),
        "mean": mean,
        "std": std,
        "note": "pooled over all samples/prompts; used as the gate's fixed reference",
    }
    json.dump(out, open(args.out, "w"), indent=1)
    print(f"wrote {args.out}  (n={n}, {L} layers)")
    for l in (0, L // 2, L - 1):
        print(f"   layer {l:2d}: mean={mean[l]:+.4f} std={std[l]:.4f}")


if __name__ == "__main__":
    main()
