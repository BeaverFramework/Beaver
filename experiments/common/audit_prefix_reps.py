"""What, if anything, do the prefix extractions actually support?

The prefix run labelled each truncated generation with the verdict of the one
continuation that happened to follow it. That is a single Bernoulli draw from
P(violation | prefix), which is not enough to calibrate a threshold that is
supposed to mean "prune below X% violation probability".

But two things in that data are still sound, and this separates them:

  k=0 IS a proper experiment. The prefix is the prompt alone, so all 10
  completions of a prompt share it -- M=10 draws from the same prefix, giving a
  real p_hat per prefix for 600 prefixes. Scoring the PROMPT's own hidden state
  against that p_hat is exactly the question the NVF has to answer, at prefix
  length zero.

  k>0 supports a weaker claim: AUC of prefix score against the realized outcome
  of its own continuation. Noisy and depressed relative to the true signal
  (the outcome depends on tokens not yet written), but a value near 0.5 still
  means no signal and a value well above it still means some.

Also reports how many samples SHARE a k-token prefix. Where several do, M>1
comes for free and those prefixes support a real p_hat without new sampling.

Scores are on DirectionNVF's scale -- (S @ w)/sum|w| -- so z is comparable to
what a run sees.
"""

import argparse
import json
from collections import defaultdict

import numpy as np


def auc(scores, labels):
    labels = np.asarray(labels, dtype=bool)
    n1 = int(labels.sum())
    n0 = len(labels) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = np.argsort(np.argsort(scores)) + 1
    return float((r[labels].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    return float((ra @ rb) / (np.linalg.norm(ra) * np.linalg.norm(rb) + 1e-12))


def combined_score(reps_path, D, w_l1, w):
    z = np.load(reps_path)
    R, y, idx = z["reps"], z["y"].astype(bool), z["idx"]
    n, L, _ = R.shape
    S = np.zeros((n, L), dtype=np.float32)
    for l in range(L):
        H = R[:, l, :]
        S[:, l] = (H @ D[l]) / (np.linalg.norm(H, axis=1) + 1e-8)
    return (S @ w) / w_l1, y, idx, R


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--reps-glob", required=True,
                    help="e.g. repr_out/qwen3_4b/secure_code/prefix_reps_k{k}.npz")
    ap.add_argument("--ks", default="0,8,16,32,48")
    ap.add_argument("--directions", required=True)
    ap.add_argument("--lda", required=True)
    ap.add_argument("--gate-stats", required=True)
    ap.add_argument("--samples", required=True)
    ap.add_argument("--model", default=None,
                    help="tokenizer for prefix-sharing counts; skipped if absent")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    dz = np.load(args.directions)
    D = dz[dz.files[0]][1:]
    D = D / (np.linalg.norm(D, axis=1, keepdims=True) + 1e-8)
    w = np.asarray(json.load(open(args.lda))["weights"], dtype=np.float32)
    w_l1 = float(np.abs(w).sum())
    gs = json.load(open(args.gate_stats))
    mu, sd = float(gs["combined_mean"]), float(gs["combined_std"])

    records = [json.loads(l) for l in open(args.samples)]
    out = {"w_l1": w_l1, "gate_mu": mu, "gate_sd": sd, "per_k": {}}

    # ── how many samples share a k-token prefix? -> free M>1 where they do ────
    if args.model:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.model)
        gen_ids = [tok.encode(r["generation"], add_special_tokens=False)
                   for r in records]
        share = {}
        for k in [int(x) for x in args.ks.split(",")]:
            groups = defaultdict(list)
            for r, g in zip(records, gen_ids):
                groups[(r["idx"], tuple(g[:k]))].append(bool(not r["is_secure"]))
            sizes = [len(v) for v in groups.values()]
            share[str(k)] = {
                "n_distinct_prefixes": len(groups),
                "mean_samples_per_prefix": float(np.mean(sizes)),
                "prefixes_with_M_ge_2": int(sum(1 for s in sizes if s >= 2)),
                "prefixes_with_M_ge_5": int(sum(1 for s in sizes if s >= 5)),
                "max_M": int(max(sizes)),
            }
        out["prefix_sharing"] = share

    for k in [int(x) for x in args.ks.split(",")]:
        path = args.reps_glob.format(k=k)
        try:
            comb, y, idx, R = combined_score(path, D, w_l1, w)
        except FileNotFoundError:
            print(f"k={k}: missing {path}, skipping")
            continue
        zs = (comb - mu) / sd
        ent = {
            "n": int(len(y)),
            "auc_vs_realized_label": auc(comb, y),
            "z_min": float(zs.min()), "z_p50": float(np.median(zs)),
            "z_max": float(zs.max()),
            "z_violating_mean": float(zs[y].mean()),
            "z_safe_mean": float(zs[~y].mean()),
            "frac_z_ge_0": float((zs >= 0).mean()),
        }

        # k=0: every completion of a prompt shares the prompt as its prefix, so
        # the 10 rows per prompt must be the SAME state. Verifying that is a
        # direct correctness check on the extractor, and it is what makes the
        # per-prompt p_hat below a real M=10 estimate rather than 10 draws of
        # one Bernoulli treated as independent prefixes.
        prompts = sorted(set(idx.tolist()))
        within = []
        for p in prompts[:50]:
            m = idx == p
            if m.sum() > 1:
                within.append(float(comb[m].std()))
        ent["within_prompt_score_std_mean"] = float(np.mean(within)) if within else None

        pm = np.array([comb[idx == p].mean() for p in prompts])
        ph = np.array([y[idx == p].mean() for p in prompts])
        ent["prompt_level"] = {
            "n_prompts": len(prompts),
            "spearman_score_vs_phat": spearman(pm, ph),
            "auc_score_vs_any_violation": auc(pm, ph > 0),
            "mixed_prompts": int(((ph > 0) & (ph < 1)).sum()),
        }
        # Calibration against a real p_hat (M=10 per prompt).
        order = np.argsort(pm)
        ent["prompt_calibration"] = [
            {"bin": i, "n": int(len(c)), "score_z_mean": float(((pm[c] - mu) / sd).mean()),
             "phat_mean": float(ph[c].mean())}
            for i, c in enumerate(np.array_split(order, 10))
        ]
        out["per_k"][str(k)] = ent
        del R

        print(f"k={k:>2}  n={ent['n']}  AUC(vs realized)={ent['auc_vs_realized_label']:.4f}  "
              f"z[{ent['z_min']:+.2f},{ent['z_max']:+.2f}] p50={ent['z_p50']:+.2f}  "
              f"frac z>=0 {ent['frac_z_ge_0']:.3f}")
        print(f"      within-prompt score sd = {ent['within_prompt_score_std_mean']}"
              f"   prompt spearman(score, p_hat) = "
              f"{ent['prompt_level']['spearman_score_vs_phat']:.4f}")

    json.dump(out, open(args.out, "w"), indent=1)
    if "prefix_sharing" in out:
        print("\nprefix sharing (how many completions share a k-token prefix):")
        for k, v in out["prefix_sharing"].items():
            print(f"  k={k:>2}: {v['n_distinct_prefixes']:>5} distinct  "
                  f"mean M={v['mean_samples_per_prefix']:.2f}  "
                  f"M>=2: {v['prefixes_with_M_ge_2']:>4}  M>=5: {v['prefixes_with_M_ge_5']:>4}  "
                  f"max M={v['max_M']}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
