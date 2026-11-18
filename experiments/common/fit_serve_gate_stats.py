"""Calibrate the gate through the SERVER, on the exact score the gate sees.

Three things separate the offline score from the online one, and all three
were biasing theta:

  1. DirectionNVF divides the layer weights by sum|w| (and scales b to match),
     so the online score is the offline one shrunk ~4.6x for a 32-layer LDA
     weighting -- mean AND spread. Standardising an online score against an
     unnormalised offline mu/sd put every candidate at z ~ +0.95, which is
     exactly the +0.94 median observed in a live run. That alone made theta=0
     pass 100% of candidates and collapsed `gated` onto `max-mu`.
  2. The score carries a b/(sum|w| * depth) length-decay term, worth ~0.26 sd
     at depth 1 and ~0.01 sd at depth 32 -- a shallow-depth bonus that a
     completion-only fit never sees.
  3. The states themselves come from hf_server's forward hook, not from
     TransformerLens, which folds LayerNorm and centers writing weights.

Posting prefixes to the running server sidesteps all three: same weights, same
dtype, same hook, same arithmetic as DirectionNVF.predict. Prefix depths are
sampled uniformly over the generated span, which matches the depth profile of
a real frontier (measured mean 15.9 / median 16 of 32 on a completed run).

No labels are involved -- a mean and a variance do not need them. The LDA
weights, which do, still come from the labelled completions.
"""

import argparse
import base64
import json

import numpy as np
import requests


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--samples", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--server_addr", required=True)
    ap.add_argument("--directions", required=True)
    ap.add_argument("--lda", required=True)
    ap.add_argument("--gate-stats", required=True)
    ap.add_argument("--n_records", type=int, default=300)
    ap.add_argument("--k_positions", type=int, default=6)
    ap.add_argument("--max_gen_tokens", type=int, default=32)
    ap.add_argument("--b", type=float, default=1e-2, help="DirectionNVF b")
    ap.add_argument("--timeout", type=float, default=180.0)
    args = ap.parse_args()

    w = np.asarray(json.load(open(args.lda))["weights"], dtype=np.float32)
    dz = np.load(args.directions)
    D = dz[dz.files[0]][1:]
    D = D / (np.linalg.norm(D, axis=1, keepdims=True) + 1e-8)
    L = D.shape[0]
    assert len(w) == L, f"{len(w)} weights vs {L} directions"
    total = float(np.abs(w).sum())          # exactly DirectionNVF's normaliser
    wn = w / total
    b = args.b / total

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)

    records = [json.loads(l) for l in open(args.samples)]
    rng = np.random.RandomState(0)
    if len(records) > args.n_records:
        records = [records[i] for i in rng.choice(len(records), args.n_records, False)]

    url = args.server_addr.rstrip("/") + "/v1/completions"
    scores, depths, missing = [], [], 0
    for n, r in enumerate(records):
        p_ids = tok.encode(r["exact_prompt"], add_special_tokens=False)
        g_ids = tok.encode(r["generation"], add_special_tokens=False)[: args.max_gen_tokens]
        if len(g_ids) < 2:
            continue
        js = rng.choice(len(g_ids), min(args.k_positions, len(g_ids)), False) + 1
        rows = [p_ids + g_ids[:int(j)] for j in js]
        try:
            resp = requests.post(url, json={
                "model": args.model, "prompt": rows, "max_tokens": 1,
                "logprobs": 1, "return_hidden_states": True,
            }, timeout=args.timeout)
            resp.raise_for_status()
            choices = sorted(resp.json()["choices"], key=lambda c: c.get("index", 0))
        except Exception as e:                     # the server is shared with a
            print(f"  [skip {n}] {type(e).__name__}: {e}", flush=True)
            continue

        for j, ch in zip(js, choices):
            hs = ch.get("hidden_states") or []
            by_layer = {}
            for h in hs:
                v = np.frombuffer(base64.b64decode(h["data"]), dtype=np.float32)
                by_layer[int(h["layer"])] = v
            if len(by_layer) < L:
                missing += 1
                continue
            H = np.stack([by_layer[l] for l in range(L)])         # [L, d]
            cos = (H * D).sum(1) / (np.linalg.norm(H, axis=1) + 1e-8)
            scores.append(float((wn * cos).sum() + b / max(int(j), 1)))
            depths.append(int(j))
        if (n + 1) % 50 == 0:
            print(f"[{n+1}/{len(records)}] {len(scores)} states", flush=True)

    if not scores:
        raise SystemExit("no states collected -- is the server up and hooking all layers?")
    s = np.asarray(scores)
    gs = json.load(open(args.gate_stats))
    gs["offline_mean"] = gs.get("combined_mean")
    gs["offline_std"] = gs.get("combined_std")
    gs["combined_mean"], gs["combined_std"] = float(s.mean()), float(s.std())
    gs["serve_calibration"] = {
        "n_states": int(s.size), "n_records": len(records),
        "mean_depth": float(np.mean(depths)), "sum_abs_w": total,
        "missing_layer_rows": int(missing),
    }
    # Atomic: a `gated` arm may start reading this file at any moment, and a
    # half-written gate_stats.json is a crashed run.
    import os
    tmp = args.gate_stats + ".tmp"
    json.dump(gs, open(tmp, "w"), indent=1)
    os.replace(tmp, args.gate_stats)
    print(f"  served mu={s.mean():+.6f} sd={s.std():.6f} (n={s.size}, sum|w|={total:.3f})")
    print(f"  theta=0 passes {(s >= s.mean()).mean():.1%} of served prefix states")
    print(f"  -> {args.gate_stats}")


if __name__ == "__main__":
    main()
