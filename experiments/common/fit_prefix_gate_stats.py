"""Re-reference the gate to the states it will actually score: PREFIXES.

The gate compares (score - mu)/sd against theta, and mu/sd were being taken
from the direction-fitting set -- the last token of a *completed* 32/64-token
generation. But the frontier never scores a completed generation. It scores a
partial prefix, mostly a shallow one, whose last-token residual is still
dominated by the prompt. Measured on qwen3_4b/toxicity those two populations
are ~1 sd apart: every search-time candidate landed at z in [0.81, 1.36], so
theta=0 passed 100% of them and `gated` came out identical to `max-mu`. The
same mismatch, in the same direction, is what made the earlier fit-set theta
(-1.199) inert.

The weights are not the problem and are not touched -- LDA still needs labels,
and only completions carry them. Only the reference point moves: mu/sd are
recomputed over prefix states, which need no labels at all. theta=0 then means
"above the average prefix", which is what a fixed threshold should mean.

Positions are sampled uniformly inside the generated span, which matches a
frontier holding candidates at many depths. Scores are accumulated on the fly,
so this stores nothing and costs one forward pass per sampled record.
"""

import argparse
import json

import numpy as np
import torch

_HOOKS = {"resid_post": "hook_resid_post", "mlp_post": "mlp.hook_post"}


def load_tl(model_name, tl_recipe, dtype, n_devices, tok):
    """Load the checkpoint under test into TransformerLens.

    TransformerLens keys everything off a registered model name. For a
    checkpoint it does not know (Qwen3-4B-Instruct-2507), the previous code
    passed the recipe name in place of the model -- which silently loaded the
    RECIPE's weights. The qwen3_4b directions were therefore fitted on
    Qwen3-4B *base* activations and then scored against Qwen3-4B-Instruct-2507
    states at search time: a different network. Passing hf_model keeps the
    recipe for the architecture/config and the real checkpoint for the weights.
    """
    from transformer_lens import HookedTransformer
    kw = {}
    if tl_recipe:
        from transformers import AutoModelForCausalLM
        kw["hf_model"] = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype
        )
        kw["tokenizer"] = tok
    m = HookedTransformer.from_pretrained(
        tl_recipe or model_name, device="cuda", dtype=dtype,
        n_devices=n_devices, **kw
    )
    m.eval()
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--samples", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--directions", required=True)
    ap.add_argument("--lda", required=True)
    ap.add_argument("--gate-stats", required=True)
    ap.add_argument("--hook", default="resid_post", choices=sorted(_HOOKS))
    ap.add_argument("--n_records", type=int, default=500)
    ap.add_argument("--k_positions", type=int, default=8)
    ap.add_argument("--max_gen_tokens", type=int, default=32)
    ap.add_argument("--tl_recipe", default=None)
    ap.add_argument("--n_devices", type=int, default=1)
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    args = ap.parse_args()

    w = np.asarray(json.load(open(args.lda))["weights"], dtype=np.float32)
    dz = np.load(args.directions)
    D = dz[dz.files[0]][1:]
    D = D / (np.linalg.norm(D, axis=1, keepdims=True) + 1e-8)
    L = D.shape[0]
    assert len(w) == L, f"{len(w)} weights vs {L} directions"

    records = [json.loads(l) for l in open(args.samples)]
    rng = np.random.RandomState(0)
    if len(records) > args.n_records:
        records = [records[i] for i in rng.choice(len(records), args.n_records, False)]

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    model = load_tl(args.model, args.tl_recipe, getattr(torch, args.dtype),
                    args.n_devices, tok)
    suffix = _HOOKS[args.hook]
    names = [f"blocks.{l}.{suffix}" for l in range(L)]
    Dt = torch.tensor(D, device="cuda", dtype=torch.float32)
    wt = torch.tensor(w, device="cuda", dtype=torch.float32)

    scores, depths = [], []
    with torch.no_grad():
        for n, r in enumerate(records):
            p_ids = tok.encode(r["exact_prompt"], add_special_tokens=False)
            g_ids = tok.encode(r["generation"], add_special_tokens=False)[: args.max_gen_tokens]
            if len(g_ids) < 2:
                continue
            ids = p_ids + g_ids
            # positions inside the generated span only: index len(p_ids)-1+j is
            # the state after j generated tokens, j = 1..len(g_ids)
            js = rng.choice(len(g_ids), min(args.k_positions, len(g_ids)), False) + 1
            pos = [len(p_ids) - 1 + int(j) for j in js]
            _, cache = model.run_with_cache(
                torch.tensor([ids], device="cuda"), names_filter=names,
                stop_at_layer=model.cfg.n_layers,   # skip the 151k-wide unembed
            )
            H = torch.stack([cache[nm][0] for nm in names])          # [L, T, d]
            del cache
            h = H[:, pos, :].float()                                  # [L, K, d]
            cos = torch.einsum("lkd,ld->lk", h, Dt) / (h.norm(dim=-1) + 1e-8)
            scores.extend((wt @ cos).cpu().numpy().tolist())          # [K]
            depths.extend(int(j) for j in js)
            if (n + 1) % 100 == 0:
                print(f"[{n+1}/{len(records)}] {len(scores)} states", flush=True)

    s = np.asarray(scores, dtype=np.float64)
    gs = json.load(open(args.gate_stats))
    old_mu = gs.get("combined_mean")
    old_sd = gs.get("combined_std")
    gs["completion_mean"], gs["completion_std"] = old_mu, old_sd
    gs["combined_mean"], gs["combined_std"] = float(s.mean()), float(s.std())
    gs["prefix_calibration"] = {
        "n_states": int(s.size),
        "n_records": len(records),
        "k_positions": args.k_positions,
        "mean_depth": float(np.mean(depths)),
    }
    json.dump(gs, open(args.gate_stats, "w"), indent=1)

    shift = (s.mean() - old_mu) / old_sd if old_sd else float("nan")
    print(f"  prefix  mu={s.mean():+.5f} sd={s.std():.5f}  (n={s.size})")
    print(f"  compl.  mu={old_mu:+.5f} sd={old_sd:.5f}")
    print(f"  shift   {shift:+.2f} sd  -- that is what made theta=0 inert")
    print(f"  theta=0 now passes {(s >= s.mean()).mean():.1%} of prefix states")


if __name__ == "__main__":
    main()
