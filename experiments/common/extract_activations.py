"""Extract last-token activations with TransformerLens, in-process.

extract_gen_last_representations.py POSTs to beaver.hf_server, which only
hooks the residual stream (a plain forward hook on model.model.layers[i]).
To read MLP neurons you need TransformerLens's named hooks, so this does the
forward passes locally instead of over HTTP -- the same backend beaver's
"hf-local" uses, so what gets fitted matches what gets scored at search time.

  --hook resid_post   blocks.{l}.hook_resid_post   [d_model]  (== hf_server)
  --hook mlp_post     blocks.{l}.mlp.hook_post     [d_mlp]    MLP neurons

Output npz matches extract_gen_last_representations.py, so
compute_gen_last_direction.py and fit_layer_weights.py consume it unchanged.

d_mlp is 8192 for Llama-3.2-3B vs d_model 3072, so all 28 MLP layers over
5000 samples is ~4.6 GB. Extract every layer unless you also pass matching
--direction_layers at run time: the fitted stack is indexed positionally, so
a gap in --layers silently shifts which layer each direction is scored
against (see the --allow_subset guard below).

Example:
    python -m experiments.common.extract_activations \
        --samples repr_out/llama32_3b/toxicity/samples_500x10.jsonl \
        --model meta-llama/Llama-3.2-3B-Instruct \
        --hook mlp_post \
        --out repr_out/llama32_3b/toxicity/mlp_reps.npz
"""

import argparse
import json
import time

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
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--samples", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--hook", default="resid_post", choices=sorted(_HOOKS))
    ap.add_argument("--layers", default=None, help="comma-separated; default all")
    ap.add_argument("--max_gen_tokens", type=int, default=64)
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    ap.add_argument("--tl_recipe", default=None)
    # 14B in fp32 is ~59 GB and does not fit one 40 GB card. TransformerLens can
    # shard the blocks across visible devices; the hook outputs still come back
    # on cuda:0, so nothing downstream changes.
    ap.add_argument("--n_devices", type=int, default=1)
    ap.add_argument("--allow_subset", action="store_true",
                    help="permit a non-contiguous --layers subset (see below)")
    args = ap.parse_args()

    records = [json.loads(l) for l in open(args.samples)]
    print(f"{len(records)} samples")

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    model = load_tl(args.model, args.tl_recipe, getattr(torch, args.dtype),
                    args.n_devices, tok)

    layers = (
        [int(x) for x in args.layers.split(",")]
        if args.layers
        else list(range(model.cfg.n_layers))
    )
    # compute_gen_last_direction.py stacks layers positionally and prepends one
    # zero row, and DirectionNVF then reads stack row i+1 as layer i. So a
    # layer subset that is not exactly 0..k-1 would be scored against the
    # WRONG layers, silently. Refuse rather than produce a mis-indexed probe.
    if layers != list(range(len(layers))) and not args.allow_subset:
        raise SystemExit(
            f"--layers {layers} is not contiguous from 0. The direction stack "
            f"is indexed positionally, so DirectionNVF would map these to "
            f"layers {list(range(len(layers)))}. Extract all layers, or pass "
            f"--allow_subset and give the run matching --direction_layers."
        )

    suffix = _HOOKS[args.hook]
    names = [f"blocks.{l}.{suffix}" for l in layers]
    dim = model.cfg.d_model if args.hook == "resid_post" else model.cfg.d_mlp
    print(f"hook={args.hook} layers={layers} dim={dim}")

    reps = np.zeros((len(records), len(layers), dim), dtype=np.float32)
    t0 = time.time()
    with torch.no_grad():
        for i, r in enumerate(records):
            ids = tok.encode(r["exact_prompt"], add_special_tokens=False) + tok.encode(
                r["generation"], add_special_tokens=False
            )[: args.max_gen_tokens]
            t = torch.tensor([ids], device="cuda")
            # stop_at_layer skips the unembed: W_U is [d_model, 151k] for Qwen
            # and materialising logits costs more memory than the whole cache.
            _, cache = model.run_with_cache(
                t, names_filter=names, stop_at_layer=model.cfg.n_layers
            )
            for li, nm in enumerate(names):
                reps[i, li] = cache[nm][0, -1, :].float().cpu().numpy()
            del cache
            if (i + 1) % 250 == 0 or i + 1 == len(records):
                print(f"[{i+1}/{len(records)}] {time.time()-t0:.0f}s", flush=True)

    def _int_field(name):
        return np.array(
            [(-1 if r.get(name) is None else int(r[name])) for r in records],
            dtype=np.int64,
        )

    np.savez_compressed(
        args.out,
        reps=reps,
        hf_server_layers=np.array(layers),
        y=np.array([0 if r["is_secure"] else 1 for r in records], dtype=np.int64),
        idx=np.array([r["idx"] for r in records]),
        template_id=_int_field("template_id"),
        task_idx=_int_field("task_idx"),
        hook=args.hook,
    )
    print(f"wrote {args.out} reps={reps.shape}")


if __name__ == "__main__":
    main()
