"""Extract last-token activations with the in-process vLLM backend.

Why this exists alongside extract_activations.py
------------------------------------------------
extract_activations.py reads activations through TransformerLens, which is the
same backend beaver's ``hf-local`` uses -- so what gets fitted matches what
gets scored *only if the run also uses hf/hf-local*. A run on ``--backend
vllm-local`` scores against vLLM's own residual stream instead, and the two
are not interchangeable in practice:

  * TransformerLens keys off a *registered* model name. For a checkpoint it
    does not know (Qwen3-4B-Instruct-2507) it needs ``--tl_recipe``, and
    getting that wrong silently loads the recipe's weights instead of the
    checkpoint's -- the exact bug documented in extract_activations.py's
    ``load_tl`` docstring, which is how the qwen3_4b directions once came to
    be fitted on Qwen3-4B *base* activations.
  * vLLM indexes aux hidden states by the *input* to block N, beaver by the
    *output* of block N. ``local_vllm._vllm_capture_layers`` owns that +1
    shift. Re-deriving it here would be a second place to get it wrong.

So this script does not talk to vLLM directly. It sets up the same ``_w``
worker state ``api.py`` builds for a ``vllm-local`` run and calls
``local_vllm.completion(..., return_hidden_states=True)`` -- the identical
call the representation verifier makes at search time, layer shift and all.
Fitting and scoring then read the same numbers by construction.

Output npz schema is byte-for-byte the one extract_activations.py writes, so
compute_gen_last_direction.py, fit_layer_weights.py, fit_gate_stats.py and
fit_lda_direction.py consume it unchanged.

Only ``resid_post`` is available here: vLLM's ``extract_hidden_states`` method
captures the residual stream, and has no equivalent of TransformerLens's
``mlp.hook_post``. Use extract_activations.py for MLP neurons.

Example:
    python -m experiments.common.extract_activations_vllm \\
        --samples repr_out/qwen3_4b/secure_code/samples.jsonl \\
        --model Qwen/Qwen3-4B-Instruct-2507 \\
        --out repr_out/qwen3_4b/secure_code/gen_last_reps.npz \\
        --max_model_len 4096 --gpu_memory_utilization 0.75
"""

import argparse
import base64
import json
import time

import numpy as np


def _decode(hs_entry):
    """One ``{"layer","data","shape","dtype"}`` dict -> float32 vector.

    ``local_vllm._b64`` always emits little-endian float32 regardless of the
    engine dtype, so the decode is fixed rather than read from the entry.
    """
    assert hs_entry["dtype"] == "float32", hs_entry["dtype"]
    return np.frombuffer(base64.b64decode(hs_entry["data"]), dtype="<f4")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--samples", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_gen_tokens", type=int, default=64)
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    ap.add_argument("--max_model_len", type=int, default=4096)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.75)
    # The connector clones [num_tokens, n_layers*d] per in-flight request and
    # holds it on GPU until popped. At 36 layers x 2560 x fp32 that is ~370 MB
    # per 1000-token prompt, so the batch is deliberately small; it is popped
    # down to just the last token as soon as each batch returns.
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--layers", default=None,
                    help="comma-separated; default all. Must be contiguous "
                         "from 0 -- see the guard below.")
    ap.add_argument("--allow_subset", action="store_true")
    args = ap.parse_args()

    records = [json.loads(l) for l in open(args.samples)]
    print(f"{len(records)} samples", flush=True)

    from transformers import AutoConfig, AutoTokenizer

    cfg = AutoConfig.from_pretrained(args.model)
    n_layers = cfg.num_hidden_layers
    layers = (
        [int(x) for x in args.layers.split(",")]
        if args.layers
        else list(range(n_layers))
    )

    # Same guard, same reason, as extract_activations.py: the direction stack
    # is indexed positionally and DirectionNVF reads stack row i+1 as layer i,
    # so a subset that is not exactly 0..k-1 is scored against the WRONG
    # layers with no error raised anywhere.
    if layers != list(range(len(layers))) and not args.allow_subset:
        raise SystemExit(
            f"--layers {layers} is not contiguous from 0. The direction stack "
            f"is indexed positionally, so DirectionNVF would map these to "
            f"layers {list(range(len(layers)))}. Extract all layers, or pass "
            f"--allow_subset and give the run matching --direction_layers."
        )

    # ── Build the worker state a vllm-local run would have ────────────────
    # api.py populates these from --direction_layers/--probe_layer; here the
    # caller has said which layers directly. local_vllm reads them off _w.
    from beaver.model import local_vllm
    from beaver.verifiers.worker_common import _w

    _w.model_name = args.model
    _w.tokenizer = AutoTokenizer.from_pretrained(args.model)
    _w.vllm_hidden_layers = list(layers)
    _w.vllm_dtype = args.dtype
    _w.vllm_max_model_len = args.max_model_len
    _w.vllm_gpu_memory_utilization = args.gpu_memory_utilization
    # Scoring, not warm-starting: every sequence here is distinct, so there is
    # no shared-prefix antichain to preserve and the default read path is fine.
    _w.vllm_skip_prefix_cache_read = False

    def _no_timing(*a, **k):
        pass

    dim = cfg.hidden_size
    print(f"hook=resid_post layers={layers} dim={dim} (vllm-local)", flush=True)

    reps = np.zeros((len(records), len(layers), dim), dtype=np.float32)
    tok = _w.tokenizer
    t0 = time.time()

    for start in range(0, len(records), args.batch_size):
        chunk = records[start : start + args.batch_size]
        # Identical construction to extract_activations.py: the exact prompt
        # the sample was drawn under, plus its generation truncated to the
        # same budget the search runs under.
        rows = [
            tok.encode(r["exact_prompt"], add_special_tokens=False)
            + tok.encode(r["generation"], add_special_tokens=False)[: args.max_gen_tokens]
            for r in chunk
        ]
        choices = local_vllm.completion(
            rows,
            max_tokens=1,
            logprobs=None,
            echo=False,
            temperature=0.0,
            top_p=1.0,
            top_k=-1,
            return_hidden_states=True,
            label="extract",
            record_timing=_no_timing,
        )
        for j, ch in enumerate(choices):
            hs = {e["layer"]: _decode(e) for e in ch.hidden_states}
            for li, layer in enumerate(layers):
                reps[start + j, li] = hs[layer]

        done = start + len(chunk)
        if done % 250 < args.batch_size or done == len(records):
            print(f"[{done}/{len(records)}] {time.time() - t0:.0f}s", flush=True)

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
        hook="resid_post",
    )
    print(f"wrote {args.out} reps={reps.shape}")


if __name__ == "__main__":
    main()
