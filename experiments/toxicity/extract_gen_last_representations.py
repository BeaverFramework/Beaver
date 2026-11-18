"""Extract the last-token (prompt + completion) representation, per sample.

Unlike the prompt-only representation (extract_prompt_representations.py),
this depends on the actual completion, so it's computed once per SAMPLE, not
once per prompt. Matches the paper's y = generated completion, r^(l)(y) =
residual stream reading it after each layer (see arXiv:2607.20286 Eq. 3 --
D_harmful(x)/D_harmless(x) are full "responses", not prompts).

Example:
    python -m experiments.toxicity.extract_gen_last_representations \
        --samples logging/tox_samples_500x10.jsonl \
        --model meta-llama/Llama-3.2-3B-Instruct \
        --server_addr http://localhost:8112 \
        --out logging/tox_gen_last_reps_5000.npz
"""

import argparse
import base64
import json
import time

import numpy as np
import requests
from transformers import AutoTokenizer


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--samples", required=True, help="JSONL from sample_completions.py")
    ap.add_argument("--model", required=True)
    ap.add_argument("--server_addr", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_gen_tokens", type=int, default=64, help="truncate generation before appending")
    args = ap.parse_args()

    records = [json.loads(l) for l in open(args.samples)]
    n = len(records)
    print(f"{n} samples")

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    token_ids = []
    for r in records:
        prompt_ids = tokenizer.encode(r["exact_prompt"], add_special_tokens=False)
        gen_ids = tokenizer.encode(r["generation"], add_special_tokens=False)[: args.max_gen_tokens]
        token_ids.append(prompt_ids + gen_ids)

    url = f"{args.server_addr}/v1/completions"
    all_reps = None
    layer_order = None
    t0 = time.time()

    for start in range(0, n, args.batch_size):
        batch = token_ids[start : start + args.batch_size]
        resp = requests.post(
            url,
            json={
                "model": args.model,
                "prompt": batch,
                "max_tokens": 1,
                "return_hidden_states": True,
            },
            timeout=180,
        )
        resp.raise_for_status()
        data = resp.json()
        for row in range(len(batch)):
            i = start + row
            hs_raw = data["choices"][row]["hidden_states"]
            if layer_order is None:
                layer_order = sorted(h["layer"] for h in hs_raw)
                n_layers = len(layer_order)
                dim = None
            layer_vecs = {}
            for h in hs_raw:
                b = base64.b64decode(h["data"])
                v = np.frombuffer(b, dtype=np.float32).reshape(h["shape"])
                layer_vecs[h["layer"]] = v
                if dim is None:
                    dim = v.shape[-1]
            if all_reps is None:
                all_reps = np.zeros((n, n_layers, dim), dtype=np.float32)
            for li, layer in enumerate(layer_order):
                all_reps[i, li, :] = layer_vecs[layer]
        done = start + len(batch)
        if done % (args.batch_size * 5) == 0 or done == n:
            print(f"[{done}/{n}] elapsed={time.time()-t0:.0f}s")

    y = np.array([0 if r["is_secure"] else 1 for r in records], dtype=np.int64)
    idx = np.array([r["idx"] for r in records])

    def _int_field(name):
        # Only toxicity carries template_id/task_idx; other datasets (e.g.
        # secure_code, whose labels come from a local static analyser) have
        # neither. -1 keeps the array integer-typed so the confound analysis
        # in compute_gen_last_direction.py still runs -- it just finds one
        # degenerate group instead of erroring on an object array of None.
        return np.array(
            [(-1 if r.get(name) is None else int(r[name])) for r in records],
            dtype=np.int64,
        )

    saved = dict(
        reps=all_reps,  # [n_samples, n_layers, d_model], hf_server layer order
        hf_server_layers=np.array(layer_order),
        y=y,
        idx=idx,
        template_id=_int_field("template_id"),
        task_idx=_int_field("task_idx"),
    )
    # Toxicity-only extras; absent for every other dataset.
    if records and "toxicity_score" in records[0]:
        saved["toxicity_score"] = np.array(
            [r["toxicity_score"] for r in records], dtype=np.float32
        )
        saved["severe_toxicity_score"] = np.array(
            [r["severe_toxicity_score"] for r in records], dtype=np.float32
        )

    np.savez_compressed(args.out, **saved)
    print(f"\nwrote {args.out}  reps shape={all_reps.shape}")


if __name__ == "__main__":
    main()
