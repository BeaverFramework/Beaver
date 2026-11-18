"""Extract the last-prompt-token representation at every layer, once per prompt.

Reads a sample file written by sample_completions.py, takes the unique
exact_prompt per `idx` (this representation is prompt-only -- it doesn't
depend on which completion was sampled, so it's computed once per prompt,
not once per sample), and queries an hf_server instance (started with
--hidden-state-layers covering every layer) for the hidden state at the
prompt's last token, at every layer.

Example:
    python -m experiments.toxicity.extract_prompt_representations \
        --samples logging/tox_samples_500x10.jsonl \
        --model meta-llama/Llama-3.2-3B-Instruct \
        --server_addr http://localhost:8112 \
        --out logging/tox_prompt_reps_500.npz
"""

import argparse
import base64
import json

import numpy as np
import requests
from transformers import AutoTokenizer


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--samples", required=True, help="JSONL from sample_completions.py")
    ap.add_argument("--model", required=True)
    ap.add_argument("--server_addr", required=True, help="hf_server base URL, e.g. http://localhost:8112")
    ap.add_argument("--out", required=True, help="output .npz path")
    ap.add_argument("--batch_size", type=int, default=16)
    args = ap.parse_args()

    records = [json.loads(l) for l in open(args.samples)]
    by_idx = {}
    for r in records:
        by_idx.setdefault(r["idx"], r)  # first occurrence -> same exact_prompt for all samples of this idx
    idxs = sorted(by_idx)
    print(f"{len(records)} samples -> {len(idxs)} unique prompts")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    token_ids = {i: tokenizer.encode(by_idx[i]["exact_prompt"], add_special_tokens=False) for i in idxs}

    url = f"{args.server_addr}/v1/completions"
    all_reps = None  # [n_prompts, n_layers, dim], filled in per batch
    layer_order = None

    for start in range(0, len(idxs), args.batch_size):
        batch_idxs = idxs[start : start + args.batch_size]
        prompt_batch = [token_ids[i] for i in batch_idxs]
        resp = requests.post(
            url,
            json={
                "model": args.model,
                "prompt": prompt_batch,
                "max_tokens": 1,
                "return_hidden_states": True,
            },
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        for row, i in enumerate(batch_idxs):
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
                all_reps = np.zeros((len(idxs), n_layers, dim), dtype=np.float32)
            for li, layer in enumerate(layer_order):
                all_reps[idxs.index(i), li, :] = layer_vecs[layer]
        done = start + len(batch_idxs)
        print(f"[{done}/{len(idxs)}] prompts extracted")

    template_ids = np.array([by_idx[i].get("template_id") for i in idxs])
    task_idxs = np.array([by_idx[i].get("task_idx") for i in idxs])

    np.savez_compressed(
        args.out,
        idx=np.array(idxs),
        reps=all_reps,  # [n_prompts, 28, 3072], hf_server layer indices 0..27 (heretic index - 1)
        hf_server_layers=np.array(layer_order),
        template_id=template_ids,
        task_idx=task_idxs,
    )
    print(f"\nwrote {args.out}  reps shape={all_reps.shape}")


if __name__ == "__main__":
    main()
