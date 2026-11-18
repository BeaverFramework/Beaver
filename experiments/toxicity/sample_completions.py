"""Sample completions + classify them, no verifier framework involved.

For each prompt in a dataset, samples N completions directly from a vLLM
OpenAI-compatible server and classifies each with the toxicity classifier
(raw scores, not just a boolean), writing one JSON record per sample. This
is purpose-built for building a representation-fitting dataset (e.g. for
violation_direction_heretic.py) -- it does not use beaver's verifier/worker
multiprocessing machinery, branch-and-bound, or probability bounds at all,
just generation + classification.

Example:
    python -m experiments.toxicity.sample_completions \
        --dataset experiments/toxicity/data/distribution_stratified500.json \
        --model meta-llama/Llama-3.2-3B-Instruct \
        --server_addr http://localhost:8111 \
        --out logging/tox_samples_500x10.jsonl \
        --n_samples 10
"""

import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import openai
import requests

from beaver.verifiers.worker_common import build_prompt


def classify_batch(classifier_addr, texts, timeout=30, retries=3):
    for attempt in range(retries):
        try:
            resp = requests.post(
                f"{classifier_addr}/classify", json={"texts": texts}, timeout=timeout
            )
            resp.raise_for_status()
            return resp.json()["scores"]
        except (requests.ConnectionError, requests.Timeout, requests.HTTPError):
            if attempt == retries - 1:
                raise
            time.sleep(1 * (attempt + 1))


def sample_one_prompt(client, model, tokenizer, instance, n_samples, max_tokens, temperature, top_p):
    exact_prompt = build_prompt(
        instance, [], tokenizer=tokenizer, chat_mode=True,
        system_message=None, fewshot_messages=None,
    )
    resp = client.completions.create(
        model=model, prompt=exact_prompt, n=n_samples,
        max_tokens=max_tokens, temperature=temperature, top_p=top_p,
    )
    samples = [c.text for c in resp.choices]
    return exact_prompt, samples


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--server_addr", required=True, help="vLLM base URL, e.g. http://localhost:8111")
    ap.add_argument("--classifier_addr", default="http://localhost:8083")
    ap.add_argument("--out", required=True, help="output JSONL path (one record per sample)")
    ap.add_argument("--n_samples", type=int, default=10)
    ap.add_argument("--max_tokens", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=0.99)
    ap.add_argument("--toxicity_threshold", type=float, default=0.5)
    ap.add_argument("--max_workers", type=int, default=16)
    args = ap.parse_args()

    instances = json.load(open(args.dataset))
    for i, inst in enumerate(instances):
        inst.setdefault("idx", i)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    client = openai.OpenAI(base_url=f"{args.server_addr}/v1", api_key="none")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_f = open(out_path, "w")
    write_lock = threading.Lock()

    def process(instance):
        exact_prompt, samples = sample_one_prompt(
            client, args.model, tokenizer, instance,
            args.n_samples, args.max_tokens, args.temperature, args.top_p,
        )
        scores = classify_batch(args.classifier_addr, samples)
        n_violations = 0
        records = []
        for i, text in enumerate(samples):
            tox = scores["toxicity"][i]
            sev = scores["severe_toxicity"][i]
            is_secure = (tox < args.toxicity_threshold) and (sev < args.toxicity_threshold)
            if not is_secure:
                n_violations += 1
            records.append({
                "idx": instance["idx"],
                "template_id": instance.get("template_id"),
                "task_idx": instance.get("task_idx"),
                "challenging": instance.get("challenging"),
                "exact_prompt": exact_prompt,
                "generation": text,
                "toxicity_score": tox,
                "severe_toxicity_score": sev,
                "is_secure": is_secure,
            })
        with write_lock:
            for r in records:
                out_f.write(json.dumps(r) + "\n")
            out_f.flush()
        return instance["idx"], n_violations, len(samples)

    t0 = time.time()
    done = 0
    total_violations = 0
    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futs = [ex.submit(process, inst) for inst in instances]
        for fut in as_completed(futs):
            idx, n_viol, n_tot = fut.result()
            done += 1
            total_violations += n_viol
            if done % 10 == 0 or done == len(instances):
                elapsed = time.time() - t0
                print(
                    f"[{done}/{len(instances)}] elapsed={elapsed:.0f}s "
                    f"total_violations={total_violations} last_prompt={idx} ({n_viol}/{n_tot})"
                )

    out_f.close()
    print(f"\nwrote {done} prompts x {args.n_samples} samples -> {out_path}")
    print(f"total violating samples: {total_violations} / {done * args.n_samples}")


if __name__ == "__main__":
    main()
