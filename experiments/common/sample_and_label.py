"""Dataset-agnostic completion sampler + labeller for direction fitting.

experiments/toxicity/sample_completions.py does this for toxicity only: it
labels by POSTing to the detoxify classifier server, so it cannot be pointed
at secure_code (a local static analyser) or anything else. This script takes
the experiment YAML instead and labels with whatever ``constraint_fn`` that
experiment registers -- the same function the verifier itself calls, so the
fitted direction is trained against exactly the labels the search resolves.

Output schema matches sample_completions.py's, so
extract_gen_last_representations.py and compute_gen_last_direction.py consume
it unchanged. ``is_secure`` is the constraint's own verdict (True = satisfied
= no violation).

Example:
    python -m experiments.common.sample_and_label \
        --experiment experiments/secure_code/secure_code_new.yaml \
        --model meta-llama/Llama-3.2-3B-Instruct \
        --server_addr http://localhost:8113 \
        --out repr_out/llama32_3b/secure_code/samples_500x10.jsonl \
        --n_prompts 500 --n_samples 10 --max_tokens 64
"""

import argparse
import importlib.util
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import openai
import yaml

from beaver.verifiers.worker_common import build_prompt


def _import_module_from_path(path: Path):
    # The experiment dir must be importable: secure_code.py does
    # `from insecure_code_detector import ...`, a sibling package.
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_experiment(exp_yaml: Path, overrides=None):
    """Load the experiment's callables and prompts.

    ``overrides`` are merged over the YAML before load_prompts kwargs are
    introspected, so the sampler can point at a different dataset than the
    YAML's default. That default is toxic_subset318.jsonl, whose ``prompt``
    field is a nested dict -- sampling it hands the model a stringified dict
    and yields a 0% violation rate. Always pass --dataset_path explicitly.
    """
    cfg = yaml.safe_load(open(exp_yaml))
    if overrides:
        cfg.update({k: v for k, v in overrides.items() if v is not None})
    mod = _import_module_from_path(exp_yaml.parent / cfg["experiment_file"])
    load_prompts_fn = getattr(mod, cfg["load_prompts_fn"])
    constraint_fn = getattr(mod, cfg["constraint_fn"])
    import inspect

    kwargs = {
        k: cfg[k] for k in inspect.signature(load_prompts_fn).parameters if k in cfg
    }
    return cfg, load_prompts_fn(**kwargs), constraint_fn


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--server_addr", required=True, help="OpenAI-compatible base URL")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_prompts", type=int, default=500)
    ap.add_argument("--n_samples", type=int, default=10)
    ap.add_argument("--max_tokens", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=0.99)
    ap.add_argument("--max_workers", type=int, default=8)
    ap.add_argument("--dataset_path", default=None,
                    help="overrides the YAML's dataset; pass it explicitly")
    ap.add_argument("--template_id", type=int, default=None)
    args = ap.parse_args()

    cfg, prompts, constraint_fn = load_experiment(
        Path(args.experiment).resolve(),
        overrides={"dataset_path": args.dataset_path,
                   "template_id": args.template_id},
    )
    for i, inst in enumerate(prompts):
        inst.setdefault("idx", i)
    if args.n_prompts > 0:
        prompts = prompts[: args.n_prompts]
    print(f"{len(prompts)} prompts from {args.experiment}")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    client = openai.OpenAI(base_url=f"{args.server_addr}/v1", api_key="none")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_f = open(out_path, "w")

    def generate(instance):
        """Generation only -- safe to run on worker threads (pure HTTP)."""
        exact_prompt = build_prompt(
            instance, [], tokenizer=tokenizer, chat_mode=True,
            system_message=None, fewshot_messages=None,
        )
        resp = client.completions.create(
            model=args.model, prompt=exact_prompt, n=args.n_samples,
            max_tokens=args.max_tokens, temperature=args.temperature,
            top_p=args.top_p,
        )
        return instance, exact_prompt, [c.text for c in resp.choices]

    def label(instance, exact_prompt, texts):
        """Labelling -- MAIN THREAD ONLY.

        secure_code's constraint_fn arms a SIGALRM handler around its
        subprocess analysers (secure_code.py:130), and signal.signal() raises
        "signal only works in main thread of the main interpreter" anywhere
        else. Calling this from the generation pool silently failed every
        prompt, so generation is threaded and labelling is not.
        """
        # constraint_fn returns True where the constraint HOLDS (no violation).
        verdicts = list(constraint_fn(instance, texts))
        recs = [
            {
                "idx": instance["idx"],
                "template_id": instance.get("template_id"),
                "task_idx": instance.get("task_idx"),
                "language": instance.get("language"),
                "group": instance.get("group"),
                "exact_prompt": exact_prompt,
                "generation": t,
                "is_secure": bool(v),
            }
            for t, v in zip(texts, verdicts)
        ]
        for r in recs:
            out_f.write(json.dumps(r) + "\n")
        out_f.flush()
        return sum(1 for v in verdicts if not v), len(texts)

    t0, done, viol, failed = time.time(), 0, 0, 0
    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futs = [ex.submit(generate, p) for p in prompts]
        for fut in as_completed(futs):
            try:
                inst, prompt_text, texts = fut.result()
            except Exception as e:
                failed += 1
                print(f"  generation failed: {type(e).__name__}: {str(e)[:160]}")
                continue
            try:
                nv, nt = label(inst, prompt_text, texts)   # main thread
            except Exception as e:
                failed += 1
                print(f"  labelling failed: {type(e).__name__}: {str(e)[:160]}")
                continue
            done += 1
            viol += nv
            if done % 25 == 0 or done == len(prompts):
                print(
                    f"[{done}/{len(prompts)}] {time.time()-t0:.0f}s "
                    f"violations={viol}/{done*args.n_samples} failed={failed}"
                )
    out_f.close()
    if failed:
        print(f"WARNING: {failed}/{len(prompts)} prompts failed")
    print(f"\nwrote -> {out_path}")
    rate = viol / max(done * args.n_samples, 1)
    print(f"violation rate: {viol}/{done*args.n_samples} = {rate:.1%}")
    if viol == 0:
        raise SystemExit(
            "ABORT: zero violations across every sample. A direction cannot be "
            "fit from single-class labels. Usually the prompts are wrong -- "
            "check --dataset_path (the toxicity YAML defaults to "
            "toxic_subset318.jsonl, whose prompt field is a nested dict)."
        )


if __name__ == "__main__":
    main()
