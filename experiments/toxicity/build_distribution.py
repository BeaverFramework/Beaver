"""Build a fixed distribution of toxicity prompts and write to JSON.

Cross-product of every system-prompt template in data/toxicity_prompts.py
(PROMPTS list) with every toxic task prompt in data/toxic.jsonl (or a subset
like toxic_subset318.jsonl). Each emitted entry records both the
``template_id`` (index into PROMPTS) and ``task_idx`` (index into the jsonl)
so downstream analysis can slice by system prompt or by task.

Examples:
    # Full distribution: every template × every task in toxic_subset318.jsonl
    python build_distribution.py --output distribution_full.json

    # Same but against the full toxic.jsonl (larger)
    python build_distribution.py --output distribution_all.json \\
        --dataset_path data/toxic.jsonl

    # Uniform-sampled 100 entries from the full cross product
    python build_distribution.py --output distribution_uniform100.json \\
        --num_seqs 100
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from collections import Counter
from pathlib import Path

import numpy as np

_THIS_DIR = Path(__file__).parent
_DATA_DIR = _THIS_DIR / "data"
_DEFAULT_DATASET_PATH = _DATA_DIR / "toxic.jsonl"
_PROMPTS_PATH = _DATA_DIR / "toxicity_prompts.py"


def _load_templates() -> list[dict]:
    spec = importlib.util.spec_from_file_location("toxicity_prompts", _PROMPTS_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.PROMPTS


def _load_tasks(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def build_distribution(
    dataset_path: Path,
    template_ids: list[int] | None = None,
    task_start_idx: int = 0,
    task_end_idx: int = -1,
) -> list[dict]:
    templates = _load_templates()
    tasks = _load_tasks(dataset_path)

    if template_ids is None:
        template_ids = list(range(len(templates)))

    end = task_end_idx if task_end_idx != -1 else len(tasks)
    task_range = range(task_start_idx, end)

    result = []
    for tid in template_ids:
        tmpl = templates[tid]
        for task_idx in task_range:
            item = tasks[task_idx]
            result.append(
                {
                    "prompt": tmpl["task_desc"] + item["prompt"]["text"],
                    "system_prompt": tmpl["sys_prompt"],
                    "template_id": tid,
                    "task_idx": task_idx,
                    "task_text": item["prompt"]["text"],
                    "challenging": item.get("challenging", False),
                }
            )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build a fixed JSON distribution of toxicity prompts.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSON filename (written to experiments/toxicity/data/).",
    )
    parser.add_argument(
        "--dataset_path",
        default=str(_DEFAULT_DATASET_PATH),
        help="Path to toxic jsonl (default: data/toxic_subset318.jsonl).",
    )
    parser.add_argument(
        "--template_ids",
        type=str,
        default=None,
        help="Comma-separated subset of template ids. Default: all templates in PROMPTS.",
    )
    parser.add_argument("--task_start_idx", type=int, default=0)
    parser.add_argument("--task_end_idx", type=int, default=-1)
    parser.add_argument(
        "--num_seqs",
        type=int,
        default=None,
        help="If set, uniformly sample this many entries from the full cross product.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    _DATA_DIR.mkdir(exist_ok=True)
    out_path = _DATA_DIR / args.output

    template_ids = (
        [int(x) for x in args.template_ids.split(",")]
        if args.template_ids
        else None
    )

    prompts = build_distribution(
        dataset_path=Path(args.dataset_path),
        template_ids=template_ids,
        task_start_idx=args.task_start_idx,
        task_end_idx=args.task_end_idx,
    )

    if args.num_seqs is not None and args.num_seqs < len(prompts):
        rng = np.random.RandomState(args.seed)
        indices = rng.choice(len(prompts), size=args.num_seqs, replace=False)
        prompts = [prompts[i] for i in sorted(indices)]
        print(f"Sampled {len(prompts)} from full distribution")

    with open(out_path, "w") as f:
        json.dump(prompts, f, indent=2)

    tmpl_counts = Counter(p["template_id"] for p in prompts)
    print(f"Wrote {len(prompts)} prompts to {out_path}")
    print(f"  unique template_ids: {len(tmpl_counts)}")
    print(f"  unique task_idx:     {len({p['task_idx'] for p in prompts})}")
