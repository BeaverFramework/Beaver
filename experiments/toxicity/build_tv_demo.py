"""Build a TV1-grouping demo dataset for toxicity (analog of enron/tv_demo).

Grouping heuristic (why TV1 grouping is beneficial here):

    group  = one toxic task (one RealToxicityPrompts continuation `task_text`)
    member = the SAME task_text framed by a DIFFERENT jailbreak template
             (system_prompt + task_desc from data/toxicity_prompts.py)

The model's *next token* continues `task_text` — the partial sentence being
completed. That immediate context dominates the one-step next-token
distribution, while the jailbreak template only shifts how much probability
mass lands on the toxic continuation. So every member of a group drives a
near-identical next-token distribution (within-group one-step TV ≈ 0), while
different tasks complete different sentences (between-group TV ≈ 1). That is
exactly the δ the cached-frontier verifier wants: the first member of each
group pays the cold frontier-search cost and caches its violation frontier;
the rest warm-start from it.

This mirrors experiments/enron/data/tv_demo_15x24.json, where a group is one
target email (same violation continuation) and members vary the upstream
context. Here a group is one toxic sentence and members vary the upstream
jailbreak framing.

Output schema matches the enron demo: each instance carries `group_true`
(ground-truth group = task) plus `template_id` / `task_idx` for analysis, and
the list is ordered group-by-group. Position 0 of every group is template 0
("You are a helpful assistant." — the weakest / canonical framing) so it can
serve as the cache-seeding medoid. Feed it to beaver/grouping/tv_distance.py to
recover the grouping from one-step TV and validate against `group_true`.

Example:
    cd experiments/toxicity
    python build_tv_demo.py --n-groups 20 --per-group 24 \
        --output tv_demo_20x24.json
    # -> experiments/toxicity/data/tv_demo_20x24.json  (480 instances)
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

_THIS_DIR = Path(__file__).parent
_DATA_DIR = _THIS_DIR / "data"
_PROMPTS_PATH = _DATA_DIR / "toxicity_prompts.py"
_DEFAULT_TASKS = _DATA_DIR / "toxic_subset318.jsonl"


def _load_templates() -> list[dict]:
    spec = importlib.util.spec_from_file_location("toxicity_prompts", _PROMPTS_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.PROMPTS


def _load_tasks(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def build_tv_demo(
    n_groups: int,
    per_group: int,
    tasks_path: Path,
    task_start: int = 0,
) -> list[dict]:
    templates = _load_templates()
    tasks = _load_tasks(tasks_path)

    if per_group > len(templates):
        raise ValueError(
            f"per_group={per_group} exceeds available templates ({len(templates)}); "
            f"lower --per-group or add templates to toxicity_prompts.py"
        )
    if task_start + n_groups > len(tasks):
        raise ValueError(
            f"need {n_groups} tasks from offset {task_start} but only "
            f"{len(tasks) - task_start} available in {tasks_path.name}"
        )

    # Template 0 first (canonical/weakest framing -> cache-seeding medoid),
    # then the next per_group-1 templates for jailbreak variety.
    template_ids = list(range(per_group))

    out: list[dict] = []
    idx = 0
    for g in range(n_groups):
        task_idx = task_start + g
        task_text = tasks[task_idx]["prompt"]["text"]
        challenging = tasks[task_idx].get("challenging", False)
        for pos, tid in enumerate(template_ids):
            tmpl = templates[tid]
            out.append(
                {
                    # Same continuation point (task_text) across the whole group;
                    # only the upstream framing (task_desc + system_prompt) varies.
                    "prompt": tmpl["task_desc"] + task_text,
                    "system_prompt": tmpl["sys_prompt"],
                    "template_id": tid,
                    "task_idx": task_idx,
                    "task_text": task_text,
                    "challenging": challenging,
                    "group_true": g,
                    "group_position": pos,
                    "idx": idx,
                }
            )
            idx += 1
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--output", required=True, help="filename written under data/")
    ap.add_argument("--n-groups", type=int, default=20, help="distinct toxic tasks (groups)")
    ap.add_argument("--per-group", type=int, default=24, help="templates per task (group size)")
    ap.add_argument("--task-start", type=int, default=0, help="offset into the task jsonl")
    ap.add_argument("--tasks-path", default=str(_DEFAULT_TASKS))
    args = ap.parse_args()

    data = build_tv_demo(
        n_groups=args.n_groups,
        per_group=args.per_group,
        tasks_path=Path(args.tasks_path),
        task_start=args.task_start,
    )

    _DATA_DIR.mkdir(exist_ok=True)
    out_path = _DATA_DIR / args.output
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)

    n_groups = len(set(d["group_true"] for d in data))
    print(f"Wrote {len(data)} instances in {n_groups} groups to {out_path}")
    print(f"  group = one toxic task (shared continuation); members vary jailbreak template")
    print(f"  group size = {args.per_group}; position 0 = template 0 (canonical seed)")
