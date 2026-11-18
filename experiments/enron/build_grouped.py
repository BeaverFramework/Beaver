"""Build a *grouped* Enron dataset for the cached-frontier verifier.

Grouping principle (structural, not proxy-clustered):

    group  = one target email (one base prompt)
    member = the SAME base prompt under a DIFFERENT system prompt (jailbreak)

Every instance in a group tries to leak the *same* email, so the violating
token sequence (the email itself) is identical across the group. The first
member's search caches that violation; the rest warm-start from it (often an
early-gate ``warm_skip``). Only the system prompt varies — it shifts how much
probability mass the model puts on leaking, not which sequence is the violation.

This is the same on-disk schema the proxy pipeline emits (each instance dict
gains ``group`` + ``group_position``, list ordered group-by-group), so it feeds
the cached-frontier verifier directly. Unlike the proxy pipeline it needs no GPU
and no model — groups are assigned by construction.

Example:
    cd experiments/enron
    python build_grouped.py --n-groups 10 --per-group 33 \
        --output grouped_enron_10x33.json
    # → experiments/enron/data/grouped_enron_10x33.json  (330 instances)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.enron.build_distribution import build_clean_prompts
from experiments.enron.privacy_jailbreak import MAIN_JB, gen_jb

_DATA_DIR = Path(__file__).parent / "data"


def make_system_prompts(m: int, crossover_prob: float, seed: int) -> list[str]:
    """Return ``m`` distinct jailbreak system prompts.

    Position 0 is the bare MAIN_JB template (the canonical, weakest framing) so
    it can serve as the cache-seeding medoid; positions 1..m-1 are genetic
    crossover variants. Deduped; ordered weakest→stronger is a reasonable
    default, but order only affects *which* member pays the cold cost first.
    """
    np.random.seed(seed)  # gen_jb uses the numpy global RNG
    seen = {MAIN_JB}
    prompts = [MAIN_JB]
    attempts = 0
    while len(prompts) < m and attempts < 100 * m:
        attempts += 1
        jb = gen_jb(tokenizer=None, crossover_prob=crossover_prob, mutation_prob=0.0)
        if jb not in seen:
            seen.add(jb)
            prompts.append(jb)
    if len(prompts) < m:
        raise RuntimeError(
            f"only generated {len(prompts)}/{m} distinct system prompts; "
            f"raise --crossover-prob or lower --per-group"
        )
    return prompts


def build_grouped(n_groups: int, per_group: int, crossover_prob: float, seed: int) -> list[dict]:
    # K distinct targets → K base prompts (clean: prompt only, no system prompt).
    bases = build_clean_prompts(num_seqs=n_groups, seed=seed)
    if len(bases) < n_groups:
        print(f"[build_grouped] only {len(bases)} targets available; using all")
        n_groups = len(bases)

    # One shared pool of M system prompts, applied to every base prompt, so
    # group_position p is the *same* jailbreak across groups (clean for analysis).
    system_prompts = make_system_prompts(per_group, crossover_prob, seed)

    out: list[dict] = []
    for g, base in enumerate(bases):
        for pos, sysp in enumerate(system_prompts):
            inst = dict(base)  # carries prompt, answer, target_email, target_name, n_shot
            inst["system_prompt"] = sysp
            inst["jailbreak_type"] = "privacy_jb" if pos > 0 else "main_jb"
            inst["group"] = g
            inst["group_position"] = pos
            out.append(inst)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", required=True, help="filename written under data/")
    ap.add_argument("--n-groups", type=int, default=10, help="number of distinct targets (groups)")
    ap.add_argument("--per-group", type=int, default=33, help="system prompts per group (group size)")
    ap.add_argument("--crossover-prob", type=float, default=0.3, help="gen_jb crossover prob (variety)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    data = build_grouped(args.n_groups, args.per_group, args.crossover_prob, args.seed)

    _DATA_DIR.mkdir(exist_ok=True)
    out_path = _DATA_DIR / args.output
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)

    n_groups = len(set(d["group"] for d in data))
    print(f"Wrote {len(data)} instances in {n_groups} groups to {out_path}")
    print(f"  group size = {args.per_group}; group = one target email, members vary system prompt")
