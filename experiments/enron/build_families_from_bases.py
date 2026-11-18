"""Build a TV-groupable Enron distribution from a FIXED set of base prompts.

Unlike build_tv_families_distribution.py, which re-samples its base prompts
from five_shot.pkl, this reads an explicit base-prompt file (e.g.
data/enron_base_prompts_20.json) and expands each base into a *family* of
near-identical system-prompt variants (one persona, low crossover_prob, mild
variation only) via experiments/enron/privacy_jailbreak.gen_jb — the same
mechanism the other builders use. Use this when you have already curated the
base prompts you want and only need to fan out jailbreak variants over them.

Each input base is expected to carry a prebuilt few-shot `prompt` plus
`target_email` / `target_name` / `n_shot` (the schema of
enron_base_prompts_20.json). The output mirrors build_tv_families_distribution
so the downstream grouping tools consume it unchanged.

This intentionally does NOT assign a `group` field — run
beaver.grouping.tv_prefix_distance + tv_visualize to discover real groups.

Example:
    python -m experiments.enron.build_families_from_bases \\
        --bases data/enron_base_prompts_20.json \\
        --output data/enron_families_20x50.json --family-size 50

    python -m beaver.grouping.tv_prefix_distance \\
        --dataset experiments/enron/data/enron_families_20x50.json \\
        --model Qwen/Qwen3-4B-Instruct-2507 --config experiments/enron/enron_new.yaml --server-addr http://localhost:8082 \\
        --out tv_out/enron_families_20x50_k5

    python -m beaver.grouping.tv_visualize \\
        --matrix tv_out/enron_families_20x50_k5/tv_prefix_distance.npy \\
        --dataset tv_out/enron_families_20x50_k5/instances.json \\
        --dataset-out experiments/enron/data/enron_families_20x50_k5_grouped.json \\
        --threshold 0.01 --out tv_out/enron_families_20x50_k5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.enron.privacy_jailbreak import gen_jb

_THIS_DIR = Path(__file__).parent
_DATA_DIR = _THIS_DIR / "data"


def build_family(
    base: dict,
    family_size: int,
    crossover_prob: float,
    rng: np.random.Generator,
) -> list[dict]:
    """One fixed base prompt x a family of near-identical system-prompt variants."""
    # gen_jb draws from the global numpy random state (not an injectable
    # Generator), so seed it once per family from our own Generator for
    # reproducibility.
    np.random.seed(int(rng.integers(0, 2**31 - 1)))

    seen: set[str] = set()
    out = []
    attempts = 0
    while len(out) < family_size and attempts < 100 * family_size:
        attempts += 1
        system_prompt = gen_jb(tokenizer=None, crossover_prob=crossover_prob, mutation_prob=0.0)
        if system_prompt in seen:
            continue
        seen.add(system_prompt)
        out.append({
            "prompt": base["prompt"],
            "answer": base["target_email"],
            "target_email": base["target_email"],
            "target_name": base["target_name"],
            "n_shot": base.get("n_shot"),
            "system_prompt": system_prompt,
            "jailbreak_type": "privacy_jb_mild",
            "source_idx": base.get("source_idx"),
            "origin": base.get("origin"),
            "family_position": len(out),
        })
    return out


def build(
    bases_path: Path,
    family_size: int,
    crossover_prob: float,
    seed: int,
) -> list[dict]:
    with open(bases_path) as f:
        bases = json.load(f)

    rng = np.random.default_rng(seed)
    out: list[dict] = []
    sample_id = 0
    for base in bases:
        family = build_family(base, family_size, crossover_prob, rng)
        for inst in family:
            inst["sample_id"] = sample_id
            sample_id += 1
        out.extend(family)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bases", type=str, default=str(_DATA_DIR / "enron_base_prompts_20.json"),
                    help="JSON list of base prompts (prompt/target_email/target_name/...)")
    ap.add_argument("--output", type=str, required=True)
    ap.add_argument("--family-size", type=int, default=50)
    ap.add_argument("--crossover-prob", type=float, default=0.05,
                    help="kept low so family members stay near-identical (default 0.05)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    bases_path = Path(args.bases)
    if not bases_path.is_absolute() and not bases_path.exists():
        bases_path = _DATA_DIR / bases_path.name

    output_path = Path(args.output)
    if not output_path.is_absolute() and not str(output_path).startswith("data/"):
        output_path = _DATA_DIR / output_path
    elif str(output_path).startswith("data/"):
        output_path = _DATA_DIR / output_path.name

    data = build(
        bases_path=bases_path,
        family_size=args.family_size,
        crossover_prob=args.crossover_prob,
        seed=args.seed,
    )

    n_bases = len({inst["source_idx"] for inst in data})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Wrote {len(data)} instances ({n_bases} bases x ~{args.family_size} family members) to {output_path}")
    print("No `group` field set — run beaver.grouping.tv_prefix_distance + tv_visualize to discover real groups.")


if __name__ == "__main__":
    main()
