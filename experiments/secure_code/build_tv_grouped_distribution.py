"""Build a TV-groupable secure_code distribution for the cached-frontier verifier.

Applies the same lesson experiments/enron/TV_GROUPING_DESIGN.md derived
empirically: sharing a base prompt is *necessary* but not *sufficient* for
good caching — members also need within-family TV-tightness, which independent
per-instance persona+crossover sampling breaks (~29% of same-base pairs missed
threshold in the earlier pilot). So each base gets a *family* of near-identical
system prompts: one fixed persona, a low crossover_prob, mild variation only —
instead of resampling a persona independently per instance.

This intentionally does NOT assign a `group` field. Grouping is left to real
one-step TV clustering (beaver.grouping.tv_distance + tv_visualize) so the
`group` labels reflect actual measured next-token similarity, not an assumed
base-prompt structure — TV clustering discovers it; this script just makes
that discovery easy by removing the persona-driven noise that used to break it.

Base prompts default to the 12 c/cpp prompts already found to be
persona-stable in the earlier pilot (max intra-base TV < ~0.03 across all 9
personas) — pass --bases to override.

Example:
    python -m experiments.secure_code.build_tv_grouped_distribution \\
        --output data/tv_families_12x50.json --family-size 50

    python -m beaver.grouping.tv_distance --dataset data/tv_families_12x50.json \\
        --model Qwen/Qwen3-4B-Instruct-2507 --out ../../tv_out/tv_families_12x50

    python -m beaver.grouping.tv_visualize \\
        --matrix ../../tv_out/tv_families_12x50/tv_distance.npy \\
        --dataset data/tv_families_12x50.json \\
        --dataset-out data/tv_families_12x50_grouped.json \\
        --threshold 0.01 --out ../../tv_out/tv_families_12x50
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.secure_code.genetic_jailbreak import MAIN_JBS, gen_jb

_THIS_DIR = Path(__file__).parent
_DATA_DIR = _THIS_DIR / "data"

# The 12/25 bases found to fully TV-merge across all 9 personas in the earlier
# pilot (experiments/secure_code/data/autocomplete_ccpp_genetic_cross_25x400_test200.json
# analysis) — persona-stable, so a mild-variation family should stay TV-tight.
STABLE_BASES = [241, 204, 201, 88, 449, 348, 179, 379, 403, 327, 41, 341]


def build_family(
    base: dict,
    base_idx: int,
    family_size: int,
    crossover_prob: float,
    rng: np.random.Generator,
    persona_per_seed: bool = False,
) -> list[dict]:
    """One base prompt x a family of near-identical system-prompt variants.

    A single persona is fixed for the whole family (persona doesn't affect
    grouping, per the earlier pilot) and crossover_prob is kept low so members
    differ by at most a side-instruction or two — mild variation, matching
    Enron's few-shot-reorder analogue, instead of independent full resampling.
    """
    # persona_per_seed: every member draws its own persona (gen_jb main_jb=None).
    main_jb = None if persona_per_seed else MAIN_JBS[int(rng.integers(0, len(MAIN_JBS)))]
    seen: set[str] = set()
    out = []
    attempts = 0
    while len(out) < family_size and attempts < 100 * family_size:
        attempts += 1
        system_prompt = gen_jb(crossover_prob=crossover_prob, rng=rng, main_jb=main_jb)
        if system_prompt in seen:
            continue
        seen.add(system_prompt)
        inst = dict(base)
        inst["system_prompt"] = system_prompt
        inst["jailbreak_type"] = "genetic_persona" if persona_per_seed else "genetic_mild"
        inst["source_idx"] = base_idx
        inst["family_position"] = len(out)
        out.append(inst)
    return out


def build(
    input_path: Path,
    bases: list[int],
    family_size: int,
    crossover_prob: float,
    seed: int,
    persona_per_seed: bool = False,
) -> list[dict]:
    with open(input_path) as f:
        data = json.load(f)
    ccpp = [p for p in data if p["language"] in ("c", "cpp")]
    if len(ccpp) == 0:
        raise ValueError(f"No c/cpp prompts found in {input_path}")

    rng = np.random.default_rng(seed)

    out: list[dict] = []
    sample_id = 0
    for base_idx in bases:
        if base_idx >= len(ccpp):
            raise ValueError(f"base index {base_idx} out of range (pool has {len(ccpp)})")
        family = build_family(ccpp[base_idx], base_idx, family_size, crossover_prob, rng, persona_per_seed)
        for inst in family:
            inst["sample_id"] = sample_id
            sample_id += 1
        out.extend(family)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=str, default=str(_DATA_DIR / "autocomplete.json"))
    ap.add_argument("--output", type=str, required=True)
    ap.add_argument("--bases", type=str, default=None,
                     help="comma-separated base indices (default: the 12 known-stable bases)")
    ap.add_argument("--family-size", type=int, default=50)
    ap.add_argument("--crossover-prob", type=float, default=0.05,
                     help="kept low so family members stay near-identical (default 0.05)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--persona-per-seed", action="store_true",
                    help="draw a persona for every member, not one per family")
    args = ap.parse_args(argv)

    bases = [int(x) for x in args.bases.split(",")] if args.bases else STABLE_BASES

    input_path = Path(args.input)
    if not input_path.exists() and not input_path.is_absolute():
        input_path = _DATA_DIR / input_path.name

    output_path = Path(args.output)
    if not output_path.is_absolute() and not str(output_path).startswith("data/"):
        output_path = _DATA_DIR / output_path

    data = build(
        input_path=input_path,
        bases=bases,
        family_size=args.family_size,
        crossover_prob=args.crossover_prob,
        seed=args.seed,
        persona_per_seed=args.persona_per_seed,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Wrote {len(data)} instances ({len(bases)} bases x ~{args.family_size} family members) to {output_path}")
    print("No `group` field set — run beaver.grouping.tv_distance + tv_visualize to discover real groups.")


if __name__ == "__main__":
    main()
