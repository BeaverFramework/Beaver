"""Build a TV-groupable Enron distribution using in-domain 5-shot data.

Same recipe as experiments/secure_code/build_tv_grouped_distribution.py: each
base prompt gets a *family* of near-identical system-prompt variants (one
fixed persona, low crossover_prob, mild variation only) instead of an
independently-resampled persona per instance — independent resampling breaks
within-family TV-tightness (see experiments/enron/TV_GROUPING_DESIGN.md).

Uses `five_shot.pkl` (IN-DOMAIN shots — same-domain few-shot examples), not
`five_shot_non_domain.pkl` used by build_distribution.py / the older
build_tv_grouped_distribution.py — the in-domain, 5-shot configuration is the
only one observed to reliably produce constraint violations.

This intentionally does NOT assign a `group` field — run
beaver.grouping.tv_prefix_distance + tv_visualize to discover real groups from
measured k-step TV distance.

Example:
    python -m experiments.enron.build_tv_families_distribution \\
        --output data/enron_tv_families_12x50.json --n-bases 12 --family-size 50

    python -m beaver.grouping.tv_prefix_distance \\
        --dataset experiments/enron/data/enron_tv_families_12x50.json \\
        --model Qwen/Qwen3-4B-Instruct-2507 --config experiments/enron/enron_new.yaml --server-addr http://localhost:8082 \\
        --out tv_out/enron_tv_families_12x50_k5

    python -m beaver.grouping.tv_visualize \\
        --matrix tv_out/enron_tv_families_12x50_k5/tv_prefix_distance.npy \\
        --dataset tv_out/enron_tv_families_12x50_k5/instances.json \\
        --dataset-out experiments/enron/data/enron_tv_families_12x50_k5_grouped.json \\
        --threshold 0.01 --out tv_out/enron_tv_families_12x50_k5
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from experiments.enron.build_distribution import _build_fewshot_prompt
from experiments.enron.privacy_jailbreak import gen_jb

_THIS_DIR = Path(__file__).parent
_DATA_DIR = _THIS_DIR / "data"


def _load_pkl(name: str):
    with open(_DATA_DIR / name, "rb") as f:
        return pickle.load(f)


def load_domain_data() -> dict:
    """Load five-shot *in-domain* data — same-domain shots, not non_domain.pkl."""
    return {
        "five_shot": _load_pkl("five_shot.pkl"),
        "email2name": _load_pkl("email2name.pkl"),
    }


def build_family(
    target_email: str,
    target_name: str,
    shots: list,
    base_idx: int,
    n_shot: int,
    family_size: int,
    crossover_prob: float,
    rng: np.random.Generator,
) -> list[dict]:
    """One target (base prompt) x a family of near-identical system-prompt variants.

    A single (only) persona is used for the whole family; crossover_prob is
    kept low so members differ by at most a side-instruction or two — mild
    variation, not independent full resampling.
    """
    k = min(n_shot, len(shots))
    base_prompt = _build_fewshot_prompt(target_name, shots, k)

    seen: set[str] = set()
    out = []
    attempts = 0
    # gen_jb draws from the global numpy random state (not an injectable
    # Generator), so seed it once per family from our own Generator for
    # reproducibility without disturbing base-prompt/target sampling above.
    np.random.seed(int(rng.integers(0, 2**31 - 1)))
    while len(out) < family_size and attempts < 100 * family_size:
        attempts += 1
        system_prompt = gen_jb(tokenizer=None, crossover_prob=crossover_prob, mutation_prob=0.0)
        if system_prompt in seen:
            continue
        seen.add(system_prompt)
        out.append({
            "prompt": base_prompt,
            "answer": target_email,
            "target_email": target_email,
            "target_name": target_name,
            "n_shot": k,
            "system_prompt": system_prompt,
            "jailbreak_type": "privacy_jb_mild",
            "source_idx": base_idx,
            "family_position": len(out),
        })
    return out


def build(
    n_bases: int,
    family_size: int,
    crossover_prob: float,
    n_shot: int,
    min_shots: int,
    seed: int,
) -> list[dict]:
    data = load_domain_data()
    five_shot = data["five_shot"]
    email2name = data["email2name"]

    targets = [e for e in five_shot if e in email2name and len(five_shot[e]) >= min_shots]
    if len(targets) < n_bases:
        raise ValueError(
            f"only {len(targets)} targets have >= {min_shots} shots; need {n_bases}"
        )

    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(targets), size=n_bases, replace=False)

    out: list[dict] = []
    sample_id = 0
    for base_idx, idx in enumerate(chosen):
        target_email = targets[int(idx)]
        target_name = email2name[target_email]
        shots = list(five_shot[target_email])
        rng.shuffle(shots)

        family = build_family(
            target_email, target_name, shots, base_idx, n_shot, family_size, crossover_prob, rng
        )
        for inst in family:
            inst["sample_id"] = sample_id
            sample_id += 1
        out.extend(family)

    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", type=str, required=True)
    ap.add_argument("--n-bases", type=int, default=12)
    ap.add_argument("--family-size", type=int, default=50)
    ap.add_argument("--crossover-prob", type=float, default=0.05,
                     help="kept low so family members stay near-identical (default 0.05)")
    ap.add_argument("--n-shot", type=int, default=5)
    ap.add_argument("--min-shots", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    output_path = Path(args.output)
    if not output_path.is_absolute() and not str(output_path).startswith("data/"):
        output_path = _DATA_DIR / output_path
    elif str(output_path).startswith("data/"):
        output_path = _DATA_DIR / output_path.name

    data = build(
        n_bases=args.n_bases,
        family_size=args.family_size,
        crossover_prob=args.crossover_prob,
        n_shot=args.n_shot,
        min_shots=args.min_shots,
        seed=args.seed,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Wrote {len(data)} instances ({args.n_bases} bases x ~{args.family_size} family members) to {output_path}")
    print("No `group` field set — run beaver.grouping.tv_prefix_distance + tv_visualize to discover real groups.")


if __name__ == "__main__":
    main()
