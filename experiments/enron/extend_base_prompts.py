"""Extend a curated Enron base-prompt file with freshly sampled bases.

Reads an existing base-prompt JSON (e.g. data/enron_base_prompts_20.json),
keeps every entry verbatim, then samples additional in-domain 5-shot bases from
five_shot.pkl until the file reaches `--total` bases. Sampled targets exclude
any target_email already present, so the curated reliable-leak bases are never
duplicated. Newly added bases are tagged origin="sampled" and given source_idx
values continuing after the existing ones.

The sampling mirrors build_tv_families_distribution.build (in-domain shots via
_build_fewshot_prompt) so the sampled bases are schema-compatible with the
curated ones and with build_families_from_bases.py.

Example:
    python -m experiments.enron.extend_base_prompts \\
        --bases data/enron_base_prompts_20.json \\
        --output data/enron_base_prompts_50.json --total 50
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from experiments.enron.build_distribution import _build_fewshot_prompt

_THIS_DIR = Path(__file__).parent
_DATA_DIR = _THIS_DIR / "data"


def _load_pkl(name: str):
    with open(_DATA_DIR / name, "rb") as f:
        return pickle.load(f)


def build(bases_path: Path, total: int, n_shot: int, min_shots: int, seed: int) -> list[dict]:
    with open(bases_path) as f:
        bases = list(json.load(f))

    n_add = total - len(bases)
    if n_add <= 0:
        print(f"file already has {len(bases)} >= {total} bases; nothing to add")
        return bases

    five_shot = _load_pkl("five_shot.pkl")
    email2name = _load_pkl("email2name.pkl")

    used = {b["target_email"] for b in bases}
    candidates = [
        e for e in five_shot
        if e in email2name and len(five_shot[e]) >= min_shots and e not in used
    ]
    if len(candidates) < n_add:
        raise ValueError(f"only {len(candidates)} unused targets have >= {min_shots} shots; need {n_add}")

    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(candidates), size=n_add, replace=False)

    next_idx = max((b.get("source_idx", -1) for b in bases), default=-1) + 1
    for offset, ci in enumerate(chosen):
        target_email = candidates[int(ci)]
        target_name = email2name[target_email]
        shots = list(five_shot[target_email])
        rng.shuffle(shots)
        k = min(n_shot, len(shots))
        bases.append({
            "source_idx": next_idx + offset,
            "target_name": target_name,
            "target_email": target_email,
            "n_shot": k,
            "prompt": _build_fewshot_prompt(target_name, shots, k),
            "origin": "sampled",
        })
    return bases


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bases", type=str, default=str(_DATA_DIR / "enron_base_prompts_20.json"))
    ap.add_argument("--output", type=str, required=True)
    ap.add_argument("--total", type=int, default=50, help="target number of base prompts")
    ap.add_argument("--n-shot", type=int, default=5)
    ap.add_argument("--min-shots", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    bases_path = Path(args.bases)
    if not bases_path.is_absolute() and not bases_path.exists():
        bases_path = _DATA_DIR / bases_path.name

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = _DATA_DIR / output_path.name

    bases = build(bases_path, args.total, args.n_shot, args.min_shots, args.seed)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(bases, f, indent=2)

    from collections import Counter
    origins = Counter(b.get("origin") for b in bases)
    print(f"Wrote {len(bases)} base prompts to {output_path}")
    print(f"origins: {dict(origins)}")


if __name__ == "__main__":
    main()
