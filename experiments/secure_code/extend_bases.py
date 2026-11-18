"""Select a set of secure_code c/cpp base indices: keep Nalin's known-stable
bases and top up with sampled ones.

build_tv_grouped_distribution.py already fans each base index into a family of
near-identical system-prompt variants; it just defaults to the 12 persona-stable
bases. To build a larger N-base dataset the aligned way, this picks those 12
verbatim and samples the remaining (N-12) indices from the same c/cpp pool
(experiments/secure_code/data/autocomplete.json), excluding the stable ones, so
the extra bases come from the identical distribution.

Prints a comma-separated index list ready to pass to
build_tv_grouped_distribution.py's --bases, and (optionally) writes it to JSON.

Example:
    python -m experiments.secure_code.extend_bases --total 50 > /tmp/bases50.txt
    python -m experiments.secure_code.build_tv_grouped_distribution \\
        --bases "$(cat /tmp/bases50.txt)" \\
        --output data/secure_code_tv_families_50x200.json --family-size 200
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from experiments.secure_code.build_tv_grouped_distribution import STABLE_BASES

_THIS_DIR = Path(__file__).parent
_DATA_DIR = _THIS_DIR / "data"


def select(total: int, input_path: Path, seed: int) -> list[int]:
    with open(input_path) as f:
        data = json.load(f)
    ccpp = [p for p in data if p.get("language") in ("c", "cpp")]
    pool_n = len(ccpp)

    bases = list(STABLE_BASES)
    if any(b >= pool_n for b in bases):
        raise ValueError(f"a stable base index is out of range for pool size {pool_n}")

    n_add = total - len(bases)
    if n_add < 0:
        return bases[:total]
    if n_add > 0:
        used = set(bases)
        candidates = [i for i in range(pool_n) if i not in used]
        if len(candidates) < n_add:
            raise ValueError(f"only {len(candidates)} unused c/cpp bases; need {n_add}")
        rng = np.random.default_rng(seed)
        chosen = rng.choice(len(candidates), size=n_add, replace=False)
        bases.extend(sorted(candidates[int(c)] for c in chosen))
    return bases


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--total", type=int, default=50, help="target number of base indices")
    ap.add_argument("--input", type=str, default=str(_DATA_DIR / "autocomplete.json"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--json-out", type=str, default=None,
                    help="optional path to also write the index list + provenance as JSON")
    args = ap.parse_args(argv)

    input_path = Path(args.input)
    if not input_path.exists() and not input_path.is_absolute():
        input_path = _DATA_DIR / input_path.name

    bases = select(args.total, input_path, args.seed)

    # comma-separated list to stdout (pipe straight into --bases)
    print(",".join(str(b) for b in bases))

    stable = set(STABLE_BASES)
    sys.stderr.write(
        f"{len(bases)} bases: {len(stable & set(bases))} stable + "
        f"{len(bases) - len(stable & set(bases))} sampled (seed={args.seed})\n"
    )

    if args.json_out:
        out_path = Path(args.json_out)
        if not out_path.is_absolute():
            out_path = _DATA_DIR / out_path.name
        record = [{"base_idx": b, "origin": "stable" if b in stable else "sampled"} for b in bases]
        with open(out_path, "w") as f:
            json.dump(record, f, indent=2)
        sys.stderr.write(f"wrote {out_path}\n")


if __name__ == "__main__":
    main()
