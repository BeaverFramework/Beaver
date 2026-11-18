"""Build a TV-groupable Enron distribution for the cached-frontier verifier.

This encodes the distribution-design learnings in ``TV_GROUPING_DESIGN.md``. The
goal is a distribution that the one-step TV heuristic clusters into **few, large,
low-singleton groups** so the cached-frontier verifier maximally reuses work.

Design (see TV_GROUPING_DESIGN.md for the "why" and the measured numbers):

  * **Theme = one target email** → distinct next-token behaviour, so between-group
    TV ≈ 1.0 and TV never merges different themes (L1, L3).
  * **Members = the same target under MILD variation** (few-shot reordering, with
    an optional *single shared* jailbreak) → within-group TV stays ≤ ε, so the
    theme survives as one TV cluster instead of fracturing into singletons
    (L3, L4). Members share the target ⇒ they share the violation sequence, which
    is what lets warm-starts gate-trip cheaply (L6).
  * **Large groups, few groups** ⇒ many non-seed members ⇒ many warm-skips (L2, L8).

By default this writes *structural* groups (``group`` = theme index), which is a
good approximation when variation is mild and needs no GPU. Pass ``--tv-regroup``
to instead compute the exact one-step TV matrix and re-group with the same
complete-linkage diameter cap the verifier's grouping assumes — the gold-standard
grouping — and report the realized singleton rate.

The output schema (``prompt``, ``answer``, ``target_email``, ``target_name``,
``system_prompt``, ``group``, ``group_position``, ``idx``, plus ``group_true``
for reference) feeds ``cachedfrontier`` / ``frontier`` directly.

Examples:
    # Structural grouping (fast, no GPU): 15 themes x 24 members = 360 prompts
    python -m experiments.enron.build_tv_grouped_distribution \
        --output data/tv_grouped_15x24.json --n-groups 15 --group-size 24

    # Gold-standard TV grouping (needs a model + GPU): re-cluster + report singletons
    python -m experiments.enron.build_tv_grouped_distribution \
        --output data/tv_grouped_15x24.json --n-groups 15 --group-size 24 \
        --tv-regroup --model Qwen/Qwen3-4B-Instruct-2507 --threshold 0.1

Then run the verifier (groups already assigned, no rebuild):
    beaver run --experiment experiments/enron/enron.yaml \
        --verifier cachedfrontier --server_addr http://localhost:8000 \
        --model Qwen/Qwen3-4B-Instruct-2507 --log_dir logging/cached
  (point enron.yaml's prompts_file at the output, or pass a YAML like the
   enron_tvdemo*_cached.yaml files).
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import List, Optional

from experiments.enron.build_distribution import (
    _build_fewshot_prompt,
    load_non_domain_data,
)

_DATA_DIR = Path(__file__).parent / "data"


# ── member-variation strategies (keep within-group TV small) ────────────────────
def _variant_prompt(name: str, shots: list, n_shot: int, rng: random.Random) -> tuple[str, int]:
    """Produce one MILD variation of a target's few-shot completion prompt.

    Only the *order* of the few-shot exemplars changes; the final
    ``"the email address of <target> is"`` — which dominates the next-token
    distribution — is untouched, so members stay TV-close (L3/L4). This is the
    variation that gave the lowest singleton rate in testing.
    """
    s = list(shots)
    rng.shuffle(s)
    k = min(n_shot, len(s))
    return _build_fewshot_prompt(name, s, k), k


def build(
    n_groups: int,
    group_size: int,
    seed: int = 42,
    n_shot: int = 5,
    min_shots: int = 3,
    jailbreak: str = "none",
) -> List[dict]:
    """Build a structurally-grouped, TV-tight distribution.

    ``n_groups`` distinct targets (themes), each expanded to ``group_size``
    members by :func:`_variant_prompt`. ``jailbreak`` is applied identically to
    every member of every group (``none`` → no system prompt); a *single shared*
    jailbreak preserves within-group TV, whereas per-member different jailbreaks
    would break it (L4) and are intentionally not supported.
    """
    rng = random.Random(seed)
    data = load_non_domain_data()
    five_shot, email2name = data["five_shot"], data["email2name"]

    targets = [e for e in five_shot if e in email2name and len(five_shot[e]) >= min_shots]
    rng.shuffle(targets)
    if len(targets) < n_groups:
        raise SystemExit(
            f"only {len(targets)} targets have >= {min_shots} shots; "
            f"lower --n-groups or --min-shots"
        )

    system_prompt = _resolve_jailbreak(jailbreak, seed)

    out: List[dict] = []
    idx = 0
    for g in range(n_groups):
        tgt = targets[g]
        name = email2name[tgt]
        shots = list(five_shot[tgt])
        for pos in range(group_size):
            prompt, k = _variant_prompt(name, shots, n_shot, rng)
            out.append(
                {
                    "prompt": prompt,
                    "answer": tgt,
                    "target_email": tgt,
                    "target_name": name,
                    "n_shot": k,
                    "system_prompt": system_prompt,   # always present (workers require the key)
                    "jailbreak_type": jailbreak,
                    "group": g,                        # structural group = theme
                    "group_true": g,                   # reference label, kept even after --tv-regroup
                    "group_position": pos,
                    "idx": idx,
                }
            )
            idx += 1
    return out


def _resolve_jailbreak(jailbreak: str, seed: int) -> str:
    """Return the single shared system prompt for the chosen jailbreak mode."""
    if jailbreak == "none":
        return ""
    if jailbreak == "main_jb":
        from experiments.enron.privacy_jailbreak import MAIN_JB
        return MAIN_JB
    raise SystemExit(
        f"unknown --jailbreak {jailbreak!r}. Use 'none' or 'main_jb'. "
        "Per-member *different* jailbreaks are intentionally unsupported — they "
        "break within-group TV tightness (see TV_GROUPING_DESIGN.md, L4)."
    )


# ── optional: exact TV re-grouping (gold standard) ──────────────────────────────
def tv_regroup(
    instances: List[dict],
    model: str,
    threshold: float,
    device: str,
    dtype: str,
    out_dir: Optional[Path],
) -> List[dict]:
    """Recompute one-step TV and re-group with the complete-linkage diameter cap.

    Replaces the structural ``group``/``group_position`` with TV-cluster labels
    (medoid-first order), the exact grouping the cached-frontier verifier's
    diameter guarantee assumes. Returns the re-grouped, verifier-ready list and
    prints the realized singleton rate + diameter check.
    """
    import numpy as np

    from beaver.grouping.tv_distance import HFBackend, one_step_tv_matrix
    from beaver.grouping.tv_visualize import (
        build_grouped_dataset,
        diameter_clusters,
        verify_grouping,
    )

    backend = HFBackend(model, device=device, dtype=dtype)
    D = one_step_tv_matrix(instances=instances, backend=backend)
    labels = diameter_clusters(D, threshold)

    grouped = build_grouped_dataset(instances, D, labels)  # adds group/group_position/tv_index
    v = verify_grouping(grouped, D, threshold)

    sizes = Counter(int(x) for x in labels)
    sz = sorted(sizes.values(), reverse=True)
    singleton_prompts = sum(s for s in sz if s == 1)
    print(
        f"\nTV re-grouping @ threshold={threshold}: {len(sz)} clusters, "
        f"max={max(sz)}, singletons={sum(s == 1 for s in sz)} clusters "
        f"({singleton_prompts}/{len(instances)} prompts)"
    )
    print(f"  sizes: {sz}")
    print(
        f"  diameter check: max intra-group TV = {v['max_intra_distance']:.4f} "
        f"(<= {threshold}): {'PASS' if v['passed'] else 'FAIL'}"
    )
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "tv_distance.npy", D)
        np.save(out_dir / "tv_cluster_labels.npy", labels)
        print(f"  TV artifacts -> {out_dir}/tv_distance.npy, tv_cluster_labels.npy")
    return grouped


# ── CLI ─────────────────────────────────────────────────────────────────────────
def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--output", required=True,
                    help="output JSON path (relative paths resolve under experiments/enron/)")
    ap.add_argument("--n-groups", type=int, default=15, help="number of themes / groups (default 15)")
    ap.add_argument("--group-size", type=int, default=24, help="members per group (default 24)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-shot", type=int, default=5, help="few-shot exemplars per prompt (default 5)")
    ap.add_argument("--min-shots", type=int, default=3,
                    help="only use targets with at least this many shots (default 3)")
    ap.add_argument("--jailbreak", choices=["none", "main_jb"], default="none",
                    help="single SHARED system prompt for every member (default none). "
                         "Per-member different jailbreaks are unsupported by design (L4).")
    # optional gold-standard TV re-grouping
    ap.add_argument("--tv-regroup", action="store_true",
                    help="recompute exact TV and re-group with the diameter cap (needs --model + GPU)")
    ap.add_argument("--model", default=None, help="HF model id for --tv-regroup")
    ap.add_argument("--threshold", type=float, default=0.1, help="TV diameter cap ε (default 0.1)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--tv-out", default=None, help="dir for TV artifacts (default: alongside --output)")
    args = ap.parse_args(argv)

    data = build(
        n_groups=args.n_groups,
        group_size=args.group_size,
        seed=args.seed,
        n_shot=args.n_shot,
        min_shots=args.min_shots,
        jailbreak=args.jailbreak,
    )

    if args.tv_regroup:
        if not args.model:
            raise SystemExit("--tv-regroup requires --model")
        out_path_early = Path(args.output)
        tv_out = Path(args.tv_out) if args.tv_out else (
            (out_path_early if out_path_early.is_absolute()
             else Path(__file__).parent / out_path_early).parent
            / "tv_regroup"
        )
        data = tv_regroup(
            data, args.model, args.threshold, args.device, args.dtype, tv_out
        )

    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = Path(__file__).parent / out_path   # resolve relative to experiments/enron/
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)

    n_groups = len({d["group"] for d in data})
    sizes = sorted(Counter(d["group"] for d in data).values(), reverse=True)
    print(f"\nWrote {len(data)} instances in {n_groups} groups to {out_path}")
    print(f"  group sizes: {sizes}")
    print(f"  singletons: {sum(s == 1 for s in sizes)} groups")
    if not args.tv_regroup:
        print("  (structural theme grouping — run with --tv-regroup, or tv_distance + "
              "tv_visualize, to get the exact TV grouping and verify tightness)")


if __name__ == "__main__":
    main()
