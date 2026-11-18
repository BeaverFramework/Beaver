"""Slice a grouped families dataset down to the first N bases x M members.

Groups are subsets of families (zero mixed-base groups on every grouping we
have), so restricting each base to its first M family members keeps every
group coherent; groups are renumbered densely, ``group_position`` is
recomputed within the slice and the original group order is preserved so the
cached-frontier verifier still meets each group's first member first.

    python -m experiments.enron.slice_grouped_families \\
        --dataset tv_out/enron_families_50x200_k5_qwen3_4b/grouped_dataset_tv.json \\
        --n-bases 12 --family-size 50 \\
        --output experiments/enron/data/enron_tv_families_12x50_k5_grouped.json
"""

from __future__ import annotations

import argparse
import collections
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--n-bases", type=int, default=12)
    ap.add_argument("--family-size", type=int, default=50)
    ap.add_argument("--max-groups", type=int, default=None,
                    help="optional: keep only the first K (renumbered) groups, for pilots")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    data = json.load(open(args.dataset))
    bases = sorted({x["source_idx"] for x in data})[: args.n_bases]
    keep = [
        x for x in data
        if x["source_idx"] in bases and x["family_position"] < args.family_size
    ]
    # Preserve the file's (group, group_position) order, renumber densely.
    keep.sort(key=lambda x: (x["group"], x["group_position"]))
    gmap, out, pos = {}, [], collections.Counter()
    for x in keep:
        g = gmap.setdefault(x["group"], len(gmap))
        if args.max_groups is not None and g >= args.max_groups:
            continue
        y = dict(x)
        y["group"] = g
        y["group_position"] = pos[g]
        pos[g] += 1
        out.append(y)
    json.dump(out, open(args.output, "w"), indent=1)
    sizes = collections.Counter(y["group"] for y in out)
    mixed = sum(
        1 for g in sizes
        if len({y["source_idx"] for y in out if y["group"] == g}) > 1
    )
    print(
        f"{len(out)} prompts, {len(bases)} bases x <= {args.family_size}, "
        f"{len(sizes)} groups (sizes {sorted(sizes.values(), reverse=True)}), "
        f"{mixed} mixed-base groups -> {args.output}"
    )


if __name__ == "__main__":
    main()
