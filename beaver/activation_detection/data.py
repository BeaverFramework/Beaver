"""Label building from BEAVER sample logs, shared by both direction scripts.

`build_labels` turns a log dir + dataset file into a flat list of
(prompt, generation, is_secure, cwe, group) rows that every downstream function
(residual extraction, analyze_layers, grouped_cv_auc, ...) consumes.
"""

import json
from pathlib import Path

from beaver.logging import get_log_data


def build_labels(
    log_file: Path,
    dataset_file: Path,
    keep_zero_violation: bool = False,
):
    """-> list of dicts: {prompt, generation, is_secure, cwe, group, gen_ids?}.

    is_secure=True  == non-violating sample
    is_secure=False == violating sample
    group           == prompt id (so grouped CV never splits one prompt).
    prompt          == `exact_prompt`: the fully chat-templated string fed to the
                       model (generations were sampled directly after it).

    Generations are de-duplicated per prompt (a set) but sorted so row order is
    deterministic across processes -- otherwise set iteration order
    (PYTHONHASHSEED) silently reshuffles the data and changes CV folds run-to-run.
    Prompts with zero violations are dropped unless keep_zero_violation=True.
    If logged_gen_tokenids_key is given, the per-generation token ids logged under
    that key are attached as `gen_ids` (used verbatim instead of re-tokenizing).
    """
    ld = get_log_data(log_file)
    with open(dataset_file) as f:
        dataset = {i: j for i, j in enumerate(json.load(f))}

    labels = []
    for f, d in ld.items():
        last = d[-1]  # last log entry == final genetic generation
        n_viol = last["n_violations_found"]
        if n_viol <= 0 and not keep_zero_violation:
            continue
        violating = set(last.get("violation_texts", []))
        non_violating = set(last["samples"]) - violating
        dentry = dataset[int(f)]
        prompt = last["exact_prompt"]
        group = int(f)


        for gen in sorted(violating):  # sorted -> deterministic order
            labels.append(
                dict(
                    prompt=prompt,
                    generation=gen,
                    is_secure=False,
                    group=group,
                )
            )
        for gen in sorted(non_violating):
            labels.append(
                dict(
                    prompt=prompt,
                    generation=gen,
                    is_secure=True,
                    group=group,
                )
            )
    return labels
