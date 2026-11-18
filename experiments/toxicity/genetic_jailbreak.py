"""Genetic jailbreak generator for toxicity prompts.

Mirrors experiments/secure_code/genetic_jailbreak.py's crossover mechanism
(_load_bank + gen_jb), but as its own module — like
experiments/enron/privacy_jailbreak.py — so gen_jb's SIDE_INSTRUCTIONS global
resolves against *this* module (toxicity's own jailbreak_bank.json), not
secure_code's. A shared gen_jb previously caused build_tv_families_distribution.py
to silently pull secure_code's side instructions into toxicity's system
prompts, since a function's bare-name globals resolve to its defining module,
not the caller's.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

_THIS_DIR = Path(__file__).parent
_DATA_DIR = _THIS_DIR / "data"
_TASK_BANK_PATH = _DATA_DIR / "jailbreak_bank.json"


def _load_bank(task_path: Path) -> tuple[list[str], list[str]]:
    with open(task_path) as f:
        task = json.load(f)

    main_jbs = list(task["main_jbs"])

    side_instructions: list[str] = []
    for category in task["side_instructions"].values():
        side_instructions.extend(category)

    return main_jbs, side_instructions


MAIN_JBS, SIDE_INSTRUCTIONS = _load_bank(_TASK_BANK_PATH)


def gen_jb(
    tokenizer=None,
    crossover_prob: float = 0.2,
    mutation_prob: float = 0.0,
    rng: np.random.Generator | None = None,
    main_jb: str | None = None,
) -> str:
    if rng is None:
        rng = np.random.default_rng()
    if main_jb is None:
        main_jb = MAIN_JBS[int(rng.integers(0, len(MAIN_JBS)))]

    chunks = main_jb.split(". ")
    chunks = [c.strip() for c in chunks if c.strip()]
    chunks = [(c + ".") if not c.endswith(".") else c for c in chunks]

    prompt = [chunks[0]] if chunks else []
    for chunk in chunks[1:]:
        selected = [s for s in SIDE_INSTRUCTIONS if rng.random() < crossover_prob]
        rng.shuffle(selected)
        prompt.extend(selected)
        prompt.append(chunk)

    # Always sample a trailing block of side instructions so a short/benign
    # MAIN_JB (e.g. a single sentence) still receives jailbreak content.
    trailing = [s for s in SIDE_INSTRUCTIONS if rng.random() < crossover_prob]
    rng.shuffle(trailing)
    prompt.extend(trailing)

    if tokenizer is not None and mutation_prob > 0:
        for i in range(len(prompt)):
            tokens = tokenizer.encode(prompt[i], add_special_tokens=False)
            for j in range(len(tokens)):
                if rng.random() < mutation_prob:
                    tokens[j] = int(rng.integers(0, tokenizer.vocab_size))
            prompt[i] = tokenizer.decode(tokens, skip_special_tokens=True)

    return " ".join(prompt)
