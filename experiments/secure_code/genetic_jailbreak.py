"""Genetic jailbreak generator for secure_code prompts.

Given an input prompts file (e.g. data/instruct-v2.json), this script
generates a jailbroken version of each prompt by sampling a system prompt
from a genetic distribution of insecure-coding jailbreaks (crossover of
sentence chunks + optional token-level mutation), then samples `num_seqs`
jailbroken prompts and writes them to an output file.

Example:
    python genetic_jailbreak.py \\
        --input data/instruct-v2.json \\
        --output data/instruct-v2_genetic.json \\
        --num_seqs 1000 \\
        --seed 42

Alternatively, `--num_base`/`--num_system` build a cross product: `num_base`
distinct C/C++ base prompts x `num_system` distinct genetic system prompts,
i.e. every sampled base prompt paired with every sampled adversarial system
prompt.

Example:
    python genetic_jailbreak.py \\
        --input data/autocomplete_ccpp.json \\
        --output data/autocomplete_ccpp_genetic_cross.json \\
        --num_base 100 \\
        --num_system 20 \\
        --seed 42
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


_THIS_DIR = Path(__file__).parent
_DATA_DIR = _THIS_DIR / "data"
_COMMON_BANK_PATH = _THIS_DIR.parent / "common" / "jailbreak_bank.json"
_TASK_BANK_PATH = _DATA_DIR / "jailbreak_bank.json"


# ── Jailbreak template bank ────────────────────────────────────────────────
#
# main_jbs: alternate personas/pretexts for the insecure-code-completion
# framing; gen_jb samples one per generated system prompt.
# side_instructions: domain-agnostic pressure (refusal suppression, format
# enforcement, authority framing, anti-hedging) from the shared common bank,
# merged with secure_code-specific pressure (insecure-coding practices) from
# the task bank. See experiments/common/jailbreak_bank.json and
# experiments/secure_code/data/jailbreak_bank.json.


def _load_bank(common_path: Path, task_path: Path) -> tuple[list[str], list[str]]:
    with open(common_path) as f:
        common = json.load(f)
    with open(task_path) as f:
        task = json.load(f)

    main_jbs = list(task["main_jbs"])

    side_instructions: list[str] = []
    for category in common["side_instructions"].values():
        side_instructions.extend(category)
    for category in task["side_instructions"].values():
        side_instructions.extend(category)

    return main_jbs, side_instructions


MAIN_JBS, SIDE_INSTRUCTIONS = _load_bank(_COMMON_BANK_PATH, _TASK_BANK_PATH)
MAIN_JB = MAIN_JBS[0]  # default/primary framing, kept for backward compat


# ── Generator ──────────────────────────────────────────────────────────────


def gen_jb(
    tokenizer=None,
    crossover_prob: float = 0.2,
    mutation_prob: float = 0.0,
    rng: np.random.Generator | None = None,
    main_jb: str | None = None,
) -> str:
    """Generate one genetic jailbreak system prompt.

    Crossover: sample a main_jb persona (uniformly from MAIN_JBS, unless one
    is passed explicitly) and randomly interleave SIDE_INSTRUCTIONS between
    its sentence chunks. Mutation: optionally flip random tokens.
    """
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


# ── Build jailbroken prompts ───────────────────────────────────────────────


def build_jailbroken_prompts(
    input_path: Path,
    num_seqs: int,
    crossover_prob: float = 0.2,
    mutation_prob: float = 0.0,
    tokenizer=None,
    seed: int = 42,
    with_replacement: bool = True,
) -> list[dict]:
    """Load input prompts and produce `num_seqs` jailbroken variants.

    Each output entry is a shallow copy of the source prompt with an added
    `system_prompt` (the genetic jailbreak) and `jailbreak_type = "genetic"`.
    """
    with open(input_path) as f:
        base_prompts = json.load(f)
    
    # filter prompts
    base_prompts = [p for p in base_prompts if p["language"] in ("c", "cpp")]


    if len(base_prompts) == 0:
        raise ValueError(f"No prompts found in {input_path}")

    rng = np.random.default_rng(seed)

    if with_replacement or num_seqs > len(base_prompts):
        idxs = rng.integers(0, len(base_prompts), size=num_seqs)
    else:
        idxs = rng.choice(len(base_prompts), size=num_seqs, replace=False)

    out = []
    for sample_id, idx in enumerate(idxs):
        base = dict(base_prompts[int(idx)])
        system_prompt = gen_jb(
            tokenizer=tokenizer,
            crossover_prob=crossover_prob,
            mutation_prob=mutation_prob,
            rng=rng,
        )
        base["system_prompt"] = system_prompt
        base["jailbreak_type"] = "genetic"
        base["source_idx"] = int(idx)
        base["sample_id"] = sample_id
        out.append(base)

    return out


def make_system_prompt_pool(
    num_system: int,
    crossover_prob: float = 0.2,
    mutation_prob: float = 0.0,
    tokenizer=None,
    rng: np.random.Generator | None = None,
) -> list[str]:
    """Return `num_system` distinct genetic jailbreak system prompts.

    Retries on collision so the pool has no duplicates (crossover_prob values
    near 0 or 1 shrink the space of distinct outputs, so keep it away from
    those extremes when num_system is large).
    """
    if rng is None:
        rng = np.random.default_rng()

    seen: set[str] = set()
    prompts: list[str] = []
    attempts = 0
    max_attempts = 100 * num_system
    while len(prompts) < num_system and attempts < max_attempts:
        attempts += 1
        jb = gen_jb(
            tokenizer=tokenizer,
            crossover_prob=crossover_prob,
            mutation_prob=mutation_prob,
            rng=rng,
        )
        if jb not in seen:
            seen.add(jb)
            prompts.append(jb)

    if len(prompts) < num_system:
        raise RuntimeError(
            f"only generated {len(prompts)}/{num_system} distinct system"
            f" prompts after {attempts} attempts; raise --crossover_prob or"
            f" lower --num_system"
        )
    return prompts


def build_cross_product_prompts(
    input_path: Path,
    num_base: int,
    num_system: int,
    crossover_prob: float = 0.2,
    mutation_prob: float = 0.0,
    tokenizer=None,
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[str]]:
    """Cross num_base base prompts with num_system adversarial system prompts.

    Samples `num_base` distinct C/C++ base prompts (without replacement) and
    a pool of `num_system` distinct genetic system prompts, then emits every
    (base, system) pair — `num_base * num_system` instances total, ordered
    with base prompts outermost so same-base instances are contiguous.

    Returns `(cross_product, sampled_base_prompts, sampled_system_prompts)`
    so callers can inspect/save the two sampled pools independently of the
    joined output.
    """
    with open(input_path) as f:
        base_prompts = json.load(f)

    base_prompts = [p for p in base_prompts if p["language"] in ("c", "cpp")]
    if len(base_prompts) == 0:
        raise ValueError(f"No prompts found in {input_path}")
    if num_base > len(base_prompts):
        raise ValueError(
            f"num_base={num_base} exceeds available c/cpp base prompts"
            f" ({len(base_prompts)})"
        )

    rng = np.random.default_rng(seed)

    base_idxs = rng.choice(len(base_prompts), size=num_base, replace=False)
    system_prompts = make_system_prompt_pool(
        num_system,
        crossover_prob=crossover_prob,
        mutation_prob=mutation_prob,
        tokenizer=tokenizer,
        rng=rng,
    )

    sampled_bases = [base_prompts[int(i)] for i in base_idxs]

    out = []
    sample_id = 0
    for base_idx, base in zip(base_idxs, sampled_bases):
        base_idx = int(base_idx)
        for system_idx, system_prompt in enumerate(system_prompts):
            inst = dict(base)
            inst["system_prompt"] = system_prompt
            inst["jailbreak_type"] = "genetic"
            inst["source_idx"] = base_idx
            inst["system_idx"] = system_idx
            inst["sample_id"] = sample_id
            out.append(inst)
            sample_id += 1

    return out, sampled_bases, system_prompts


# ── CLI ────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=str,
        default=str(_DATA_DIR / "instruct-v2.json"),
        help="Input prompts JSON (instruct-v2.json format).",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output JSON path (written under data/ if relative).",
    )
    parser.add_argument("--num_seqs", type=int, default=10000)
    parser.add_argument(
        "--num_base",
        type=int,
        default=None,
        help=(
            "Number of distinct base prompts to sample for cross-product mode."
            " Must be given together with --num_system; when set, this"
            " replaces --num_seqs sampling with a full base x system cross"
            " product."
        ),
    )
    parser.add_argument(
        "--num_system",
        type=int,
        default=None,
        help=(
            "Number of distinct adversarial system prompts to sample for"
            " cross-product mode. Must be given together with --num_base."
        ),
    )
    parser.add_argument("--crossover_prob", type=float, default=0.2)
    parser.add_argument("--mutation_prob", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="HF tokenizer name for mutation (only used if mutation_prob > 0).",
    )
    parser.add_argument(
        "--no_replacement",
        action="store_true",
        help="Sample input prompts without replacement (requires num_seqs <= len(input)).",
    )
    parser.add_argument(
        "--save_base_prompts",
        type=str,
        default=None,
        help=(
            "Cross-product mode only: also write the sampled base prompts"
            " (the num_base entries, pre-cross-product) to this JSON path"
            " for review."
        ),
    )
    parser.add_argument(
        "--save_system_prompts",
        type=str,
        default=None,
        help=(
            "Cross-product mode only: also write the sampled adversarial"
            " system prompts (the num_system distinct strings) to this JSON"
            " path for review."
        ),
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists() and not input_path.is_absolute():
        input_path = _DATA_DIR / input_path.name

    output_path = Path(args.output)
    if not output_path.is_absolute() and not str(output_path).startswith("data/"):
        output_path = _DATA_DIR / output_path

    if (args.num_base is None) != (args.num_system is None):
        parser.error("--num_base and --num_system must be given together")
    if args.num_base is None and (args.save_base_prompts or args.save_system_prompts):
        parser.error(
            "--save_base_prompts/--save_system_prompts require cross-product"
            " mode (--num_base/--num_system)"
        )

    tokenizer = None
    if args.mutation_prob > 0 and args.tokenizer:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    def _resolve(p: str) -> Path:
        path = Path(p)
        if not path.is_absolute() and not str(path).startswith("data/"):
            path = _DATA_DIR / path
        return path

    if args.num_base is not None:
        prompts, sampled_bases, system_prompts = build_cross_product_prompts(
            input_path=input_path,
            num_base=args.num_base,
            num_system=args.num_system,
            crossover_prob=args.crossover_prob,
            mutation_prob=args.mutation_prob,
            tokenizer=tokenizer,
            seed=args.seed,
        )
        mode_desc = f"{args.num_base} base x {args.num_system} system"

        if args.save_base_prompts:
            base_out = _resolve(args.save_base_prompts)
            base_out.parent.mkdir(parents=True, exist_ok=True)
            with open(base_out, "w") as f:
                json.dump(sampled_bases, f, indent=2)
            print(f"Wrote {len(sampled_bases)} sampled base prompts to {base_out}")

        if args.save_system_prompts:
            sys_out = _resolve(args.save_system_prompts)
            sys_out.parent.mkdir(parents=True, exist_ok=True)
            with open(sys_out, "w") as f:
                json.dump(system_prompts, f, indent=2)
            print(f"Wrote {len(system_prompts)} sampled system prompts to {sys_out}")
    else:
        prompts = build_jailbroken_prompts(
            input_path=input_path,
            num_seqs=args.num_seqs,
            crossover_prob=args.crossover_prob,
            mutation_prob=args.mutation_prob,
            tokenizer=tokenizer,
            seed=args.seed,
            with_replacement=not args.no_replacement,
        )
        mode_desc = f"num_seqs={args.num_seqs}"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(prompts, f, indent=2)

    print(f"Wrote {len(prompts)} genetic-jailbroken prompts ({mode_desc}) to {output_path}")
    print(f"  Source: {input_path} ({len(json.load(open(input_path)))} base prompts)")
    print(f"  Example system_prompt (truncated):\n  {prompts[0]['system_prompt'][:300]}...")
