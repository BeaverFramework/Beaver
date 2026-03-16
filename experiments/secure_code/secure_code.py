"""Secure code experiment — model must generate code without security vulnerabilities."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import numpy as np


DATASET_NAME = "secure_code"

_DATA_DIR = Path(__file__).parent / "data"
_DEFAULT_DATASET_PATH = str(_DATA_DIR / "instruct-v2.json")

_LANG_TO_GRAMMAR = {
    "c": "c",
    "cpp": "c",
    "go": "go",
    "python": "python",
    "rust": "rust",
}


# ── Constraint ──────────────────────────────────────────────────────────────


def check_call_fn(
    instance: dict, decoded_sequences: list[str], token_lists
) -> np.ndarray:
    token_lists = np.array(token_lists, dtype=object)
    should_check = token_lists.shape[1] % 10 == 0
    return np.full(len(decoded_sequences), should_check, dtype=bool)
    return np.full(len(decoded_sequences), True, dtype=bool)



def _get_context_code(instance: dict) -> str | None:
    if instance.get("variant") == "autocomplete":
        line_text = instance.get("line_text")
        origin_code = instance.get("origin_code")
        if line_text and origin_code:
            try:
                return origin_code.split(line_text)[0]
            except (ValueError, IndexError):
                pass
    return None


def constraint_fn(instance: dict, seq: str) -> bool:
    from insecure_code_detector import insecure_code_detector
    from insecure_code_detector.languages import Language

    _language_map = {
        "c": Language.C,
        "cpp": Language.CPP,
        "c++": Language.CPP,
        "csharp": Language.CSHARP,
        "c#": Language.CSHARP,
        "java": Language.JAVA,
        "javascript": Language.JAVASCRIPT,
        "js": Language.JAVASCRIPT,
        "python": Language.PYTHON,
        "py": Language.PYTHON,
        "php": Language.PHP,
        "rust": Language.RUST,
    }

    language_str = instance.get("language", "python")
    language = _language_map.get(language_str.lower(), Language.PYTHON)
    context_code = _get_context_code(instance)

    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        issues = loop.run_until_complete(
            asyncio.wait_for(
                insecure_code_detector.analyze(language, seq, code_before=context_code),
                timeout=2.0,
            )
        )
        loop.close()
        return len(issues) == 0
    except Exception:
        return True


def instance_context_fn(instance: dict) -> str:
    return f"{instance.get('language', 'python')}\x00{(_get_context_code(instance) or '').strip()}"


def grammar_for_language(language: str) -> str | None:
    return _LANG_TO_GRAMMAR.get(language.lower())


# ── Data loading ────────────────────────────────────────────────────────────


def load_prompts(
    dataset_path: str | None = None,
    language: str = "python",
    question_prefix: str = "",
    question_prefix_file: str | None = None,
    start_idx: int = 0,
    end_idx: int = -1,
    debug_ids: str | list[int] | None = None,
) -> list[dict]:
    if question_prefix_file:
        prefix_path = _DATA_DIR / question_prefix_file
        with open(prefix_path) as f:
            question_prefix = f.read()

    path = str(_DATA_DIR / dataset_path) if dataset_path else _DEFAULT_DATASET_PATH
    with open(path) as f:
        data = json.load(f)

    data = [
        item for item in data if item.get("language", "").lower() == language.lower()
    ]
    if not data:
        raise ValueError(f"No instances found for language: {language}")

    new_data = []
    for i, item in enumerate(data):
        item = dict(item)
        item["idx"] = i
        if "instruct" in path:
            item["prompt"] = (
                question_prefix
                + item["test_case_prompt"].replace("in C", f"in {item['language']}")
                + f"\nYou must only output code. Nothing but code no other text."
                f" ALWAYS start with ```{item['language']}\n and end with ```"
            )
        else:
            item["prompt"] = question_prefix + item["test_case_prompt"]
        new_data.append(item)

    print(f"Loaded {len(new_data)} secure_code instances for language: {language}")
    print(f"Using grammar: {grammar_for_language(language)}")

    if debug_ids is not None:
        if isinstance(debug_ids, str):
            debug_ids = [int(i) for i in debug_ids.split(",")]
        new_data = [new_data[i] for i in debug_ids]
    else:
        end = end_idx if end_idx != -1 else len(new_data)
        new_data = new_data[start_idx:end]

    return new_data


# ── CLI ─────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    import argparse
    import beaver

    parser = argparse.ArgumentParser(description="Run Secure Code experiment.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--server_addr", default=None)
    parser.add_argument("--auto_server", action="store_true")
    parser.add_argument("--server_port", type=int, default=8000)
    parser.add_argument("--dataset_path", default=None)
    parser.add_argument("--language", default="python")
    parser.add_argument("--question_prefix", default="")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=-1)
    parser.add_argument("--debug_ids", type=str, default=None)
    parser.add_argument("--log_dir", default="beaver_logs")
    args, _ = parser.parse_known_args()

    beaver.run(
        prompts=load_prompts(
            dataset_path=args.dataset_path,
            language=args.language,
            question_prefix=args.question_prefix,
            start_idx=args.start_idx,
            end_idx=args.end_idx,
            debug_ids=args.debug_ids,
        ),
        constraint_fn=constraint_fn,
        check_call_fn=check_call_fn,
        cache=True,
        cache_dataset_name=DATASET_NAME,
        instance_context_fn=instance_context_fn,
        grammar=grammar_for_language(args.language),
        model=args.model,
        server_addr=args.server_addr,
        auto_server=args.auto_server,
        server_port=args.server_port,
        log_dir=args.log_dir,
    )
