"""GSM Symbolic experiment — math reasoning with symbolic variables.

Data: downloaded on first use from the HuggingFace dataset anon2525/gsms.
"""

from __future__ import annotations

import re
import ast

import numpy as np
from concurrent.futures import ProcessPoolExecutor

DATASET_NAME = "gsm_symbolic"
GRAMMAR = "gsm"
SEMANTIC_SYMBOL = ">>"


# ── Constraint ──────────────────────────────────────────────────────────────
from z3 import And, If, ToInt, ToReal, Real, Solver, unsat

VAR_PATTERN = re.compile(r"\b[a-zA-Z_]\w*\b")
INT_SUB_PATTERN = re.compile(r"\bint\(")
FLOOR_DIV_PATTERN = re.compile(
    r"(?P<left>[^/]+?)\s*//\s*(?P<right>[^/]+?)(?=\s*[+\-*/%)]|$)"
)


def _z3_floor_div(x, y):
    return If(y != 0, ToInt(x / y), 0)


def _validate_expression_equivalence(expr1, expr2, var_types):
    # 2. FAST VARIABLE EXTRACTION
    var_names = set(VAR_PATTERN.findall(expr1 + " " + expr2)) - {
        "int",
        "ToInt",
        "z3_floor_div",
    }

    vars_dict = {}
    constraints = []

    # Z3 definitions
    def _floor(x):
        return If(x >= 0, ToInt(x), ToInt(x) - If(ToReal(ToInt(x)) == x, 0, 1))

    def _ceiling(x):
        return If(x >= 0, ToInt(x) + If(ToReal(ToInt(x)) == x, 0, 1), ToInt(x))

    for name in var_names:
        var = Real(name)
        vars_dict[name] = var
        v_type = var_types.get(name, "str")
        if v_type == "int":
            constraints.extend([var > 0, var == _floor(var), var == _ceiling(var)])
        elif v_type == "float" or v_type == "float between 0 and 1":
            constraints.append(var > 0)
            if "between 0 and 1" in v_type:
                constraints.append(var <= 1)
        elif v_type != "str":
            return False

    # 3. FAST STRING PROCESSING
    expr1 = INT_SUB_PATTERN.sub("ToInt(", expr1)
    expr2 = INT_SUB_PATTERN.sub("ToInt(", expr2)

    for _ in range(3):  # Simple loop is faster than while-prev for shallow nesting
        expr1 = FLOOR_DIV_PATTERN.sub(r"z3_floor_div(\1, \2)", expr1)
        expr2 = FLOOR_DIV_PATTERN.sub(r"z3_floor_div(\1, \2)", expr2)

    safe_env = {
        "__builtins__": None,
        **vars_dict,
        "ToInt": ToInt,
        "z3_floor_div": _z3_floor_div,
    }

    try:
        solver = Solver()
        solver.set("timeout", 5000)
        solver.add(constraints)
        solver.add(eval(expr1, safe_env) != eval(expr2, safe_env))
        return solver.check() == unsat
    except:
        return False


# def _validate_expression_equivalence(expr1, expr2, var_types):

#     var_names = set(re.findall(r"\b[a-zA-Z_]\w*\b", expr1 + " " + expr2))
#     var_names -= {"int"}

#     vars_dict = {}
#     constraints = []

#     def _floor(x):
#         return If(x >= 0, ToInt(x), ToInt(x) - If(ToReal(ToInt(x)) == x, 0, 1))

#     def _ceiling(x):
#         return If(x >= 0, ToInt(x) + If(ToReal(ToInt(x)) == x, 0, 1), ToInt(x))

#     for name in var_names:
#         var = Real(name)
#         vars_dict[name] = var
#         var_type = var_types.get(name, "str")
#         if var_type == "int":
#             constraints.append(var > 0)
#             constraints.append(And(var == _floor(var), var == _ceiling(var)))
#         elif var_type == "float":
#             constraints.append(var > 0)
#         elif var_type == "float between 0 and 1":
#             constraints.append(var > 0)
#             constraints.append(var <= 1)
#         elif var_type == "str":
#             continue
#         else:
#             return False

#     expr1 = re.sub(r"\bint\(", "ToInt(", expr1)
#     expr2 = re.sub(r"\bint\(", "ToInt(", expr2)

#     if "round(" in expr1 or "round(" in expr2:
#         return False

#     def _convert_floor_div(expression):
#         def replace(match):
#             return f"z3_floor_div({match.group('left').strip()}, {match.group('right').strip()})"

#         pattern = r"(?P<left>[^/]+?)\s*//\s*(?P<right>[^/]+?)(?=\s*[+\-*/%)]|$)"
#         prev = None
#         while prev != expression:
#             prev = expression
#             expression = re.sub(pattern, replace, expression)
#         return expression

#     expr1 = _convert_floor_div(expr1)
#     expr2 = _convert_floor_div(expr2)

#     def _z3_floor_div(x, y):
#         return If(y != 0, ToInt(x / y), 0)

#     safe_env = {
#         "__builtins__": None,
#         **vars_dict,
#         "ToInt": ToInt,
#         "z3_floor_div": _z3_floor_div,
#     }

#     try:
#         expr1_z3 = eval(expr1, safe_env)
#         expr2_z3 = eval(expr2, safe_env)
#     except Exception:
#         return False

#     solver = Solver()
#     solver.set("timeout", 5000)
#     solver.add(constraints)
#     try:
#         solver.add(expr1_z3 != expr2_z3)
#     except Exception:
#         return False
#     try:
#         return solver.check() == unsat
#     except Exception:
#         return False


def check_call_fn(
    instance: dict, decoded_sequences: list[str], token_lists
) -> np.ndarray:
    return np.array([(">>" in seq) for seq in decoded_sequences])


def constraint_fn(instance: dict, seqs: list[str]) -> list[bool]:
    # seq = seq.strip()
    # if "<<" in seq and ">>" in seq:
    #     seq = seq.split("<<")[1].split(">>")[0].strip()
    # else:
    #     return False

    expected_answer = instance.get("answer", "")
    v_types_str = instance.get("variable_types", "{}")

    try:
        # Use ast.literal_eval for safety and speed over eval()
        variable_types = ast.literal_eval(v_types_str)
    except:
        return [False] * len(seqs)

    cleaned_seqs = []
    for seq in seqs:
        s = seq.strip()
        if "<<" in s and ">>" in s:
            cleaned_seqs.append(s.split("<<")[1].split(">>")[0].strip())

        else:
            cleaned_seqs.append(None)  # Mark as invalid

    # with ProcessPoolExecutor() as executor:
    #     # Create a list of tasks for valid sequences
    #     future_to_idx = {
    #         executor.submit(
    #             _validate_expression_equivalence, s, expected_answer, variable_types
    #         ): i
    #         for i, s in enumerate(cleaned_seqs)
    #         if s is not None
    #     }

    #     for future in future_to_idx:
    #         idx = future_to_idx[future]
    #         try:
    #             results[idx] = future.result()
    #         except:
    #             results[idx] = False

    # for clean_seq in cleaned_seqs:
    #     if clean_seq is not None:
    #         results.append(
    #             _validate_expression_equivalence(
    #                 clean_seq, expected_answer, variable_types
    #             )
    #         )
    #     else:
    #         results.append(False)
    results = [
        (
            _validate_expression_equivalence(clean_seq, expected_answer, variable_types)
            if clean_seq is not None
            else False
        )
        for clean_seq in cleaned_seqs
    ]

    # try:
    #     variable_types = eval(variable_types_str)
    # except Exception:
    #     return False

    return _validate_expression_equivalence(seq, expected_answer, variable_types)


def instance_context_fn(instance: dict) -> str:
    return f"{instance.get('answer', '')}\x00{instance.get('variable_types', '')}"


# ── Data loading ────────────────────────────────────────────────────────────

import os
import yaml as _yaml

_TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), "gsm_symbolic_template.yaml")


def _load_template():
    with open(_TEMPLATE_PATH) as f:
        return _yaml.safe_load(f)


def load_prompts(
    num_shots: int = 8,
    start_idx: int = 0,
    end_idx: int = -1,
    debug_ids: str | list[int] | None = None,
) -> list[dict]:
    from datasets import load_dataset

    template = _load_template()
    ds = load_dataset("anon2525/gsms", split="train")

    if debug_ids is not None:
        if isinstance(debug_ids, str):
            debug_ids = [int(i) for i in debug_ids.split(",")]
        ds = ds.select(debug_ids)
    else:
        end = end_idx if end_idx != -1 else len(ds)
        ds = ds.select(range(start_idx, end))

    system_prompt = template["instruction"].strip()
    fewshot_messages = [
        {"prompt": shot["question"], "response": shot["response"]}
        for shot in template["fewshots"][:num_shots]
    ]

    data = []
    for row in ds:
        d = dict(row)
        d["prompt"] = d.pop("question")
        d["system_prompt"] = system_prompt
        d["fewshot_messages"] = fewshot_messages
        data.append(d)
    return data


# ── CLI ─────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    import argparse
    import beaver

    parser = argparse.ArgumentParser(description="Run GSM Symbolic experiment.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--server_addr", default=None)
    parser.add_argument("--auto_server", action="store_true")
    parser.add_argument("--server_port", type=int, default=8000)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=-1)
    parser.add_argument("--debug_ids", type=str, default=None)
    parser.add_argument("--log_dir", default="beaver_logs")
    args, _ = parser.parse_known_args()

    beaver.run(
        prompts=load_prompts(
            start_idx=args.start_idx,
            end_idx=args.end_idx,
            debug_ids=args.debug_ids,
        ),
        constraint_fn=constraint_fn,
        check_call_fn=check_call_fn,
        cache=True,
        cache_dataset_name=DATASET_NAME,
        instance_context_fn=instance_context_fn,
        grammar=GRAMMAR,
        semantic_symbol=SEMANTIC_SYMBOL,
        model=args.model,
        server_addr=args.server_addr,
        auto_server=args.auto_server,
        server_port=args.server_port,
        log_dir=args.log_dir,
    )
