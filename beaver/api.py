"""High-level BEAVER API.

Example — custom constraint::

    import beaver

    results = beaver.run(
        prompts=[{"question": "What is 2+2?"}],
        constraint_fn=lambda instance, seq: "4" in seq,
        model="meta-llama/Llama-3.1-8B-Instruct",
        server_addr="http://localhost:8000",
    )

Example — experiment file::

    from experiments.gsm_symbolic.gsm_symbolic import (
        load_prompts, constraint_fn, check_call_fn, instance_context_fn
    )

    results = beaver.run(
        prompts=load_prompts(start_idx=0, end_idx=100),
        constraint_fn=constraint_fn,
        check_call_fn=check_call_fn,
        cache=True,
        cache_dataset_name="gsm_symbolic",
        instance_context_fn=instance_context_fn,
        grammar="gsm",
        semantic_symbol=">>",
        model="meta-llama/Llama-3.1-8B-Instruct",
        auto_server=True,
    )
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Callable

import numpy as np

from beaver.utils import new_log_dir
from beaver.logging import get_log_data, summarize_log_data, summarize_profile_data
from beaver.verifiers.frontier_verifier import FrontierVerifier
from beaver.verifiers.sampling_verifier import SamplingVerifier
from beaver.constraints.base_constraints import register_constraint



def _default_check_call_fn(_inst, seqs, _toks):
    return np.ones(len(seqs), dtype=bool)


def _default_instance_context_fn(_inst):
    return ""


def _prepare_dataset_from_prompts(prompts: list[dict]):
    from datasets import Dataset

    data = []
    for i, item in enumerate(prompts):
        row = dict(item)
        if "prompt" not in row:
            raise ValueError(
                f"Each prompt dict must contain a 'prompt' key. Got: {row}"
            )
        row.setdefault("idx", i)
        data.append(row)
    return Dataset.from_list(data)


# ── Main run() function ────────────────────────────────────────────────────


def run(
    *,
    prompts: list[dict],
    constraint_fn: Callable,
    check_call_fn: Callable | None = None,
    cache: bool = False,
    cache_dataset_name: str | None = None,
    instance_context_fn: Callable | None = None,
    model: str,
    server_addr: str | None = None,
    auto_server: bool = True,
    # Model config — for auto_server
    model_config: str | Path | dict | None = None,
    # Experiment params
    verifier: str = "frontier",
    gen_length: int = 32,
    temperature: float = 1.0,
    top_p: float = 0.99,
    top_k: int = -1,
    max_iterations: int = 100,
    epsilon: float = 0.01,
    max_workers: int = 16,
    num_logprobs: int = 100,
    max_frontier_size: int = 10000,
    max_frontier_prob: float = 1.0,
    frontier_scoring_strategy: str = "highest-prob",
    use_grammar: bool = False,
    use_chat_template: bool = True,
    system_message: str | None = None,
    fewshot_messages: list | None = None,
    # Grammar / semantic symbol
    grammar: str | None = None,
    semantic_symbol: str | None = None,
    # Output
    log_dir: str = "logging",
    verbose: bool = False,
    # Server auto-management
    server_port: int | None = None,
    gpu_visible_devices: str | None = None,
    extra_vllm_args: list[str] | None = None,
) -> list[dict]:
    """Run BEAVER verification on a model.

    Args:
        prompts: List of dicts with at least ``"question"`` key.
        constraint_fn: ``(instance, seq) -> bool``. Required.
        check_call_fn: ``(instance, seqs, token_lists) -> np.ndarray[bool]``.
            Optional pre-filter called before ``constraint_fn`` to skip sequences
            that trivially don't need the full check (e.g. too short, wrong prefix).
            Return ``True`` for sequences that *should* be checked.
            Defaults to always checking all sequences.
        cache: Enable constraint result caching. Defaults to ``False``.
            When ``True``, ``cache_dataset_name`` is required and
            ``instance_context_fn`` should be provided if the constraint result
            depends on per-instance fields (e.g. expected answer).
        cache_dataset_name: Cache namespace key (e.g. ``"gsm_symbolic"``).
            Required when ``cache=True``.
        instance_context_fn: ``(instance) -> str``. Returns a string that
            varies per instance and is included in the cache key. Only used
            when ``cache=True``.
        model: HuggingFace model ID or local path (required).
        server_addr: URL of a running vLLM server. Either this or
            ``auto_server=True`` must be set.
        auto_server: Start and stop a vLLM server automatically.
        model_config: Dict or path to a YAML file with vLLM server overrides.
            Merged on top of ``configs/models/default.yaml``.
        verifier: ``"frontier"`` or ``"sampling"``.
        gen_length: Max tokens per sequence.
        temperature / top_p / top_k: Sampling parameters.
        max_iterations: Max verification iterations per instance.
        epsilon: Convergence threshold.
        max_workers: Parallel worker processes.
        num_logprobs: Top-logprob tokens per step.
        max_frontier_size / max_frontier_prob / frontier_scoring_strategy:
            Frontier pruning settings.
        use_grammar: Apply grammar constraints.
        use_chat_template: Apply model chat template.
        num_shots: Few-shot examples to prepend.
        system_message / fewshot_messages: Chat template content.
        grammar: Grammar name (looked up in ``beaver/grammars/``).
        semantic_symbol: Symbol used to mark semantic completion (e.g. ``">>``").
        log_dir: Directory for run logs.
        verbose: Verbose output.
        server_port: Port for the auto-managed vLLM server.
        gpu_visible_devices: ``CUDA_VISIBLE_DEVICES`` for the auto server.
        extra_vllm_args: Extra raw flags appended to ``vllm serve``.

    Returns:
        List of per-instance result dicts.
    """
    if prompts is None:
        raise ValueError("'prompts' is required.")
    if constraint_fn is None:
        raise ValueError(
            "A 'constraint_fn' is required. "
            "Signature: constraint_fn(instance: dict, sequence: str) -> bool"
        )
    if cache and (cache_dataset_name is None):
        raise ValueError("'cache_dataset_name' is required when 'cache=True'.")
    if instance_context_fn is not None and not cache:
        raise ValueError("'instance_context_fn' requires 'cache=True'.")

    fewshot_messages = fewshot_messages or []


    # ── Prepare dataset and register constraint ────────────────────────────
    ds = _prepare_dataset_from_prompts(prompts)


    if cache_dataset_name is None:
        effective_dataset_name = "custom"
    else:
        effective_dataset_name = cache_dataset_name
    register_constraint(
            effective_dataset_name,
            check_call_fn=check_call_fn if check_call_fn else _default_check_call_fn,
            instance_context_fn=instance_context_fn or _default_instance_context_fn,
            check_fn=constraint_fn,
    )

    # ── Create log dir & save args ─────────────────────────────────────────
    run_log_dir = new_log_dir(Path(log_dir))

    # ── Auto server management ─────────────────────────────────────────────
    server_proc = None
    if auto_server and server_addr is None:
        from beaver.server import start_server, stop_server
        from beaver.server import load_model_config as _load_model_cfg

        _mcfg = _load_model_cfg(
            model,
            model_config if isinstance(model_config, (str, Path)) else None,
        )
        resolved_port = (
            server_port if server_port is not None else int(_mcfg.get("port", 8000))
        )
        server_addr = f"http://localhost:{resolved_port}"
        server_proc = start_server(
            model_id=model,
            port=resolved_port,
            gpu_visible_devices=gpu_visible_devices,
            model_config_path=(
                model_config if isinstance(model_config, (str, Path)) else None
            ),
            extra_vllm_args=extra_vllm_args,
            log_dir=run_log_dir,
        )
        if server_proc is None:
            raise RuntimeError(
                f"Failed to start vLLM server for model '{model}' on port {resolved_port}."
            )
    elif server_addr is None:
        raise ValueError(
            "Provide 'server_addr' (URL of a running vLLM server) or set auto_server=True."
        )

    try:
        return _run_inner(
            dataset=ds,
            dataset_name=effective_dataset_name,
            use_cache=cache,
            model=model,
            server_addr=server_addr,
            verifier=verifier,
            gen_length=gen_length,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_iterations=max_iterations,
            epsilon=epsilon,
            max_workers=max_workers,
            num_logprobs=num_logprobs,
            max_frontier_size=max_frontier_size,
            max_frontier_prob=max_frontier_prob,
            frontier_scoring_strategy=frontier_scoring_strategy,
            use_grammar=use_grammar,
            use_chat_template=use_chat_template,
            system_message=system_message,
            fewshot_messages=fewshot_messages,
            grammar=grammar,
            semantic_symbol=semantic_symbol,
            log_dir=run_log_dir,
            verbose=verbose,
        )
    finally:
        if server_proc is not None:
            from beaver.server import stop_server

            stop_server(server_proc)


# ── Console tee ───────────────────────────────────────────────────────────


class _TeeStream:
    """Write to both the original stream and a file handle simultaneously."""

    def __init__(self, stream, fh):
        self._stream = stream
        self._fh = fh

    def write(self, data):
        self._stream.write(data)
        if not self._fh.closed:
            try:
                self._fh.write(data)
            except (ValueError, OSError):
                pass
        return len(data)

    def flush(self):
        self._stream.flush()
        if not self._fh.closed:
            try:
                self._fh.flush()
            except (ValueError, OSError):
                pass

    def __getattr__(self, name):
        return getattr(self._stream, name)


# ── Inner run (all params fully resolved) ─────────────────────────────────


def _run_inner(
    dataset,
    dataset_name,
    use_cache,
    model,
    server_addr,
    verifier,
    gen_length,
    temperature,
    top_p,
    top_k,
    max_iterations,
    epsilon,
    max_workers,
    num_logprobs,
    max_frontier_size,
    max_frontier_prob,
    frontier_scoring_strategy,
    use_grammar,
    use_chat_template,
    system_message,
    fewshot_messages,
    grammar,
    semantic_symbol,
    log_dir,
    verbose,
) -> list[dict]:

    _console_fh = open(log_dir / "console.log", "w")
    _orig_stdout, _orig_stderr = sys.stdout, sys.stderr
    sys.stdout = _TeeStream(sys.stdout, _console_fh)
    sys.stderr = _TeeStream(sys.stderr, _console_fh)

    try:
        run_args = dict(
            model=model,
            dataset=dataset_name,
            server_addr=server_addr,
            verifier=verifier,
            gen_length=gen_length,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_iterations=max_iterations,
            epsilon=epsilon,
            max_workers=max_workers,
            num_logprobs=num_logprobs,
            max_frontier_size=max_frontier_size,
            max_frontier_prob=max_frontier_prob,
            frontier_scoring_strategy=frontier_scoring_strategy,
            use_grammar=use_grammar,
            use_chat_template=use_chat_template,
            log_dir=str(log_dir),
            verbose=verbose,
        )
        with open(log_dir / "run_args.json", "w") as f:
            json.dump(run_args, f, indent=2)

        # ── Instantiate and run verifier ───────────────────────────────────────
        common_kwargs = dict(
            grammar=grammar,
            gen_length=gen_length,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            semantic_symbol=semantic_symbol,
            max_iterations=max_iterations,
            epsilon=epsilon,
            verbose=verbose,
            max_workers=max_workers,
            num_logprobs=num_logprobs,
            use_grammar=use_grammar,
            chat_mode=use_chat_template,
            system_message=system_message,
            fewshot_messages=fewshot_messages,
            use_cache=use_cache,
        )

        if verifier == "frontier":
            llm = FrontierVerifier(
                model,
                dataset_name,
                server_addr,
                max_frontier_size=max_frontier_size,
                max_frontier_prob=max_frontier_prob,
                frontier_scoring_strategy=frontier_scoring_strategy,
                **common_kwargs,
            )
        elif verifier == "sampling":
            llm = SamplingVerifier(
                model, dataset_name, server_addr, **common_kwargs
            )
        else:
            raise ValueError(
                f"Unknown verifier: '{verifier}'. Choose 'frontier' or 'sampling'."
            )

        results = llm(dataset, log_dir)

        print(f"\n[beaver] Run logs: {log_dir}")
        all_data = get_log_data(log_dir)
        if all_data:
            summarize_log_data(all_data, log_dir)
            summarize_profile_data(log_dir)

        return results
    finally:
        sys.stdout = _orig_stdout
        sys.stderr = _orig_stderr
        _console_fh.close()
