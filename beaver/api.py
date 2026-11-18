"""High-level BEAVER API.

Example — custom constraint::

    import beaver

    results = beaver.run(
        prompts=[{"prompt": "What is 2+2?"}],
        constraint_fn=lambda instance, seqs: ["4" in s for s in seqs],
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
import os
import sys
from pathlib import Path
from typing import Callable

import numpy as np

from beaver.constraints.base_constraints import register_constraint
from beaver.logging import get_log_data, summarize_log_data, summarize_profile_data
from beaver.utils import new_log_dir
from beaver.verifiers import SamplingVerifier, FrontierVerifier
from beaver.verifiers.cached_frontier_verifier import CachedFrontierVerifier
from beaver.verifiers.distribution_verifier import DistributionVerifier
from beaver.verifiers.iterative_verifier import IterativeVerifier


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
    complete_fn: Callable | None = None,
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
    backend: str | None = None,
    vllm_max_model_len: int | None = None,
    vllm_gpu_memory_utilization: float | None = None,
    vllm_skip_prefix_cache_read: bool = False,
    hf_local_tl_recipe: str | None = None,
    hf_local_dtype: str = "bfloat16",
    hf_local_hidden_layers: list[int] | None = None,
    hf_local_hook: str = "resid_post",
    gen_length: int = 32,
    temperature: float = 1.0,
    top_p: float = 0.99,
    top_k: int = -1,
    decoded_dist: bool = False,
    max_iterations: int = 500,
    epsilon: float = 0.1,
    early_termination_threshold: float | None = 0.9,
    prefix_closed: bool = False,
    max_workers: int = 16,
    num_logprobs: int = 200,
    score_chunk_size: int = 64,
    max_frontier_size: int = 10000,
    max_frontier_prob: float = 1.0,
    frontier_scoring_strategy: str | None = None,
    use_grammar: bool = False,
    use_chat_template: bool = True,
    system_message: str | None = None,
    fewshot_messages: list | None = None,
    enable_thinking: bool = False,
    chat_template_kwargs: dict | None = None,
    # Grammar / semantic symbol
    grammar: str | None = None,
    semantic_symbol: str | None = None,
    # Probe / Neural Value Function (representation verifier)
    probe_path: str | None = None,
    probe_layer: int | None = 16,
    probe_gpu: str | None = None,
    direction_layers: list[int] | str | None = None,
    direction_layer_weights: list[float] | str | None = None,
    stopping_criterion: str = "max_seq_prob",
    expand_batch_size: int = 1,
    gate_theta: float = 0.0,
    gate_pool_size: int = 64,
    gate_warmup: int = 20,
    ucb_c: float = 1.0,
    nvf_type: str = "knn",
    nvf_data_path: str | None = None,
    knn_k: int = 5,
    knn_normalize: bool = True,
    surrogate_model: str | None = None,
    surrogate_server_addr: str | None = None,
    # Cache (cachedfrontier verifier)
    cache_path: str | None = None,
    # Iterative verifier (beam search over system prompts). The mutator LLM
    # uses surrogate_server_addr and surrogate_model.
    beam_width: int = 4,
    max_rounds: int = 10,
    patience: int = 3,
    mutator_num_prompts: int | None = None,
    mutator_max_calls: int = 3,
    stop_violation_prob: float | None = None,
    score_mode: str = "frontier",
    num_samples: int = 10,
    tree_threshold: float | None = None,
    tree_layer: int | None = None,
    dump_trie: bool = False,
    mutator_system_prompt: bool = False,
    prune_unreachable: bool = True,
    cache_hit_prob: float = 1e-2,
    cache_evict_after: int | None = None,
    cache_max_size: int | None = None,
    # Iterative verifier: start each group from the token trie an earlier run
    # saved in cache_path/<group>.json (same dataset, e.g. a model after a
    # training step). Labels depend only on the text, so bounds stay sound.
    cache_warm_start: bool = False,
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
        prompts: List of dicts with at least a ``"prompt"`` key. Optional
            per-instance keys: ``"system_prompt"``, ``"fewshot_messages"``,
            and ``"group"`` (required by ``verifier="cachedfrontier"`` /
            ``"iterative"``: instances in a group share a token-trie cache).
        constraint_fn: ``(instance, seqs: list[str]) -> list[bool]``.
            Required. Batched: receives every sequence that needs checking
            in one call and returns one bool per sequence (``True`` =
            constraint satisfied).
        complete_fn: ``(instance, seqs, token_lists) -> np.ndarray[bool]``.
            Optional early-completion hook: ``True`` marks a sequence as
            complete (and triggers the constraint check) before EOS /
            ``gen_length``, e.g. once a refusal has been detected.
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
        verifier: ``"frontier"`` (branch-and-bound, default), ``"sampling"``
            (Monte Carlo baseline), ``"cachedfrontier"`` (frontier with a
            per-group token-trie cache), ``"iterative"`` (beam search over
            system prompts on top of the cached frontier),
            ``"representation"`` (frontier guided by a neural value
            function over hidden states) or ``"distribution"``.
        backend: ``"vllm"`` (default; HTTP to ``vllm serve``), ``"hf"``
            (HTTP to ``beaver.hf_server``, can return hidden states),
            ``"vllm-local"`` or ``"hf-local"`` (in-process engines, no
            server).
        gen_length: Max tokens per sequence.
        temperature / top_p / top_k: Sampling parameters.
        decoded_dist: If True, verify the *decoded* distribution -- what a
            sampler at (temperature, top_p, top_k) actually draws from, with
            the truncated tokens at probability zero and the rest
            renormalised. The default (False) is the historical reading:
            top_p / top_k only limit which children are explored, the mass
            they drop is carried as unresolved, and temperature is ignored,
            so the bounds refer to the raw distribution. Temperature must be
            in [0, 1] in decoded mode (0 = greedy).
        max_iterations: Max verification iterations per instance.
        epsilon: Convergence threshold.
        early_termination_threshold: If not None, stop verification of an
            instance as soon as its ``upper_bound`` drops below this value.
            Useful when the caller only cares whether the upper bound can
            ever reach a given confidence level.
        prefix_closed: If False (default), the constraint is only ever
            evaluated once a sequence actually completes (EOS /
            ``gen_length`` / a dataset's ``complete_fn``) — never on a
            partial prefix. If True, a dataset's ``check_call_fn`` may
            trigger the semantic constraint check on an incomplete sequence,
            treating a mid-generation violation as terminal.
        max_workers: Parallel worker processes.
        num_logprobs: Top-logprob tokens per step.
        score_chunk_size: Max sequences per cached-sequence scoring request.
            Smaller chunks send sequentially so vLLM prefix caching reuses the
            shared prefix and requests stay under the read timeout.
        max_frontier_size / max_frontier_prob / frontier_scoring_strategy:
            Frontier pruning settings.
        use_grammar: Apply grammar constraints.
        use_chat_template: Apply model chat template.
        system_message / fewshot_messages: Chat template content.
        enable_thinking: If True, the model's chat template renders a
            reasoning block (e.g. Qwen's ``<think>...</think>``) that the
            model fills in before its actual output. Defaults to False.
        chat_template_kwargs: Dict forwarded verbatim to the model's own
            ``tokenizer.apply_chat_template``. Different templates read
            different keys -- Qwen3-Thinking reads ``enable_thinking``,
            gpt-oss reads ``reasoning_effort`` -- and Jinja ignores unknown
            ones silently, so a key the template does not read has no effect
            and raises nothing. Defaults to
            ``{"enable_thinking": enable_thinking}``, which is what every
            pre-existing config rendered with.
        grammar: Grammar name (looked up in ``beaver/grammars/``).
        semantic_symbol: Symbol used to mark semantic completion (e.g. ``">>``").
        nvf_type: Neural value function type for ``"representation"`` verifier.
            ``"knn"`` (default), ``"linear"`` (linear/MLP ViolationProbe),
            ``"nonlinear"`` (NonlinearProbe) or ``"direction"`` (violation
            direction). KNN requires ``nvf_data_path`` pointing to a ``.pt``
            file; linear/nonlinear require ``probe_path`` pointing to a saved
            checkpoint; direction requires ``probe_path`` pointing to a
            directory containing ``directions.npz``.
        nvf_data_path: Path to ``.pt`` training data file for KNN estimator
            (output of ``extract_data.py``).
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
            "Signature: constraint_fn(instance: dict, sequences: list[str]) -> list[bool]"
        )
    if cache and (cache_dataset_name is None):
        raise ValueError("'cache_dataset_name' is required when 'cache=True'.")
    if instance_context_fn is not None and not cache:
        raise ValueError("'instance_context_fn' requires 'cache=True'.")

    # Resolve the node-selection strategy once, here, so run_args.json records
    # what actually ran. Previously the representation branch *forced*
    # "fitness" unconditionally, silently discarding an explicit caller choice.
    if frontier_scoring_strategy is None:
        frontier_scoring_strategy = (
            "fitness" if verifier == "representation" else "max-mu"
        )

    # CLI/YAML pass these as comma-separated strings; normalize to lists so
    # run_args.json records them in a form that can be fed straight back in.
    if isinstance(direction_layers, str):
        direction_layers = [int(x) for x in direction_layers.split(",") if x.strip()]
    if isinstance(hf_local_hidden_layers, str):
        hf_local_hidden_layers = [
            int(x) for x in hf_local_hidden_layers.split(",") if x.strip()
        ]
    if isinstance(direction_layer_weights, str):
        direction_layer_weights = [
            float(x) for x in direction_layer_weights.split(",") if x.strip()
        ]

    if verifier == "representation":
        # The NVF needs transformer hidden states, which only the two HF-based
        # backends can produce (vLLM exposes none: its OpenAI server has no
        # hook and offline RequestOutput carries no states). Previously this
        # branch *forced* backend="hf", silently discarding an explicit
        # --backend hf-local. Now an unset backend defaults to "hf" and an
        # explicit one is honoured or rejected, never overridden.
        from beaver.model import HIDDEN_STATE_BACKENDS

        if backend is None:
            backend = "hf"
        elif backend not in HIDDEN_STATE_BACKENDS:
            raise ValueError(
                f"verifier='representation' needs hidden states, which "
                f"backend={backend!r} cannot return. Use one of "
                f"{HIDDEN_STATE_BACKENDS}."
            )
        if nvf_type in ("linear", "nonlinear") and (
            probe_path is None or probe_layer is None
        ):
            raise ValueError(
                f"Define both probe_path and probe_layer to use NVF of type linear / nonlinear"
            )
        elif nvf_type == "knn" and nvf_data_path is None:
            raise ValueError(f"Define nvf_data_path to use NVF of type KNN")
        elif nvf_type == "direction" and probe_path is None:
            raise ValueError(
                "Define probe_path (a directory containing directions.npz from "
                "violation_direction_heretic.py) to use NVF of type direction"
            )

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
        complete_fn=complete_fn,
    )

    # ── Create log dir & save args ─────────────────────────────────────────
    run_log_dir = new_log_dir(Path(log_dir))

    # ── Backend validation ─────────────────────────────────────────────────
    # CUDA_VISIBLE_DEVICES must be set before *anything* creates a CUDA
    # context. For the spawn-pool path init_worker_state() sets it in the
    # fresh child, but local backends clamp max_workers to 1 and run
    # init_fn in-process (base_verifier._run_pool), where it would land too
    # late to matter. Setting it here, at the top of run(), covers both.
    if gpu_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_visible_devices)

    from beaver.model import BACKENDS

    if backend is None:
        backend = "vllm"
    if backend not in BACKENDS:
        raise ValueError(f"Unknown backend {backend!r}. Expected one of {BACKENDS}.")

    if backend in ("vllm-local", "hf-local") and max_workers and max_workers > 1:
        # One GPU-resident engine per process, so the multiprocessing pool
        # can't be used: each worker would try to load its own copy.
        print(
            f"[beaver] backend='{backend}' runs in a single process; "
            f"clamping max_workers {max_workers} -> 1."
        )
        max_workers = 1

    # ── Auto server management ─────────────────────────────────────────────
    server_proc = None
    server_kind = None  # "vllm" or "hf"
    if backend in ("vllm-local", "hf-local"):
        # No server at all — the engine is loaded in-process on first call.
        # server_addr stays None; nothing on this path opens a socket.
        server_addr = None
        # vllm-local reads the same list (see local_vllm._hidden_layers), so
        # both in-process backends derive their capture set identically.
        if backend in ("hf-local", "vllm-local") and hf_local_hidden_layers is None:
            # Same layer set hf_server would have been launched with, so the
            # two backends are interchangeable. Without this hf_local falls
            # back to [probe_layer] and DirectionNVF silently sums one term
            # instead of all of them.
            if verifier == "representation" and nvf_type == "direction" and probe_path:
                if direction_layers:
                    # Only capture what the NVF will actually score -- one
                    # layer instead of 28 is 28x less activation to move.
                    hf_local_hidden_layers = list(direction_layers)
                else:
                    from beaver.nvf import load_direction_stack

                    hf_local_hidden_layers = list(
                        range(load_direction_stack(probe_path).shape[0] - 1)
                    )
            # Only the representation verifier ever reads hidden states.
            # probe_layer defaults to 16, so without this guard a plain
            # frontier run builds the engine with the capture connector,
            # which clones a GPU tensor per request that nothing pops
            # (OOM after ~2.4k enron instances) and pays the
            # CacheOnlyAttentionLayer cost on every forward.
            elif verifier == "representation" and probe_layer is not None:
                hf_local_hidden_layers = [probe_layer]
    elif backend == "hf" and server_addr is None:
        # Start a beaver.hf_server process that exposes an OpenAI-compatible
        # /v1/completions endpoint plus hidden-state hooks.
        from beaver.hf_server import start_server as start_hf_server

        resolved_port = server_port if server_port is not None else 8082
        server_addr = f"http://localhost:{resolved_port}"
        if verifier == "representation" and nvf_type == "direction" and probe_path is not None:
            from beaver.nvf import load_direction_stack

            hidden_layers = list(range(load_direction_stack(probe_path).shape[0] - 1))
        else:
            hidden_layers = [probe_layer] if probe_layer is not None else None
        server_proc = start_hf_server(
            model_id=model,
            port=resolved_port,
            gpu_visible_devices=gpu_visible_devices,
            hidden_state_layers=hidden_layers,
            log_dir=run_log_dir,
            quiet=True,
        )
        if server_proc is None:
            raise RuntimeError(
                f"Failed to start HF server for model '{model}' on port {resolved_port}."
            )
        server_kind = "hf"
    elif backend == "hf":
        # server_addr provided — assume an hf_server is already running there.
        pass
    elif auto_server and server_addr is None:
        from beaver.server import load_model_config as _load_model_cfg
        from beaver.server import start_server, stop_server

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
        server_kind = "vllm"
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
            backend=backend,
            vllm_max_model_len=vllm_max_model_len,
            vllm_gpu_memory_utilization=vllm_gpu_memory_utilization,
            vllm_skip_prefix_cache_read=vllm_skip_prefix_cache_read,
            hf_local_tl_recipe=hf_local_tl_recipe,
            hf_local_dtype=hf_local_dtype,
            hf_local_hidden_layers=hf_local_hidden_layers,
            hf_local_hook=hf_local_hook,
            gpu_visible_devices=gpu_visible_devices,
            gen_length=gen_length,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            decoded_dist=decoded_dist,
            max_iterations=max_iterations,
            epsilon=epsilon,
            early_termination_threshold=early_termination_threshold,
            prefix_closed=prefix_closed,
            max_workers=max_workers,
            num_logprobs=num_logprobs,
            score_chunk_size=score_chunk_size,
            max_frontier_size=max_frontier_size,
            max_frontier_prob=max_frontier_prob,
            frontier_scoring_strategy=frontier_scoring_strategy,
            use_grammar=use_grammar,
            use_chat_template=use_chat_template,
            system_message=system_message,
            fewshot_messages=fewshot_messages,
            enable_thinking=enable_thinking,
            chat_template_kwargs=chat_template_kwargs,
            grammar=grammar,
            semantic_symbol=semantic_symbol,
            probe_path=probe_path,
            probe_layer=probe_layer,
            probe_gpu=probe_gpu,
            direction_layers=direction_layers,
            direction_layer_weights=direction_layer_weights,
            stopping_criterion=stopping_criterion,
            expand_batch_size=expand_batch_size,
            gate_theta=gate_theta,
            gate_pool_size=gate_pool_size,
            gate_warmup=gate_warmup,
            ucb_c=ucb_c,
            nvf_type=nvf_type,
            nvf_data_path=nvf_data_path,
            knn_k=knn_k,
            knn_normalize=knn_normalize,
            surrogate_model=surrogate_model,
            surrogate_server_addr=surrogate_server_addr,
            log_dir=run_log_dir,
            verbose=verbose,
            cache_path=cache_path,
            beam_width=beam_width,
            max_rounds=max_rounds,
            patience=patience,
            mutator_num_prompts=mutator_num_prompts,
            mutator_max_calls=mutator_max_calls,
            stop_violation_prob=stop_violation_prob,
            score_mode=score_mode,
            num_samples=num_samples,
            tree_threshold=tree_threshold,
            tree_layer=tree_layer,
            dump_trie=dump_trie,
            mutator_system_prompt=mutator_system_prompt,
            prune_unreachable=prune_unreachable,
            cache_hit_prob=cache_hit_prob,
            cache_evict_after=cache_evict_after,
            cache_max_size=cache_max_size,
            cache_warm_start=cache_warm_start,
        )
    finally:
        if server_proc is not None:
            if server_kind == "hf":
                from beaver.hf_server import stop_server as stop_hf_server

                stop_hf_server(server_proc)
            else:
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
    backend,
    vllm_max_model_len,
    vllm_gpu_memory_utilization,
    vllm_skip_prefix_cache_read,
    hf_local_tl_recipe,
    hf_local_dtype,
    hf_local_hidden_layers,
    hf_local_hook,
    gpu_visible_devices,
    gen_length,
    temperature,
    top_p,
    top_k,
    decoded_dist,
    max_iterations,
    epsilon,
    early_termination_threshold,
    prefix_closed,
    max_workers,
    num_logprobs,
    score_chunk_size,
    max_frontier_size,
    max_frontier_prob,
    frontier_scoring_strategy,
    use_grammar,
    use_chat_template,
    system_message,
    fewshot_messages,
    enable_thinking,
    chat_template_kwargs,
    grammar,
    semantic_symbol,
    probe_path,
    probe_layer,
    probe_gpu,
    direction_layers,
    direction_layer_weights,
    stopping_criterion,
    expand_batch_size,
    gate_theta,
    gate_pool_size,
    gate_warmup,
    ucb_c,
    nvf_type,
    nvf_data_path,
    knn_k,
    knn_normalize,
    surrogate_server_addr,
    surrogate_model,
    log_dir,
    verbose,
    cache_path=None,
    beam_width=4,
    max_rounds=10,
    patience=3,
    mutator_num_prompts=None,
    mutator_max_calls=3,
    stop_violation_prob=None,
    score_mode="frontier",
    num_samples=10,
    tree_threshold=None,
    tree_layer=None,
    dump_trie=False,
    mutator_system_prompt=False,
    prune_unreachable=True,
    cache_hit_prob=1e-2,
    cache_evict_after=None,
    cache_max_size=None,
    cache_warm_start=False,
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
            decoded_dist=decoded_dist,
            max_iterations=max_iterations,
            epsilon=epsilon,
            early_termination_threshold=early_termination_threshold,
            max_workers=max_workers,
            num_logprobs=num_logprobs,
            score_chunk_size=score_chunk_size,
            max_frontier_size=max_frontier_size,
            max_frontier_prob=max_frontier_prob,
            frontier_scoring_strategy=frontier_scoring_strategy,
            use_grammar=use_grammar,
            use_chat_template=use_chat_template,
            enable_thinking=enable_thinking,
            chat_template_kwargs=chat_template_kwargs,
            backend=backend,
            vllm_max_model_len=vllm_max_model_len,
            vllm_gpu_memory_utilization=vllm_gpu_memory_utilization,
            gpu_visible_devices=gpu_visible_devices,
            nvf_type=nvf_type,
            nvf_data_path=nvf_data_path,
            knn_k=knn_k,
            knn_normalize=knn_normalize,
            probe_path=probe_path,
            probe_layer=probe_layer,
            probe_gpu=probe_gpu,
            direction_layers=direction_layers,
            direction_layer_weights=direction_layer_weights,
            prefix_closed=prefix_closed,
            hf_local_dtype=hf_local_dtype,
            hf_local_hidden_layers=hf_local_hidden_layers,
            hf_local_hook=hf_local_hook,
            stopping_criterion=stopping_criterion,
            expand_batch_size=expand_batch_size,
            gate_theta=gate_theta,
            gate_pool_size=gate_pool_size,
            gate_warmup=gate_warmup,
            ucb_c=ucb_c,
            log_dir=str(log_dir),
            verbose=verbose,
            surrogate_server_addr=surrogate_server_addr,
            surrogate_model=surrogate_model,
            beam_width=beam_width,
            max_rounds=max_rounds,
            patience=patience,
            mutator_num_prompts=mutator_num_prompts,
            mutator_max_calls=mutator_max_calls,
            stop_violation_prob=stop_violation_prob,
            score_mode=score_mode,
            num_samples=num_samples,
            tree_threshold=tree_threshold,
            tree_layer=tree_layer,
            dump_trie=dump_trie,
            mutator_system_prompt=mutator_system_prompt,
            prune_unreachable=prune_unreachable,
            cache_hit_prob=cache_hit_prob,
            cache_evict_after=cache_evict_after,
            cache_max_size=cache_max_size,
            cache_warm_start=cache_warm_start,
        )
        with open(log_dir / "run_args.json", "w") as f:
            json.dump(run_args, f, indent=2)

        # ── Instantiate and run verifier ───────────────────────────────────────
        common_kwargs = dict(
            grammar=grammar,
            gpu_visible_devices=gpu_visible_devices,
            gen_length=gen_length,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            decoded_dist=decoded_dist,
            semantic_symbol=semantic_symbol,
            max_iterations=max_iterations,
            epsilon=epsilon,
            early_termination_threshold=early_termination_threshold,
            prefix_closed=prefix_closed,
            verbose=verbose,
            max_workers=max_workers,
            num_logprobs=num_logprobs,
            score_chunk_size=score_chunk_size,
            use_grammar=use_grammar,
            chat_mode=use_chat_template,
            system_message=system_message,
            fewshot_messages=fewshot_messages,
            enable_thinking=enable_thinking,
            chat_template_kwargs=chat_template_kwargs,
            use_cache=use_cache,
            backend=backend,
            vllm_max_model_len=vllm_max_model_len,
            vllm_gpu_memory_utilization=vllm_gpu_memory_utilization,
            vllm_skip_prefix_cache_read=vllm_skip_prefix_cache_read,
            hf_local_tl_recipe=hf_local_tl_recipe,
            hf_local_dtype=hf_local_dtype,
            hf_local_hidden_layers=hf_local_hidden_layers,
            hf_local_hook=hf_local_hook,
            # Frontier-family verifiers read these; the others absorb them via
            # BaseVerifier(**kwargs). They live here rather than in the
            # per-verifier branches because BaseVerifier silently swallows
            # unknown keywords -- a branch that forgot to forward them made
            # --expand_batch_size a no-op with no error (every k ran as k=1).
            expand_batch_size=expand_batch_size,
            stopping_criterion=stopping_criterion,
            ucb_c=ucb_c,
        )

        if verifier == "cachedfrontier":
            llm = CachedFrontierVerifier(
                model,
                dataset_name,
                server_addr,
                max_frontier_size=max_frontier_size,
                max_frontier_prob=max_frontier_prob,
                frontier_scoring_strategy=frontier_scoring_strategy,
                cache_path=cache_path,
                **common_kwargs,
            )
        elif verifier == "iterative":
            llm = IterativeVerifier(
                model,
                dataset_name,
                server_addr,
                max_frontier_size=max_frontier_size,
                max_frontier_prob=max_frontier_prob,
                frontier_scoring_strategy=frontier_scoring_strategy,
                cache_path=cache_path,
                surrogate_server_addr=surrogate_server_addr,
                surrogate_model=surrogate_model,
                beam_width=beam_width,
                max_rounds=max_rounds,
                patience=patience,
                mutator_num_prompts=mutator_num_prompts,
                mutator_max_calls=mutator_max_calls,
                stop_violation_prob=stop_violation_prob,
                score_mode=score_mode,
                num_samples=num_samples,
                tree_threshold=tree_threshold,
                tree_layer=tree_layer,
                dump_trie=dump_trie,
                mutator_system_prompt=mutator_system_prompt,
                prune_unreachable=prune_unreachable,
                cache_hit_prob=cache_hit_prob,
                cache_evict_after=cache_evict_after,
                cache_max_size=cache_max_size,
                cache_warm_start=cache_warm_start,
                **common_kwargs,
            )
        elif verifier == "frontier":
            llm = FrontierVerifier(
                model,
                dataset_name,
                server_addr,
                max_frontier_size=max_frontier_size,
                max_frontier_prob=max_frontier_prob,
                frontier_scoring_strategy=frontier_scoring_strategy,
                **common_kwargs,
            )
        elif verifier == "representation":
            llm = FrontierVerifier(
                model,
                dataset_name,
                server_addr,
                max_frontier_size=max_frontier_size,
                max_frontier_prob=max_frontier_prob,
                frontier_scoring_strategy=frontier_scoring_strategy,
                node_selection="representation",
                # backend / vllm_* already come in via **common_kwargs; passing
                # them again here is a duplicate-keyword TypeError.
                probe_path=probe_path,
                probe_layer=probe_layer,
                probe_gpu=probe_gpu,
                direction_layers=direction_layers,
                direction_layer_weights=direction_layer_weights,
                gate_theta=gate_theta,
                gate_pool_size=gate_pool_size,
                gate_warmup=gate_warmup,
                nvf_type=nvf_type,
                nvf_data_path=nvf_data_path,
                knn_k=knn_k,
                knn_normalize=knn_normalize,
                **common_kwargs,
            )
        elif verifier == "sampling":
            llm = SamplingVerifier(model, dataset_name, server_addr, **common_kwargs)
        elif verifier == "distribution":
            llm = DistributionVerifier(
                model,
                dataset_name,
                server_addr,
                max_frontier_size=max_frontier_size,
                max_frontier_prob=max_frontier_prob,
                probe_path=probe_path,
                frontier_scoring_strategy=frontier_scoring_strategy,
                surrogate_server_addr=surrogate_server_addr,
                surrogate_model=surrogate_model,
                **common_kwargs,
            )
        else:
            raise ValueError(
                f"Unknown verifier: '{verifier}'. "
                f"Choose 'frontier', 'cachedfrontier', 'iterative', 'representation', 'sampling', or 'distribution'."
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
