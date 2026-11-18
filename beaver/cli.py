"""BEAVER command-line interface.

Sub-commands:
    beaver run    — run a single verification experiment (experiment YAML)
    beaver batch  — orchestrate batch experiments across models
    beaver logs   — summarise an existing run-logs directory
"""

import argparse
import importlib.util
import json
import inspect
import sys
from pathlib import Path
import yaml

# ── Helpers ───────────────────────────────────────────────────────────────


def _load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _import_module_from_path(module_path: Path):
    """Dynamically import a Python file as a module."""
    parent_dir = str(module_path.parent.resolve())
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

    module_name = module_path.stem
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod  # Required for pickle to find it in workers
    spec.loader.exec_module(mod)
    return mod


def _get_load_prompts_kwargs(load_prompts_fn, merged_cfg: dict) -> dict:
    """Return kwargs to pass to load_prompts_fn by introspecting its signature."""
    sig = inspect.signature(load_prompts_fn)
    kwargs = {k: merged_cfg[k] for k in sig.parameters if k in merged_cfg}

    # --dataset_path is a generic CLI/YAML knob for "where's the data file".
    # Most load_prompts_fn's take it directly, but a few (e.g. enron.py)
    # named their param differently before this flag existed; map onto the
    # first alias the function actually accepts so --dataset_path works
    # everywhere without renaming those params (which would break existing
    # YAML configs that already set the old name).
    if "dataset_path" in merged_cfg and "dataset_path" not in sig.parameters:
        for alias in ("prompts_file",):
            if alias in sig.parameters:
                kwargs[alias] = merged_cfg["dataset_path"]
                break

    return kwargs


def build_run_kwargs(experiment: str | Path, overrides: dict | None = None) -> dict:
    """``beaver.run`` keyword arguments from an experiment YAML.

    Loads the experiment file the YAML names, resolves its callables
    (load_prompts_fn, constraint_fn, check_call_fn, instance_context_fn,
    complete_fn), merges ``overrides`` on top of the YAML, loads the prompts
    with the merged settings and keeps only the keys ``beaver.run`` accepts.
    ``beaver run`` and callers that run BEAVER from Python 
    share this, so both read a YAML the same way.
    """
    exp_yaml_path = Path(experiment).resolve(None)
    if not exp_yaml_path.exists():
        raise FileNotFoundError(f"experiment YAML not found: {exp_yaml_path}")
    cfg = _load_yaml(exp_yaml_path)
    exp_file = exp_yaml_path.parent / cfg["experiment_file"]
    if not exp_file.exists():
        raise FileNotFoundError(f"experiment file not found: {exp_file}")

    mod = _import_module_from_path(exp_file)

    def fn(key):
        return getattr(mod, cfg[key]) if cfg.get(key) else None

    # Layer configurations: base YAML, then overrides.
    merged_cfg = {**cfg, **(overrides or {})}
    load_prompts_fn = fn("load_prompts_fn")
    prompts = load_prompts_fn(**_get_load_prompts_kwargs(load_prompts_fn, merged_cfg))

    import beaver

    run_sig = inspect.signature(beaver.run)
    run_kwargs = {k: v for k, v in merged_cfg.items() if k in run_sig.parameters}
    run_kwargs["prompts"] = prompts
    run_kwargs["constraint_fn"] = fn("constraint_fn")
    run_kwargs["check_call_fn"] = fn("check_call_fn")
    run_kwargs["instance_context_fn"] = fn("instance_context_fn")
    # Without this, a YAML `complete_fn:` would reach beaver.run() as a bare
    # string (it is a real run() parameter), so the dataset's early-completion
    # hook would never be callable.
    run_kwargs["complete_fn"] = fn("complete_fn")
    return run_kwargs


# ── beaver run ────────────────────────────────────────────────────────────


def _run_cmd(argv):
    """Single-experiment CLI driven by an experiment YAML file."""
    parser = argparse.ArgumentParser(
        prog="beaver run", description="Run a single BEAVER verification experiment."
    )

    # Core Infrastructure
    parser.add_argument(
        "--experiment", type=str, required=True, help="Path to experiment YAML."
    )
    parser.add_argument(
        "--model", type=str, required=True, help="HuggingFace model ID or local path."
    )
    parser.add_argument("--server_addr", type=str, default=None)
    parser.add_argument("--auto_server", action="store_true")
    parser.add_argument("--server_port", type=int, default=None)
    parser.add_argument("--gpu_visible_devices", type=str, default=None)

    # Prompt Slicing / Generation Context (Forwarded to load_prompts_fn)
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="Path to the dataset/prompts file, forwarded to load_prompts_fn "
        "(param name 'dataset_path', or 'prompts_file' for experiments that "
        "use that name, e.g. enron).",
    )
    # Namespace for the semantic-constraint sqlite cache (overrides the
    # YAML's cache_dataset_name). Parallel jobs on the same dataset must not
    # share one .db file: they would contend for its lock and any per-arm
    # cache reset in one job would delete the other's live cache.
    parser.add_argument("--cache_dataset_name", type=str, default=None)
    parser.add_argument("--template_id", type=int, default=None)
    parser.add_argument("--start_idx", type=int, default=None)
    parser.add_argument("--end_idx", type=int, default=None)
    parser.add_argument("--debug_ids", type=str, default=None)
    parser.add_argument("--classifier_addr", type=str, default=None)
    parser.add_argument("--num_shots", type=int, default=None)

    # Core Experiment Hyperparameters
    parser.add_argument(
        "--verifier",
        type=str,
        # None (not "frontier"): every value here that isn't None gets
        # layered over the YAML config below via cli_flags, so a non-None
        # default here would silently clobber the YAML's `verifier:` on every
        # invocation that doesn't repeat --verifier on the command line.
        # beaver.run()'s own default ("frontier") applies when neither the
        # YAML nor the CLI specify one.
        default=None,
        choices=["frontier", "cachedfrontier", "iterative", "representation", "sampling", "distribution"],
    )
    parser.add_argument(
        "--backend",
        default=None,
        choices=["vllm", "hf", "vllm-local", "hf-local"],
        help=(
            "Where model calls go. 'vllm': OpenAI-compatible HTTP to a "
            "`vllm serve` process. 'hf': HTTP to beaver.hf_server (the only "
            "backend that can return hidden states). 'vllm-local': in-process "
            "vLLM engine, no server -- the only one that can read vLLM's "
            "prefix cache for echo/scoring calls. Default: vllm."
        ),
    )
    parser.add_argument(
        "--vllm_max_model_len", type=int, default=None,
        help="backend=vllm-local only: engine max_model_len.",
    )
    parser.add_argument(
        "--vllm_gpu_memory_utilization", type=float, default=None,
        help="backend=vllm-local only: engine gpu_memory_utilization.",
    )
    parser.add_argument(
        "--hf_local_tl_recipe", default=None,
        help="backend=hf-local only: TL-registered model whose conversion "
             "recipe fits this checkpoint (e.g. Qwen/Qwen3-4B for "
             "Qwen/Qwen3-4B-Instruct-2507).",
    )
    parser.add_argument(
        "--hf_local_dtype", default=None,
        help="backend=hf-local only: bfloat16 (default) or float32. float32 "
             "costs ~2x memory but cuts per-token logprob error from ~0.07 "
             "nats to ~1e-4 (measured); bf16 error compounds along a path.",
    )
    parser.add_argument(
        "--hf_local_hook", default=None, choices=["resid_post", "mlp_post"],
        help="backend=hf-local only: which activation the NVF reads. "
             "resid_post (default) is the residual stream, identical to "
             "hf_server's hook. mlp_post is the post-activation MLP neurons.",
    )
    parser.add_argument(
        "--hf_local_hidden_layers", type=str, default=None,
        help="backend=hf-local only: comma-separated layers to capture. "
             "Defaults to whatever the direction stack needs.",
    )
    parser.add_argument(
        "--vllm_skip_prefix_cache_read",
        action="store_true",
        default=None,
        help=(
            "backend=vllm-local only: set SamplingParams."
            "skip_reading_prefix_cache=True, reproducing what vLLM's HTTP "
            "server forces (full prompt recompute on every echo/scoring "
            "call). Off by default; use it to A/B the prefix-cache win."
        ),
    )
    parser.add_argument("--gen_length", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument(
        "--decoded_dist", action=argparse.BooleanOptionalAction, default=None,
        help="Verify the decoded distribution (temperature/top-k/top-p applied "
             "and renormalised, truncated tokens at probability 0) instead of "
             "the raw one with top-p/top-k as an exploration cutoff. "
             "Temperature must be in [0, 1]; 0 is greedy.",
    )
    parser.add_argument("--max_iterations", type=int, default=None)
    parser.add_argument("--epsilon", type=float, default=None)
    parser.add_argument("--early_termination_threshold", type=float, default=None)
    parser.add_argument(
        "--prefix_closed", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument("--max_workers", type=int, default=None)
    parser.add_argument("--num_logprobs", type=int, default=None)
    parser.add_argument("--score_chunk_size", type=int, default=None)

    # Frontier Optimization Params
    parser.add_argument("--max_frontier_size", type=int, default=None)
    parser.add_argument("--max_frontier_prob", type=float, default=None)
    parser.add_argument(
        "--frontier_scoring_strategy",
        type=str,
        choices=["max-mu", "sample-mu", "random", "fitness", "gated", "ucb"],
    )

    # Grammar & Chat Structure
    parser.add_argument("--use_grammar", action="store_true", default=None)
    parser.add_argument("--use_chat_template", action="store_true", default=None)
    parser.add_argument("--grammar", type=str, default=None)
    parser.add_argument("--semantic_symbol", type=str, default=None)
    parser.add_argument("--enable_thinking", action="store_true", default=None)
    parser.add_argument(
        "--chat_template_kwargs", type=json.loads, default=None,
        help='JSON dict forwarded to the model\'s own chat template, e.g. '
             '\'{"reasoning_effort": "medium"}\' for gpt-oss or '
             '\'{"enable_thinking": true}\' for Qwen3-Thinking. Defaults to '
             '{"enable_thinking": <--enable_thinking>}. Templates read '
             'different keys and Jinja ignores unknown ones silently, so a key '
             'the template does not read has no effect and raises nothing.')

    # Neural Value Function Configuration
    parser.add_argument("--nvf_type", type=str, choices=["knn", "linear", "nonlinear", "direction"])
    parser.add_argument("--nvf_data_path", type=str, default=None)
    parser.add_argument("--probe_path", type=str, default=None)
    parser.add_argument("--probe_layer", type=int, default=None)
    parser.add_argument("--probe_gpu", type=str, default=None)
    parser.add_argument("--knn_k", type=int, default=None)
    parser.add_argument("--knn_normalize", action="store_true", default=None)
    # "gated" frontier_scoring_strategy knobs. gate_theta is in per-instance
    # standard deviations of the NVF score, not raw score units.
    # Comma-separated, e.g. --direction_layers 12  or  --direction_layers 10,12,14
    parser.add_argument("--direction_layers", type=str, default=None)
    parser.add_argument("--direction_layer_weights", type=str, default=None)
    # Frontier nodes expanded per round. 1 = original single-node behaviour.
    # >1 sends them as one batched forward pass -- the only parallelism
    # hf-local can use, since it clamps max_workers to 1.
    parser.add_argument("--expand_batch_size", type=int, default=None)
    # Stop rule. max_seq_prob (default) stops when the top incomplete leaf
    # falls below epsilon -- which with prefix_closed=False fires before any
    # sequence completes, so the constraint never runs. "bounds" stops when
    # the instance is actually decided.
    parser.add_argument("--stopping_criterion", type=str, default=None,
                        choices=["max_seq_prob", "bounds"])
    parser.add_argument("--gate_theta", type=float, default=None)
    parser.add_argument("--gate_pool_size", type=int, default=None)
    parser.add_argument("--gate_warmup", type=int, default=None)
    # "ucb" frontier_scoring_strategy exploration constant.
    parser.add_argument("--ucb_c", type=float, default=None)

    # Surrogate Model Settings (distribution / iterative verifiers)
    parser.add_argument("--surrogate_model", type=str, default=None)
    parser.add_argument("--surrogate_server_addr", type=str, default=None)

    # Cached Frontier Settings
    parser.add_argument("--cache_path", type=str, default=None)

    # Iterative Verifier Settings (mutator LLM: --surrogate_server_addr / --surrogate_model)
    parser.add_argument("--beam_width", type=int, default=None)
    parser.add_argument("--max_rounds", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--mutator_num_prompts", type=int, default=None)
    parser.add_argument("--mutator_max_calls", type=int, default=None)
    parser.add_argument("--stop_violation_prob", type=float, default=None)
    parser.add_argument("--score_mode", choices=["frontier", "sampling"], default=None)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--tree_threshold", type=float, default=None)
    parser.add_argument("--tree_layer", type=int, default=None)
    parser.add_argument("--dump_trie", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--mutator_system_prompt", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument(
        "--prune_unreachable", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument("--cache_hit_prob", type=float, default=None)
    parser.add_argument("--cache_evict_after", type=int, default=None)
    parser.add_argument("--cache_max_size", type=int, default=None)
    parser.add_argument("--cache_warm_start", action=argparse.BooleanOptionalAction, default=None)

    # Reporting / Outputs
    parser.add_argument("--log_dir", type=str, default=None)
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args(argv)

    cli_flags = {k: v for k, v in vars(args).items() if v is not None}
    run_kwargs = build_run_kwargs(args.experiment, cli_flags)
    run_kwargs["model"] = args.model
    run_kwargs["auto_server"] = args.auto_server
    run_kwargs["server_port"] = args.server_port
    run_kwargs["gpu_visible_devices"] = args.gpu_visible_devices
    run_kwargs["server_addr"] = args.server_addr if not args.auto_server else None

    import beaver

    beaver.run(**run_kwargs)


# ── beaver batch ──────────────────────────────────────────────────────────


def _batch_cmd(argv):
    """Batch experiments — orchestrate multiple experiments across models."""
    parser = argparse.ArgumentParser(
        prog="beaver batch", description="Orchestrate batch BEAVER experiments."
    )
    parser.add_argument("--batch", type=str, required=True, help="Path to batch YAML.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    batch_path = Path(args.batch).resolve()
    if not batch_path.exists():
        print(f"Error: batch config not found: {batch_path}", file=sys.stderr)
        sys.exit(1)

    from beaver.batch_runner import load_batch_config, run_batch

    success = run_batch(
        load_batch_config(batch_path),
        batch_path,
        dry_run=args.dry_run,
        verbose=args.verbose,
    )
    sys.exit(0 if success else 1)


# ── beaver logs ───────────────────────────────────────────────────────────


def _logs_cmd(argv):
    """Summarise an existing run-logs directory."""
    parser = argparse.ArgumentParser(
        prog="beaver logs", description="Summarise a BEAVER run-logs directory."
    )
    parser.add_argument("run_logs_folder", type=str)
    parser.add_argument("--threshold", type=float, default=0.9)
    args = parser.parse_args(argv)

    folder = Path(args.run_logs_folder)
    if not folder.exists():
        print(f"Error: {folder} does not exist", file=sys.stderr)
        sys.exit(1)

    from beaver.logging import get_log_data, summarize_log_data, summarize_profile_data

    all_data = get_log_data(folder)
    if all_data:
        summarize_log_data(all_data, folder, threshold=args.threshold)
        summarize_profile_data(folder)
    else:
        print("No log data found.")


# ── entry point ───────────────────────────────────────────────────────────


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    parser = argparse.ArgumentParser(
        prog="beaver",
        description="BEAVER: Formal verification bounds for LLM generation.",
    )
    parser.add_argument(
        "command", choices=["run", "batch", "logs"], help="Sub-command to execute."
    )

    if not argv:
        parser.print_help()
        sys.exit(0)

    cmd, rest = argv[0], argv[1:]
    if cmd == "run":
        _run_cmd(rest)
    elif cmd == "batch":
        _batch_cmd(rest)
    elif cmd == "logs":
        _logs_cmd(rest)


if __name__ == "__main__":
    main()
