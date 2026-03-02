"""BEAVER command-line interface.

Sub-commands:
  beaver run    — run a single verification experiment (experiment YAML)
  beaver batch  — orchestrate batch experiments across models
  beaver logs   — summarise an existing run-logs directory
"""

import argparse
import importlib.util
import inspect
import sys
from pathlib import Path

import yaml


# ── Helpers ───────────────────────────────────────────────────────────────


def _load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _import_module_from_path(module_path: Path):
    """Dynamically import a Python file as a module.

    Uses the file's stem as the module name and adds its parent directory to
    sys.path so that spawned worker processes (which inherit sys.path) can
    re-import the module when unpickling constraint functions.
    """
    import sys

    parent_dir = str(module_path.parent.resolve())
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

    module_name = module_path.stem
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod  # Required for pickle to find it in workers
    spec.loader.exec_module(mod)
    return mod


# Keys in experiment YAML that are meta/structural (never passed to load_prompts or run)
_YAML_META_KEYS = frozenset(
    {
        "experiment_file",
        "load_prompts_fn",
        "constraint_fn",
        "check_call_fn",
        "instance_context_fn",
        "cache",
        "cache_dataset_name",
        "grammar",
        "semantic_symbol",
    }
)

# Keys that are beaver.run() algorithm params
_BEAVER_RUN_KEYS = frozenset(
    {
        "verifier",
        "gen_length",
        "temperature",
        "top_p",
        "top_k",
        "max_iterations",
        "epsilon",
        "max_workers",
        "num_logprobs",
        "max_frontier_size",
        "max_frontier_prob",
        "frontier_scoring_strategy",
        "use_grammar",
        "use_chat_template",
        "num_shots",
        "verbose",
        "log_dir",
    }
)


def _get_load_prompts_kwargs(load_prompts_fn, merged_cfg: dict, extra: dict) -> dict:
    """Return kwargs to pass to load_prompts_fn by introspecting its signature."""
    sig = inspect.signature(load_prompts_fn)
    skip = _YAML_META_KEYS | _BEAVER_RUN_KEYS
    result = {}
    for param_name in sig.parameters:
        if param_name in extra:
            result[param_name] = extra[param_name]
        elif param_name in merged_cfg and param_name not in skip:
            result[param_name] = merged_cfg[param_name]
    return result


def _get_run_kwargs(merged_cfg: dict) -> dict:
    """Extract beaver.run() kwargs from merged config."""
    return {
        k: v for k, v in merged_cfg.items() if k in _BEAVER_RUN_KEYS and v is not None
    }


# ── beaver run ────────────────────────────────────────────────────────────


def _run_cmd(argv):
    """Single-experiment CLI driven by an experiment YAML file."""
    parser = argparse.ArgumentParser(
        prog="beaver run",
        description="Run a single BEAVER verification experiment.",
    )

    # Experiment
    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        help="Path to experiment YAML (e.g. experiments/gsm_symbolic/experiment.yaml).",
    )

    # Server
    parser.add_argument("--server_addr", type=str, default=None)
    parser.add_argument("--classifier_addr", type=str, default=None)
    parser.add_argument(
        "--auto_server",
        action="store_true",
        help="Automatically start and stop a vLLM server.",
    )
    parser.add_argument("--server_port", type=int, default=None)
    parser.add_argument("--gpu_visible_devices", type=str, default=None)

    # Model
    parser.add_argument("--model", type=str, required=True)

    # Data slicing (forwarded to load_prompts_fn)
    parser.add_argument("--start_idx", type=int, default=None)
    parser.add_argument("--end_idx", type=int, default=None)
    parser.add_argument("--debug_ids", type=str, default=None)

    # Experiment param overrides (any key from experiment.yaml can be overridden)
    parser.add_argument(
        "--verifier", type=str, default=None, choices=["frontier", "sampling"]
    )
    parser.add_argument("--gen_length", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--max_iterations", type=int, default=None)
    parser.add_argument("--epsilon", type=float, default=None)
    parser.add_argument("--max_workers", type=int, default=None)
    parser.add_argument("--num_logprobs", type=int, default=None)
    parser.add_argument("--max_frontier_size", type=int, default=None)
    parser.add_argument("--max_frontier_prob", type=float, default=None)
    parser.add_argument(
        "--frontier_scoring_strategy",
        type=str,
        default=None,
        choices=["highest-prob", "length-bias", "random-select", "sample-select"],
    )
    parser.add_argument(
        "--use_grammar",
        type=lambda x: x.lower() in ["true", "1", "yes"],
        default=None,
    )
    parser.add_argument(
        "--use_chat_template",
        type=lambda x: x.lower() in ["true", "1", "yes"],
        default=None,
    )
    parser.add_argument("--num_shots", type=int, default=None)
    parser.add_argument(
        "--verbose",
        action="store_true",
    )
    parser.add_argument("--log_dir", type=str, default=None)

    args = parser.parse_args(argv)

    # ── Load experiment YAML ───────────────────────────────────────────────
    exp_yaml_path = Path(args.experiment).resolve()
    if not exp_yaml_path.exists():
        print(f"Error: experiment YAML not found: {exp_yaml_path}", file=sys.stderr)
        sys.exit(1)

    exp_dir = exp_yaml_path.parent
    cfg = _load_yaml(exp_yaml_path)

    # ── Import experiment module ───────────────────────────────────────────
    exp_file = exp_dir / cfg["experiment_file"]
    if not exp_file.exists():
        print(f"Error: experiment file not found: {exp_file}", file=sys.stderr)
        sys.exit(1)

    mod = _import_module_from_path(exp_file)

    load_prompts_fn = getattr(mod, cfg["load_prompts_fn"])
    constraint_fn = getattr(mod, cfg["constraint_fn"])
    check_call_fn = (
        getattr(mod, cfg.get("check_call_fn") or "", None)
        if cfg.get("check_call_fn")
        else None
    )
    instance_context_fn = (
        getattr(mod, cfg.get("instance_context_fn") or "", None)
        if cfg.get("instance_context_fn")
        else None
    )

    # ── Merge YAML params with explicit CLI flags (CLI wins) ───────────────
    merged_cfg = dict(cfg)
    cli_overrides = {
        k: v for k, v in vars(args).items() if v is not None and k in _BEAVER_RUN_KEYS
    }
    merged_cfg.update(cli_overrides)

    # ── Build load_prompts kwargs ──────────────────────────────────────────
    slicing_kwargs = {}
    if args.start_idx is not None:
        slicing_kwargs["start_idx"] = args.start_idx
    if args.end_idx is not None:
        slicing_kwargs["end_idx"] = args.end_idx
    if args.debug_ids is not None:
        slicing_kwargs["debug_ids"] = args.debug_ids
    if args.classifier_addr is not None:
        slicing_kwargs["classifier_addr"] = args.classifier_addr

    load_kwargs = _get_load_prompts_kwargs(load_prompts_fn, merged_cfg, slicing_kwargs)

    # ── Load prompts ───────────────────────────────────────────────────────
    prompts = load_prompts_fn(**load_kwargs)

    # ── Build beaver.run() kwargs ──────────────────────────────────────────
    run_kwargs = _get_run_kwargs(merged_cfg)

    # ── Determine server ───────────────────────────────────────────────────
    server_addr = args.server_addr if not args.auto_server else None

    # ── Call beaver.run() ──────────────────────────────────────────────────
    import beaver

    beaver.run(
        prompts=prompts,
        constraint_fn=constraint_fn,
        check_call_fn=check_call_fn,
        cache=cfg.get("cache", False),
        cache_dataset_name=cfg.get("cache_dataset_name"),
        instance_context_fn=instance_context_fn,
        grammar=cfg.get("grammar"),
        semantic_symbol=cfg.get("semantic_symbol"),
        model=args.model,
        server_addr=server_addr,
        auto_server=args.auto_server,
        server_port=args.server_port,
        gpu_visible_devices=args.gpu_visible_devices,
        **run_kwargs,
    )


# ── beaver batch ──────────────────────────────────────────────────────────


def _batch_cmd(argv):
    """Batch experiments — orchestrate multiple experiments across models."""
    parser = argparse.ArgumentParser(
        prog="beaver batch",
        description="Orchestrate batch BEAVER experiments across models.",
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

    batch_cfg = load_batch_config(batch_path)
    success = run_batch(
        batch_cfg,
        batch_path,
        dry_run=args.dry_run,
        verbose=args.verbose,
    )
    sys.exit(0 if success else 1)


# ── beaver logs ───────────────────────────────────────────────────────────


def _logs_cmd(argv):
    """Summarise an existing run-logs directory."""
    parser = argparse.ArgumentParser(
        prog="beaver logs",
        description="Summarise a BEAVER run-logs directory.",
    )
    parser.add_argument("run_logs_folder", type=str)
    parser.add_argument("--threshold", type=float, default=0.9)
    args = parser.parse_args(argv)

    from beaver.logging import get_log_data, summarize_log_data, summarize_profile_data

    folder = Path(args.run_logs_folder)
    if not folder.exists():
        print(f"Error: {folder} does not exist", file=sys.stderr)
        sys.exit(1)

    all_data = get_log_data(folder)
    if all_data:
        summarize_log_data(all_data, folder, threshold=args.threshold)
        summarize_profile_data(folder)
    else:
        print("No log data found.")


# ── entry point ───────────────────────────────────────────────────────────


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(
        prog="beaver",
        description="BEAVER: Formal verification bounds for LLM generation.",
    )
    parser.add_argument(
        "command",
        choices=["run", "batch", "logs"],
        help="Sub-command to execute.",
    )

    if not argv:
        parser.print_help()
        sys.exit(0)

    cmd = argv[0]
    rest = argv[1:]

    if cmd == "run":
        _run_cmd(rest)
    elif cmd == "batch":
        _batch_cmd(rest)
    elif cmd == "logs":
        _logs_cmd(rest)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
