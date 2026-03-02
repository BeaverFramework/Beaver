#!/usr/bin/env python3
"""
Batch runner for LLM verification experiments.

Orchestrates running multiple experiments across multiple models:
  1. For each model, starts a vLLM/HF server on all available GPUs
  2. Runs each specified experiment against that server (in-process)
  3. Organises logs as  {output_dir}/{model_name}/{experiment_name}/logs_*/

Batch YAML format:
  experiments: list of paths to experiment YAMLs (relative to the batch YAML)

Usage:
    beaver batch --batch configs/batches/example_batch.yaml
    beaver batch --batch configs/batches/example_batch.yaml --dry-run
"""

import concurrent.futures
import copy
import importlib.util
import inspect
import io
import json
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from huggingface_hub import snapshot_download

from beaver import run
from beaver.server import load_model_config, start_server, stop_server
from beaver.logging import (
    get_log_data,
    summarize_log_data,
    create_plots,
    create_time_plots,
    get_profile_data,
    summarize_profile_data,
)


# ── helpers ───────────────────────────────────────────────────────────────


def deep_merge(base: dict, override: dict) -> dict:
    """Return a new dict with *override* values merged on top of *base*."""
    merged = copy.deepcopy(base)
    for key, val in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(val, dict):
            merged[key] = deep_merge(merged[key], val)
        else:
            merged[key] = copy.deepcopy(val)
    return merged


def load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _sanitise_model_name(model_id: str) -> str:
    """Turn 'meta-llama/Llama-3.1-8B-Instruct' into 'meta-llama--Llama-3.1-8B-Instruct'."""
    return model_id.replace("/", "--")


def _find_logs_dir(base_dir: Path):
    """Find the most recent logs_* subdirectory inside base_dir, if any."""
    logs_dirs = sorted(
        [d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith("logs_")],
        reverse=True,
    )
    return logs_dirs[0] if logs_dirs else None


def is_experiment_completed(exp_base_dir: Path) -> bool:
    """Check whether a previous run of this experiment completed successfully.

    An experiment is considered complete if any ``logs_*/summary.json``
    exists under *exp_base_dir*.
    """
    if not exp_base_dir.is_dir():
        return False
    logs_dir = _find_logs_dir(exp_base_dir)
    return logs_dir is not None and (logs_dir / "summary.json").is_file()


def generate_summary(exp_base_dir: Path):
    """Run plot_logs summary on the most recent logs dir that has data."""
    if not exp_base_dir.is_dir():
        return
    logs_dir = _find_logs_dir(exp_base_dir)
    if logs_dir is None:
        return
    if (logs_dir / "summary.json").is_file():
        return  # already summarised
    try:
        all_data = get_log_data(logs_dir)
        all_profile_data = get_profile_data(logs_dir)
        if all_data:
            summarize_log_data(all_data, logs_dir)
            create_plots(all_data, logs_dir)
            create_time_plots(all_data, all_profile_data, logs_dir)
            summarize_profile_data(logs_dir)
            _log(f"  Generated summary: {logs_dir / 'summary.json'}")
    except Exception as e:
        _log(f"  Warning: failed to generate summary: {e}", "WARN")


# ── config loading ────────────────────────────────────────────────────────

# Structural keys in experiment YAML — never passed to load_prompts or run()
_YAML_META_KEYS = frozenset({
    "experiment_file",
    "load_prompts_fn",
    "constraint_fn",
    "check_call_fn",
    "instance_context_fn",
    "cache",
    "cache_dataset_name",
    "grammar",
    "semantic_symbol",
})

# Keys that go to beaver.run() as algo/config params
_BEAVER_RUN_KEYS = frozenset({
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
})


def load_experiment_config(exp_yaml_path: Path) -> dict[str, Any]:
    """Load experiment config from a file path."""
    if not exp_yaml_path.exists():
        raise FileNotFoundError(f"Experiment YAML not found: {exp_yaml_path}")
    cfg = load_yaml(exp_yaml_path)
    cfg["_yaml_path"] = str(exp_yaml_path)
    cfg["_yaml_dir"] = str(exp_yaml_path.parent)
    cfg.setdefault("_name", exp_yaml_path.stem)
    return cfg


# ── batch config ──────────────────────────────────────────────────────────


def load_batch_config(path: Path) -> dict[str, Any]:
    config = load_yaml(path)
    config.setdefault("output_dir", "./batch_results")
    config.setdefault("models", [])
    config.setdefault("experiments", [])
    config.setdefault("server", {})
    config.setdefault("gpu", {})
    config.setdefault("execution", {})
    return config


# ── server management ─────────────────────────────────────────────────────


def ensure_model_downloaded(model_id: str) -> None:
    _log(f"Ensuring model is downloaded: {model_id}")
    try:
        path = snapshot_download(model_id)
        _log(f"  Model ready at: {path}")
    except Exception as e:
        _log(f"  Failed to download model {model_id}: {e}", "ERROR")
        raise


# ── experiment execution (in-process) ────────────────────────────────────


_import_lock = threading.Lock()


def _import_module_from_path(module_path: Path):
    """Dynamically import a Python file as a module.

    Uses the file's stem as the module name and adds its parent directory to
    sys.path so that spawned worker processes (which inherit sys.path) can
    re-import the module when unpickling constraint functions.

    Thread-safe: if two parallel experiments load the same file, only one
    import happens and both get the same module object.  This is required for
    pickle to work correctly — if two threads each create a fresh module for
    the same file, the second ``sys.modules[name] = mod`` clobbers the first,
    making the first thread's function objects un-picklable.
    """
    module_name = module_path.stem  # e.g. "enron" from "enron.py"
    parent_dir = str(module_path.parent.resolve())

    with _import_lock:
        if module_name in sys.modules:
            return sys.modules[module_name]

        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)

        spec = importlib.util.spec_from_file_location(module_name, module_path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = mod  # Required for pickle to find it in workers
        spec.loader.exec_module(mod)

    return mod


def _get_load_prompts_kwargs(load_prompts_fn, exp_cfg: dict) -> dict:
    """Return kwargs for load_prompts_fn by introspecting its signature."""
    sig = inspect.signature(load_prompts_fn)
    skip = _YAML_META_KEYS | _BEAVER_RUN_KEYS
    result = {}
    for param_name in sig.parameters:
        if param_name in exp_cfg and param_name not in skip:
            val = exp_cfg[param_name]
            if val is not None:
                result[param_name] = val
    return result


EXPERIMENT_TIMEOUT: int = 60 * 60  # 60 minutes


def run_experiment(
    exp_cfg: dict,
    model_id: str,
    port: int,
    exp_base_dir: Path,
    dry_run: bool = False,
    verbose: bool = False,
    timeout: int = EXPERIMENT_TIMEOUT,
) -> bool:
    """Run a single experiment in-process.

    *exp_base_dir* is the experiment-level directory, e.g.
    ``batch_results/model/enron``. Logs go to ``exp_base_dir/logs_*/``
    (created by ``beaver.run()`` via ``new_log_dir()``).
    """
    exp_name = exp_cfg["_name"]
    exp_dir = Path(exp_cfg["_yaml_dir"])

    _log(f"Running experiment: {exp_name}")

    if dry_run:
        _log(f"  [DRY RUN] Would run: {exp_name} for model {model_id}")
        return True

    # ── Import experiment module ───────────────────────────────────────────
    exp_file = exp_dir / exp_cfg["experiment_file"]
    if not exp_file.exists():
        _log(f"  Error: experiment file not found: {exp_file}", "ERROR")
        return False

    try:
        mod = _import_module_from_path(exp_file)
    except Exception as e:
        _log(f"  Error importing {exp_file}: {e}", "ERROR")
        return False

    load_prompts_fn = getattr(mod, exp_cfg["load_prompts_fn"])
    constraint_fn = getattr(mod, exp_cfg["constraint_fn"])
    check_call_fn_name = exp_cfg.get("check_call_fn")
    check_call_fn = getattr(mod, check_call_fn_name) if check_call_fn_name else None
    instance_context_fn_name = exp_cfg.get("instance_context_fn")
    instance_context_fn = getattr(mod, instance_context_fn_name) if instance_context_fn_name else None

    # ── Build load_prompts kwargs ──────────────────────────────────────────
    load_kwargs = _get_load_prompts_kwargs(load_prompts_fn, exp_cfg)
    if verbose:
        _log(f"  load_prompts kwargs: {load_kwargs}")

    # ── Build beaver.run() kwargs ──────────────────────────────────────────
    run_kwargs = {
        k: exp_cfg[k]
        for k in _BEAVER_RUN_KEYS
        if k in exp_cfg and exp_cfg[k] is not None
    }
    # Override log_dir to point to exp_base_dir
    run_kwargs["log_dir"] = str(exp_base_dir)

    exp_base_dir.mkdir(parents=True, exist_ok=True)

    # Save experiment config for debugging
    with open(exp_base_dir / "experiment_config.json", "w") as f:
        json.dump(
            {k: v for k, v in exp_cfg.items() if not k.startswith("_")}, f, indent=2
        )

    # ── Run in a thread with timeout ───────────────────────────────────────
    result: dict = {"ok": False, "error": None}

    def _run():
        try:

            prompts = load_prompts_fn(**load_kwargs)
            run(
                prompts=prompts,
                constraint_fn=constraint_fn,
                check_call_fn=check_call_fn,
                cache=exp_cfg.get("cache", False),
                cache_dataset_name=exp_cfg.get("cache_dataset_name"),
                instance_context_fn=instance_context_fn,
                grammar=exp_cfg.get("grammar"),
                semantic_symbol=exp_cfg.get("semantic_symbol"),
                model=model_id,
                server_addr=f"http://localhost:{port}",
                **run_kwargs,
            )
            result["ok"] = True
        except Exception as e:
            result["error"] = e

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    thread.join(timeout=timeout)

    if thread.is_alive():
        _log(
            f"  Experiment {exp_name} timed out after {timeout}s",
            "ERROR",
        )
        return False

    if result["ok"]:
        _log(f"  Experiment {exp_name} completed successfully")
        return True

    _log(f"  Experiment {exp_name} failed: {result['error']}", "ERROR")
    return False


# ── logging ───────────────────────────────────────────────────────────────


_log_fh: io.TextIOWrapper | None = None
_log_lock = threading.Lock()


def _log(msg: str, level: str = "INFO"):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] [{level}] {msg}"
    with _log_lock:
        print(line)
        if _log_fh is not None:
            _log_fh.write(line + "\n")
            _log_fh.flush()


# ── main loop ─────────────────────────────────────────────────────────────


def run_batch(
    batch_cfg: dict,
    batch_path: Path,
    dry_run: bool = False,
    verbose: bool = False,
) -> bool:
    global _log_fh

    output_dir = Path(batch_cfg["output_dir"])
    batch_dir = batch_path.parent

    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        batch_log_path = output_dir / f"batch_runner_{datetime.now():%Y%m%d_%H%M%S}.log"
        _log_fh = open(batch_log_path, "w")

    server_cfg = batch_cfg["server"]
    gpu_cfg = batch_cfg["gpu"]
    exec_cfg = batch_cfg["execution"]

    base_port = server_cfg.get("base_port", 8000)
    startup_timeout = server_cfg.get("startup_timeout", 300)
    health_interval = server_cfg.get("health_check_interval", 5)
    stop_on_failure = exec_cfg.get("stop_on_failure", False)
    cooldown = exec_cfg.get("cooldown_between_experiments", 10)
    parallel_experiments = exec_cfg.get("parallel_experiments", 1)
    skip_completed = exec_cfg.get("skip_completed", False)

    model_names: list[str] = batch_cfg["models"]
    exp_paths: list[str] = batch_cfg["experiments"]

    _log("=" * 60)
    _log("Starting batch run")
    _log(f"  Output dir  : {output_dir}")
    _log(f"  Models      : {model_names}")
    _log(f"  Experiments : {exp_paths}")
    _log("=" * 60)

    # Pre-load all experiment configs (fail fast if any are missing)
    exp_cfgs: list[dict] = []
    for exp_path in exp_paths:
        abs_path = (batch_dir / exp_path).resolve()
        exp_cfgs.append(load_experiment_config(abs_path))

    results: list[dict] = []
    t_batch = time.time()
    server_proc: subprocess.Popen | None = None

    def _cleanup(_sig=None, _frame=None):
        nonlocal server_proc
        print("\n[batch_runner] Shutting down ...")
        if server_proc is not None:
            stop_server(server_proc)
            server_proc = None
        if _sig is not None:
            sys.exit(1)

    signal.signal(signal.SIGINT, _cleanup)
    signal.signal(signal.SIGTERM, _cleanup)

    total_experiments = len(model_names) * len(exp_cfgs)
    completed_experiments = 0

    for model_idx, model_name in enumerate(model_names, 1):
        _log("-" * 40)
        model_cfg = load_model_config(model_name)
        model_id = model_cfg["model"]
        safe_name = _sanitise_model_name(model_id)
        _log(
            f"Model [{model_idx}/{len(model_names)}]: {model_id}  (config: {model_name})"
        )

        model_log_dir = output_dir / safe_name

        if skip_completed:
            all_done = all(
                is_experiment_completed(model_log_dir / cfg["_name"])
                for cfg in exp_cfgs
            )
            if all_done:
                _log(f"  All experiments already completed for {model_id} — skipping model")
                for exp_cfg in exp_cfgs:
                    completed_experiments += 1
                    results.append(
                        {
                            "model": model_id,
                            "experiment": exp_cfg["_name"],
                            "success": True,
                            "log_dir": str(model_log_dir / exp_cfg["_name"]),
                            "skipped": True,
                        }
                    )
                continue

        if not dry_run:
            ensure_model_downloaded(model_id)

        if dry_run:
            _log("  [DRY RUN] Would start server")
            server_proc = None
        else:
            gpu_visible_devices = (
                str(gpu_cfg["visible_devices"])
                if gpu_cfg.get("visible_devices") is not None
                else None
            )
            _log(f"Starting server for {model_id} ...")
            server_proc = start_server(
                model_id,
                port=base_port,
                gpu_visible_devices=gpu_visible_devices,
                startup_timeout=startup_timeout,
                health_check_interval=health_interval,
                log_dir=model_log_dir,
            )
            if server_proc is None:
                _log(f"Failed to start server for {model_id}", "ERROR")
                if stop_on_failure:
                    return False
                continue

        try:
            if parallel_experiments <= 1:
                for exp_idx, exp_cfg in enumerate(exp_cfgs, 1):
                    exp_name = exp_cfg["_name"]
                    exp_log_dir = model_log_dir / exp_name
                    completed_experiments += 1

                    _log(
                        f"  Experiment [{exp_idx}/{len(exp_cfgs)}] "
                        f"(overall {completed_experiments}/{total_experiments}): "
                        f"{exp_name}"
                    )

                    if skip_completed and is_experiment_completed(exp_log_dir):
                        _log(f"  Skipping {exp_name} — already completed")
                        results.append(
                            {
                                "model": model_id,
                                "experiment": exp_name,
                                "success": True,
                                "log_dir": str(exp_log_dir),
                                "skipped": True,
                            }
                        )
                        continue

                    ok = run_experiment(
                        exp_cfg,
                        model_id,
                        base_port,
                        exp_log_dir,
                        dry_run=dry_run,
                        verbose=verbose,
                    )

                    results.append(
                        {
                            "model": model_id,
                            "experiment": exp_name,
                            "success": ok,
                            "log_dir": str(exp_log_dir),
                        }
                    )

                    if not ok and stop_on_failure:
                        _log("Stopping batch due to failure", "ERROR")
                        return False

                    if cooldown > 0:
                        time.sleep(cooldown)
            else:
                _log(
                    f"Running {len(exp_cfgs)} experiments with parallelism={parallel_experiments}"
                )

                exps_to_run = []
                for exp_cfg in exp_cfgs:
                    exp_name = exp_cfg["_name"]
                    exp_log_dir = model_log_dir / exp_name
                    if skip_completed and is_experiment_completed(exp_log_dir):
                        _log(f"  Skipping {exp_name} — already completed")
                        completed_experiments += 1
                        results.append(
                            {
                                "model": model_id,
                                "experiment": exp_name,
                                "success": True,
                                "log_dir": str(exp_log_dir),
                                "skipped": True,
                            }
                        )
                    else:
                        exps_to_run.append(exp_cfg)

                futures: dict[concurrent.futures.Future, dict] = {}

                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=parallel_experiments
                ) as pool:
                    for exp_cfg in exps_to_run:
                        exp_name = exp_cfg["_name"]
                        exp_log_dir = model_log_dir / exp_name

                        fut = pool.submit(
                            run_experiment,
                            exp_cfg,
                            model_id,
                            base_port,
                            exp_log_dir,
                            dry_run=dry_run,
                            verbose=verbose,
                        )
                        futures[fut] = {
                            "model": model_id,
                            "experiment": exp_name,
                            "log_dir": str(exp_log_dir),
                        }

                    for fut in concurrent.futures.as_completed(futures):
                        info = futures[fut]
                        ok = fut.result()
                        info["success"] = ok
                        results.append(info)
                        completed_experiments += 1

                        if ok and not dry_run:
                            generate_summary(Path(info["log_dir"]))

                        _log(
                            f"  Completed (overall {completed_experiments}/{total_experiments}): "
                            f"{info['experiment']} — {'OK' if ok else 'FAIL'}"
                        )

                        if not ok and stop_on_failure:
                            _log("Stopping batch due to failure", "ERROR")
                            for f in futures:
                                f.cancel()
                            return False
        finally:
            stop_server(server_proc)
            server_proc = None

    # ── summary ───────────────────────────────────────────────────────────
    duration = time.time() - t_batch
    ok_count = sum(1 for r in results if r["success"])
    total = len(results)

    _log("=" * 60)
    _log("Batch complete")
    _log(f"  Time: {duration:.1f}s")
    skipped_count = sum(1 for r in results if r.get("skipped"))
    for r in results:
        if r.get("skipped"):
            tag = "SKIP"
        elif r["success"]:
            tag = "OK"
        else:
            tag = "FAIL"
        _log(f"    [{tag}] {r['model']} / {r['experiment']}")
    _log(f"  {ok_count}/{total} succeeded ({skipped_count} skipped)")
    _log("=" * 60)

    if not dry_run:
        summary_path = output_dir / f"batch_summary_{datetime.now():%Y%m%d_%H%M%S}.yaml"
        with open(summary_path, "w") as f:
            yaml.dump(
                {
                    "duration_seconds": round(duration, 1),
                    "success_count": ok_count,
                    "total_count": total,
                    "results": results,
                },
                f,
            )
        _log(f"Summary: {summary_path}")

    if _log_fh is not None:
        _log_fh.close()
        _log_fh = None

    return ok_count == total
