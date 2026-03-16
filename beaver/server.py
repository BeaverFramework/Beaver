"""Server management utilities for beaver.

Provides start/stop helpers for vLLM inference servers.
When ``auto_server=True`` is used in ``beaver.run()``, the server is started
with the same vLLM flags as the batch runner — loaded from
``beaver/configs/models/default.yaml`` and optionally a per-model override.
"""

from __future__ import annotations

import copy
import json
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

import requests
import yaml


# ── Config directory bundled with the package ─────────────────────────────

_CONFIGS_DIR = Path(__file__).parent / "configs"


# ── Config helpers (mirrors batch_runner logic) ────────────────────────────


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _deep_merge(base: dict, override: dict) -> dict:
    """Return a new dict with *override* merged on top of *base* (recursive)."""
    merged = copy.deepcopy(base)
    for key, val in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(val, dict):
            merged[key] = _deep_merge(merged[key], val)
        else:
            merged[key] = copy.deepcopy(val)
    return merged


def load_model_config(
    model_id: str,
    model_config_path: str | Path | None = None,
) -> dict[str, Any]:
    """Load the merged model config for *model_id*.

    Merging order (later overrides earlier):
      1. ``beaver/configs/models/default.yaml``   (package default vLLM flags)
      2. ``beaver/configs/models/<short_name>.yaml``  (if it exists — e.g. "llama31")
      3. *model_config_path* (caller-supplied yaml)
      4. ``{"model": model_id}`` always wins for the model field

    Args:
        model_id: HuggingFace model ID or local path.
        model_config_path: Optional path to a YAML file whose contents are
            deep-merged on top of the defaults.  Use this to override
            ``vllm.gpu_memory_utilization``, ``vllm.max_model_len``, etc.

    Returns:
        Merged config dict ready to pass to :func:`_build_server_command`.
    """
    cfg = _load_yaml(_CONFIGS_DIR / "models" / "default.yaml")
    cfg["model"] = model_id

    # Try to find a per-model yaml by short name (last path component, sanitised)
    short_name = model_id.replace("/", "--")
    per_model = _CONFIGS_DIR / "models" / f"{short_name}.yaml"
    if per_model.exists():
        cfg = _deep_merge(cfg, _load_yaml(per_model))

    if model_config_path is not None:
        cfg = _deep_merge(cfg, _load_yaml(Path(model_config_path)))

    # Caller-supplied model_id always takes precedence
    cfg["model"] = model_id
    return cfg


def _get_gpu_count(visible_devices: str | None = None) -> int:
    if visible_devices is not None:
        return len([d for d in str(visible_devices).split(",") if d.strip()])
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return len(result.stdout.strip().split("\n"))
    except FileNotFoundError:
        pass
    return 0


def _append_server_flags(
    cmd: list[str],
    cfg: dict,
    visible_devices: str | None = None,
) -> None:
    """Append vLLM CLI flags from *cfg* dict to *cmd* in-place.

    Mirrors batch_runner._append_server_flags exactly:
    - bool True  → bare --flag
    - bool False → skipped
    - None / "none" / "auto" → skipped (except tensor_parallel_size)
    - tensor_parallel_size "auto" → resolved to GPU count
    - everything else → --flag value (with dashes, not underscores)
    """
    for key, val in cfg.items():
        if key.startswith("_"):
            continue

        if key == "tensor_parallel_size":
            if val == "auto":
                val = _get_gpu_count(visible_devices) or 1
            if int(val) <= 1:
                continue

        if val is None:
            continue
        if isinstance(val, str) and val.lower() in ("none", "auto"):
            continue

        flag = "--" + key.replace("_", "-")

        if isinstance(val, bool):
            if val:
                cmd.append(flag)
        else:
            cmd.extend([flag, str(val)])


def _build_server_command(
    model_cfg: dict,
    port: int,
    visible_devices: str | None = None,
) -> tuple[list[str], dict[str, str]]:
    """Build the ``vllm serve`` command and environment from *model_cfg*."""
    env = os.environ.copy()
    if visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(visible_devices)
        print(f"CUDA_VISIBLE_DEVICES={visible_devices}")

    model_id = model_cfg["model"]
    cmd = ["vllm", "serve", model_id, "--port", str(port)]
    _append_server_flags(
        cmd,
        model_cfg.get("vllm", {}),
        visible_devices=visible_devices,
    )
    return cmd, env


# ── Public API ─────────────────────────────────────────────────────────────


_PORT_UNSET = object()


def start_server(
    model_id: str,
    port: int | None = None,
    gpu_visible_devices: str | None = None,
    startup_timeout: int = 600,
    health_check_interval: int = 5,
    model_config_path: str | Path | None = None,
    extra_vllm_args: list[str] | None = None,
    log_dir: Path | str | None = None,
) -> subprocess.Popen | None:
    """Start a vLLM server and wait until it is healthy.

    The server is launched with all flags from
    ``beaver/configs/models/default.yaml`` (the same defaults used by the
    batch runner), plus any overrides from *model_config_path*.

    Critical flags that are always applied from the defaults:
    - ``--max-logprobs -1`` — required for full-vocabulary logprobs
    - ``--return-tokens-as-token-ids`` — required for token-ID parsing
    - ``--logprobs-mode processed_logprobs``
    - ``--trust-remote-code``

    Args:
        model_id: HuggingFace model ID or local path.
        port: Port to serve on.  If ``None`` (default), uses the ``port``
            field from ``configs/models/default.yaml`` (8081), falling back
            to 8000 if not set.
        gpu_visible_devices: Comma-separated CUDA device IDs (e.g. ``"0,1"``).
        startup_timeout: Seconds to wait for the server to become healthy.
        health_check_interval: Seconds between health-check polls.
        model_config_path: Optional path to a YAML file with vLLM overrides.
            Merged on top of the package defaults.  Keys live under a ``vllm:``
            section, same format as ``configs/models/default.yaml``::

                vllm:
                  gpu_memory_utilization: 0.90
                  max_model_len: 8192
        extra_vllm_args: Raw extra CLI flags appended after all config-derived
            flags (e.g. ``["--enforce-eager"]``).

    Returns:
        The server :class:`subprocess.Popen`, or ``None`` if startup failed.
    """
    model_cfg = load_model_config(model_id, model_config_path)
    if port is None:
        port = int(model_cfg.get("port", 8000))
    cmd, env = _build_server_command(model_cfg, port, gpu_visible_devices)

    if extra_vllm_args:
        cmd.extend(extra_vllm_args)

    print(f"[beaver] Starting vLLM server for {model_id} on port {port}")
    print(f"[beaver]   Command: {' '.join(cmd)}")

    server_log: Path | None = None
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        with open(log_dir / "server_config.json", "w") as f:
            json.dump(
                {k: v for k, v in model_cfg.items() if not k.startswith("_")},
                f,
                indent=2,
            )
        with open(log_dir / "server_cmd.txt", "w") as f:
            f.write(" ".join(cmd) + "\n")
        server_log = log_dir / "server.log"
        with open(server_log, "w") as fh:
            proc = subprocess.Popen(
                cmd,
                env=env,
                stdout=fh,
                stderr=subprocess.STDOUT,
                preexec_fn=os.setsid,
            )
    else:
        proc = subprocess.Popen(
            cmd,
            env=env,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
        )

    health_url = f"http://localhost:{port}/health"
    print(f"[beaver] Waiting for server (timeout: {startup_timeout}s) ...")
    t0 = time.time()

    try:
        while time.time() - t0 < startup_timeout:
            if proc.poll() is not None:
                print(f"[beaver] Server process exited with code {proc.returncode}")
                if server_log:
                    print(f"[beaver] Check log: {server_log}")
                return None
            try:
                r = requests.get(health_url, timeout=5)
                if r.status_code == 200:
                    print(f"[beaver] Server healthy ({time.time() - t0:.1f}s)")
                    return proc
            except requests.exceptions.RequestException:
                pass
            time.sleep(health_check_interval)
    except KeyboardInterrupt:
        print("\n[beaver] Interrupted — stopping server ...")
        stop_server(proc)
        raise

    print(f"[beaver] Server failed to start within {startup_timeout}s")
    if server_log:
        print(f"[beaver] Check log: {server_log}")
    stop_server(proc)
    return None


def stop_server(proc: subprocess.Popen | None) -> None:
    """Gracefully stop a server subprocess."""
    if proc is None:
        return
    print("[beaver] Stopping server ...")
    try:
        pgid = os.getpgid(proc.pid)
        os.killpg(pgid, signal.SIGTERM)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(pgid, signal.SIGKILL)
            proc.wait(timeout=5)
    except ProcessLookupError:
        pass
    print("[beaver] Server stopped")
