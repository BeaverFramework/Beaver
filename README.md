<h1 align="center">
  <img width="60" alt="beaver" src="https://beaverframework.pages.dev/assets/icon.png" />
  &nbsp;Beaver Framework
</h1>

<p align="left">
    🌐&nbsp;<a href="https://beaverframework.pages.dev">Website</a>
    | 📊&nbsp;<a href="https://beaverframework.pages.dev/#leaderboard">Leaderboard</a>
    | 📄&nbsp;<a href="https://arxiv.org/abs/2512.05439">Paper</a>
    | 💻&nbsp;<a href="https://github.com/uiuc-focal-lab/Beaver">GitHub</a>
</p>

<p>
    <a href="https://arxiv.org/abs/2512.05439"><img src="https://img.shields.io/badge/arXiv-2512.05439-b31b1b.svg"></a>
    <a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/Python-3.10+-1f425f.svg?color=purple"></a>
    <a href="LICENSE.md"><img alt="License" src="https://img.shields.io/badge/License-CC_BY--SA_4.0-blue"></a>
</p>

## ℹ️ About

- **BEAVER** computes certified probability bounds `[PLB, PUB]` on the likelihood that an LLM violates a behavioral constraint — guaranteed to bracket the true probability.
- Unlike sampling-based evaluation, BEAVER's **FrontierVerifier** uses branch-and-bound search over the token tree to produce provably correct intervals that tighten with more compute.
- Evaluate any binary constraint: security vulnerabilities, toxicity, privacy leakage, hallucination, stereotype, and more.

## 📚 Features

|                                                                                                          |
| -------------------------------------------------------------------------------------------------------- |
| 🔒 Certified `[PLB, PUB]` bounds — the true probability is guaranteed inside the interval                |
| 🌲 Branch-and-bound **FrontierVerifier** reaches tight bounds with far fewer model queries than sampling |
| ♻️ **CachedFrontierVerifier** reuses a token-trie cache across groups of similar prompts                 |
| 🧭 **Representation-guided** search: a neural value function over hidden states picks which node to expand |
| 🎯 **IterativeVerifier**: beam search over system prompts for the worst certified violation probability  |
| 📐 Optional EBNF grammar masking (Python, Rust, C, Go) for syntax-constrained generation                 |
| ⚡ SQLite constraint-result caching — avoid re-running expensive checks                                  |
| 🤖 Works with any HuggingFace model via vLLM or the bundled HuggingFace server                           |
| 🔌 Plug in any binary constraint — one Python function is all you need                                   |

## ⚙️ Requirements

- Python 3.10+
- A CUDA-capable GPU (vLLM does not run on CPU)
- [vLLM](https://github.com/vllm-project/vllm) installed and accessible on `PATH` (install it separately; it is not a dependency of this package)

## 🚀 Quick Start

```bash
pip install -e .
```

Optional extras:

| Extra | Install | Needed for |
| --- | --- | --- |
| `classifiers` | `pip install -e ".[classifiers]"` | `toxicity` (Detoxify server) and `secure_code` (semgrep, pinned to 1.79.0) |
| `server` | `pip install -e ".[server]"` | `backend="hf"` (`beaver.hf_server`) |
| `representation` | `pip install -e ".[representation]"` | `verifier="representation"` / `"distribution"`, `backend="hf-local"`, `beaver.activation_detection` |
| `grouping` | `pip install -e ".[grouping]"` | `beaver.grouping` (building grouped datasets for `cachedfrontier`) |

### Per-experiment constraint tools

Some experiments shell out to an external analyser. A missing one is easy to miss: the verifier catches a per-instance constraint exception, logs it, and the job still exits 0. If a run finishes suspiciously fast, check `console.log`.

| Experiment | Tool | Notes |
|---|---|---|
| `secure_code` (`constraint_fn`) | `semgrep==1.79.0` | Invoked through `insecure_code_detector`. Other versions break it, one of them silently; `experiments/secure_code/verify_semgrep.py` checks that semgrep actually fires. |
| `secure_code` (`weggli_constraint`) | [`weggli`](https://github.com/weggli-rs/weggli) | Must be **built from `main`** (the constraint passes `-n`, which the released 0.2.x does not support): `cargo install --git https://github.com/weggli-rs/weggli`. Resolved via `$WEGGLI_BIN`, else `PATH`. |
| `toxicity` | Detoxify server | `python experiments/toxicity/launch_detoxify_server.py`, passed via `--classifier_addr`. |

## 🖥️ Running BEAVER

BEAVER can be used in two ways: as a **Python API** or as a **CLI tool**.

### Backends

The `backend` argument selects where model calls go:

| `backend` | What it is | Notes |
| --- | --- | --- |
| `vllm` (default) | HTTP to a `vllm serve` process | Fastest for plain `frontier` / `sampling` runs. |
| `hf` | HTTP to `beaver.hf_server` | The bundled HuggingFace server. Can return hidden states, so it is the default for `verifier="representation"`. |
| `vllm-local` | In-process vLLM engine, no server | Single process (`max_workers` is clamped to 1). |
| `hf-local` | In-process TransformerLens model, no server | Single process; hidden states and prefix caching. |

**Option 1 — Automatic server (`auto_server=True`) (Recommended)**

Pass `auto_server=True` to `beaver.run()` (or `--auto_server` on the CLI) and BEAVER will start and stop the vLLM server for you, using the default flags from `beaver/configs/models/default.yaml`.

**Option 2 — Manual server**

Start vLLM yourself before running BEAVER, then point BEAVER at it with `server_addr`. Example for `google/gemma-3-12b-it`:

```bash
vllm serve google/gemma-3-12b-it \
    --port 8091 \
    --max-logprobs -1 \
    --max-model-len 4092 \
    --gpu-memory-utilization 0.85 \
    --enable-prefix-caching \
    --max-num-seqs 256 \
    --logprobs-mode processed_logprobs \
    --return-tokens-as-token-ids
```

> **Flag notes:**
>
> - `--max-logprobs -1` and `--logprobs-mode processed_logprobs` and `--return-tokens-as-token-ids` are **required** — BEAVER needs full-vocabulary logprobs to perform its branch-and-bound search.
> - `--gpu-memory-utilization`, `--enable-prefix-caching`, and `--max-num-seqs` are performance flags that significantly speed up computation; tune them to your hardware.

To use the HuggingFace server instead (`backend="hf"`), either let BEAVER start it (leave `server_addr` unset) or start it yourself:

```bash
python -m beaver.hf_server --model Qwen/Qwen3-4B-Instruct-2507 --port 8082 \
    --hidden-state-layers 18
```

## 👀 Example Usage

### Python API

`constraint_fn` is **batched**: it receives the instance and a list of decoded sequences, and returns one bool per sequence (`True` = constraint satisfied).

```python
import beaver

results = beaver.run(
    prompts=[{"prompt": "Write a Python function to parse user input."}],
    constraint_fn=lambda instance, seqs: ["eval(" not in s for s in seqs],
    model="meta-llama/Llama-3.1-8B-Instruct",
    auto_server=True,
    gen_length=64,
    epsilon=0.05,
)
```

```python
# Connect to a running vLLM server instead
results = beaver.run(
    prompts=[{"prompt": "Write a Python function to parse user input."}],
    constraint_fn=lambda instance, seqs: ["eval(" not in s for s in seqs],
    model="meta-llama/Llama-3.1-8B-Instruct",
    server_addr="http://localhost:8091",
    gen_length=64,
    epsilon=0.05,
)
```

By default (`prefix_closed=False`) the constraint is only evaluated on **complete** sequences (EOS, `gen_length`, or the experiment's `complete_fn`). Set `prefix_closed=True` if a violation on a prefix is final (e.g. "never emits `eval(`"), which lets the search prune violating prefixes early.

### CLI

```bash
# Single experiment (auto-start server)
beaver run \
    --experiment experiments/enron/enron.yaml \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --auto_server

# Single experiment (connect to running server)
beaver run \
    --experiment experiments/enron/enron.yaml \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --server_addr http://localhost:8091

# Multi-model batch
beaver batch --batch configs/batches/my_batch.yaml

# Summarize logs
beaver logs logging/logs_20250225123456/
```

Any `beaver.run()` argument can be set in the experiment YAML or overridden on the command line (`beaver run --help`).

## 🧭 Verifiers

| `verifier` | Description |
| --- | --- |
| `frontier` (default) | Branch-and-bound over the token tree. |
| `sampling` | Monte Carlo baseline: samples full sequences and accumulates their mass. |
| `cachedfrontier` | Frontier that warm-starts each instance from a token-trie cache grown by the earlier instances of its **group**. Every instance needs a `group` key. |
| `iterative` | Beam search over system prompts on top of the cached frontier: a mutator LLM (`surrogate_server_addr` / `surrogate_model`) proposes new system prompts, each one is verified, and the search keeps the ones with the largest certified violation probability. |
| `representation` | Frontier whose node selection is guided by a neural value function (NVF) over hidden states. Needs a hidden-state backend (`hf`, `hf-local`, `vllm-local`). |
| `distribution` | Ranks a prompt distribution with a cheap probe (`probe_path`), then fully verifies the riskiest prompts under a budget. |

Node selection for the frontier family is set by `frontier_scoring_strategy`: `max-mu` (most probable leaf, default), `sample-mu`, `random`, `ucb`, and, with `verifier=representation`, `gated` and `fitness`. `expand_batch_size` expands several nodes per round in one batched forward pass. `stopping_criterion` is `max_seq_prob` (default: stop when the most probable incomplete leaf drops below `epsilon`) or `bounds` (stop when the instance is decided against `early_termination_threshold`, or `PUB − PLB ≤ epsilon`).

### Cached frontier and grouping

`cachedfrontier` pays off when the prompts in a group induce similar output distributions. `beaver.grouping` builds such groups from any dataset JSON by clustering prompts on the total-variation distance between their output distributions:

```bash
python -m beaver.grouping.tv_group_dataset \
    --dataset my_prompts.json \
    --output my_prompts_grouped.json \
    --model Qwen/Qwen3-4B-Instruct-2507 --threshold 0.1

beaver run --experiment experiments/enron/enron_new.yaml \
    --model Qwen/Qwen3-4B-Instruct-2507 --auto_server \
    --dataset_path "$PWD/my_prompts_grouped.json"   # absolute: enron resolves relative paths against experiments/enron/
```

`tv_distance` / `tv_prefix_distance` compute the distance matrices and `tv_visualize` clusters them; see the module docstrings and `experiments/enron/TV_GROUPING_DESIGN.md`.

### Representation-guided verification

`verifier=representation` scores expanded nodes with an NVF (`nvf_type`): `direction` (a violation direction; `probe_path` is a directory with `directions.npz`), `linear` / `nonlinear` (a trained probe checkpoint at `probe_path`), or `knn` (`nvf_data_path`). Directions are fitted from BEAVER run logs with `beaver/activation_detection/violation_direction_heretic.py` (needs the `heretic` package); the supporting scripts for extracting activations and calibrating the `gated` strategy are in `experiments/common/`.

## 🧪 Experiments

The `experiments/` folder contains the experiments from the [paper](https://arxiv.org/abs/2512.05439) and [leaderboard](https://beaverframework.pages.dev/#leaderboard). Each experiment has YAML configs and a Python file with its prompt loader and constraint.

| Experiment | What it checks |
| --- | --- |
| `enron` | Privacy: does the model reproduce email addresses from the Enron corpus? |
| `secure_code` | Security: does the model generate code with known vulnerabilities (CWE-level checks)? |
| `toxicity` | Safety: does the model produce toxic text? |
| `stereotype` | Fairness: does the model agree with stereotypical statements? |
| `agent_safety` | Agent security: does the model call a harmful tool (AgentHarm)? |
| `gsm_symbolic` | Correctness: does the model produce the correct symbolic answer to math problems? |
| `hallucination` | Factuality: does the model answer SimpleQA questions correctly? |

To run a single experiment:

```bash
beaver run \
    --experiment experiments/toxicity/toxicity.yaml \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --auto_server
```

To reproduce the leaderboard, use the batch config (edit it to select models):

```bash
beaver batch --batch configs/batches/new_leaderboard.yaml
```

### Datasets

**The dataset files are not included in this repository.** Each experiment's Python file documents in its header where its data comes from; the scripts that build the derived prompt distributions are included. Put the files under `experiments/<name>/data/` (git-ignored) or pass `--dataset_path`.

| Experiment | Source | How to get it |
| --- | --- | --- |
| `enron` | HuggingFace `tarsur909/enron_direct_100` | Downloaded automatically. Jailbreak / few-shot / grouped distributions: `experiments/enron/build_distribution.py` and the other `build_*.py` scripts (these read the DecodingTrust Enron privacy `.pkl` files). |
| `gsm_symbolic` | HuggingFace `anon2525/gsms` | Downloaded automatically. |
| `agent_safety` | HuggingFace `ai-safety-institute/AgentHarm` | `experiments/agent_safety/build_distribution.py` writes the `data/*.json` files the YAMLs reference. |
| `toxicity` | [DecodingTrust](https://github.com/AI-secure/DecodingTrust) toxicity prompts (`toxic.jsonl`) | Place `toxic.jsonl` / a subset in `data/`. Distributions: `build_distribution.py`, `genetic_jailbreak.py`, `build_tv_families_distribution.py`. |
| `stereotype` | [DecodingTrust](https://github.com/AI-secure/DecodingTrust) stereotype data (`stereotype_bias_data.jsonl`) | Place it / a subset in `data/` next to the included `system_prompts.csv`. |
| `secure_code` | [CyberSecEval](https://github.com/meta-llama/PurpleLlama) `instruct` / `autocomplete` prompt sets | Place the JSON files in `data/`. Distributions: `genetic_jailbreak.py`, `build_tv_grouped_distribution.py`. |
| `hallucination` | SimpleQA (`simple_qa_test_set.csv`, [simple-evals](https://github.com/openai/simple-evals)) | Place the CSV in `data/`. |

YAMLs with `verifier: cachedfrontier` (`toxicity_cachedfrontier.yaml`, `enron_new.yaml`, `secure_code_new.yaml`, `secure_code_weggli.yaml`) need a dataset with `group` labels; YAMLs with `verifier: iterative` need a mutator LLM server.

## 📋 Logging & Results

Every BEAVER run creates a timestamped log directory (default: `logging/logs_<timestamp>/`) containing detailed outputs for inspection and post-hoc analysis.

```
logging/logs_20260316185322/
├── run_args.json            # Full configuration used for this run
├── summary.json             # Aggregate statistics across all instances
├── console.log              # Full console output captured during the run
├── profiling_summary.json   # Timing breakdowns (per-transition and per-instance)
├── server_cmd.txt           # Server command (if BEAVER started the server)
├── server_config.json       # Resolved server configuration
├── server.log               # Server stdout/stderr
├── <idx>.jsonl              # Per-instance log: instance_start, one entry per transition, instance_complete
└── <idx>.profile.json       # Per-instance timing profile (one entry per transition)
```

- **`summary.json`** — Aggregated metrics: average bounds, transition and forward-pass counts, and safe / risky / open verdicts at a configurable threshold.
- **`<idx>.jsonl`** — Per-transition log for instance `idx`: expanded nodes, decoded text, violations found, current bounds and frontier sizes. The final `instance_complete` entry holds the final masses and the termination reason.

Re-summarize any previous run with:

```bash
beaver logs logging/logs_20260316185322/ --threshold 0.9
```

Analysis utilities in `beaver/utils/`: `run_analysis.py` (verdicts and costs of a run), `compare_runs.py` / `diff_runs.py` (compare two runs), `cost_analysis.py` (forward-pass cost, cached vs. uncached), `frontier_cache_stats.py`, `pruning_stats.py`, `analyze_iterative_run.py`, `distribution_analysis.py`, `make_leaderboard.py`, `show_cache_stats.py`.

## ✍️ Writing a Custom Constraint

Each prompt dict must have a `"prompt"` key (string). Optionally include per-instance `"system_prompt"` and/or `"fewshot_messages"` to override the global defaults when using chat mode, and `"group"` for the cached verifiers.

```python
# experiments/my_eval/my_eval.py
import numpy as np

def load_prompts(**kwargs) -> list[dict]:
    return [
        {
            "prompt": f"What is {i}+{i}?",
            "answer": 2*i,
            # Optional per-instance overrides:
            # "system_prompt": "You are a math tutor.",
            # "fewshot_messages": [{"prompt": "1+1?", "response": "2"}],
        }
        for i in range(100)
    ]

def constraint_fn(instance: dict, sequences: list[str]) -> list[bool]:
    """Batched. One bool per sequence: True = acceptable, False = violation."""
    return [str(instance["answer"]) in seq for seq in sequences]

def check_call_fn(instance, decoded_sequences, token_lists) -> np.ndarray:
    """Optional fast pre-filter, used with prefix_closed=True: True for the
    prefixes that should be sent to constraint_fn."""
    return np.ones(len(decoded_sequences), dtype=bool)

def complete_fn(instance, decoded_sequences, token_lists) -> np.ndarray:
    """Optional: True marks a sequence as complete before EOS / gen_length."""
    return np.zeros(len(decoded_sequences), dtype=bool)

def instance_context_fn(instance: dict) -> str:
    """Per-instance part of the constraint-cache key (cache: true)."""
    return str(instance["answer"])
```

```yaml
# experiments/my_eval/experiment.yaml
experiment_file: my_eval.py
load_prompts_fn: load_prompts
constraint_fn: constraint_fn
check_call_fn: check_call_fn
complete_fn: complete_fn
instance_context_fn: instance_context_fn
cache: true
cache_dataset_name: my_eval
verifier: frontier
gen_length: 32
epsilon: 0.05
max_workers: 8
```

```bash
beaver run --experiment experiments/my_eval/experiment.yaml \
           --model meta-llama/Llama-3.1-8B-Instruct \
           --server_addr http://localhost:8000
```

<details>
<summary><b>BEAVER Arguments</b></summary>

Defaults are those of `beaver.run()`; see its docstring and `beaver run --help` for the full list.

| Argument                          | Default           | Description                                                           |
| --------------------------------- | ----------------- | --------------------------------------------------------------------- |
| `verifier`                        | `frontier`        | `frontier` \| `sampling` \| `cachedfrontier` \| `iterative` \| `representation` \| `distribution` |
| `backend`                         | `vllm`            | `vllm` \| `hf` \| `vllm-local` \| `hf-local`                          |
| `gen_length`                      | `32`              | Max tokens per generated sequence                                     |
| `epsilon`                         | `0.1`             | Convergence threshold (see `stopping_criterion`)                      |
| `stopping_criterion`              | `max_seq_prob`    | `max_seq_prob` \| `bounds`                                            |
| `early_termination_threshold`     | `0.9`             | Stop an instance once `PUB` drops below this value (`null` to disable) |
| `prefix_closed`                   | `false`           | Evaluate the constraint on prefixes (via `check_call_fn`), not only on complete sequences |
| `max_iterations`                  | `500`             | Hard iteration cap per instance                                       |
| `max_frontier_size`               | `10000`           | Max partial sequences tracked in the frontier                         |
| `max_frontier_prob`               | `1.0`             | Max cumulative probability tracked in the frontier                    |
| `frontier_scoring_strategy`       | `max-mu`          | `max-mu` \| `sample-mu` \| `random` \| `ucb` \| `gated` \| `fitness`  |
| `expand_batch_size`               | `1`               | Frontier nodes expanded per round (batched forward pass)              |
| `num_logprobs`                    | `200`             | Top-k log-probs requested per expansion step                          |
| `temperature` / `top_p` / `top_k` | `1.0 / 0.99 / -1` | Sampling distribution parameters                                      |
| `decoded_dist`                    | `false`           | Bound the *decoded* distribution (temperature / top-p / top-k applied and renormalised) instead of the raw one |
| `use_grammar`                     | `false`           | Apply EBNF grammar mask during expansion                              |
| `grammar`                         | —                 | `rust` \| `python` \| `c` \| `go` \| `gsm`                            |
| `use_chat_template`               | `true`            | Apply the model's chat template to prompts                            |
| `enable_thinking` / `chat_template_kwargs` | `false` / — | Forwarded to the model's chat template (e.g. reasoning models)      |
| `system_message`                  | —                 | Global system message (can be overridden per-instance via `system_prompt` key) |
| `fewshot_messages`                | `[]`              | Global few-shot examples (can be overridden per-instance via `fewshot_messages` key) |
| `max_workers`                     | `16`              | Parallel worker processes                                             |
| `cache`                           | `false`           | Enable SQLite constraint-result cache (`BEAVER_SEMANTIC_CACHE_DIR` sets its location) |
| `cache_path`                      | —                 | Where `cachedfrontier` / `iterative` write their token-trie caches    |
| `auto_server`                     | `true` (API) / off (CLI) | Start/stop the vLLM server automatically                       |

</details>

## 🔖 Citation

```bibtex
@misc{suresh2025beaverefficientdeterministicllm,
      title={BEAVER: An Efficient Deterministic LLM Verifier},
      author={Tarun Suresh and Nalin Wadhwa and Debangshu Banerjee and Gagandeep Singh},
      year={2025},
      eprint={2512.05439},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2512.05439},
}
```

<a href="https://github.com/uiuc-focal-lab/Beaver">Beaver</a> © 2026 by <a href="https://nalinwadhwa02.github.io">Nalin Wadhwa</a> is licensed under <a href="https://creativecommons.org/licenses/by-sa/4.0/">CC BY-SA 4.0</a>
