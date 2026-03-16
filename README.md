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
| 📐 Optional EBNF grammar masking (Python, Rust, C, Go) for syntax-constrained generation                 |
| ⚡ SQLite constraint-result caching — avoid re-running expensive checks                                  |
| 🤖 Works with any HuggingFace model via a vLLM backend                                                   |
| 🔌 Plug in any binary constraint — one Python function is all you need                                   |

## ⚙️ Requirements

- Python 3.10+
- A CUDA-capable GPU (vLLM does not run on CPU)
- [vLLM](https://github.com/vllm-project/vllm) installed and accessible on `PATH`

## 🚀 Quick Start

```bash
pip install -e .
```

## 🖥️ Running BEAVER

BEAVER can be used in two ways: as a **Python API** or as a **CLI tool**.

### Backend: vLLM Server

BEAVER uses [vLLM](https://github.com/vllm-project/vllm) to serve models. You have two options:

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

## 👀 Example Usage

### Python API

```python
import beaver

results = beaver.run(
    prompts=[{"prompt": "Write a Python function to parse user input."}],
    constraint_fn=lambda instance, seq: valid_prefix(seq),  # Your prefix closed constraint here
    model="meta-llama/Llama-3.1-8B-Instruct",
    auto_server=True,
    gen_length=64,
    epsilon=0.05,
)

# Each result: {"idx": 0, "lower_bound": 0.91, "upper_bound": 0.97, "transition": 42, ...}
```

```python
# Connect to a running vLLM server instead
results = beaver.run(
    prompts=[{"prompt": "Write a Python function to parse user input."}],
    constraint_fn=lambda instance, seq: "eval(" not in seq,
    model="meta-llama/Llama-3.1-8B-Instruct",
    server_addr="http://localhost:8091",
    gen_length=64,
    epsilon=0.05,
)
```

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

## 🧪 Reproducing Paper Experiments

The `experiments/` folder contains all experiments from the [paper](https://arxiv.org/abs/2512.05439) and [leaderboard](https://beaverframework.pages.dev/#leaderboard). Each experiment has a YAML config and a Python constraint file.

Available experiments: `enron` (privacy), `secure_code` (security), `toxicity`, `stereotype`, `gsm_symbolic` (correctness).

To reproduce the full leaderboard, use the batch config:

```bash
beaver batch --batch configs/batches/new_leaderboard.yaml
```

This runs all leaderboard experiments across a broad set of instruction-tuned models. Edit `new_leaderboard.yaml` to select the specific models you want to evaluate (many are commented out by default).

To run a single experiment:

```bash
beaver run \
    --experiment experiments/toxicity/toxicity.yaml \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --auto_server
```

## 📋 Logging & Results

Every BEAVER run creates a timestamped log directory (default: `logging/logs_<timestamp>/`) containing detailed outputs for inspection and post-hoc analysis.

### Log directory structure

```
logging/logs_20260316185322/
├── run_args.json            # Full configuration used for this run
├── bounds.csv               # Per-instance lower/upper bounds and transition counts
├── summary.json             # Aggregate statistics across all instances
├── console.log              # Full console output captured during the run
├── profiling_summary.json   # Timing breakdowns (per-transition and per-instance)
├── server_cmd.txt           # vLLM server command (if auto_server was used)
├── server_config.json       # Resolved server configuration
├── server.log               # vLLM server stdout/stderr
├── <idx>.jsonl              # Per-instance transition log (one JSON object per step)
└── <idx>.profile.json       # Per-instance timing profile (one entry per step)
```

### Key files

- **`bounds.csv`** — Quick overview of results. Each row contains `idx`, `lower_bound`, `upper_bound`, and `num_transitions` for one instance.
- **`summary.json`** — Aggregated metrics: average bounds, transition counts, and constraint satisfaction rates at a configurable threshold.
- **`<idx>.jsonl`** — Detailed per-transition log for instance `idx`, including expanded tokens, decoded text, current bounds, and frontier state at each step.
- **`<idx>.profile.json`** — Timing breakdown per transition (model generation, grammar masking, semantic checks, frontier updates, etc.).

### Summarizing logs after a run

Use the `beaver logs` CLI command to re-summarize any previous run:

```bash
beaver logs logging/logs_20260316185322/
```

This prints aggregate statistics (average bounds, transition counts, constraint satisfaction) and saves `summary.json` and `profiling_summary.json` to the log directory.

## 📦 Built-in Datasets

BEAVER ships with the following datasets used in the paper:

- **enron** — Privacy leakage: does the model reproduce personal information from the Enron email corpus?
- **secure_code** — Security: does the model generate code with known vulnerabilities (CWE-level checks)?
- **toxicity** — Safety: does the model produce toxic or harmful text?
- **stereotype** — Fairness: does the model generate stereotypical statements about demographic groups?
- **gsm_symbolic** — Correctness: does the model produce the correct answer to symbolic math problems?

## ✍️ Writing a Custom Constraint

Each prompt dict must have a `"prompt"` key (string). Optionally include per-instance `"system_prompt"` and/or `"fewshot_messages"` to override the global defaults when using chat mode.

```python
# experiments/my_eval/my_eval.py

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

def constraint_fn(instance: dict, sequence: str) -> bool:
    """True = acceptable, False = violation."""
    pass

def check_call_fn(instance, decoded_sequences, token_lists):
    """Optional fast pre-filter — skip expensive checks on short prefixes."""
    pass

def instance_context_fn(instance: dict) -> str:
    """Cache key for this instance's constraint context."""
    pass
```

```yaml
# experiments/my_eval/experiment.yaml
experiment_file: my_eval.py
load_prompts_fn: load_prompts
constraint_fn: constraint_fn
check_call_fn: check_call_fn
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

| Argument                          | Default           | Description                                                           |
| --------------------------------- | ----------------- | --------------------------------------------------------------------- |
| `verifier`                        | `frontier`        | `frontier` (branch-and-bound) or `sampling` (Monte Carlo)             |
| `gen_length`                      | `32`              | Max tokens per generated sequence                                     |
| `epsilon`                         | `0.01`            | Convergence threshold — stop when `PUB − PLB ≤ ε`                     |
| `max_iterations`                  | `100`             | Hard iteration cap per instance                                       |
| `max_frontier_size`               | `10000`           | Max partial sequences tracked in the frontier                         |
| `max_frontier_prob`               | `1.0`             | Max cumulative probability tracked in the frontier                    |
| `frontier_scoring_strategy`       | `highest-prob`    | `highest-prob` \| `length-bias` \| `random-select` \| `sample-select` |
| `num_logprobs`                    | `100`             | Top-k log-probs requested per expansion step                          |
| `use_grammar`                     | `false`           | Apply EBNF grammar mask during expansion                              |
| `grammar`                         | —                 | `rust` \| `python` \| `c` \| `go` \| path to `.lark` file             |
| `temperature` / `top_p` / `top_k` | `1.0 / 0.99 / -1` | Sampling distribution parameters                                      |
| `use_chat_template`               | `true`            | Apply the model's chat template to prompts                            |
| `system_message`                  | —                 | Global system message (can be overridden per-instance via `system_prompt` key) |
| `fewshot_messages`                | `[]`              | Global few-shot examples (can be overridden per-instance via `fewshot_messages` key) |
| `max_workers`                     | `16`              | Parallel worker processes                                             |
| `cache`                           | `false`           | Enable SQLite constraint-result cache                                 |
| `auto_server`                     | `true`            | Start/stop vLLM server automatically                                  |

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
