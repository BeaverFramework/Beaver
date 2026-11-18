import multiprocessing as mp
import os
from abc import ABC, abstractmethod
from typing import Optional

from tqdm import tqdm
from transformers import AutoTokenizer

import beaver.constraints  # ensures all dataset modules are registered


def _resolve_eos_tokens(tokenizer, model_name, override=None) -> list[int]:
    """Token ids that end a generation.

    Historically only the tokenizer's ``eos_token_id`` and ``pad_token_id``
    were used. That misses ids declared only in the model's
    ``generation_config.json`` -- e.g. Gemma-3-it ends turns with
    ``<end_of_turn>`` (106) while its tokenizer's eos is ``<eos>`` (1), and
    Llama-3.x lists ``<|eom_id|>`` next to ``<|eot_id|>``. A missed id means
    the frontier keeps expanding past the model's own stop token, inflating
    forward passes and shifting mass, so both sources are unioned here.
    ``override`` (an int or list) replaces the lot when given.
    """
    if override is not None:
        return list(override) if isinstance(override, (list, tuple)) else [int(override)]
    ids: list[int] = []
    for attr in ("eos_token_id", "pad_token_id"):
        v = getattr(tokenizer, attr, None)
        if v is not None:
            ids.append(int(v))
    try:
        from transformers import GenerationConfig

        gc = GenerationConfig.from_pretrained(model_name)
        v = getattr(gc, "eos_token_id", None)
        for t in (v if isinstance(v, (list, tuple)) else [v]):
            if t is not None:
                ids.append(int(t))
    except Exception as exc:  # no generation_config, offline, gated repo...
        print(f"[BaseVerifier] generation_config eos lookup failed for {model_name}: {exc}")
    return sorted(set(ids))


class BaseVerifier(ABC):
    def __init__(
        self,
        model,
        dataset,
        server_addr,
        grammar,
        semantic_symbol=None,
        **kwargs,
    ) -> None:
        self.server_addr = server_addr
        self.grammar = grammar

        # Grammar loading
        current_dir = os.path.dirname(os.path.abspath(__file__))
        grammars_dir = os.path.join(current_dir, "..", "grammars")

        grammar_file = None
        if grammar is not None:
            for suffix in ["_grammar_sglang.lark", "_grammar.lark"]:
                potential_file = os.path.join(grammars_dir, f"{grammar}{suffix}")
                if os.path.exists(potential_file):
                    grammar_file = potential_file
                    break
            if grammar_file is None:
                raise FileNotFoundError(
                    f"Could not find grammar file for '{grammar}' in {grammars_dir}"
                )
            with open(grammar_file, "r") as f:
                self.ebnf = f.read()
        else:
            self.ebnf = None

        from beaver.constraints.base_constraints import _REGISTRY
        if dataset not in _REGISTRY:
            raise ValueError(
                f"Unknown dataset: {dataset}. "
                f"Available: {list(_REGISTRY.keys())}"
            )
        self.dataset_name = dataset
        self.use_cache: bool = kwargs.get("use_cache", True)
        self.prefix_closed: bool = kwargs.get("prefix_closed", False)

        self.model_name = model
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.semantic_symbol = semantic_symbol

        # Common generation parameters (previously duplicated in subclasses)
        self.temperature = kwargs.get("temperature", 1.0)
        self.top_p = kwargs.get("top_p", 1.0)
        self.top_k = kwargs.get("top_k", -1)
        self.decoded_dist = kwargs.get("decoded_dist", False)
        self.max_iterations = kwargs.get("max_iterations", 1000)
        self.epsilon = kwargs.get("epsilon", 0.01)
        self.early_termination_threshold = kwargs.get(
            "early_termination_threshold", None
        )
        self.eos_tokens: list[int] = _resolve_eos_tokens(
            self.tokenizer, self.model_name, kwargs.get("eos_token_id")
        )
        self.gen_length: int = kwargs.get("gen_length", 128)
        self.verbose: bool = kwargs.get("verbose", False)
        self.max_workers = kwargs.get("max_workers", 1)
        self.num_logprobs = kwargs.get("num_logprobs", 100)
        self.score_chunk_size = kwargs.get("score_chunk_size", 64)
        self.use_grammar = kwargs.get("use_grammar", True)
        self.chat_mode = kwargs.get("chat_mode", True)
        self.system_message = kwargs.get("system_message", None)
        self.fewshot_messages = kwargs.get("fewshot_messages", [])
        self.gpu_visible_devices = kwargs.get("gpu_visible_devices", None)
        # Which model backend beaver.model routes calls to; see beaver/model/.
        self.backend = kwargs.get("backend", "vllm")
        self.vllm_max_model_len = kwargs.get("vllm_max_model_len", None)
        self.vllm_gpu_memory_utilization = kwargs.get(
            "vllm_gpu_memory_utilization", None
        )
        self.vllm_skip_prefix_cache_read = kwargs.get(
            "vllm_skip_prefix_cache_read", False
        )
        self.hf_local_dtype = kwargs.get("hf_local_dtype", "bfloat16")
        self.hf_local_tl_recipe = kwargs.get("hf_local_tl_recipe", None)
        self.hf_local_hidden_layers = kwargs.get("hf_local_hidden_layers", None)
        # Which TransformerLens activation the NVF reads: "resid_post" (the
        # residual stream, equivalent to hf_server's layer forward-hook) or
        # "mlp_post" (post-activation MLP neurons, d_mlp-wide).
        self.hf_local_hook = kwargs.get("hf_local_hook", "resid_post")
        self.enable_thinking = kwargs.get("enable_thinking", False)
        # Forwarded verbatim to tokenizer.apply_chat_template. Defaults to
        # {"enable_thinking": enable_thinking}, which is what every config
        # rendered with before this was configurable.
        self.chat_template_kwargs = kwargs.get("chat_template_kwargs", None)

    def _build_worker_config(self):
        """Build a pickleable config dict for init_worker_state()."""
        from beaver.constraints.base_constraints import _REGISTRY

        check_call_fn, instance_context_fn, check_fn, complete_fn = _REGISTRY[self.dataset_name]
        return {
            "model_name": self.model_name,
            "server_addr": self.server_addr,
            "ebnf": self.ebnf,
            "dataset_name": self.dataset_name,
            "use_cache": self.use_cache,
            "prefix_closed": self.prefix_closed,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "decoded_dist": self.decoded_dist,
            "eos_tokens": self.eos_tokens,
            "gen_length": self.gen_length,
            "epsilon": self.epsilon,
            "early_termination_threshold": self.early_termination_threshold,
            "num_logprobs": self.num_logprobs,
            "score_chunk_size": self.score_chunk_size,
            "verbose": self.verbose,
            "max_iterations": self.max_iterations,
            "semantic_symbol": self.semantic_symbol,
            "use_grammar": self.use_grammar,
            "chat_mode": self.chat_mode,
            "system_message": self.system_message,
            "fewshot_messages": self.fewshot_messages,
            "enable_thinking": self.enable_thinking,
            "chat_template_kwargs": self.chat_template_kwargs,
            "check_call_fn": check_call_fn,
            "instance_context_fn": instance_context_fn,
            "check_fn": check_fn,
            "complete_fn": complete_fn,
            "gpu_visible_devices": self.gpu_visible_devices,
            "backend": self.backend,
            "vllm_max_model_len": self.vllm_max_model_len,
            "vllm_gpu_memory_utilization": self.vllm_gpu_memory_utilization,
            "vllm_skip_prefix_cache_read": self.vllm_skip_prefix_cache_read,
            "hf_local_dtype": self.hf_local_dtype,
            "hf_local_tl_recipe": self.hf_local_tl_recipe,
            "hf_local_hidden_layers": self.hf_local_hidden_layers,
            "hf_local_hook": self.hf_local_hook,
        }

    def _run_pool(self, dataset, run_log_dir, worker_fn, init_fn, config):
        """Run worker_fn over dataset using multiprocessing or single-thread.

        Workers should return a dict with at least:
            idx, a forward-pass count, a wall-clock time, and verifier-specific
            fields. Verifiers disagree on the exact key names (frontier_verifier
            logs total_transitions/instance_run_time via build_instance_complete;
            the sampling verifier logs transitions/time_s directly) so both are
            checked here rather than standardizing the on-disk log schema.
        The tqdm bar updates as each instance finishes (unordered) and
        shows running averages in the postfix.
        """
        results = []
        worker_args = [(instance, run_log_dir) for instance in dataset]
        total_time = 0.0
        total_fp = 0

        def _update_bar(bar, result):
            nonlocal total_time, total_fp
            if result is not None:
                total_time += result.get("instance_run_time", result.get("time_s", 0))
                total_fp += result.get("total_transitions", result.get("transitions", 0))
                n_done = bar.n + 1
                bar.set_postfix_str(
                    f"total_fp={total_fp}  avg={total_fp / n_done:.1f}/inst, "
                    f"avg {total_time / n_done:.1f}s/inst"
                )
            bar.update(1)

        if self.max_workers > 1:
            ctx = mp.get_context("spawn")
            with ctx.Pool(
                processes=self.max_workers,
                initializer=init_fn,
                initargs=(config,),
            ) as pool:
                bar = tqdm(total=len(dataset), desc="Processing instances")
                for result in pool.imap_unordered(worker_fn, worker_args, chunksize=1):
                    results.append(result)
                    _update_bar(bar, result)
                bar.close()
        else:
            init_fn(config)
            bar = tqdm(total=len(dataset), desc="Processing instances")
            for instance in dataset:
                result = worker_fn((instance, run_log_dir))
                results.append(result)
                _update_bar(bar, result)
            bar.close()

        return results

    @abstractmethod
    def __call__(self, dataset, run_log_dir):
        raise NotImplementedError
