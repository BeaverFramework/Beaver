"""Offline (in-process) TransformerLens backend, with prefix caching + hidden states.

Why this exists
---------------
Neither other backend can give hidden states *and* prefix-cache reuse together:

  * ``hf`` (beaver.hf_server) returns hidden states but has no KV cache at all --
    every request recomputes the whole prompt from scratch.
  * ``vllm-local`` has a real prefix cache but vLLM's offline ``RequestOutput``
    exposes no hidden states.

This backend runs a TransformerLens ``HookedTransformer`` in-process, which
gives both: ``run_with_cache(names_filter=...)`` for residual-stream hooks, and
a frozen KV cache for the shared prompt prefix.

The prefix-cache mechanism
--------------------------
TL's ``TransformerLensKeyValueCache`` can be *frozen*, and its own docstring
names our use case: "The cache can be frozen so that it is not updated during
the forward pass. This is useful when we want to run many inputs with the same
prefix." So:

    kv = KV.init_cache(cfg, device); model(prefix, past_kv_cache=kv)  # prefix ONCE
    kv.freeze()
    for each continuation:  model(cont, past_kv_cache=kv)             # prefix reused

Because it is frozen, each continuation leaves the cache untouched and the next
one reuses it -- the shared prompt is never recomputed.

Validated before this file was written (throwaway probe, Qwen3-4B, 177-token
prefix): the frozen cache stays at exactly the prefix length across repeated
continuation runs (no pollution), argmax matches a full-prompt forward at every
position, and in **float32** the logprob delta vs a full forward is 1.1e-4 --
i.e. the larger bf16 deltas (~0.4) seen at reduced precision are pure
quantization noise, not a structural error.

Two TransformerLens gotchas this file works around
--------------------------------------------------
1. ``init_cache_entry`` allocates the KV tensors with
   ``torch.get_default_dtype()`` (float32), ignoring ``cfg.dtype``. Against a
   bf16 model the first cached attention then dies with "expected scalar type
   BFloat16 but found Float". We cast the entries explicitly rather than
   mutating the global default dtype.
2. ``_load_checkpoint_with_recipe`` (grouping/features.py) patches
   ``cfg.rotary_base`` from the checkpoint's ``rope_theta``, but newer
   transformers nests that key under ``rope_parameters``, so the patch silently
   no-ops and the model loads with the *base* model's rotary base. See
   ``_rope_theta_of``; this backend patches it correctly.

Deliberately simple: one sequence per forward pass, no async, no request
batching, no padding. TL is instrumented and slower per pass than vLLM, so this
backend's value is hidden states + prefix reuse, not raw throughput.
"""

import os
import time
import types

import numpy as np

from beaver.verifiers.worker_common import _w

_MODEL = None
_MODEL_KEY = None
# prefix-KV cache: {prefix_token_tuple: frozen KV cache}, module-level so it
# survives across calls, transitions and instances for the life of the worker
# process (hf-local runs single-process by construction).
#
# Keyed on the exact token tuple, which is what makes reuse safe across
# instances: a different prompt is a different key, so a stale entry can never
# be served to the wrong sequence. Same content-addressing property vLLM gets
# from chain-hashed blocks, just at whole-prefix granularity.
_PREFIX_CACHE = {}
_PREFIX_ORDER = []
# 32 entries is well under a GB of KV even for a 14B (see _kv_bytes_per_token);
# the old value of 4 was sized for one multi-row scoring call at a time and
# thrashes badly once the frontier loop starts caching continuation prefixes.
_MAX_PREFIX_ENTRIES = 32
# Shortest prefix worth caching. Below this the KV write costs more than the
# prefill it saves. Settable via BEAVER_HF_LOCAL_MIN_PREFIX so an ablation can
# disable caching entirely (set it huge) without editing code.
_MIN_PREFIX_TOKENS = int(os.environ.get("BEAVER_HF_LOCAL_MIN_PREFIX", "16"))
# Last single-row request, used to DISCOVER a shared boundary by longest
# common prefix -- the same thing vLLM's block hashes discover, without
# needing the caller to declare where the prompt ends.
_LAST_ROW = None

# TransformerLens hook names, by what the NVF wants to read.
#   resid_post -> [d_model]  residual stream after the block (== hf_server)
#   mlp_post   -> [d_mlp]    MLP neuron activations after the nonlinearity
_HOOK_SUFFIX = {
    "resid_post": "hook_resid_post",
    "mlp_post": "mlp.hook_post",
}


def _rope_theta_of(hf_cfg):
    """Checkpoint's true rope_theta, handling both config layouts.

    Older transformers put ``rope_theta`` at the top level; newer nests it in
    ``rope_parameters``. grouping/features.py only checks the flat key, so on
    newer transformers its rotary-base patch never fires and e.g.
    Qwen3-4B-Instruct-2507 (rope_theta=5000000) silently loads with the base
    Qwen3-4B value (1000000) -- every attention score computed at the wrong
    rotation frequency.
    """
    if hf_cfg.get("rope_theta") is not None:
        return hf_cfg["rope_theta"]
    return (hf_cfg.get("rope_parameters") or {}).get("rope_theta")


# Checkpoints TL has no conversion recipe for, mapped to a registered model of
# the same architecture. Kept here rather than inferred because the mapping is
# a claim about weight layout that only a human should make.
#
# Safe to apply automatically *in this file* only because ``_load_model``
# re-patches ``cfg.rotary_base`` from the checkpoint's own config (see
# ``_rope_theta_of``) -- the rotary base is the one thing the base model's
# recipe gets wrong for these fine-tunes.
_TL_ARCH_ALIAS = {
    "Qwen/Qwen3-4B-Instruct-2507": "Qwen/Qwen3-4B",
    "Qwen/Qwen3-8B-Instruct-2507": "Qwen/Qwen3-8B",
}


def _resolve_tl_recipe(model_name):
    """Recipe to convert ``model_name`` with, or None if TL knows it directly.

    Raises with an actionable message when the checkpoint is neither registered
    with TransformerLens nor aliased above. This check has to happen *before*
    the HF weights are loaded: the old code discovered the same failure after
    ``AutoModelForCausalLM.from_pretrained``, so every instance in a run paid a
    full checkpoint load to arrive at the identical error.
    """
    from transformer_lens import loading_from_pretrained as loading

    if model_name in _TL_ARCH_ALIAS:
        return _TL_ARCH_ALIAS[model_name]
    try:
        loading.get_official_model_name(model_name)
    except ValueError:
        raise ValueError(
            f"hf-local: TransformerLens has no conversion recipe for "
            f"{model_name!r}. Pass --hf_local_tl_recipe with a registered "
            f"model of the same architecture (e.g. Qwen/Qwen3-4B for "
            f"Qwen/Qwen3-4B-Instruct-2507), or add it to "
            f"beaver/model/hf_local.py:_TL_ARCH_ALIAS."
        ) from None
    return None


def _load_model(model_name, device, dtype, tl_recipe=None):
    """Load ``model_name`` into a HookedTransformer.

    ``tl_recipe`` names a TL-registered model whose conversion recipe fits this
    checkpoint's architecture (e.g. "Qwen/Qwen3-4B" for
    "Qwen/Qwen3-4B-Instruct-2507"). When omitted it is resolved from
    ``_TL_ARCH_ALIAS``; a checkpoint TL knows directly needs no recipe and
    takes the plain path, falling back to using itself as its own recipe.
    """
    import torch
    from transformer_lens import HookedTransformer, loading_from_pretrained as loading
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if tl_recipe is None:
        tl_recipe = _resolve_tl_recipe(model_name)

    if tl_recipe is None:
        # Registered name: the plain path works. If it still fails it failed
        # for some *other* reason, so retry by driving the conversion by hand.
        try:
            m = HookedTransformer.from_pretrained(model_name, device=device, dtype=dtype)
            m.eval()
            return m
        except Exception as e:
            print(f"[hf_local] direct load failed ({e}); retrying via recipe")
            tl_recipe = model_name
    else:
        print(f"[hf_local] converting {model_name} via recipe {tl_recipe}")

    hf_model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    hf_cfg = hf_model.config.to_dict()

    official = loading.get_official_model_name(tl_recipe)
    cfg = loading.get_pretrained_model_config(
        official, hf_cfg=hf_cfg, fold_ln=True, device=device, dtype=dtype
    )
    rope = _rope_theta_of(hf_cfg)
    if rope is not None and cfg.rotary_base != rope:
        print(f"[hf_local] rotary_base {cfg.rotary_base} -> {rope} (from checkpoint)")
        cfg.rotary_base = rope

    state_dict = loading.get_pretrained_state_dict(official, cfg, hf_model, dtype=dtype)
    model = HookedTransformer(cfg, tokenizer, move_to_device=False)
    model.load_and_process_state_dict(
        state_dict, fold_ln=True, center_writing_weights=True,
        center_unembed=True, fold_value_biases=True,
    )
    model.move_model_modules_to_device()
    model.eval()
    return model


def get_model():
    """Process-wide HookedTransformer, loaded on first use."""
    global _MODEL, _MODEL_KEY
    import torch

    dtype_name = getattr(_w, "hf_local_dtype", "bfloat16")
    key = (_w.model_name, dtype_name, getattr(_w, "hf_local_tl_recipe", None))
    if _MODEL is not None and _MODEL_KEY == key:
        return _MODEL
    if _MODEL is not None:
        raise RuntimeError(
            f"hf-local model already loaded for {_MODEL_KEY}, cannot switch to "
            f"{key} in the same process."
        )
    print(f"[hf_local] loading {key[0]} (dtype={dtype_name}) ...")
    t0 = time.time()
    _MODEL = _load_model(
        _w.model_name, "cuda", getattr(torch, dtype_name),
        tl_recipe=getattr(_w, "hf_local_tl_recipe", None),
    )
    _MODEL_KEY = key
    print(f"[hf_local] ready in {time.time() - t0:.1f}s "
          f"(n_layers={_MODEL.cfg.n_layers}, rotary_base={_MODEL.cfg.rotary_base})")
    return _MODEL


def _get_prefix_kv(model, prefix_ids):
    """Frozen KV cache for ``prefix_ids``, computing it on a miss.

    The returned cache is frozen, so callers may run any number of
    continuations against it without mutating it.
    """
    import torch
    from transformer_lens.cache.key_value_cache import TransformerLensKeyValueCache as KV

    key = tuple(prefix_ids)
    if key in _PREFIX_CACHE:
        _PREFIX_ORDER.remove(key)
        _PREFIX_ORDER.append(key)
        return _PREFIX_CACHE[key]

    dtype = next(model.parameters()).dtype
    kv = KV.init_cache(model.cfg, model.cfg.device, batch_size=1)
    # TL allocates these with torch.get_default_dtype() (float32) regardless of
    # cfg.dtype -- cast so a bf16 model doesn't trip on the first cached attn.
    for e in kv.entries:
        e.past_keys = e.past_keys.to(dtype)
        e.past_values = e.past_values.to(dtype)

    toks = torch.tensor([list(prefix_ids)], device=model.cfg.device, dtype=torch.long)
    with torch.no_grad():
        model(toks, past_kv_cache=kv)
    kv.freeze()

    _PREFIX_CACHE[key] = kv
    _PREFIX_ORDER.append(key)
    while len(_PREFIX_ORDER) > _MAX_PREFIX_ENTRIES:
        old = _PREFIX_ORDER.pop(0)
        _PREFIX_CACHE.pop(old, None)
    return kv


def _common_prefix_len(a, b):
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def _longest_cached_prefix(row):
    """Longest cached prefix that is a PROPER prefix of ``row`` (or 0).

    Proper matters: serving the whole row from cache would leave no tokens for
    the forward pass and hence no logits.
    """
    best = 0
    for key in _PREFIX_ORDER:
        n = len(key)
        if n < len(row) and n > best and tuple(row[:n]) == key:
            best = n
    return best


def _as_token_ids(prompt):
    """Normalize prompt to a list of token-ID rows (same contract as local_vllm).

    Text is tokenized with ``add_special_tokens=False``: build_prompt() has
    already applied the chat template, which carries the model's own BOS.
    """
    tok = _w.tokenizer
    if isinstance(prompt, str):
        return [tok.encode(prompt, add_special_tokens=False)]
    if isinstance(prompt, list) and prompt and isinstance(prompt[0], str):
        return [tok.encode(p, add_special_tokens=False) for p in prompt]
    if isinstance(prompt, list) and prompt and isinstance(prompt[0], list):
        return [[int(t) for t in r] for r in prompt]
    return [[int(t) for t in prompt]]


def _topk_dict(logprobs_row, k):
    """One position's top-k as the wire-format {"token_id:N": float} dict."""
    import torch

    k = min(k, logprobs_row.shape[-1])
    vals, idx = torch.topk(logprobs_row, k)
    return {f"token_id:{int(i)}": float(v) for i, v in zip(idx.tolist(), vals.tolist())}


def _b64(tensor):
    import base64
    return base64.b64encode(
        tensor.detach().cpu().float().numpy().astype(np.float32).tobytes()
    ).decode("ascii")


def completion(
    prompt,
    *,
    max_tokens,
    n=1,
    logprobs=None,
    echo=False,
    temperature,
    top_p,
    top_k,
    timeout=None,          # accepted for signature parity; no sockets here
    return_hidden_states=False,
    label="model_call",
    record_timing,
):
    """Run rows through the in-process HookedTransformer.

    Returns the same list-of-choices shape as the other backends
    (see beaver/model/parsing.py).
    """
    import torch

    model = get_model()
    rows = _as_token_ids(prompt)
    k = logprobs or 0

    hidden_layers = getattr(_w, "hf_local_hidden_layers", None)
    if return_hidden_states and not hidden_layers:
        layer = getattr(_w, "probe_layer", None)
        hidden_layers = [layer] if layer is not None else [model.cfg.n_layers // 2]
    # "resid_post" reads the residual stream after each block -- the same
    # tensor beaver.hf_server's forward hook on model.model.layers[i] returns
    # (verified numerically identical at float32). "mlp_post" reads the
    # post-activation MLP neurons instead, which are d_mlp-wide (8192 for
    # Llama-3.2-3B vs d_model 3072), sparser, and closer to single features.
    hook_kind = getattr(_w, "hf_local_hook", "resid_post") or "resid_post"
    if hook_kind not in _HOOK_SUFFIX:
        raise ValueError(
            f"Unknown hf_local_hook {hook_kind!r}; expected one of "
            f"{sorted(_HOOK_SUFFIX)}"
        )
    suffix = _HOOK_SUFFIX[hook_kind]
    hooks = ([f"blocks.{i}.{suffix}" for i in hidden_layers]
             if return_hidden_states else [])
    if return_hidden_states:
        bad = [i for i in hidden_layers if not 0 <= i < model.cfg.n_layers]
        if bad:
            raise ValueError(
                f"hf_local_hidden_layers {bad} out of range for a "
                f"{model.cfg.n_layers}-layer model"
            )

    # How much of each row can be served from a frozen prefix cache.
    #
    # This must never exceed the true shared *prompt* prefix: an echo/scoring
    # call needs a distribution at every continuation position, and positions
    # covered by the cache produce none. Over-caching a single row (e.g.
    # len-1) silently returns empty distributions for the whole path, which
    # starves TokenTrieCache's rebuild of children and makes the verifier
    # converge far worse while staying technically sound -- a bug that is
    # invisible in bounds but shows up as a large forward-pass regression.
    #
    # Multi-row calls: use the longest common prefix across rows, which is
    # exactly the shared prompt (that is how scoring batches are built).
    #
    # Single-row calls: reuse the longest ALREADY-CACHED prefix, and when
    # there is no hit, seed the cache from the longest common prefix with the
    # previous single-row request. That LCP is a *discovered* boundary, not an
    # invented one -- by construction those tokens are shared, which is the
    # property this code has always required. Without the seeding step the
    # cache could never bootstrap (only multi-row calls ever wrote to it, and
    # the frontier verifier makes none), so num_cached_tokens was 0 on every
    # call and the whole mechanism was dead.
    global _LAST_ROW
    shared = 0
    if len(rows) > 1:
        shortest = min(len(r) for r in rows)
        while shared < shortest - 1 and all(r[shared] == rows[0][shared] for r in rows):
            shared += 1
    else:
        row = rows[0]
        shared = _longest_cached_prefix(row)
        if shared == 0 and _LAST_ROW is not None:
            cand = _common_prefix_len(row, _LAST_ROW)
            # Cap at len(row)-1 so the live pass always has a real token left:
            # a fully-cached row would produce no logits at all.
            cand = min(cand, len(row) - 1)
            if cand >= _MIN_PREFIX_TOKENS:
                _get_prefix_kv(get_model(), row[:cand])   # seed for next time
                shared = cand
        _LAST_ROW = list(row)

    t0 = time.perf_counter()
    choices = []
    cached_tokens = 0
    # Fine-grained breakdown of where a call's time actually goes. The
    # transition-level profile only sees one `model_generate` number, which on
    # this backend is ~99% of the instance and says nothing about *why*.
    tail_tokens = 0
    prefix_kv_s = forward_s = logprob_s = hidden_s = 0.0
    # CUDA is asynchronous: without an explicit sync the forward's cost lands
    # on whichever later op first forces one (here .float()/.item() in the
    # logprob assembly), which would blame post-processing for GPU work. The
    # sync moves no wall time -- the very next line would have blocked anyway
    # -- it only attributes it to the right bucket.
    _sync = torch.cuda.synchronize if torch.cuda.is_available() else (lambda: None)
    with torch.no_grad():
        # Serve one token FEWER than the shared prefix from the cache.
        #
        # top_logprobs[j] must hold the distribution that PREDICTS token j,
        # i.e. the logits at position j-1. The first slot any caller reads is
        # j = start (the prompt length), which therefore needs logits at
        # position start-1 -- the last prompt token. If the cache covered all
        # `shared` positions we would never compute those logits and slot
        # `start` would come back empty, starving the trie rebuild of children
        # at depth 0 and silently costing the verifier its lower bound.
        # `shared` is the longest common prefix across ROWS. For a scoring
        # batch the rows are prompt+parent_path, and sibling parent paths
        # routinely share leading tokens, so `shared` can extend PAST the
        # prompt into the path. Caching that far silently drops the
        # distributions for the overlapped path positions -- the rebuild then
        # finds no expandable leaves and every warm-started instance dies with
        # `frontier_empty`.
        #
        # We cannot see the prompt/path boundary from here, but a path is at
        # most gen_length tokens, so any position at least gen_length+1 back
        # from the shortest row is provably still prompt. Bound the cache by
        # that as well as by shared-1 (needed so the last prompt position's
        # logits, which predict the first path token, are always computed).
        gen_len = int(getattr(_w, "gen_length", 0) or 0)
        shortest_row = min(len(r) for r in rows)
        cache_len = max(0, min(shared - 1, shortest_row - gen_len - 1))
        kv = None
        if cache_len > 1:
            _t = time.perf_counter()
            kv = _get_prefix_kv(model, rows[0][:cache_len])
            _sync()
            prefix_kv_s = time.perf_counter() - _t

        for i, ids in enumerate(rows):
            tail = ids[cache_len:] if kv is not None else ids
            tail_tokens += len(tail)
            toks = torch.tensor([tail], device=model.cfg.device, dtype=torch.long)
            _t = time.perf_counter()
            if hooks:
                lg, cache = model.run_with_cache(
                    toks, past_kv_cache=kv, names_filter=hooks
                )
            else:
                lg, cache = model(toks, past_kv_cache=kv), None
            _sync()
            forward_s += time.perf_counter() - _t
            if kv is not None:
                cached_tokens += cache_len

            _t = time.perf_counter()
            lg = lg[0].float()                       # [tail_len, vocab]
            lps = torch.log_softmax(lg, dim=-1)

            if echo:
                L = len(ids)
                # Positions before `shared` were served from the frozen cache and
                # have no logits here. Emit empty dicts / None so the 1:1 length
                # invariant ops.model_score_sequences_per_token slices against
                # still holds -- exactly how vLLM reports cache-hit positions.
                token_logprobs = [None] * L
                top_logprobs = [{} for _ in range(L + 1)]
                base = cache_len if kv is not None else 0
                for j in range(lps.shape[0]):
                    p_abs = base + j          # absolute position of these logits
                    nxt = p_abs + 1           # the token this row predicts
                    # Slot `nxt` holds the distribution predicting token `nxt`
                    # -- same convention as hf_server.py's echo builder. Storing
                    # it at p_abs instead shifts every distribution one level
                    # deeper in the rebuilt trie.
                    if nxt <= L:
                        top_logprobs[nxt] = _topk_dict(lps[j], k) if k else {}
                    if 1 <= nxt <= L - 1:
                        token_logprobs[nxt] = float(lps[j, ids[nxt]])
                tokens = [f"token_id:{t}" for t in ids] + ["token_id:0"]
                text = ""
            else:
                last = lps[-1]
                if temperature and temperature > 0:
                    probs = torch.softmax(lg[-1] / temperature, dim=-1)
                    tid = int(torch.multinomial(probs, 1).item())
                else:
                    tid = int(last.argmax().item())
                tokens = [f"token_id:{tid}"]
                token_logprobs = [float(last[tid])]
                d = _topk_dict(last, k) if k else {}
                d.setdefault(f"token_id:{tid}", float(last[tid]))
                top_logprobs = [d]
                text = _w.tokenizer.decode([tid], skip_special_tokens=False)
            logprob_s += time.perf_counter() - _t

            _t = time.perf_counter()
            hs_out = None
            if hooks and cache is not None:
                hs_out = []
                for layer in hidden_layers:
                    act = cache[f"blocks.{layer}.{suffix}"][0, -1, :]
                    hs_out.append({
                        "layer": layer,
                        "data": _b64(act),
                        "shape": [int(act.shape[-1])],
                        "dtype": "float32",
                    })
                del cache
            hidden_s += time.perf_counter() - _t

            choices.append(types.SimpleNamespace(
                index=i, text=text, hidden_states=hs_out,
                logprobs=types.SimpleNamespace(
                    tokens=tokens, token_logprobs=token_logprobs,
                    top_logprobs=top_logprobs, text_offset=[],
                ),
            ))

    net_s = time.perf_counter() - t0
    record_timing(
        label, prefix="local", net_s=net_s, parse_s=0.0,
        n_rows=len(rows), resp_bytes=0,
        num_cached_tokens=cached_tokens,
        # Breakdown of net_s. These are summed across calls by
        # drain_model_timings, so they land in the per-transition profile
        # alongside num_cached_tokens with no aggregation changes.
        prefix_kv_s=prefix_kv_s,     # building/fetching the frozen prefix KV
        forward_s=forward_s,         # the per-row forward passes themselves
        logprob_s=logprob_s,         # log_softmax + top-k dict assembly
        hidden_s=hidden_s,           # hidden-state slice + base64 encode
        n_forwards=len(rows),        # forwards per call (one per row: no batching)
        tail_tokens=tail_tokens,     # tokens actually recomputed, vs num_cached_tokens
    )
    return choices
