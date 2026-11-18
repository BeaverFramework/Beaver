"""One-step (single next-token) total-variation distance between prompts — exact.

Given ``n`` prompts we let the model emit exactly one token from each and compare
the resulting next-token distributions pairwise in total variation:

    TV(p, q) = 1/2 * sum_token |p(token) - q(token)|    in [0, 1].

This is the δ of the regime note: ``TV(mu_p_i, mu_p_j) <= δ``, in the same units as
ε (the verifier's ``early_termination_threshold``), so a TV matrix can gate grouping
directly (co-group only pairs with TV < ε).

This is the *exact* computation: every distribution is the full softmax over the
whole vocabulary (``[V]``), so there is no truncation and no approximation. The
full distribution is read straight from a HuggingFace ``AutoModelForCausalLM`` —
any checkpoint loads, with no TransformerLens model-registry restriction — by
taking the last-position logits and softmaxing over the whole vocab.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np


def tv_pair(p: np.ndarray, q: np.ndarray) -> float:
    """Exact total-variation distance between two full next-token distributions.

    ``p`` and ``q`` are dense probability vectors over the whole vocabulary
    (``[V]``, summing to 1). Returns ``1/2 * sum_token |p - q|`` in ``[0, 1]``.
    """
    return 0.5 * float(np.sum(np.abs(p - q)))



def top_p_sparse(
    probs: np.ndarray, top_p: float = 0.999, max_k: int = 256
) -> tuple[np.ndarray, np.ndarray, float]:
    """Truncate a full `[V]` probability vector to its top-p mass.

    Returns `(token_ids, probs_desc, covered_mass)` sorted by probability
    descending, containing the smallest prefix whose cumulative mass reaches
    `top_p` (capped at `max_k` entries either way). Restricting a TV
    computation to the union of two distributions' top_p_sparse output changes
    the result by at most `(1-top_p)` versus the full-vocab TV, since each
    side's excluded tail mass is bounded by `1-top_p` by construction.
    side's excluded tail mass is bounded by `1-top_p` by construction —
    but only when `covered_mass >= top_p` is actually reached. If `max_k`
    caps the truncation before cumulative mass hits `top_p` (e.g. a flat,
    non-peaky distribution), the excluded tail is `1 - covered_mass` instead,
    which can be far larger than `1 - top_p`. Callers that care about the
    error bound must check the returned `covered_mass` against `top_p`.
    """
    order = np.argsort(probs)[::-1]
    sorted_probs = probs[order]
    cumsum = np.cumsum(sorted_probs)
    # Tolerance guards against float64 cumsum rounding landing just below the
    # target (e.g. 0.7 + 0.2 evaluates to 0.8999999999999999, not 0.9), which
    # would otherwise push the cutoff index off by one.
    cutoff = int(np.searchsorted(cumsum, top_p - 1e-10) + 1)
    cutoff = min(cutoff, max_k, len(probs))
    ids = order[:cutoff]
    p = sorted_probs[:cutoff]
    covered = float(cumsum[cutoff - 1])
    return ids, p, covered


def sparse_tv_pair(
    ids_a: np.ndarray, probs_a: np.ndarray, ids_b: np.ndarray, probs_b: np.ndarray
) -> float:
    """TV distance between two top_p_sparse-truncated distributions.

    Computes ``0.5 * sum(|p(v) - q(v)|)`` over the union of the two token-id
    sets, treating a token missing from one side as probability 0 on that
    side. Accurate to within the top_p_sparse truncation's error bound versus
    the true full-vocab TV (see top_p_sparse's docstring).
    the true full-vocab TV *only if both inputs' `covered_mass >= top_p`* (see
    top_p_sparse's docstring) — if `max_k` capped either side's truncation
    first, the actual error can be much larger than `1 - top_p`.
    """
    a = dict(zip(ids_a.tolist(), probs_a.tolist()))
    b = dict(zip(ids_b.tolist(), probs_b.tolist()))
    union = set(a) | set(b)
    total = sum(abs(a.get(tok, 0.0) - b.get(tok, 0.0)) for tok in union)
    return 0.5 * total


def one_step_tv_matrix(
    instances: Optional[Sequence[dict]] = None,
    backend=None,
    probs: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Symmetric ``[n, n]`` matrix of exact one-step TV distances (zero diagonal).

    Provide a precomputed ``[n, V]`` ``probs`` matrix (each row a full next-token
    distribution), or ``instances`` + a TL ``backend`` to compute them (one
    forward pass per prompt via ``backend.next_token_full_dists``).
    """
    if probs is None:
        if instances is None or backend is None:
            raise ValueError("pass either `probs`, or `instances` and `backend`")
        probs = backend.next_token_full_dists(instances)

    probs = np.asarray(probs, dtype=np.float64)
    n = probs.shape[0]
    D = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            d = tv_pair(probs[i], probs[j])
            D[i, j] = D[j, i] = d
    return D


# ── HuggingFace backend (full-vocab logits, any checkpoint) ────────────────────
class HFBackend:
    """Minimal HuggingFace backend exposing the full next-token distribution.

    Unlike the TransformerLens backend it loads any ``AutoModelForCausalLM``
    checkpoint (no model-registry restriction), which is what the exact TV needs.
    Prompts are chat-templated through the same :func:`build_prompt` the verifier
    uses, so the distribution matches the real run (system prompt, few-shot, etc.).
    """

    def __init__(self, model_name: str, device: str = "cuda", dtype: str = "bfloat16"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        # device="auto" shards the model across the visible GPUs (accelerate).
        # Restrict/choose GPUs via CUDA_VISIBLE_DEVICES, e.g.
        #   CUDA_VISIBLE_DEVICES=0,1,3 ... --device auto
        # Inputs go to the first shard; accelerate hooks move activations.
        self.device = "cuda:0" if device == "auto" else device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Batched last-token logits need left padding so logits[:, -1] is the real
        # final token for every row regardless of that row's unpadded length.
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=getattr(torch, dtype), device_map=device
        )
        self.model.eval()
        if device == "auto":
            self.device = next(self.model.parameters()).device

    def next_token_full_dists(self, instances: Sequence[dict]) -> np.ndarray:
        from tqdm import tqdm

        from beaver.verifiers.worker_common import build_prompt

        rows: List[np.ndarray] = []
        with self._torch.no_grad():
            for inst in tqdm(instances, desc="next-token dists"):
                text = build_prompt(
                    inst, [], tokenizer=self.tokenizer, chat_mode=True,
                    system_message=None,
                    fewshot_messages=inst.get("fewshot_messages") or [],
                )
                # add_special_tokens=False: `text` is already chat-templated
                # (build_prompt renders the full template to a string, which
                # for e.g. Llama-3 already embeds the literal BOS token) —
                # the default add_special_tokens=True would add a second,
                # redundant BOS for any tokenizer that auto-prepends one.
                toks = self.tokenizer(
                    text, return_tensors="pt", add_special_tokens=False
                ).to(self.device)
                logits = self.model(**toks).logits[0, -1]  # [vocab]
                probs = self._torch.softmax(logits.float(), dim=-1)
                rows.append(probs.cpu().numpy().astype(np.float64))
        return np.stack(rows, axis=0)

    def sparse_next_token_dist(
        self,
        instance: dict,
        prefix_tokens: Sequence[int],
        top_p: float = 0.999,
        max_k: int = 256,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Top-p-truncated next-token distribution after prompt + prefix_tokens.

        Same softmax computation as next_token_full_dists, but conditions on
        prefix_tokens appended after the prompt — this is what makes k-step
        greedy rollout possible — and returns only the top_p_sparse-truncated
        (ids, probs, covered_mass) instead of the full [V] vector.

        The returned `covered_mass` is only guaranteed close to `top_p` when
        `top_p_sparse` reaches it before `max_k` binds; callers relying on the
        `<=0.001`-scale truncation error should check `covered_mass >= top_p`
        themselves (see `top_p_sparse`'s docstring) — for low-peakiness
        distributions `max_k` can cap the truncation well short of `top_p`,
        leaving an excluded tail of `1 - covered_mass` rather than `1 - top_p`.
        """
        from beaver.verifiers.worker_common import build_prompt

        text = build_prompt(
            instance, list(prefix_tokens), tokenizer=self.tokenizer, chat_mode=True,
            system_message=None, fewshot_messages=instance.get("fewshot_messages") or [],
        )
        # add_special_tokens=False: see next_token_full_dists — `text` is
        # already chat-templated and may already contain a literal BOS.
        toks = self.tokenizer(
            text, return_tensors="pt", add_special_tokens=False
        ).to(self.device)
        with self._torch.no_grad():
            logits = self.model(**toks).logits[0, -1]
        probs = self._torch.softmax(logits.float(), dim=-1).cpu().numpy().astype(np.float64)
        return top_p_sparse(probs, top_p=top_p, max_k=max_k)

    def sparse_next_token_dists(
        self,
        instances: Sequence[dict],
        prefixes: Sequence[Sequence[int]],
        top_p: float = 0.999,
        max_k: int = 256,
        batch_size: int = 32,
    ) -> List[tuple[np.ndarray, np.ndarray, float]]:
        """Batched :meth:`sparse_next_token_dist` over many (instance, prefix) pairs.

        Same per-pair result as calling ``sparse_next_token_dist`` in a loop, but
        prompts are padded and pushed through the model ``batch_size`` at a time.
        Left padding (set in ``__init__``) keeps ``logits[:, -1]`` aligned to each
        row's true last token, and the attention mask zeroes out the pad
        positions, so each row's next-token distribution matches the single-prompt
        computation up to the usual batched-matmul float noise. Returns one
        ``(ids, probs, covered_mass)`` tuple per input, in the input order.
        """
        from beaver.verifiers.worker_common import build_prompt

        if len(instances) != len(prefixes):
            raise ValueError("instances and prefixes must be the same length")

        results: List[Optional[tuple[np.ndarray, np.ndarray, float]]] = [None] * len(instances)
        for start in range(0, len(instances), batch_size):
            batch_insts = instances[start : start + batch_size]
            batch_prefixes = prefixes[start : start + batch_size]
            texts = [
                build_prompt(
                    inst, list(prefix), tokenizer=self.tokenizer, chat_mode=True,
                    system_message=None,
                    fewshot_messages=inst.get("fewshot_messages") or [],
                )
                for inst, prefix in zip(batch_insts, batch_prefixes)
            ]
            # add_special_tokens=False: see next_token_full_dists.
            toks = self.tokenizer(
                texts, return_tensors="pt", padding=True, add_special_tokens=False
            ).to(self.device)
            with self._torch.no_grad():
                # logits_to_keep=1 restricts the lm_head projection to the last
                # position; the default (0) is special-cased by HF as "keep all"
                # (slice(-0, None) == slice(0, None)), which computes full-vocab
                # logits over the whole prompt for every row and OOMs at batch.
                logits = self.model(**toks, logits_to_keep=1).logits[:, -1, :]  # [B, vocab]
            probs = (
                self._torch.softmax(logits.float(), dim=-1)
                .cpu()
                .numpy()
                .astype(np.float64)
            )
            for bi in range(len(batch_insts)):
                results[start + bi] = top_p_sparse(probs[bi], top_p=top_p, max_k=max_k)
        return results  # type: ignore[return-value]

    def greedy_rollout_kv(
        self,
        instances: Sequence[dict],
        k: int,
        top_p: float = 0.999,
        max_k: int = 256,
        batch_size: int = 32,
    ) -> List[dict]:
        """Batched greedy k-step rollout with KV-cache reuse.

        For each micro-batch of ``batch_size`` prompts: prefill all prompts once,
        then take k greedy steps feeding only the single newly-appended token per
        step (the backbone reuses the cached keys/values), so each step is a
        one-token forward instead of a full re-prefill of the growing prompt. This
        is the speedup lever — the k redundant re-prefills of :func:`sparse_next_token_dists`
        collapse to one prefill + k cheap single-token decodes, and batching keeps
        those otherwise memory-bandwidth-bound decodes GPU-efficient.

        Correctness notes:
          * Left padding (set in ``__init__``) right-aligns every sequence, so a
            single token is appended at the same cache position for the whole
            batch. The attention mask is extended by one each step and passed to
            every forward; ``position_ids`` is left to the model to derive. The
            model places real tokens at absolute positions ``pad_count .. L_pad-1``
            during prefill and continues the decode from ``cache_position = L_pad``,
            which is self-consistent with the cached keys — RoPE only depends on
            the query-key *relative* offset, so this matches the unpadded rollout
            (verified to ~1e-2 bf16 noise). Passing an explicit pad-stripped
            ``position_ids`` (e.g. ``real_len + step``) instead is WRONG: it
            mismatches the cached keys' absolute positions and corrupts every
            row that carried any left padding.
          * ``worst_covered_mass`` is tracked across all k greedy steps and the
            final fork distribution, identical to the serial reference, so the
            same ``covered_mass < top_p`` caveat applies (see ``top_p_sparse``).

        Returns one dict per instance (input order):
        ``{path, fork_ids, fork_probs, worst_covered_mass}``. ``path`` is the k
        greedy token ids; ``fork_ids``/``fork_probs`` are the top_p_sparse
        distribution at depth k+1 (the fork point).
        """
        from beaver.verifiers.worker_common import build_prompt
        from tqdm import tqdm

        torch = self._torch
        results: List[Optional[dict]] = [None] * len(instances)

        starts = range(0, len(instances), batch_size)
        for start in tqdm(
            starts,
            total=len(starts),
            desc=f"k={k} rollout (bs={batch_size})",
            unit="batch",
        ):
            batch = instances[start : start + batch_size]
            texts = [
                build_prompt(
                    inst, [], tokenizer=self.tokenizer, chat_mode=True,
                    system_message=None,
                    fewshot_messages=inst.get("fewshot_messages") or [],
                )
                for inst in batch
            ]
            # add_special_tokens=False: see next_token_full_dists.
            toks = self.tokenizer(
                texts, return_tensors="pt", padding=True, add_special_tokens=False
            ).to(self.device)
            attn = toks["attention_mask"]
            B = attn.shape[0]
            paths: List[List[int]] = [[] for _ in range(B)]
            worst = [1.0] * B

            with torch.no_grad():
                # logits_to_keep=1: see sparse_next_token_dists — without it the
                # prefill computes full-vocab logits over the whole prompt length
                # per row instead of just the last position, which is all that's
                # ever read below and is what OOMs first at realistic batch sizes.
                out = self.model(
                    input_ids=toks["input_ids"], attention_mask=attn, use_cache=True,
                    logits_to_keep=1,
                )
                past = out.past_key_values
                logits = out.logits[:, -1, :]  # [B, V] distribution before any greedy token

                # k greedy steps: pick top-1 from the current distribution, then
                # decode that one token to advance the cache to the next position.
                for _ in range(k):
                    probs = (
                        torch.softmax(logits.float(), dim=-1)
                        .cpu().numpy().astype(np.float64)
                    )
                    next_ids = []
                    for bi in range(B):
                        ids, _p, covered = top_p_sparse(probs[bi], top_p=top_p, max_k=max_k)
                        worst[bi] = min(worst[bi], covered)
                        tok = int(ids[0])
                        paths[bi].append(tok)
                        next_ids.append(tok)
                    new_tok = torch.tensor(next_ids, device=self.device).unsqueeze(1)  # [B,1]
                    attn = torch.cat(
                        [attn, torch.ones((B, 1), dtype=attn.dtype, device=self.device)],
                        dim=1,
                    )
                    out = self.model(
                        input_ids=new_tok, attention_mask=attn,
                        past_key_values=past, use_cache=True,
                    )
                    past = out.past_key_values
                    logits = out.logits[:, -1, :]

                # after k appended tokens, `logits` is the fork distribution
                fork = (
                    torch.softmax(logits.float(), dim=-1).cpu().numpy().astype(np.float64)
                )
            for bi in range(B):
                fids, fp, fcov = top_p_sparse(fork[bi], top_p=top_p, max_k=max_k)
                worst[bi] = min(worst[bi], fcov)
                results[start + bi] = {
                    "path": tuple(paths[bi]),
                    "fork_ids": fids,
                    "fork_probs": fp,
                    "worst_covered_mass": worst[bi],
                }
        return results  # type: ignore[return-value]

    def hidden_states_and_topk(
        self,
        instance: dict,
        prefixes: Sequence[Sequence[int]],
        top_p: float = 0.99,
        max_k: int = 50,
        batch_size: int = 16,
        must_include: Optional[Sequence[Sequence[int]]] = None,
    ) -> List[tuple[np.ndarray, np.ndarray]]:
        """Batched last-token hidden states (every layer) + top-p next-token
        logprobs, one forward pass per prefix -- the representation-extraction
        analogue of :meth:`sparse_next_token_dists`.

        Unlike the other methods here, ``prefixes`` are already token ids (not
        re-tokenized text): each is appended directly after the prompt's own
        token ids, so this is safe to call with the partial/incomplete
        continuations a trie rebuild walks, not just whole prompts. This
        avoids retokenizing text entirely, which sidesteps the double-BOS
        class of bug by construction -- there is no chat-template string
        round-trip here at all.

        The returned distribution is a genuine top-p nucleus (``top_p_sparse``,
        same helper :meth:`sparse_next_token_dists`/:meth:`greedy_rollout_kv`
        use) capped at ``max_k`` candidates, not a fixed top-k count: a child
        outside its parent's top-``top_p`` mass is a child that should never
        be materialized as a node at all, so this is what actually decides
        which subtrees a trie rebuild expands. ``max_k`` is just a safety
        bound for the rare very-flat distribution that would otherwise need
        many tokens to reach ``top_p`` mass.

        ``must_include`` (optional, one token-id list per prefix): token ids
        that must appear in that prefix's returned distribution even if they
        fall outside the top-p nucleus, with their *true* logprob (read off
        the same log-softmax row, not fabricated) appended as an extra row.
        This solves a different problem than top_p/max_k: a trie rebuild
        needs every cached on-path continuation to survive
        re-materialization, but this instance's own re-scored distribution
        can legitimately differ from whatever distribution originally put
        that token in cache (distribution shift between prompts in the same
        group) -- so even a correctly-computed top_p=0.99 nucleus can
        validly miss a cached token. ``must_include`` guarantees fidelity
        regardless; ``top_p``/``max_k`` independently control exploration
        breadth into genuinely new territory.

        Returns one ``(dist, hidden)`` pair per prefix, in input order:
        ``dist`` is ``[N, 2]`` ``[token_id, logprob]`` (descending by
        logprob) -- a drop-in for the ``[token_id, logprob]`` arrays
        ``TokenTrieCache._get_distribution`` normally gets over HTTP.
        ``hidden`` is ``[n_layers+1, hidden_size]`` float32 (every
        ``hidden_states`` layer HF returns, including the embedding layer),
        the last real token's residual stream at each layer -- left padding
        (set in ``__init__``) keeps position ``-1`` aligned to each row's true
        last token regardless of that row's unpadded length, same as every
        other batched method here.
        """
        from beaver.verifiers.worker_common import build_prompt

        prompt_text = build_prompt(
            instance, [], tokenizer=self.tokenizer, chat_mode=True,
            system_message=None, fewshot_messages=instance.get("fewshot_messages") or [],
        )
        prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]

        pad_id = self.tokenizer.pad_token_id
        results: List[Optional[tuple[np.ndarray, np.ndarray]]] = [None] * len(prefixes)

        for start in range(0, len(prefixes), batch_size):
            batch = [prompt_ids + list(p) for p in prefixes[start : start + batch_size]]
            max_len = max(len(s) for s in batch)
            padded = [[pad_id] * (max_len - len(s)) + s for s in batch]
            attn = [[0] * (max_len - len(s)) + [1] * len(s) for s in batch]
            input_ids = self._torch.tensor(padded, dtype=self._torch.long, device=self.device)
            attention_mask = self._torch.tensor(
                attn, dtype=self._torch.long, device=self.device
            )

            with self._torch.no_grad():
                # logits_to_keep=1: see sparse_next_token_dists -- without it
                # the lm_head projects every position (not just the last) to
                # full-vocab logits, which is both wasted compute and, under
                # memory pressure, an easy OOM (observed: a 16x60x151936
                # logits tensor at fp32 upcast is enough to fail on a
                # near-full GPU). output_hidden_states is unaffected --
                # hidden_states collection happens before this slicing.
                out = self.model(
                    input_ids=input_ids, attention_mask=attention_mask,
                    output_hidden_states=True, logits_to_keep=1,
                )
                logp = self._torch.log_softmax(out.logits[:, -1, :].float(), dim=-1)
                probs = logp.exp()
                # [n_layers+1, B, H] -> per-row [n_layers+1, H], last (real) token
                layer_stack = self._torch.stack(
                    [h[:, -1, :] for h in out.hidden_states], dim=0
                ).float().cpu().numpy()

            probs_np = probs.double().cpu().numpy()
            for bi in range(len(batch)):
                ids, p_desc, _covered = top_p_sparse(probs_np[bi], top_p=top_p, max_k=max_k)
                rows = [np.stack([ids.astype(np.float64), np.log(p_desc)], axis=1)]
                if must_include is not None:
                    have = set(ids.tolist())
                    extra_ids = [
                        t for t in must_include[start + bi] if t not in have
                    ]
                    if extra_ids:
                        extra_lp = logp[bi, extra_ids].double().cpu().numpy()
                        rows.append(np.stack(
                            [np.asarray(extra_ids, dtype=np.float64), extra_lp], axis=1
                        ))
                topk_arr = np.concatenate(rows, axis=0) if len(rows) > 1 else rows[0]
                results[start + bi] = (topk_arr, layer_stack[:, bi, :])

        return results  # type: ignore[return-value]


# ── CLI ────────────────────────────────────────────────────────────────────────
def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Compute the exact one-step (next-token) TV distance matrix "
        "over a dataset of prompts (local TransformerLens backend only)."
    )
    ap.add_argument("--dataset", required=True, help="dataset JSON (list of instances)")
    ap.add_argument("--out", default="tv_out", help="output dir")
    ap.add_argument("--group", default=None,
                    help="keep only instances whose `group` field equals this "
                    "(builds the TV matrix for a single sub-group)")
    ap.add_argument("--max-n", type=int, default=None, help="cap #prompts (debug)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args(argv)

    instances = json.load(open(args.dataset))
    if args.group is not None:
        # match by string so "--group 3" works whether group is int or str.
        instances = [x for x in instances if str(x.get("group")) == str(args.group)]
        if not instances:
            raise SystemExit(f"no instances with group={args.group!r} in {args.dataset}")
    if args.max_n:
        instances = instances[: args.max_n]
    for inst in instances:
        inst.setdefault("prompt", inst.get("test_case_prompt", inst.get("question", "")))

    backend = HFBackend(args.model, device=args.device, dtype=args.dtype)
    D = one_step_tv_matrix(instances=instances, backend=backend)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "tv_distance.npy", D)

    iu = np.triu_indices(len(D), k=1)
    pair = D[iu]
    print(f"\nexact one-step TV matrix  n={len(D)}  -> {out}/tv_distance.npy")
    if pair.size:
        print(f"  off-diagonal TV: mean={pair.mean():.4f}  min={pair.min():.4f}  "
              f"max={pair.max():.4f}  median={np.median(pair):.4f}")


if __name__ == "__main__":
    main()
