"""TV@5 exactly as Definition 5.5 states it: TV between length-k prefix distributions.

The algorithm
-------------
Per prompt, best-first expansion of the response trie until the enumerated
depth-k mass reaches ``top_p``:

    heap of incomplete prefixes keyed by reaching probability
    pop the most probable, expand it (one forward pass), push its children
    a child at depth k (or ending in EOS) is *resolved*
    stop when resolved mass >= top_p

The stopping rule is a *resolved-mass target*, not a step budget: the loop runs
however many expansions that prompt needs. This is the point -- the guarantee
below is exactly what "resolved >= 0.999" buys, 

    m_p = 1 - resolved_p 
    P = P_hat + p_r and Q = Q_hat + q_r,

        | TV@5 - TV@5_trunc |  <=  1/2||p_r - q_r||_1  <=  1/2(m_p + m_p')

Cost is dominated by the expansions, and it is data-dependent: a prompt whose
conditionals are peaky resolves 0.999 in ~k passes, while a high-entropy one
needs the whole top-p tree. ``fp_per_prompt`` in the report is the number to
look at.

Prompts: the instances and prompt settings come from a real ``beaver run`` on
``--config``/``--dataset``, stopped at the verifier call (``verifier_view``),
so the prompts grouped on are the prompts verified on. Row i of every output
matrix is ``instances.json[i]``; give that file to tv_visualize.

Partitions: one enumeration yields two matrices and two clusterings at rho --
``tv_prefix_distance.npy`` / ``tv_cluster_labels.npy`` on the point estimate
TV_trunc, and ``tv_prefix_distance_upper.npy`` / ``tv_cluster_labels_upper.npy``
on TV_trunc + 1/2(m_p + m_p'), which certifies every group's TV@k diameter
<= rho. ``--from-dists`` re-clusters a saved enumeration (other rho, either
matrix) without touching the model.

Example:
    python -m beaver.hf_server --model Qwen/Qwen3-4B-Instruct-2507 \
        --port 8082 --gpu-visible-devices 2 &

    python -m beaver.grouping.tv_prefix_distance \
        --dataset experiments/enron/data/enron_tv_families_50x200.json \
        --model Qwen/Qwen3-4B-Instruct-2507 \
        --config experiments/enron/enron_new.yaml \
        --server-addr http://localhost:8082 \
        --out tv_out/enron_50x200_prefix_k5
    # then, unchanged:
    python -m beaver.grouping.tv_visualize \
        --matrix tv_out/enron_50x200_prefix_k5/tv_prefix_distance.npy \
        --dataset ... --dataset-out ... --threshold 0.01 --out tv_out/...
"""

from __future__ import annotations

import argparse
import heapq
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np



def enumerate_prefix_dist(instance, k, top_p, batch_size, max_fp=None):
    """Best-first enumerate ``instance``'s length-k prefix distribution.

    Returns ``(resolved, resolved_mass, forward_passes)`` where ``resolved`` maps
    a prefix (tuple of token ids, length k or ending in EOS) to its probability.
    ``1 - resolved_mass`` is the ``m_p`` of the bound in the module docstring: it
    absorbs both the frontier we never reached and the tail the server's
    top-``num_logprobs`` cut off, so it needs no separate accounting.

    ``max_fp`` caps the forward passes spent on one prompt. It is a *cost* knob,
    not a correctness one: the loop stops early and the mass it never reached stays
    in ``1 - resolved_mass``, so the bracket in the module docstring still holds --
    it just widens for that prompt. Once ``m_p`` reaches rho the bracket alone spans
    the grouping threshold and no pair involving that prompt can be decided, so it
    falls out as a singleton. That is the conservative outcome, and it is how a cap
    differs from silently truncating the search: the loss is visible in ``m_p``
    rather than hidden in a distance that looks tight but is not.

    ``batch_size`` expands that many frontier nodes per request. It changes only
    how the same expansions are packed into forward passes, never which nodes get
    expanded -- with batch_size > 1 the pops are the current top-b rather than
    strictly the top-1, which can expand a node a pure best-first order would
    have resolved away. Keep it modest.
    """
    from beaver.model import model_generate_next_token_logprobs_batch
    from beaver.verifiers.worker_common import _w

    eos = set(_w.eos_tokens) or {_w.tokenizer.eos_token_id}
    resolved: dict[tuple, float] = {}
    resolved_mass = 0.0
    heap = [(0.0, ())]          # (-logprob, prefix); root has logprob 0
    fp = 0

    while heap and resolved_mass < top_p and (max_fp is None or fp < max_fp):
        batch = [heapq.heappop(heap) for _ in range(min(batch_size, len(heap)))]
        rows = model_generate_next_token_logprobs_batch(
            instance, [list(prefix) for _, prefix in batch]
        )
        fp += len(batch)
        for (neg_lp, prefix), (lp_arr, _prompt) in zip(batch, rows):
            parent_lp = -neg_lp
            for token_id, logprob in lp_arr:          # [N, 2] = [token_id, logprob]
                child = prefix + (int(token_id),)
                child_lp = parent_lp + float(logprob)
                if len(child) == k or int(token_id) in eos:
                    p = float(np.exp(child_lp))
                    resolved[child] = p
                    resolved_mass += p
                else:
                    heapq.heappush(heap, (-child_lp, child))
    # The server's top-N logprobs can sum to marginally over 1.0 (float error;
    # measured 1.000002), so resolved_mass can drift above 1 and make m_p negative
    # -- which would claim a tighter bracket than actually holds. Clamp.
    return resolved, min(resolved_mass, 1.0), fp


def truncated_tv(a, b):
    """``(TV@5_trunc, bracket_half_width)`` for two enumerated distributions.

    ``a``/``b`` are ``(resolved, resolved_mass)``. The half-width is
    ``1/2(m_p + m_p')``; the true d_k lies in ``TV_trunc +- half_width``.
    """
    pa, ma = a
    pb, mb = b
    d = 0.5 * sum(abs(pa.get(s, 0.0) - pb.get(s, 0.0)) for s in set(pa) | set(pb))
    return d, 0.5 * ((1.0 - ma) + (1.0 - mb))


def linf(a, b):
    """Max per-sequence probability difference -- the L_inf distance, NOT TV.

    Kept so the two can be compared on one enumeration (``tv_matrix(metric=...)``).
    ``L_inf <= TV`` always, since a single sequence is one of the events TV takes a
    supremum over, so ``L_inf > rho`` soundly implies ``TV > rho`` -- valid for
    *splitting* a pair, never for merging one. The gap between them grows with the
    support size: mass spread over N sequences caps L_inf near 1/N, so above
    N ~ 1/rho the threshold stops binding and unrelated prompts merge.
    """
    pa, _ = a
    pb, _ = b
    return max(abs(pa.get(s, 0.0) - pb.get(s, 0.0)) for s in set(pa) | set(pb))


def tv_matrix(dists, metric="tv", threshold=None):
    """``[n,n]`` truncated-TV matrix, same contract as ``tv_k_distance``'s output.

    COMMENT (simplification / cost): this is the naive O(n^2) loop over Python
    dicts. It is ~30 s at n=10,000 against ~minutes of model time, so it is not
    worth optimising yet. If it ever is, the cheap win is skipping pairs whose
    depth-1 marginals are already more than the threshold apart -- d_1 <= d_k
    (Proposition F.1), so that rejection is sound -- rather than making this loop
    faster.
    """
    return tv_matrices(dists, metric=metric, threshold=threshold)[0]


def tv_matrices(dists, metric="tv", threshold=None):
    """``(D, D_upper)``: the point matrix and, for metric="tv", the certified
    upper matrix ``TV_trunc + 1/2(m_p + m_p')`` from the same pass (None for
    linf). Clustering D_upper at rho certifies every group's true TV@k
    diameter <= rho, and a prompt with m_p >= rho can then join no group."""
    n = len(dists)
    D = np.ones((n, n), dtype=np.float32)
    np.fill_diagonal(D, 0.0)
    D_hi = None
    if metric == "tv":
        D_hi = D.copy()
    cand = _d1_candidates(dists, threshold) if threshold is not None else None
    it = cand if cand is not None else ((i, j) for i in range(n) for j in range(i + 1, n))
    for i, j in it:
        if metric == "linf":
            D[i, j] = D[j, i] = linf(dists[i], dists[j])
        else:
            d, hw = truncated_tv(dists[i], dists[j])
            D[i, j] = D[j, i] = d
            D_hi[i, j] = D_hi[j, i] = d + hw
    return D, D_hi


def _d1_candidates(dists, threshold):
    """Pairs that survive a sound depth-1 rejection, as an [m,2] index array.

    Proposition F.1 gives d_1 <= d_k, so a pair whose depth-1 marginals are more
    than ``threshold`` apart cannot have d_k <= threshold. The marginals come free
    from the already-enumerated depth-k distributions (sum the mass by first
    token), and they are ~vocab-sized rather than support-sized, so this is a
    dense [n, U] matrix with U in the hundreds instead of the 10^5-entry sequence
    supports. Rejected pairs keep the 1.0 placeholder, which a diameter cap at
    ``threshold`` treats identically to any other above-threshold value.

    The bound is widened by 1/2(m_p + m_p') on each side so truncation error can
    never reject a genuinely close pair.
    """
    n = len(dists)
    cols = sorted({s[0] for d, _ in dists for s in d})
    idx = {t: j for j, t in enumerate(cols)}
    M = np.zeros((n, len(cols)), dtype=np.float32)
    for i, (d, _) in enumerate(dists):
        for s, p in d.items():
            M[i, idx[s[0]]] += p
    resid = np.array([1.0 - m for _, m in dists], dtype=np.float32)

    try:
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        torch, dev = None, None

    out = []
    block = 512
    for s0 in range(0, n, block):
        e0 = min(s0 + block, n)
        if torch is not None:
            d1 = (torch.cdist(torch.from_numpy(M[s0:e0]).to(dev),
                              torch.from_numpy(M).to(dev), p=1.0) * 0.5).cpu().numpy()
        else:
            d1 = 0.5 * np.abs(M[s0:e0, None, :] - M[None, :, :]).sum(-1)
        keep = (d1 - 0.5 * (resid[s0:e0, None] + resid[None, :])) <= threshold
        r, c = np.nonzero(keep)
        r = r + s0
        m = c > r
        out.append(np.stack([r[m], c[m]], axis=1))
    return np.concatenate(out) if out else np.zeros((0, 2), dtype=np.int64)


# Keys of the verifier's worker config that decide which prompt the model sees
# (build_prompt reads all but eos_tokens; the EOS test reads eos_tokens).
PROMPT_KEYS = ("chat_mode", "system_message", "fewshot_messages", "enable_thinking",
               "chat_template_kwargs", "eos_tokens")


class _Captured(Exception):
    pass


def verifier_view(experiment, model_name, dataset_path=None):
    """The worker config and the instances a real verification run would use.

    Runs the real `beaver run` path -- experiment YAML, its load_prompts_fn,
    beaver.run's defaults and renames (use_chat_template -> chat_mode, ...),
    the datasets.Dataset round trip, BaseVerifier (EOS ids via
    _resolve_eos_tokens) -- and stops it at the verifier call, before any model
    request. Grouping on these instances with this config is grouping on
    exactly the prompts the verifier will verify, with no second copy of the
    prompt settings to drift out of sync.

    Returns ``(worker_config, instances)``; instances are in the order the
    verifier receives them.
    """
    import tempfile

    from beaver import cli
    from beaver.verifiers.frontier_verifier import FrontierVerifier

    captured = {}

    def grab(self, dataset, run_log_dir):
        captured["config"] = self._build_worker_config()
        captured["instances"] = [dict(row) for row in dataset]
        raise _Captured

    argv = ["--experiment", str(experiment), "--model", model_name,
            "--verifier", "frontier", "--backend", "vllm",
            "--server_addr", "http://unused.invalid"]
    if dataset_path is not None:
        argv += ["--dataset_path", str(Path(dataset_path).resolve())]
    original = FrontierVerifier.__call__
    FrontierVerifier.__call__ = grab
    try:
        with tempfile.TemporaryDirectory() as tmp:
            cli._run_cmd(argv + ["--log_dir", tmp])
    except _Captured:
        pass
    finally:
        FrontierVerifier.__call__ = original
    if "config" not in captured:
        raise RuntimeError("beaver run never reached the verifier; cannot capture its prompts")
    return captured["config"], captured["instances"]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", required=True,
                    help="dataset JSON, handed to the experiment's loader as --dataset_path")
    ap.add_argument("--model", required=True)
    ap.add_argument("--config", required=True,
                    help="the experiment YAML the verifier runs (e.g. experiments/enron/"
                         "enron_new.yaml). The instances and prompt settings are taken from "
                         "a real `beaver run` on it (verifier_view), so the prompt grouped "
                         "on is the prompt verified on.")
    ap.add_argument("--server-addr", default="http://localhost:8082",
                    help="a running beaver.hf_server (default port 8082). A "
                         "`vllm serve` works too -- same wire format -- but must be "
                         "started with --max-logprobs -1, and see the top_p note below.")
    ap.add_argument("--out", required=True, help="output dir")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--top-p", type=float, default=0.999,
                    help="resolved-mass target; 1-top_p bounds the TV bracket")
    ap.add_argument("--max-n", type=int, default=None,
                    help="FIRST n instances in file order. These datasets are stored "
                         "family-by-family, so a head slice is one family and every "
                         "pair is trivially close -- use --sample-n instead.")
    ap.add_argument("--sample-n", type=int, default=None,
                    help="random seeded sample of n instances, spread across families")
    ap.add_argument("--workers", type=int, default=16,
                    help="prompts in flight; >32 overloads one vLLM server")
    ap.add_argument("--batch-size", type=int, default=8,
                    help="frontier nodes per request (see enumerate_prefix_dist)")
    ap.add_argument("--max-fp", type=int, default=None,
                    help="cap forward passes per prompt. Cost knob only: the unreached "
                         "mass stays in m_p so the bracket remains valid, but prompts "
                         "whose m_p reaches the threshold cannot be grouped and become "
                         "singletons.")
    ap.add_argument("--from-dists", default=None,
                    help="reload a dists.pkl checkpoint and skip enumeration entirely")
    ap.add_argument("--num-logprobs", type=int, default=100,
                    help="children kept per expansion. Sets how "
                         "much mass one depth-k expansion can resolve, and the support "
                         "size that drives the O(n^2) pairwise stage.")
    ap.add_argument("--threshold", type=float, default=0.01,
                    help="rho: complete-linkage diameter cap for the grouping")
    ap.add_argument("--compare-linf", action="store_true",
                    help="also cluster the SAME enumeration under L_inf and report both")
    args = ap.parse_args(argv)

    import random

    from beaver.verifiers.worker_common import init_worker_state

    # The verifier's own view: its instances and prompt settings (see verifier_view).
    view_cfg, instances = verifier_view(args.config, args.model, args.dataset)
    if args.sample_n and args.max_n:
        raise SystemExit("pass at most one of --sample-n (random) or --max-n (head slice)")
    if args.sample_n and args.sample_n < len(instances):
        instances = random.Random(0).sample(instances, args.sample_n)
    elif args.max_n:
        instances = instances[: args.max_n]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Row i of every matrix below is instances[i]; tv_visualize must get this
    # file (not the source JSON) so rows and instances stay aligned.
    with open(out_dir / "instances.json", "w") as f:
        json.dump(instances, f)
    with open(out_dir / "prompt_config.json", "w") as f:
        json.dump({k: view_cfg[k] for k in PROMPT_KEYS}, f, indent=1)
    print(f"  {len(instances)} instances as the verifier sees them -> {out_dir/'instances.json'}")
    print("  prompt settings (from the verifier's worker config): "
          + ", ".join(f"{k}={view_cfg[k]!r}" for k in PROMPT_KEYS if k != "fewshot_messages")
          + f", fewshot_messages={len(view_cfg['fewshot_messages'] or [])} global", flush=True)

    cfg = {
        "model_name": args.model,
        "server_addr": args.server_addr,
        "ebnf": None,
        "dataset_name": view_cfg["dataset_name"],
        "use_grammar": False,
        # backend "hf" == beaver.hf_server; the wire format is identical to a
        # `vllm serve`, so http_backend serves both and nothing here changes.
        "backend": "hf",
        # These are *sampling* params only. Checked against hf_server.py: top_p is
        # applied in the sampling branch to choose sampled_ids, while the returned
        # distribution is topk_vals - logits.logsumexp(-1), i.e. the RAW full-vocab
        # conditionals, and top_k is not even a CompletionRequest field. So these
        # values do not touch the logprobs we build the distribution from. The only
        # truncation is `logprobs=num_logprobs` (top-N children per expansion), and
        # the tail it drops stays visible in 1 - resolved_mass, which is what the
        # bound needs. Against a real `vllm serve` re-check this before trusting it:
        # model/__init__.py notes scoring pins these to 1.0/1.0/-1 so logprobs are
        # raw, and if that server renormalises after truncation the discarded tail
        # is redistributed and m_p silently stops being valid.
        # MUST stay 1.0 / -1. These are the sampling params the request carries, and
        # a `vllm serve` started with --logprobs-mode processed_logprobs returns the
        # logprobs AFTER these processors run: top-p/top-k truncate and renormalise,
        # the discarded tail is redistributed over the survivors, resolved_mass reads
        # ~1.0 and m_p collapses to nothing -- silently voiding the bound the whole
        # method rests on. At 1.0 / -1 the processors are a no-op, so processed and
        # raw logprobs coincide and the conditionals are the model's own, which is
        # also what Definition 5.5 is stated over. (hf_server is immune either way:
        # it computes logprobs from the raw logsumexp and ignores top_k entirely.)
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": -1,
        "num_logprobs": args.num_logprobs,
    }
    cfg.update({k: view_cfg[k] for k in PROMPT_KEYS})
    init_worker_state(cfg)

    # Per-prompt logging as each finishes. Cost per prompt varies by orders of
    # magnitude -- a near-deterministic prompt resolves 0.999 in ~k passes while a
    # diffuse one expands a frontier that multiplies by num_logprobs per level
    # before ANY mass lands at depth k -- so the aggregate mean hides what is
    # actually happening. Watching the tail is the point.
    if args.from_dists:
        import pickle
        ck = pickle.load(open(args.from_dists, "rb"))
        results = [(r, m, f) for (r, m), f in zip(ck["dists"], ck["fp"])]
        t_enum = 0.0
        print(f"  loaded {len(results)} enumerated prompts from {args.from_dists}", flush=True)
        return _finish(args, instances, results, out_dir, 0.0)

    t0 = time.perf_counter()
    results = [None] * len(instances)
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(enumerate_prefix_dist, inst, args.k, args.top_p,
                            args.batch_size, args.max_fp): i
                for i, inst in enumerate(instances)}
        for fut in as_completed(futs):
            i = futs[fut]
            results[i] = fut.result()
            done += 1
            r, m, f = results[i]
            print(f"  [{done:>4}/{len(instances)}] prompt {i:>4}: {f:>6} fp, "
                  f"support {len(r):>6}, m_p {1-m:.2e}, {time.perf_counter()-t0:7.1f}s elapsed",
                  flush=True)
    t_enum = time.perf_counter() - t0

    dists = [(r, m) for r, m, _ in results]

    # Checkpoint before the O(n^2) stage: enumeration is the GPU-bound half and
    # must never have to be redone because the CPU-bound half failed. --from-dists
    # reloads this and skips straight to the matrix.
    import pickle
    with open(out_dir / "dists.pkl", "wb") as f:
        pickle.dump({"dists": dists, "fp": [f for _, _, f in results]}, f,
                    protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  checkpointed enumeration -> {out_dir/'dists.pkl'}", flush=True)

    return _finish(args, instances, results, out_dir, t_enum)


def _finish(args, instances, results, out_dir, t_enum):
    """Matrix + clustering + report. Split out so --from-dists can enter here."""
    dists = [(r, m) for r, m, _ in results]
    fp = np.array([f for _, _, f in results])
    unresolved = np.array([1.0 - m for _, m, _ in results])
    support = np.array([len(r) for r, _, _ in results])

    # Both partitions come from this one enumeration: "point" clusters on
    # TV_trunc (the original behaviour), "upper" on TV_trunc + 1/2(m_p + m_p'),
    # which certifies every group's TV@k diameter <= rho.
    t0 = time.perf_counter()
    D, D_hi = tv_matrices(dists, threshold=args.threshold)
    t_tv = time.perf_counter() - t0
    np.save(out_dir / "tv_prefix_distance.npy", D)
    np.save(out_dir / "tv_prefix_distance_upper.npy", D_hi)

    from beaver.grouping.tv_visualize import diameter_clusters

    t0 = time.perf_counter()
    labels = diameter_clusters(D, args.threshold)
    labels_hi = diameter_clusters(D_hi, args.threshold)
    t_clu = time.perf_counter() - t0
    np.save(out_dir / "tv_cluster_labels.npy", labels)
    np.save(out_dir / "tv_cluster_labels_upper.npy", labels_hi)

    def _summary(lab):
        sizes = np.bincount(lab)
        # Grouped prompts whose own m_p already spans rho: no pair involving
        # them can be certified close, so these groups are not certified.
        uncertified = int(sum(sizes[g] > 1 and (unresolved[lab == g] >= args.threshold).any()
                              for g in range(len(sizes))))
        return dict(groups=int(len(sizes)), singletons=int((sizes == 1).sum()),
                    mean_size=float(sizes.mean()), max_size=int(sizes.max()),
                    groups_with_mp_ge_rho=uncertified)

    summ = {"tv": _summary(labels), "tv_upper": _summary(labels_hi)}
    if args.compare_linf:
        # Same enumerated distributions, only the metric differs -- so any
        # difference in the partition is attributable to the metric alone.
        Dl = tv_matrix(dists, metric="linf", threshold=args.threshold)
        np.save(out_dir / "linf_prefix_distance.npy", Dl)
        summ["linf"] = _summary(diameter_clusters(Dl, args.threshold))

    n = len(instances)
    print(f"\nn={n}  k={args.k}  top_p={args.top_p}")
    print(f"  forward passes/prompt  mean {fp.mean():.1f}  median {np.median(fp):.0f}  "
          f"max {fp.max()}  total {fp.sum():,}")
    print(f"  support size/prompt    mean {support.mean():.1f}  median {np.median(support):.0f}  "
          f"max {support.max()}")
    short = int((unresolved > 1 - args.top_p + 1e-12).sum())
    undec = int((unresolved >= args.threshold).sum())
    print(f"  unresolved mass m_p    mean {unresolved.mean():.2e}  "
          f"median {np.median(unresolved):.2e}  max {unresolved.max():.2e}")
    print(f"                         {short} short of the {args.top_p} target; "
          f"{undec} with m_p >= rho={args.threshold} (undecidable: singletons under tv_upper only)")
    print(f"  enumeration {t_enum:.1f}s ({1000*t_enum/n:.1f} ms/prompt)   "
          f"TV matrix {t_tv:.1f}s   clustering {t_clu:.1f}s")
    off = D[np.triu_indices(n, k=1)] if n > 1 else np.array([0.0])
    print(f"  pairwise TV: mean {off.mean():.4f}  median {np.median(off):.4f}  "
          f"frac <= rho={args.threshold}: {float((off <= args.threshold).mean()):.4f}")
    print(f"\n  clustering at rho={args.threshold}")
    for name, sm in summ.items():
        print(f"    {name:>8}: {sm['groups']:>5} groups, {sm['singletons']:>5} singletons, "
              f"mean size {sm['mean_size']:.1f}, max {sm['max_size']}"
              + (f", {sm['groups_with_mp_ge_rho']} groups holding an m_p>=rho prompt"
                 if "groups_with_mp_ge_rho" in sm else ""))
    print(f"\n  wrote {out_dir/'tv_prefix_distance.npy'} (point) and "
          f"{out_dir/'tv_prefix_distance_upper.npy'} (certified upper); rows = {out_dir/'instances.json'}")

    json.dump({"n": n, "k": args.k, "top_p": args.top_p,
               "fp_per_prompt": fp.tolist(), "unresolved": unresolved.tolist(),
               "support": support.tolist(), "t_enum_s": t_enum, "t_tv_s": t_tv,
               "clustering": summ},
              open(out_dir / "prefix_report.json", "w"))


if __name__ == "__main__":
    main()
