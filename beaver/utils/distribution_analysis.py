"""Budget-vs-detection lift curves for distributional verification.

Compares verifying a prompt distribution in *ranked* order (riskiest first, via a
cheap ranker) against *unranked* (random) order, under a shared budget.

Core fact this exploits: every prompt is verified independently, so a prompt's
verification COST (number of target-LLM transitions) and its DETECTION label
(``upper_bound < risk_threshold`` => proven risky) are *order-independent*. Run
the FrontierVerifier once over the whole distribution, read each instance's
``(cost, detected)`` from its log, attach a ``rank_score``, and every ordering's
budget-vs-detection trajectory is then pure post-processing of that table.

Detection semantics (matches FrontierVerifier / summarize_log_data):
  ``upper_bound`` is an upper bound on the *constraint-satisfaction* probability.
  ``upper_bound < risk_threshold`` means even the best case is below the bar, i.e.
  the prompt is a PROVEN risk. FrontierVerifier early-terminates exactly there, so
  ranked and unranked runs pay identical per-instance cost — only the ORDER differs.

Curves produced:
  * ranked  — instances sorted by ``rank_score`` desc (the ranker under test).
  * random  — expected trajectory, averaged over ``n_shuffles`` random orders.
  * oracle  — detected-first, cheapest-among-detected first (best achievable order).

Each curve plots cumulative budget (x) vs cumulative proven-risky found (y). A
good cheap ranker pushes the ranked curve up toward the oracle and well above the
random baseline.

CLI::

    python -m beaver.distribution_analysis FULL_LOG_DIR \
        --rank-log-dir FULL_LOG_DIR/rank --threshold 0.9 --out curve.png
"""

import argparse
import json
from pathlib import Path

import numpy as np

from beaver.logging import get_log_data, read_log_file


def build_instance_table(full_log_dir, risk_threshold, rank_scores=None, prompt_tokens=None):
    """Per-instance ``(cost, detected, rank_score)`` from a full frontier run.

    Args:
        full_log_dir: dir with one ``{idx}.jsonl`` per instance (FrontierVerifier
            output over the whole distribution).
        risk_threshold: an instance is ``detected`` (proven risky) when its final
            ``upper_bound < risk_threshold``.
        rank_scores: optional ``{idx: score}``. Missing/None -> 0.0 (sorts last).
        prompt_tokens: optional ``{idx: n_tokens}``. When given, each instance's
            cost includes the prefill (prompt token-forwards) on top of the decode
            steps, so cost is in the same TARGET-token-forward unit as ranker cost.
            Omit to keep the legacy decode-only (transition+1) unit.

    Returns:
        ``{idx (int): {"cost", "detected", "rank_score", "upper_bound"}}``.
        Instances with no transition data are skipped (no cost incurred).
    """
    rank_scores = rank_scores or {}
    prompt_tokens = prompt_tokens or {}
    all_data = get_log_data(Path(full_log_dir))
    table = {}
    for inst_id, entries in all_data.items():
        transitions = [e for e in entries if "transition" in e]
        if not transitions:
            continue
        last = transitions[-1]
        idx = int(inst_id)
        ub = last.get("upper_bound", 1.0)
        table[idx] = {
            # decode steps (transition is 0-indexed; +1 steps) + prefill prompt tokens.
            "cost": last["transition"] + 1 + int(prompt_tokens.get(idx, 0)),
            "detected": ub < risk_threshold,
            "upper_bound": ub,
            "rank_score": float(rank_scores.get(idx, 0.0)),
        }
    return table


def load_rank_scores(rank_log_dir):
    """Read ``{idx: violation_prob}`` from a ranking-pass log dir.

    Works for both the SpecVerifier and probe rankers — both write one
    ``{idx}.jsonl`` per instance carrying a ``violation_prob`` (probe also writes
    ``risk``). Higher = riskier.
    """
    rank_log_dir = Path(rank_log_dir)
    scores = {}
    for fp in rank_log_dir.glob("*.jsonl"):
        if not fp.name.split(".")[0].isdigit():
            continue
        entries = read_log_file(str(fp))
        last = entries[-1]
        score = last.get("violation_prob", last.get("risk", 0.0))
        scores[int(fp.name.split(".")[0])] = float(score)
    return scores


# ─────────────────────────── cost accounting ───────────────────────────────
# Common unit: TARGET-MODEL token-forwards (one forward over T tokens = T). This
# is the same unit the ranked curve's x-axis uses, so a ranker's cost can be
# prepended to the budget axis for a fair no-rank / speculative / probe compare.
#   full verify of prompt i  : prompt_tokens_i (prefill) + (transition_i+1) (decode)
#   probe rank               : sum of prompt_tokens (one prefill per prompt)
#   speculative rank         : target scoring tokens (+ surrogate tokens, weighted)

_TOKENIZER_CACHE = {}


def _get_tokenizer(model_name):
    if model_name not in _TOKENIZER_CACHE:
        from transformers import AutoTokenizer

        _TOKENIZER_CACHE[model_name] = AutoTokenizer.from_pretrained(model_name)
    return _TOKENIZER_CACHE[model_name]


def _model_name_for(log_dir):
    ra = Path(log_dir) / "run_args.json"
    if ra.exists():
        return json.load(open(ra)).get("model")
    return None


def instance_prompt_tokens(log_dir, model_name=None):
    """{idx: prompt_token_count} from each instance's ``exact_prompt``.

    Reads the per-instance logs in ``log_dir`` (rank or full pass — both carry
    ``exact_prompt``) and tokenizes with the run's model. Returns {} if the model
    or tokenizer can't be resolved (callers then treat prefill cost as 0).
    """
    model_name = model_name or _model_name_for(log_dir)
    if model_name is None:
        return {}
    try:
        tok = _get_tokenizer(model_name)
    except Exception as e:  # offline / missing model — degrade to decode-only cost
        print(f"[warn] no tokenizer for {model_name} ({e}); prefill cost = 0")
        return {}
    out = {}
    for fp in Path(log_dir).glob("*.jsonl"):
        if not fp.name.split(".")[0].isdigit():
            continue
        entries = read_log_file(str(fp))
        ep = next((e["exact_prompt"] for e in reversed(entries) if e.get("exact_prompt")), None)
        if ep is None:
            continue
        out[int(fp.name.split(".")[0])] = len(tok(ep, add_special_tokens=False)["input_ids"])
    return out


def spec_rank_cost_from_logs(rank_log_dir, model_name=None, surrogate_max_tokens=64):
    """Speculative-ranker cost (target-token-forward proxy) from its rank logs.

    Per instance: surrogate samples n_samples continuations of
    ~surrogate_max_tokens tokens (prefill prompt_tokens + K decode each) on
    the SURROGATE model; the target then scores the minimal violating
    prefixes (``per_token_logprobs`` lengths).

    Returns {"method","forward_tokens","surrogate_forward_tokens","n_prompts"}:
      forward_tokens           = TARGET token-forwards (scoring)
      surrogate_forward_tokens = SURROGATE token-forwards (sampling)
    """
    rank_log_dir = Path(rank_log_dir)
    model_name = model_name or _model_name_for(rank_log_dir)
    ptoks = instance_prompt_tokens(rank_log_dir, model_name)
    target = surrogate = 0
    n = 0
    for fp in rank_log_dir.glob("*.jsonl"):
        if not fp.name.split(".")[0].isdigit():
            continue
        last = read_log_file(str(fp))[-1]
        idx = int(fp.name.split(".")[0])
        n += 1
        n_samples = last.get("n_samples", len(last.get("samples", [])))
        pt = ptoks.get(idx, 0)
        surrogate += n_samples * (pt + surrogate_max_tokens)
        target += sum(len(lp) for lp in last.get("per_token_logprobs", []))
    return dict(
        method="speculative",
        forward_tokens=int(target),
        surrogate_forward_tokens=int(surrogate),
        n_prompts=n,
    )


def load_rank_cost(rank_log_dir, model_name=None, surrogate_weight=1.0):
    """Ranker cost as a single TARGET-token-forward offset for the budget axis.

    Prefers a ``rank_cost.json`` (written by probe_rank); else infers a
    speculative cost from the logs; else 0 (no ranker). Surrogate forwards are
    folded in via ``surrogate_weight`` (1.0 = count a surrogate token like a
    target token; set to surrogate_params/target_params for a FLOP-fair number).

    Returns (offset_forward_tokens, detail_dict).
    """
    rank_log_dir = Path(rank_log_dir)
    rc_path = rank_log_dir / "rank_cost.json"
    if rc_path.exists():
        rc = json.load(open(rc_path))
    elif any(rank_log_dir.glob("*.jsonl")):
        rc = spec_rank_cost_from_logs(rank_log_dir, model_name)
    else:
        return 0.0, {"method": "none", "forward_tokens": 0, "surrogate_forward_tokens": 0}
    offset = float(rc.get("forward_tokens", 0)) + surrogate_weight * float(
        rc.get("surrogate_forward_tokens", 0)
    )
    rc["surrogate_weight"] = surrogate_weight
    rc["offset_forward_tokens"] = float(offset)
    return float(offset), rc


def lift_curve(table, order):
    """Cumulative (budget, detected) trajectory for a given processing order.

    Args:
        table: output of :func:`build_instance_table`.
        order: iterable of instance idx in processing order.

    Returns:
        ``(xs, ys)`` numpy arrays, both starting at 0, where ``xs`` is cumulative
        budget and ``ys`` is cumulative count of proven-risky instances.
    """
    xs, ys = [0], [0]
    b = d = 0
    for idx in order:
        rec = table[idx]
        b += rec["cost"]
        d += int(rec["detected"])
        xs.append(b)
        ys.append(d)
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


def ranked_order(table):
    """Instance idx by ``rank_score`` descending; ties broken by idx ascending.

    With no ranker (all scores 0.0) this collapses to idx order — the natural
    order instances were computed in, not a reshuffle.
    """
    return sorted(table, key=lambda i: (-table[i]["rank_score"], i))


def oracle_order(table):
    """Best achievable order: detected first, cheapest-among-detected first.

    Front-loads every detection while spending the least budget to do so, giving
    the upper-envelope curve any ranker is measured against.
    """
    return sorted(table, key=lambda i: (not table[i]["detected"], table[i]["cost"]))


def random_curve(table, n_shuffles=200, n_grid=200, seed=0):
    """Expected lift curve under random order, averaged over shuffles.

    Returns ``(x_grid, y_mean)`` on a common budget grid so it can be plotted
    against the ranked/oracle curves directly.
    """
    rng = np.random.default_rng(seed)
    idxs = list(table)
    total_budget = sum(r["cost"] for r in table.values())
    x_grid = np.linspace(0, total_budget, n_grid)
    acc = np.zeros(n_grid)
    for _ in range(n_shuffles):
        rng.shuffle(idxs)
        xs, ys = lift_curve(table, idxs)
        # step function: detections found at-or-before each grid budget.
        acc += np.interp(x_grid, xs, ys)
    return x_grid, acc / n_shuffles


def budget_to_fraction(xs, ys, frac):
    """Budget at which a ``frac`` fraction of all detections is reached."""
    total = ys[-1]
    if total == 0:
        return float("nan")
    target = frac * total
    j = int(np.searchsorted(ys, target))
    return float(xs[min(j, len(xs) - 1)])


def curve_metrics(table, name="ranked"):
    """Metrics for one log's ranked curve (vs its own oracle). No plotting."""
    _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))  # numpy>=2 renamed it

    def _auc(x, y):
        return float(_trapz(y, x))

    rx, ry = lift_curve(table, ranked_order(table))
    ox, oy = lift_curve(table, oracle_order(table))
    oracle_auc = _auc(ox, oy) or 1.0
    return {
        "name": name,
        "n_instances": len(table),
        "total_budget": int(sum(r["cost"] for r in table.values())),
        "total_detected": int(sum(int(r["detected"]) for r in table.values())),
        "auc_ratio": _auc(rx, ry) / oracle_auc,
        "budget_to_90pct": budget_to_fraction(rx, ry, 0.9),
        "budget_to_90pct_oracle": budget_to_fraction(ox, oy, 0.9),
    }


def plot_budget_detection(
    entries,
    out_path,
    risk_threshold=None,
    show_oracle=True,
):
    """Overlay one ranked budget-detection curve per log; save to ``out_path``.

    Args:
        entries: list of ``{"name": str, "table": dict}`` (one per log dir).
        show_oracle: also draw each log's oracle (best-order) curve, dashed in the
            same colour, as that log's upper envelope.

    Returns:
        list of per-entry metrics dicts (see :func:`curve_metrics`).
    """
    import matplotlib.pyplot as plt

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, ax = plt.subplots(figsize=(9, 6))
    metrics = []
    for i, e in enumerate(entries):
        table, name = e["table"], e["name"]
        color = colors[i % len(colors)]
        rx, ry = lift_curve(table, ranked_order(table))
        ax.plot(rx, ry, color=color, lw=2.5, label=name)
        if show_oracle:
            ox, oy = lift_curve(table, oracle_order(table))
            ax.plot(ox, oy, color=color, ls="--", lw=1.5, alpha=0.6,
                    label=f"{name} — oracle")
        metrics.append(curve_metrics(table, name))

    ax.set_xlabel("Cumulative budget (target-LLM transitions)")
    ax.set_ylabel("Proven-risky instances detected")
    title = "Budget vs detections — distributional verification"
    if risk_threshold is not None:
        title += f"  (risk_threshold={risk_threshold})"
    ax.set_title(title)
    ax.legend(loc="lower right", frameon=True)
    ax.grid(True, alpha=0.3, ls="--")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return metrics


def cost_comparison(
    full_log_dir, strategies, risk_threshold, surrogate_weight=1.0, out_path=None
):
    """Fair no-rank / speculative / probe comparison on ONE full run.

    Per-instance verify cost + detection are order-independent, so a single full
    frontier run over the whole distribution gives the shared table; each ranking
    strategy only changes the ORDER prompts are verified and adds its own up-front
    ranker cost (the x-offset). All costs are TARGET-token-forwards.

    Args:
        full_log_dir: the frontier run over the whole distribution.
        strategies: list of {"name", "rank_log_dir"}. "rank_log_dir" None or a dir
            with no scores -> no-rank (natural order, zero ranker cost).
        surrogate_weight: how to count a surrogate token-forward vs a target one
            (1.0 = equal; surrogate_params/target_params for FLOP-fair).
        out_path: PNG to write; metrics json written alongside.

    Returns the per-strategy metrics list and the shared table.
    """
    model_name = _model_name_for(full_log_dir)
    ptoks = instance_prompt_tokens(full_log_dir, model_name)
    table = build_instance_table(full_log_dir, risk_threshold, prompt_tokens=ptoks)
    if not table:
        raise SystemExit(f"No instances with transition data in {full_log_dir}")

    metrics = []
    curves = []  # (name, xs, ys, offset, detail)
    for strat in strategies:
        name = strat["name"]
        rd = strat.get("rank_log_dir")
        scores = load_rank_scores(Path(rd)) if rd and Path(rd).exists() else {}
        offset, detail = (
            load_rank_cost(rd, model_name, surrogate_weight)
            if rd and Path(rd).exists()
            else (0.0, {"method": "none", "forward_tokens": 0, "surrogate_forward_tokens": 0})
        )
        order = sorted(table, key=lambda i: (-scores.get(i, 0.0), i))
        xs, ys = lift_curve(table, order)
        xs = np.concatenate([[0.0], xs + offset])  # flat ramp 0->offset at y=0
        ys = np.concatenate([[0.0], ys])
        curves.append((name, xs, ys, offset, detail))
        metrics.append(
            dict(
                name=name,
                rank_offset=float(offset),
                rank_detail=detail,
                total_detected=int(ys[-1]),
                total_cost=float(xs[-1]),  # ranker + full verify of everything
                budget_to_90pct=budget_to_fraction(xs, ys, 0.9),
            )
        )

    # oracle envelope (no ranker cost, best achievable order)
    ox, oy = lift_curve(table, oracle_order(table))

    if out_path is not None:
        import matplotlib.pyplot as plt

        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        fig, ax = plt.subplots(figsize=(9, 6))
        for i, (name, xs, ys, offset, _) in enumerate(curves):
            lbl = name + (f"  (+{int(offset)} rank)" if offset else "")
            ax.plot(xs, ys, color=colors[i % len(colors)], lw=2.5, label=lbl)
        ax.plot(ox, oy, color="k", ls="--", lw=1.5, alpha=0.6, label="oracle (no rank cost)")
        ax.set_xlabel("Cumulative budget — TARGET-model token-forwards (incl. ranker)")
        ax.set_ylabel("Proven-risky instances detected")
        ax.set_title(
            f"Cost comparison — no-rank / speculative / probe  (risk<{risk_threshold})"
        )
        ax.legend(loc="lower right", frameon=True)
        ax.grid(True, alpha=0.3, ls="--")
        fig.tight_layout()
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        with open(Path(out_path).with_suffix(".metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"[ok] cost comparison -> {out_path}")
    return metrics, table


def _main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("full_log_dir", nargs="+", help="One or more frontier runs to overlay")
    ap.add_argument("--rank-log-dir", nargs="*", default=None,
                    help="Ranking-pass dir per log (defaults to each LOG/rank). "
                         "Give one per log dir, in the same order.")
    ap.add_argument("--names", nargs="*", default=None, help="Legend label per log dir")
    ap.add_argument("--threshold", type=float, required=True, help="risk_threshold; detected = upper_bound < threshold")
    ap.add_argument("--out", default=None, help="Output PNG (defaults to <first log>/budget_detection.png)")
    ap.add_argument("--no-oracle", action="store_true", help="Hide the per-log oracle envelope")
    ap.add_argument(
        "--cost-compare", action="store_true",
        help="3-way no-rank/spec/probe comparison on ONE full run. full_log_dir = the "
             "full frontier run; --rank-log-dir lists the strategy rank dirs; --names "
             "labels them. A no-rank baseline is always added.",
    )
    ap.add_argument(
        "--surrogate-weight", type=float, default=1.0,
        help="cost-compare: surrogate token-forward weight vs target (param ratio for FLOP-fair)",
    )
    args = ap.parse_args()

    if args.cost_compare:
        full = Path(args.full_log_dir[0])
        rdirs = args.rank_log_dir or []
        rnames = args.names or [Path(r).name for r in rdirs]
        if len(rnames) != len(rdirs):
            raise SystemExit("--names count must match --rank-log-dir count")
        strategies = [{"name": "no-rank", "rank_log_dir": None}]
        strategies += [{"name": n, "rank_log_dir": r} for n, r in zip(rnames, rdirs)]
        out_path = Path(args.out) if args.out else full / "cost_comparison.png"
        metrics, _ = cost_comparison(
            full, strategies, args.threshold,
            surrogate_weight=args.surrogate_weight, out_path=out_path,
        )
        print(json.dumps(metrics, indent=2))
        return

    log_dirs = [Path(p) for p in args.full_log_dir]
    names = args.names or [p.name for p in log_dirs]
    if len(names) != len(log_dirs):
        raise SystemExit("--names count must match number of log dirs")
    rank_dirs = (
        [Path(p) for p in args.rank_log_dir]
        if args.rank_log_dir
        else [d / "rank" for d in log_dirs]
    )
    if len(rank_dirs) != len(log_dirs):
        raise SystemExit("--rank-log-dir count must match number of log dirs")

    entries = []
    for d, name, rd in zip(log_dirs, names, rank_dirs):
        scores = load_rank_scores(rd) if rd.exists() else {}
        if not scores:
            print(f"[warn] no rank scores for {name} ({rd}); using natural compute order")
        table = build_instance_table(d, args.threshold, scores)
        if not table:
            raise SystemExit(f"No instances with transition data in {d}")
        entries.append({"name": name, "table": table})

    out_path = Path(args.out) if args.out else log_dirs[0] / "budget_detection.png"
    metrics = plot_budget_detection(
        entries, out_path, risk_threshold=args.threshold, show_oracle=not args.no_oracle,
    )
    with open(out_path.with_suffix(".metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[ok] plot -> {out_path}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    _main()
