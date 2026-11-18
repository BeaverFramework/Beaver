import json

from tqdm import tqdm
from beaver.verifiers.base_verifier import BaseVerifier
from beaver.verifiers.frontier_verifier import FrontierVerifier


class DistributionVerifier(BaseVerifier):
    """Budget-aware orchestrator: rank prompts cheaply, then full-verify the riskiest."""

    def __init__(self, model, dataset, server_addr, **kwargs):
        super().__init__(model, dataset, server_addr, **kwargs)

        # Budget config. Risk threshold reuses BaseVerifier's
        # `early_termination_threshold` (set in super().__init__), so no new arg.
        self.budget = kwargs.get("budget", 5000)  # total full-verifier transitions
        if self.early_termination_threshold is None:
            raise ValueError(
                "DistributionVerifier requires early_termination_threshold "
                "(the risk threshold) to be set."
            )

        # Frontier knobs forwarded to the full-pass FrontierVerifier
        self.frontier_topp = kwargs.get("max_frontier_prob", 1.0)
        self.frontier_topk = kwargs.get("max_frontier_size", -1)
        self.frontier_scoring_strategy = kwargs.get(
            "frontier_scoring_strategy", "max-mu"
        )

        # Ranking pass: project the last-prompt residual onto a saved violation
        # direction (cheap, no surrogate needed). Requires probe_path.
        self.ranker = kwargs.get("ranker", "probe")
        self.probe_dir = kwargs.get("probe_path", None)
        self.probe = self.probe_dir is not None

        # Surrogate config (unused by the probe ranker)
        self.surrogate_server_addr = kwargs.get("surrogate_server_addr")
        self.surrogate_model = kwargs.get("surrogate_model")

        if self.probe:
            if self.probe_dir is None:
                raise ValueError(
                    "DistributionVerifier(probe=True) requires probe_dir pointing to "
                    "the saved violation direction (detector_decision.json + "
                    "directions.npz from violation_direction_heretic.py)."
                )
        else:
            # Only the probe ranker ships in this release; the surrogate
            # (speculative) ranking pass is not included.
            raise ValueError(
                "DistributionVerifier requires probe_path: a directory with the "
                "saved violation direction (detector_decision.json + "
                "directions.npz from violation_direction_heretic.py)."
            )

        # Keep raw kwargs so common BaseVerifier settings can be re-forwarded to the
        # sub-verifiers (they are constructed identically to api._run_inner).
        self._kwargs = kwargs

    def _common(self):
        """Shared BaseVerifier kwargs to forward to the sub-verifiers."""
        return dict(
            grammar=self.grammar,
            gpu_visible_devices=self.gpu_visible_devices,
            gen_length=self.gen_length,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            semantic_symbol=self.semantic_symbol,
            max_iterations=self.max_iterations,
            epsilon=self.epsilon,
            verbose=self.verbose,
            num_logprobs=self.num_logprobs,
            use_grammar=self.use_grammar,
            chat_mode=self.chat_mode,
            system_message=self.system_message,
            fewshot_messages=self.fewshot_messages,
            use_cache=self.use_cache,
        )

    def __call__(self, dataset, run_log_dir):
        dataset = list(dataset)

        # ── Phase 1: cheap ranking ──────────────────────────────────────────
        # Both strategies write per-instance logs into run_log_dir / "rank" and
        # return a list of {"idx", "violation_prob", ...} dicts.
        rank_log_dir = run_log_dir / "rank"
        rank_log_dir.mkdir(parents=True, exist_ok=True)

        if self.ranker == "probe":
            from beaver.verifiers.probe_ranker import probe_rank

            rank_results = probe_rank(
                dataset,
                rank_log_dir,
                model_name=self.model_name,
                probe_dir=self.probe_dir,
                chat_mode=self.chat_mode,
                system_message=self.system_message,
                fewshot_messages=self.fewshot_messages,
                batch_size=self._kwargs.get("probe_batch_size", 32),
            )

        order = sorted(
            rank_results,
            key=lambda r: (r.get("violation_prob", 0.0) if r else 0.0),
            reverse=True,
        )

        # ── Phase 2: full pass top-down until budget exhausted ──────────────
        idx_to_instance = {inst["idx"]: inst for inst in dataset}
        full = FrontierVerifier(
            self.model_name,
            self.dataset_name,
            self.server_addr,
            max_frontier_size=self.frontier_topk,
            max_frontier_prob=self.frontier_topp,
            frontier_scoring_strategy=self.frontier_scoring_strategy,
            # Stop a prompt the instant it is proven risky (upper bound on the
            # satisfaction prob drops below the threshold) — no more budget needed.
            early_termination_threshold=self.early_termination_threshold,
            max_workers=self.max_workers,
            **self._common(),
        )

        batch_size = self.max_workers

        # Build the ordered list of instances to verify (skip missing).
        ordered_insts = []
        rank_prob_by_idx = {}
        for ranked in order:
            if ranked is None:
                continue
            inst = idx_to_instance.get(ranked["idx"])
            if inst is None:
                continue
            ordered_insts.append(inst)
            rank_prob_by_idx[ranked["idx"]] = ranked.get("violation_prob", 0.0)

        spent = 0
        results = []
        exceeders = []
        # Cumulative (budget, detections) trajectory in ranked processing order —
        # the ranked curve for beaver.distribution_analysis.
        trajectory = [{"spent": 0, "n_detected": 0, "idx": None}]
        bar = tqdm(total=len(ordered_insts), desc="Budget search")
        for start in range(0, len(ordered_insts), batch_size):
            if spent >= self.budget:
                break
            batch = ordered_insts[start : start + batch_size]

            # FrontierVerifier returns a list of per-instance result dicts.
            batch_res = full(batch, run_log_dir)
            for res in batch_res:
                if res is None:
                    continue

                res["rank_violation_prob"] = rank_prob_by_idx.get(res["idx"], 0.0)
                spent += res.get("transition", 0) + 1

                is_exceeder = (
                    res.get("upper_bound", 1.0) < self.early_termination_threshold
                )
                res["is_threshold_exceeder"] = is_exceeder
                if is_exceeder:
                    exceeders.append(res["idx"])

                results.append(res)
                trajectory.append(
                    {"spent": spent, "n_detected": len(exceeders), "idx": res["idx"]}
                )

            bar.update(len(batch))
            remaining = max(self.budget - spent, 0)
            bar.set_postfix_str(
                f"budget {spent}/{self.budget} "
                f"({100 * spent / self.budget:.0f}%), "
                f"~{remaining} left, "
                f"verified {len(results)}/{len(ordered_insts)}, "
                f"exceeders {len(exceeders)}"
            )
        bar.close()

        summary = {
            "budget": self.budget,
            "spent": spent,
            "n_ranked": len(order),
            "n_verified": len(results),
            "risk_threshold": self.early_termination_threshold,
            "n_exceeders": len(exceeders),
            "exceeders": exceeders,
        }
        with open(run_log_dir / "distribution_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        with open(run_log_dir / "distribution_trajectory.json", "w") as f:
            json.dump(trajectory, f, indent=2)

        return results
