"""Probe-based cheap ranking for the DistributionVerifier (HERETIC backend).

Drop-in alternative to the SpecVerifier ranking pass. Instead of surrogate-
sampling each prompt (expensive), this reads a single residual at the LAST PROMPT
TOKEN, projects it onto a saved "violation direction", and runs that projection
through a saved calibrator to get a CALIBRATED P(violation) estimate for the
prompt. No generation is sampled, so scoring the whole distribution is one
forward pass per prompt.

It scores with the SAME machinery the direction + calibrator were fit with:
``activation_detection.violation_direction_heretic.BeaverModel.get_residuals_batched``.
Using the identical backend means residuals match by construction -- no
cross-backend layer-shift to reconcile (heretic stacks hidden_states, so
L = n_layers+1 and layer 0 is the embedding; the saved layer index uses that same
convention).

Artifacts consumed (the run dir produced by violation_direction_heretic.py):
  * detector_decision.json -> risk_prior.layer  (the calibrated last-prompt layer)
  * directions.npz         -> dirs["risk_prior"] [L, d] unit vectors
  * calibrator.joblib      -> Calibrator mapping projection -> P(violation)
  * config.json            -> winsorization_quantile (matched so residuals agree)

The prompt string MUST be rendered exactly as the model saw it during the run the
direction was fit on (BEAVER's ``exact_prompt``); we reuse the chat templating for
that. ``violation_prob`` is the calibrated estimate; the raw projection is kept as
``risk`` (ranking is by ``violation_prob`` descending, identical order to ``risk``).
"""

import json
import time
from pathlib import Path

import numpy as np

from heretic.config import Settings
from beaver.activation_detection.violation_direction_heretic import BeaverModel, RawPrompt
from beaver.nvf import load_direction_stack
from beaver.utils import log_json
from beaver.verifiers.worker_common import build_prompt

# Prompts per heretic forward batch. Throughput/VRAM only -- residuals are
# identical at any batch size. Override per call via probe_rank(batch_size=...).
RANK_BATCH_SIZE = 32


def _load_artifacts(probe_dir):
    """Load (direction [d] unit, layer int, calibrator|None, winsorization float).

    Calibrator is optional: without it we still rank by the raw projection, but
    `violation_prob` is then NOT a probability (a warning is printed).
    """
    probe_dir = Path(probe_dir)
    decision_path = probe_dir / "detector_decision.json"
    if not decision_path.exists():
        raise FileNotFoundError(
            f"probe_dir {probe_dir} must contain detector_decision.json and "
            f"directions.npz (produced by violation_direction_heretic.py)"
        )

    decision = json.load(open(decision_path))
    if "risk_prior" not in decision:
        raise KeyError(
            f"detector_decision.json in {probe_dir} has no 'risk_prior' block; "
            f"re-run the heretic detection with 'risk_prior' in READ_POSITIONS"
        )
    layer = int(decision["risk_prior"]["layer"])

    stack = load_direction_stack(probe_dir, key="risk_prior")  # normalized [L, d]
    direction = stack[layer]

    # Calibrator is optional. Without it we run in RANK-ONLY mode: violation_prob
    # holds the raw projection. Since calibration is monotonic, the ranking ORDER
    # is identical either way -- the calibrator only makes the value a probability.
    calibrator = None
    cal_path = probe_dir / "calibrator.joblib"
    if cal_path.exists():
        import joblib

        calibrator = joblib.load(cal_path)
    else:
        print(f"probe_rank: rank-only (no calibrator in {probe_dir}); "
              f"violation_prob = raw projection, order is still correct")

    winsor = 1.0
    cfg_path = probe_dir / "config.json"
    if cfg_path.exists():
        winsor = float(json.load(open(cfg_path)).get("winsorization_quantile", 1.0))
    return direction, layer, calibrator, winsor


def _build_model(model_name, winsor, batch_size=RANK_BATCH_SIZE):
    """Heretic BeaverModel configured exactly as the direction-fitting run.

    Heretic's Settings hardcodes cli_parse_args=True, so constructing it reads
    sys.argv -- which under `beaver run` holds beaver's own flags and makes
    heretic's CLI parser abort. Neutralize argv for just this construction (kwargs
    still take effect; they are the highest-priority settings source).
    """
    import sys

    saved_argv = sys.argv
    sys.argv = [saved_argv[0]]
    try:
        settings = Settings(
            model=model_name,
            batch_size=batch_size,
            max_response_length=1,
            winsorization_quantile=winsor,
            offload_outputs_to_cpu=True,
        )
    finally:
        sys.argv = saved_argv
    return BeaverModel(settings)


def _score_prompts(model, prompts, layer):
    """Last-prompt-token residual at `layer` for each prompt -> feats [M, d].
    Uses heretic's get_residuals_batched (same code path as direction fitting)."""

    res = model.get_residuals_batched([RawPrompt(p) for p in prompts])  # [M, L, d]
    return res[:, layer, :].float().cpu().numpy()


def probe_rank(
    dataset,
    rank_log_dir,
    model_name,
    probe_dir,
    chat_mode=True,
    system_message=None,
    fewshot_messages=None,
    batch_size=RANK_BATCH_SIZE,
    **_,
):
    """Rank instances by calibrated P(violation) from the saved direction.

    Writes one ``{idx}.jsonl`` per instance into ``rank_log_dir`` plus a
    ``rank_cost.json`` (forward-token proxy + wall-clock), and returns a list of
    ``{"idx", "violation_prob", "risk"}`` dicts (descending order is the ranking).
    """

    rank_log_dir = Path(rank_log_dir)
    rank_log_dir.mkdir(parents=True, exist_ok=True)

    direction, layer, calibrator, winsor = _load_artifacts(probe_dir)
    model = _build_model(model_name, winsor, batch_size=batch_size)

    # Render each instance into the EXACT string the verifier feeds the model
    # (continuation=[] -> prompt only). Shared with the full-pass workers via
    # worker_common.build_prompt, with overrides passed since _w is unset here.
    prompts = [
        build_prompt(
            inst, [], tokenizer=model.tokenizer, chat_mode=chat_mode,
            system_message=system_message, fewshot_messages=fewshot_messages,
        )
        for inst in dataset
    ]

    t0 = time.time()
    feats = _score_prompts(model, prompts, layer)
    risk = feats @ direction  # [N] raw projection
    wall = time.time() - t0

    if calibrator is not None:
        violation_prob = np.asarray(calibrator.predict(risk), dtype=float)
    else:
        violation_prob = risk.astype(float)  # NOT a probability (no calibrator)

    # cost proxy: one prefill forward per prompt over the target model -> the
    # number of target-model token-forwards is the total prompt token count.
    tok_lens = [
        len(model.tokenizer(p, add_special_tokens=False)["input_ids"]) for p in prompts
    ]
    rank_cost = dict(
        method="probe",
        model=model_name,
        n_prompts=len(prompts),
        forward_tokens=int(sum(tok_lens)),       # target-model token-forwards (proxy)
        surrogate_forward_tokens=0,              # probe uses no surrogate
        wall_clock_s=float(wall),
        calibrated=calibrator is not None,
        probe_layer=layer,
    )
    with open(rank_log_dir / "rank_cost.json", "w") as f:
        json.dump(rank_cost, f, indent=2)

    results = []
    for inst, prompt, r, p in zip(dataset, prompts, risk, violation_prob):
        entry = {
            "idx": inst["idx"],
            "exact_prompt": prompt,
            "probe_layer": layer,
            "risk": float(r),
            "violation_prob": float(p),
        }
        log_json(entry, rank_log_dir / f"{inst['idx']}.jsonl")
        results.append({"idx": inst["idx"], "violation_prob": float(p), "risk": float(r)})

    print(
        f"probe_rank: {len(prompts)} prompts, calibrated={calibrator is not None}, "
        f"forward_tokens={rank_cost['forward_tokens']}, {wall:.1f}s"
    )
    return results
